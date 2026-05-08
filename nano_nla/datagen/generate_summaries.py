"""Stage 2: generate warm-start explanations with local or Groq providers.

The NLA warm-start stage needs short natural-language descriptions for each
activation row. This implementation keeps the previous strict tag parsing and
crash-safe chunk resume behavior while allowing either local Transformers
generation or the optional Groq API provider.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Protocol

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from nano_nla.models import enable_cuda_performance, resolve_torch_device, resolve_torch_dtype
from nano_nla.schema import EXPLANATION_RE, load_config
from nano_nla.training.common import ensure_pad_token

_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*+]|[0-9]+[.)]|[a-zA-Z][.)])\s+")
_BOLD_WRAP_RE = re.compile(r"^\*\*(.+?)\*\*\s*")
_MIN_FEATURES = 2
DEFAULT_SUMMARY_MODEL = {
    "provider": "groq",
    "local": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "device": "auto",
        "dtype": "auto",
        "batch_size": 8,
        "chunk_size": 128,
        "max_new_tokens": 300,
        "max_input_chars": 2000,
        "temperature": 0.3,
        "top_p": 0.9,
    },
    "groq": {
        "model": "qwen/qwen3-32b",
        "max_tokens": 300,
        "temperature": 0.7,
        "requests_per_minute": 30,
        "max_retries": 5,
        "retry_base_delay": 2.0,
        "retry_max_delay": 60.0,
        "batch_size": 1,
        "chunk_size": 10,
        "max_input_chars": 2000,
    },
}


def provider_options(summary_config: dict, provider: str) -> dict:
    nested = summary_config.get(provider, {})
    return {**summary_config, **nested}


def _base_config_candidates(config_path: Path) -> list[Path]:
    stem = config_path.stem
    if stem.endswith("_computed"):
        stem = stem[: -len("_computed")]
    return [
        config_path.with_name(f"{stem}.yaml"),
        Path("configs") / f"{stem}.yaml",
    ]


def resolve_summary_config(config: dict, config_path: str | Path) -> dict:
    datagen_cfg = config["datagen"]
    if "summary_model" in datagen_cfg:
        return datagen_cfg["summary_model"]

    current_path = Path(config_path)
    summary_cfg = None
    for candidate in _base_config_candidates(current_path):
        if not candidate.exists() or candidate.resolve() == current_path.resolve():
            continue
        candidate_cfg = load_config(candidate)
        summary_cfg = candidate_cfg.get("datagen", {}).get("summary_model")
        if summary_cfg is not None:
            print(f"[config] Loaded missing summary_model from {candidate}")
            break

    if summary_cfg is None:
        summary_cfg = dict(DEFAULT_SUMMARY_MODEL)
        print("[config] datagen.summary_model missing; using default Groq Qwen3 summary model")

    datagen_cfg["summary_model"] = summary_cfg
    if current_path.exists():
        current_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"[config] Patched config with datagen.summary_model: {current_path}")
    return summary_cfg


class SummaryGenerator(Protocol):
    batch_size: int
    max_input_chars: int
    total_rows: int
    total_batches: int

    def complete_batch(self, system_prompt: str, user_prompts: list[str]) -> list[str | None]:
        ...


class LocalSummaryGenerator:
    """Batched local teacher model for activation explanation rows."""

    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        dtype: str,
        batch_size: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        max_input_chars: int,
    ) -> None:
        device_value = str(device or "auto").lower()
        self.device = resolve_torch_device(device_value, require_cuda=device_value in {"auto", "gpu"})
        self.dtype = resolve_torch_dtype(dtype, device=self.device)
        self.batch_size = max(1, int(batch_size))
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_input_chars = int(max_input_chars)
        self.total_rows = 0
        self.total_batches = 0

        if self.device.type == "cuda":
            enable_cuda_performance()
        print(f"[summary] Loading local teacher {model_name} on {self.device} ({self.dtype})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        ensure_pad_token(self.tokenizer)
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    def _format_prompt(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"System:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n"

    @torch.inference_mode()
    def complete_batch(self, system_prompt: str, user_prompts: list[str]) -> list[str | None]:
        prompts = [self._format_prompt(system_prompt, prompt) for prompt in user_prompts]
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        do_sample = self.temperature > 0.0
        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = self.temperature
            generate_kwargs["top_p"] = self.top_p
        output_ids = self.model.generate(**encoded, **generate_kwargs)
        generated = output_ids[:, encoded["input_ids"].shape[1] :]
        self.total_rows += len(user_prompts)
        self.total_batches += 1
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)


class RateLimiter:
    """Sliding-window request limiter for hosted provider RPM caps."""

    def __init__(self, requests_per_minute: int = 30) -> None:
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60.0
        self.timestamps: deque[float] = deque()

    def wait(self) -> float:
        now = time.time()
        while self.timestamps and self.timestamps[0] < now - self.window_seconds:
            self.timestamps.popleft()
        if len(self.timestamps) < self.requests_per_minute:
            return 0.0
        sleep_for = max(0.0, self.timestamps[0] + self.window_seconds - now + 0.1)
        time.sleep(sleep_for)
        return sleep_for

    def record(self) -> None:
        self.timestamps.append(time.time())


class GroqSummaryGenerator:
    """Optional Groq provider; used only when summary_model.provider is groq."""

    def __init__(
        self,
        *,
        model: str,
        max_tokens: int,
        temperature: float,
        requests_per_minute: int,
        max_retries: int,
        retry_base_delay: float,
        retry_max_delay: float,
        batch_size: int,
        max_input_chars: int,
    ) -> None:
        try:
            from groq import Groq
        except ImportError as exc:
            raise RuntimeError("Groq provider selected; install it with `uv sync --extra groq`.") from exc

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("Groq provider selected, but GROQ_API_KEY is not set")

        self.client = Groq(api_key=api_key)
        self.model = model
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.max_retries = int(max_retries)
        self.retry_base_delay = float(retry_base_delay)
        self.retry_max_delay = float(retry_max_delay)
        self.batch_size = max(1, int(batch_size))
        self.max_input_chars = int(max_input_chars)
        self.limiter = RateLimiter(int(requests_per_minute))
        self.total_rows = 0
        self.total_batches = 0
        self.total_retries = 0
        self.total_wait = 0.0
        print(f"[summary] Using Groq provider model={self.model}")

    def _complete_one(self, system_prompt: str, user_prompt: str) -> str | None:
        for attempt in range(self.max_retries):
            self.total_wait += self.limiter.wait()
            try:
                self.limiter.record()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                content = response.choices[0].message.content
                if content and content.strip():
                    return content.strip()
            except Exception as exc:
                self.total_retries += 1
                msg = str(exc)
                extra = 5.0 if "429" in msg or "rate" in msg.lower() else 0.0
                delay = min(self.retry_base_delay * (2**attempt) + extra, self.retry_max_delay)
                print(f"[groq] retry {attempt + 1}/{self.max_retries} after {delay:.1f}s: {msg[:120]}")
                time.sleep(delay)
        return None

    def complete_batch(self, system_prompt: str, user_prompts: list[str]) -> list[str | None]:
        self.total_batches += 1
        self.total_rows += len(user_prompts)
        return [self._complete_one(system_prompt, prompt) for prompt in user_prompts]


def build_summary_generator(summary_config: dict) -> SummaryGenerator:
    provider = str(summary_config.get("provider", "local")).lower()
    options = provider_options(summary_config, provider)
    if provider == "local":
        return LocalSummaryGenerator(
            model_name=options.get("model", options.get("name", "Qwen/Qwen2.5-7B-Instruct")),
            device=options.get("device", "auto"),
            dtype=options.get("dtype", "auto"),
            batch_size=int(options.get("batch_size", 8)),
            max_new_tokens=int(options.get("max_new_tokens", options.get("max_tokens", 300))),
            temperature=float(options.get("temperature", 0.3)),
            top_p=float(options.get("top_p", 0.9)),
            max_input_chars=int(options.get("max_input_chars", 2000)),
        )
    if provider == "groq":
        return GroqSummaryGenerator(
            model=options.get("model", "qwen/qwen3-32b"),
            max_tokens=int(options.get("max_tokens", options.get("max_new_tokens", 300))),
            temperature=float(options.get("temperature", 0.7)),
            requests_per_minute=int(options.get("requests_per_minute", 30)),
            max_retries=int(options.get("max_retries", 5)),
            retry_base_delay=float(options.get("retry_base_delay", 2.0)),
            retry_max_delay=float(options.get("retry_max_delay", 60.0)),
            batch_size=int(options.get("batch_size", 1)),
            max_input_chars=int(options.get("max_input_chars", 2000)),
        )
    raise ValueError(f"unsupported summary provider: {provider}")


def _tagged_user_prompt(template: str, text: str) -> str:
    return (
        template.format(text=text)
        + "\n\nReturn exactly 2-3 concise features inside tags:\n"
        + "<explanation>\n"
        + "first feature\n\nsecond feature\n\nfinal feature about the last token and likely continuation\n"
        + "</explanation>"
    )


def extract_and_clean_explanation(raw: str | None) -> str | None:
    """Extract explanation tags, strip list markers, require at least two features."""
    if raw is None:
        return None
    match = EXPLANATION_RE.search(raw)
    if match is None:
        return None
    lines: list[str] = []
    for line in match.group(1).splitlines():
        line = _LIST_PREFIX_RE.sub("", line)
        line = _BOLD_WRAP_RE.sub(r"\1 ", line)
        line = line.strip().strip("*_")
        if line:
            lines.append(line)
    if not lines:
        return None
    text = "\n\n".join(lines)
    feature_count = text.count("\n\n") + 1
    if feature_count < _MIN_FEATURES:
        return None
    return text


def _process_chunk(
    chunk: pa.Table,
    *,
    generator: SummaryGenerator,
    system_prompt: str,
    user_prompt_template: str,
) -> tuple[pa.Table, int]:
    texts = chunk.column("detokenized_text_truncated").to_pylist()
    keep_mask: list[bool] = [False] * len(texts)
    explanations_by_row: dict[int, str] = {}
    dropped = 0

    pending_indices: list[int] = []
    pending_prompts: list[str] = []
    for row_idx, text in enumerate(texts):
        trimmed = (text or "")[: generator.max_input_chars]
        if not trimmed.strip():
            dropped += 1
            continue
        pending_indices.append(row_idx)
        pending_prompts.append(_tagged_user_prompt(user_prompt_template, trimmed))

    for start in tqdm(range(0, len(pending_prompts), generator.batch_size), desc="summary batches", leave=False):
        batch_prompts = pending_prompts[start : start + generator.batch_size]
        batch_indices = pending_indices[start : start + generator.batch_size]
        raw_outputs = generator.complete_batch(system_prompt, batch_prompts)
        for row_idx, raw in zip(batch_indices, raw_outputs, strict=True):
            cleaned = extract_and_clean_explanation(raw)
            if cleaned is None:
                dropped += 1
                continue
            keep_mask[row_idx] = True
            explanations_by_row[row_idx] = cleaned

    explanations = [explanations_by_row[idx] for idx, keep in enumerate(keep_mask) if keep]
    filtered = chunk.filter(pa.array(keep_mask, type=pa.bool_()))
    return filtered.append_column("api_explanation", pa.array(explanations, type=pa.string())), dropped


def generate_summaries(
    input_parquet: str | Path,
    output_parquet: str | Path,
    system_prompt: str,
    user_prompt_template: str,
    summary_config: dict,
    checkpoint_dir: str | Path | None = None,
) -> None:
    input_parquet = Path(input_parquet)
    output_parquet = Path(output_parquet)
    table = pq.read_table(str(input_parquet))

    chunks_dir = Path(checkpoint_dir) if checkpoint_dir is not None else output_parquet.with_suffix(".chunks")
    chunks_dir.mkdir(parents=True, exist_ok=True)
    provider = str(summary_config.get("provider", "local")).lower()
    provider_cfg = provider_options(summary_config, provider)
    chunk_size = int(provider_cfg.get("chunk_size", summary_config.get("chunk_size", 128)))
    chunk_starts = list(range(0, table.num_rows, chunk_size))

    generator = build_summary_generator(summary_config)

    dropped_total = 0
    chunk_paths: list[Path] = []
    for start in tqdm(chunk_starts, desc=f"chunks {input_parquet.name}"):
        chunk_path = chunks_dir / f"chunk_{start:08d}.parquet"
        chunk_paths.append(chunk_path)
        if chunk_path.exists():
            continue
        out_chunk, dropped = _process_chunk(
            table.slice(start, chunk_size),
            generator=generator,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
        )
        dropped_total += dropped
        tmp_path = chunk_path.with_suffix(".tmp")
        pq.write_table(out_chunk, tmp_path)
        tmp_path.rename(chunk_path)

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    writer = None
    try:
        for chunk_path in chunk_paths:
            chunk = pq.read_table(chunk_path)
            if writer is None:
                writer = pq.ParquetWriter(str(output_parquet), chunk.schema)
            writer.write_table(chunk)
            row_count += chunk.num_rows
    finally:
        if writer is not None:
            writer.close()

    if row_count == 0:
        raise RuntimeError("all local teacher explanations were dropped; check prompt format or max_new_tokens")

    print(
        f"[summaries] wrote {row_count} rows to {output_parquet}; "
        f"dropped={dropped_total}, generated_rows={generator.total_rows}, batches={generator.total_batches}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NLA warm-start explanations")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--input", default=None, help="Override input parquet path")
    parser.add_argument("--output", default=None, help="Override output parquet path")
    parser.add_argument("--split", default=None, help="Split to process: av_sft, ar_sft, or both")
    args = parser.parse_args()

    config = load_config(args.config)
    datagen_cfg = config["datagen"]
    prompts = config["prompts"]
    output_dir = Path(datagen_cfg["output_dir"])
    summary_cfg = resolve_summary_config(config, args.config)

    if args.input and args.output:
        generate_summaries(
            args.input,
            args.output,
            prompts["summary_system"],
            prompts["summary_user"],
            summary_cfg,
        )
    else:
        splits = [args.split] if args.split else ["av_sft", "ar_sft"]
        for split_name in splits:
            input_path = output_dir / "splits" / f"{split_name}_raw.parquet"
            output_path = output_dir / "splits" / f"{split_name}_explained.parquet"
            if not input_path.exists():
                print(f"[skip] {input_path} not found")
                continue
            generate_summaries(
                input_path,
                output_path,
                prompts["summary_system"],
                prompts["summary_user"],
                summary_cfg,
            )


if __name__ == "__main__":
    main()
