"""Stage 2: generate warm-start explanations with Groq.

This mirrors the NLA API-explanation stage: external model completions are
strictly parsed, bad/truncated rows are dropped, and completed chunks are
written immediately for crash-safe resume under free-tier rate limits.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from collections import deque
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from nano_nla.schema import EXPLANATION_RE, load_config

_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*+]|[0-9]+[.)]|[a-zA-Z][.)])\s+")
_BOLD_WRAP_RE = re.compile(r"^\*\*(.+?)\*\*\s*")
_MIN_FEATURES = 2


class RateLimiter:
    """Sliding-window request limiter for Groq free-tier RPM caps."""

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
    """One-at-a-time Groq client with rate limiting and exponential backoff."""

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
    ) -> None:
        from groq import Groq

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        self.client = Groq(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self.limiter = RateLimiter(requests_per_minute)
        self.total_requests = 0
        self.total_retries = 0
        self.total_wait = 0.0

    def complete(self, system_prompt: str, user_prompt: str) -> str | None:
        for attempt in range(self.max_retries):
            self.total_wait += self.limiter.wait()
            try:
                self.limiter.record()
                self.total_requests += 1
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
    generator: GroqSummaryGenerator,
    system_prompt: str,
    user_prompt_template: str,
) -> tuple[pa.Table, int]:
    texts = chunk.column("detokenized_text_truncated").to_pylist()
    keep_mask: list[bool] = []
    explanations: list[str] = []
    dropped = 0

    for text in tqdm(texts, desc="groq rows", leave=False):
        trimmed = (text or "")[:2000]
        if not trimmed.strip():
            keep_mask.append(False)
            dropped += 1
            continue
        raw = generator.complete(system_prompt, _tagged_user_prompt(user_prompt_template, trimmed))
        cleaned = extract_and_clean_explanation(raw)
        if cleaned is None:
            keep_mask.append(False)
            dropped += 1
            continue
        keep_mask.append(True)
        explanations.append(cleaned)

    filtered = chunk.filter(pa.array(keep_mask, type=pa.bool_()))
    return filtered.append_column("api_explanation", pa.array(explanations, type=pa.string())), dropped


def generate_summaries(
    input_parquet: str | Path,
    output_parquet: str | Path,
    system_prompt: str,
    user_prompt_template: str,
    groq_config: dict,
    checkpoint_dir: str | Path | None = None,
) -> None:
    input_parquet = Path(input_parquet)
    output_parquet = Path(output_parquet)
    table = pq.read_table(str(input_parquet))

    chunks_dir = Path(checkpoint_dir) if checkpoint_dir is not None else output_parquet.with_suffix(".chunks")
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = int(groq_config.get("batch_size", 10))
    chunk_starts = list(range(0, table.num_rows, chunk_size))

    generator = GroqSummaryGenerator(
        model=groq_config.get("model", "llama-3.3-70b-versatile"),
        max_tokens=int(groq_config.get("max_tokens", 300)),
        temperature=float(groq_config.get("temperature", 0.7)),
        requests_per_minute=int(groq_config.get("requests_per_minute", 30)),
        max_retries=int(groq_config.get("max_retries", 5)),
        retry_base_delay=float(groq_config.get("retry_base_delay", 2.0)),
        retry_max_delay=float(groq_config.get("retry_max_delay", 60.0)),
    )

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
        raise RuntimeError("all Groq explanations were dropped; check prompt format or max_tokens")

    print(
        f"[summaries] wrote {row_count} rows to {output_parquet}; "
        f"dropped={dropped_total}, requests={generator.total_requests}, "
        f"retries={generator.total_retries}, waited={generator.total_wait:.1f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NLA warm-start explanations with Groq")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--input", default=None, help="Override input parquet path")
    parser.add_argument("--output", default=None, help="Override output parquet path")
    parser.add_argument("--split", default=None, help="Split to process: av_sft, ar_sft, or both")
    args = parser.parse_args()

    config = load_config(args.config)
    datagen_cfg = config["datagen"]
    prompts = config["prompts"]
    output_dir = Path(datagen_cfg["output_dir"])

    if args.input and args.output:
        generate_summaries(
            args.input,
            args.output,
            prompts["summary_system"],
            prompts["summary_user"],
            datagen_cfg["groq"],
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
                datagen_cfg["groq"],
            )


if __name__ == "__main__":
    main()
