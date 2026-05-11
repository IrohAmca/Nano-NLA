"""Stage 2: generate warm-start explanations with local or hosted providers.

The NLA warm-start stage needs short natural-language descriptions for each
activation row. This implementation keeps the previous strict tag parsing and
crash-safe chunk resume behavior while allowing local Transformers generation
or hosted API providers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from nano_nla.models import enable_cuda_performance, resolve_torch_device, resolve_torch_dtype
from nano_nla.schema import EXPLANATION_RE, load_config
from nano_nla.training.common import ensure_pad_token

_LIST_PREFIX_RE = re.compile(
    r"^\s*(?:[-*+\u2022\u2013\u2014]|[0-9]+[.)]|\([0-9]+\)|[a-zA-Z][.)]|\([a-zA-Z]\)|[ivxIVX]+[.)])\s+"
)
_BOLD_WRAP_RE = re.compile(r"^\*\*(.+?)\*\*\s*")
_MIN_FEATURES = 2
SUMMARY_RESPONSE_OPEN = "<analysis>"
SUMMARY_RESPONSE_CLOSE = "</analysis>"
SUMMARY_RESPONSE_RE = re.compile(
    f"{re.escape(SUMMARY_RESPONSE_OPEN)}(.*?){re.escape(SUMMARY_RESPONSE_CLOSE)}",
    re.DOTALL,
)
SUMMARY_PROMPT_SCHEMA = (
    "\n\nReturn only this exact tagged format. Do not include bullets, numbering, "
    "Markdown, code fences, or text outside the tags:\n"
    f"{SUMMARY_RESPONSE_OPEN}\n"
    "[first feature, about 10-20 words]\n\n"
    "[second feature, about 10-20 words]\n\n"
    "[optional final feature about the final token or phrase and immediate continuation]\n"
    f"{SUMMARY_RESPONSE_CLOSE}"
)
DEFAULT_SUMMARY_MODEL = {
    "provider": "deepseek",
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
        "temperature": 0.2,
        "requests_per_minute": 30,
        "max_concurrency": 8,
        "max_retries": 5,
        "retry_base_delay": 2.0,
        "retry_max_delay": 60.0,
        "batch_size": 8,
        "chunk_size": 512,
        "max_input_chars": 2000,
    },
    "deepseek": {
        "model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
        "max_tokens": 300,
        "temperature": 0.2,
        "thinking": "disabled",
        "requests_per_minute": 0,
        "max_concurrency": 64,
        "max_retries": 5,
        "retry_base_delay": 2.0,
        "retry_max_delay": 60.0,
        "batch_size": 64,
        "chunk_size": 512,
        "max_input_chars": 2000,
        "timeout_seconds": 120,
    },
    "nvidia": {
        "model": "nvidia/nemotron-3-nano-30b-a3b",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "max_tokens": 300,
        "temperature": 0.2,
        "requests_per_minute": 40,
        "max_concurrency": 4,
        "max_retries": 5,
        "retry_base_delay": 2.0,
        "retry_max_delay": 60.0,
        "batch_size": 4,
        "chunk_size": 512,
        "max_input_chars": 2000,
        "timeout_seconds": 120,
    },
    "multi": {
        "providers": ["deepseek", "groq", "nvidia"],
        "weights": {
            "deepseek": 64,
            "groq": 8,
            "nvidia": 4,
        },
        "skip_unavailable": True,
        "batch_size": 76,
        "chunk_size": 512,
        "max_input_chars": 2000,
    },
}

_PROVIDER_EXHAUSTED_RE = re.compile(
    r"(quota|insufficient|balance|billing|credit|credits|payment|unauthori[sz]ed|forbidden|authentication)",
    re.IGNORECASE,
)


class HostedProviderError(RuntimeError):
    """Hosted summary provider failed in a way that should preserve the chunk."""


class HostedProviderExhaustedError(HostedProviderError):
    """Hosted provider quota, billing, auth, or rate limit stopped generation."""


def provider_options(summary_config: dict, provider: str) -> dict:
    nested = summary_config.get(provider, {})
    return {**summary_config, **nested}


def summary_cache_key(summary_config: dict) -> str:
    provider = str(summary_config.get("provider", "local")).lower()
    options = provider_options(summary_config, provider)
    model = str(options.get("model", options.get("name", "unknown")))
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{provider}_{model}").strip("_")


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
    current_path = Path(config_path)
    current_summary = datagen_cfg.get("summary_model")
    summary_cfg = None
    for candidate in _base_config_candidates(current_path):
        if not candidate.exists() or candidate.resolve() == current_path.resolve():
            continue
        candidate_cfg = load_config(candidate)
        summary_cfg = candidate_cfg.get("datagen", {}).get("summary_model")
        if summary_cfg is not None:
            print(f"[config] Loaded summary_model from base config: {candidate}")
            break

    if current_summary is not None and summary_cfg is None:
        return current_summary

    if current_summary is not None and summary_cfg is not None and current_summary == summary_cfg:
        return current_summary

    if summary_cfg is None:
        summary_cfg = dict(DEFAULT_SUMMARY_MODEL)
        print("[config] datagen.summary_model missing; using default DeepSeek summary model")
    elif current_summary is not None:
        print(f"[config] Synced summary_model from base config into {current_path}")

    datagen_cfg["summary_model"] = summary_cfg
    if current_path.exists():
        current_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"[config] Patched config with datagen.summary_model: {current_path}")
    return summary_cfg


def apply_summary_provider_override(summary_config: dict, provider: str | None) -> dict:
    if provider is None:
        return summary_config
    value = provider.lower()
    if value not in {"local", "groq", "deepseek", "nvidia", "multi"}:
        raise ValueError(f"unsupported summary provider: {provider}")
    updated = dict(summary_config)
    updated["provider"] = value
    print(f"[config] Overriding summary provider from CLI: {value}")
    return updated


def apply_summary_runtime_overrides(
    summary_config: dict,
    *,
    max_concurrency: int | None = None,
    requests_per_minute: int | None = None,
    batch_size: int | None = None,
    chunk_size: int | None = None,
    timeout_seconds: float | None = None,
) -> dict:
    """Patch provider-specific runtime knobs without mutating the input config."""
    provider = str(summary_config.get("provider", "local")).lower()
    updated = dict(summary_config)
    provider_section = dict(updated.get(provider, {}))
    changed: dict[str, int | float] = {}

    if max_concurrency is not None:
        provider_section["max_concurrency"] = max(1, int(max_concurrency))
        changed["max_concurrency"] = provider_section["max_concurrency"]
    if requests_per_minute is not None:
        provider_section["requests_per_minute"] = max(0, int(requests_per_minute))
        changed["requests_per_minute"] = provider_section["requests_per_minute"]
    if batch_size is not None:
        provider_section["batch_size"] = max(1, int(batch_size))
        changed["batch_size"] = provider_section["batch_size"]
    if chunk_size is not None:
        provider_section["chunk_size"] = max(1, int(chunk_size))
        changed["chunk_size"] = provider_section["chunk_size"]
    if timeout_seconds is not None:
        provider_section["timeout_seconds"] = max(1.0, float(timeout_seconds))
        changed["timeout_seconds"] = provider_section["timeout_seconds"]

    if changed:
        updated[provider] = provider_section
        rendered = ", ".join(f"{key}={value}" for key, value in changed.items())
        print(f"[config] Stage-2 runtime overrides for {provider}: {rendered}")
    return updated


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
        self._lock = threading.Lock()

    def wait(self) -> float:
        if self.requests_per_minute <= 0:
            return 0.0
        total_slept = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                while self.timestamps and self.timestamps[0] < now - self.window_seconds:
                    self.timestamps.popleft()
                if len(self.timestamps) < self.requests_per_minute:
                    self.timestamps.append(now)
                    return total_slept
                sleep_for = max(0.0, self.timestamps[0] + self.window_seconds - now + 0.1)
            time.sleep(sleep_for)
            total_slept += sleep_for

    def record(self) -> None:
        # wait() reserves a slot atomically; kept for compatibility with callers.
        return None


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
        max_concurrency: int,
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
        self.max_concurrency = max(1, int(max_concurrency))
        self.batch_size = max(self.max_concurrency, int(batch_size))
        self.max_input_chars = int(max_input_chars)
        self.limiter = RateLimiter(int(requests_per_minute))
        self.total_rows = 0
        self.total_batches = 0
        self.total_retries = 0
        self.total_wait = 0.0
        self._stats_lock = threading.Lock()
        print(
            f"[summary] Using Groq provider model={self.model}, "
            f"concurrency={self.max_concurrency}, batch_size={self.batch_size}"
        )

    def _add_wait(self, seconds: float) -> None:
        if seconds <= 0.0:
            return
        with self._stats_lock:
            self.total_wait += seconds

    def _count_retry(self) -> None:
        with self._stats_lock:
            self.total_retries += 1

    def _raise_exhausted(self, msg: str) -> None:
        raise HostedProviderExhaustedError(
            f"Groq provider stopped before the current chunk could be checkpointed: {msg[:240]}"
        )

    def _complete_one(self, system_prompt: str, user_prompt: str) -> str | None:
        last_error: str | None = None
        for attempt in range(self.max_retries):
            self._add_wait(self.limiter.wait())
            try:
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
                self._count_retry()
                msg = str(exc)
                last_error = msg
                if _PROVIDER_EXHAUSTED_RE.search(msg):
                    self._raise_exhausted(msg)
                extra = 5.0 if "429" in msg or "rate" in msg.lower() else 0.0
                delay = min(self.retry_base_delay * (2**attempt) + extra, self.retry_max_delay)
                print(f"[groq] retry {attempt + 1}/{self.max_retries} after {delay:.1f}s: {msg[:120]}")
                time.sleep(delay)
        if last_error is not None:
            self._raise_exhausted(last_error)
        return None

    def complete_batch(self, system_prompt: str, user_prompts: list[str]) -> list[str | None]:
        self.total_batches += 1
        self.total_rows += len(user_prompts)
        workers = min(self.max_concurrency, len(user_prompts))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(lambda prompt: self._complete_one(system_prompt, prompt), user_prompts))


class DeepSeekSummaryGenerator:
    """DeepSeek OpenAI-compatible provider; used when summary_model.provider is deepseek."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        max_tokens: int,
        temperature: float,
        thinking: str | None,
        requests_per_minute: int,
        max_retries: int,
        retry_base_delay: float,
        retry_max_delay: float,
        batch_size: int,
        max_input_chars: int,
        timeout_seconds: float,
        max_concurrency: int,
    ) -> None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DeepSeek provider selected, but DEEPSEEK_API_KEY is not set")

        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.thinking = None if thinking is None else str(thinking).lower()
        if self.thinking not in {None, "enabled", "disabled"}:
            raise ValueError("DeepSeek thinking must be one of: enabled, disabled")
        self.max_retries = int(max_retries)
        self.retry_base_delay = float(retry_base_delay)
        self.retry_max_delay = float(retry_max_delay)
        self.max_concurrency = max(1, int(max_concurrency))
        self.batch_size = max(self.max_concurrency, int(batch_size))
        self.max_input_chars = int(max_input_chars)
        self.timeout_seconds = float(timeout_seconds)
        self.limiter = RateLimiter(int(requests_per_minute))
        self.total_rows = 0
        self.total_batches = 0
        self.total_retries = 0
        self.total_wait = 0.0
        self._stats_lock = threading.Lock()
        print(
            f"[summary] Using DeepSeek provider model={self.model}, "
            f"concurrency={self.max_concurrency}, batch_size={self.batch_size}, "
            f"thinking={self.thinking or 'default'}"
        )

    def _add_wait(self, seconds: float) -> None:
        if seconds <= 0.0:
            return
        with self._stats_lock:
            self.total_wait += seconds

    def _count_retry(self) -> None:
        with self._stats_lock:
            self.total_retries += 1

    def _raise_exhausted(self, msg: str) -> None:
        raise HostedProviderExhaustedError(
            f"DeepSeek provider stopped before the current chunk could be checkpointed: {msg[:240]}"
        )

    def _complete_one(self, system_prompt: str, user_prompt: str) -> str | None:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        if self.thinking is not None:
            payload["thinking"] = {"type": self.thinking}
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        last_error: str | None = None
        for attempt in range(self.max_retries):
            self._add_wait(self.limiter.wait())
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    data = json.loads(response.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
                if content and content.strip():
                    return content.strip()
            except HTTPError as exc:
                self._count_retry()
                msg = exc.read().decode("utf-8", errors="replace")[:120]
                last_error = f"HTTP {exc.code} {msg}"
                if exc.code in {401, 402, 403} or _PROVIDER_EXHAUSTED_RE.search(msg):
                    self._raise_exhausted(last_error)
                extra = 5.0 if exc.code == 429 else 0.0
                delay = min(self.retry_base_delay * (2**attempt) + extra, self.retry_max_delay)
                print(f"[deepseek] retry {attempt + 1}/{self.max_retries} after {delay:.1f}s: HTTP {exc.code} {msg}")
                time.sleep(delay)
            except (URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                self._count_retry()
                last_error = str(exc)
                delay = min(self.retry_base_delay * (2**attempt), self.retry_max_delay)
                print(f"[deepseek] retry {attempt + 1}/{self.max_retries} after {delay:.1f}s: {str(exc)[:120]}")
                time.sleep(delay)
        if last_error is not None:
            self._raise_exhausted(last_error)
        return None

    def complete_batch(self, system_prompt: str, user_prompts: list[str]) -> list[str | None]:
        self.total_batches += 1
        self.total_rows += len(user_prompts)
        workers = min(self.max_concurrency, len(user_prompts))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(lambda prompt: self._complete_one(system_prompt, prompt), user_prompts))


class NVIDIASummaryGenerator:
    """NVIDIA API Catalog / hosted NIM OpenAI-compatible provider."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        max_tokens: int,
        temperature: float,
        requests_per_minute: int,
        max_retries: int,
        retry_base_delay: float,
        retry_max_delay: float,
        batch_size: int,
        max_input_chars: int,
        timeout_seconds: float,
        max_concurrency: int,
    ) -> None:
        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise RuntimeError("NVIDIA provider selected, but NVIDIA_API_KEY is not set")

        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.max_retries = int(max_retries)
        self.retry_base_delay = float(retry_base_delay)
        self.retry_max_delay = float(retry_max_delay)
        self.max_concurrency = max(1, int(max_concurrency))
        self.batch_size = max(self.max_concurrency, int(batch_size))
        self.max_input_chars = int(max_input_chars)
        self.timeout_seconds = float(timeout_seconds)
        self.limiter = RateLimiter(int(requests_per_minute))
        self.total_rows = 0
        self.total_batches = 0
        self.total_retries = 0
        self.total_wait = 0.0
        self._stats_lock = threading.Lock()
        print(
            f"[summary] Using NVIDIA provider model={self.model}, "
            f"concurrency={self.max_concurrency}, batch_size={self.batch_size}"
        )

    def _add_wait(self, seconds: float) -> None:
        if seconds <= 0.0:
            return
        with self._stats_lock:
            self.total_wait += seconds

    def _count_retry(self) -> None:
        with self._stats_lock:
            self.total_retries += 1

    def _raise_exhausted(self, msg: str) -> None:
        raise HostedProviderExhaustedError(
            f"NVIDIA provider stopped before the current chunk could be checkpointed: {msg[:240]}"
        )

    def _complete_one(self, system_prompt: str, user_prompt: str) -> str | None:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        last_error: str | None = None
        for attempt in range(self.max_retries):
            self._add_wait(self.limiter.wait())
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    data = json.loads(response.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
                if content and content.strip():
                    return content.strip()
            except HTTPError as exc:
                self._count_retry()
                msg = exc.read().decode("utf-8", errors="replace")[:120]
                last_error = f"HTTP {exc.code} {msg}"
                if exc.code in {401, 402, 403} or _PROVIDER_EXHAUSTED_RE.search(msg):
                    self._raise_exhausted(last_error)
                extra = 5.0 if exc.code in {429, 503, 504} else 0.0
                delay = min(self.retry_base_delay * (2**attempt) + extra, self.retry_max_delay)
                print(f"[nvidia] retry {attempt + 1}/{self.max_retries} after {delay:.1f}s: HTTP {exc.code} {msg}")
                time.sleep(delay)
            except (URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                self._count_retry()
                last_error = str(exc)
                delay = min(self.retry_base_delay * (2**attempt), self.retry_max_delay)
                print(f"[nvidia] retry {attempt + 1}/{self.max_retries} after {delay:.1f}s: {str(exc)[:120]}")
                time.sleep(delay)
        if last_error is not None:
            self._raise_exhausted(last_error)
        return None

    def complete_batch(self, system_prompt: str, user_prompts: list[str]) -> list[str | None]:
        self.total_batches += 1
        self.total_rows += len(user_prompts)
        workers = min(self.max_concurrency, len(user_prompts))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(lambda prompt: self._complete_one(system_prompt, prompt), user_prompts))


class MultiProviderSummaryGenerator:
    """Weighted parallel executor across hosted providers."""

    def __init__(self, summary_config: dict) -> None:
        options = provider_options(summary_config, "multi")
        provider_names = [str(name).lower() for name in options.get("providers", [])]
        weights = dict(options.get("weights", {}))
        skip_unavailable = bool(options.get("skip_unavailable", True))

        self.providers: list[tuple[str, SummaryGenerator, int]] = []
        for name in provider_names:
            if name in {"local", "multi"}:
                raise ValueError("multi summary provider can only wrap hosted providers")
            try:
                generator = build_summary_generator({**summary_config, "provider": name})
            except RuntimeError as exc:
                if skip_unavailable:
                    print(f"[summary] Skipping unavailable provider {name}: {exc}")
                    continue
                raise
            weight = max(1, int(weights.get(name, getattr(generator, "max_concurrency", generator.batch_size))))
            self.providers.append((name, generator, weight))

        if not self.providers:
            raise RuntimeError("multi summary provider has no available hosted providers")

        self.batch_size = max(1, int(options.get("batch_size", sum(gen.batch_size for _, gen, _ in self.providers))))
        self.max_input_chars = min(gen.max_input_chars for _, gen, _ in self.providers)
        self.total_rows = 0
        self.total_batches = 0
        self.total_retries = 0
        self.total_wait = 0.0
        self._cursor = 0
        self._disabled: set[int] = set()
        rendered = ", ".join(f"{name}:weight={weight}" for name, _, weight in self.providers)
        print(f"[summary] Using multi-provider executor: {rendered}; batch_size={self.batch_size}")

    def _schedule(self, active: set[int]) -> list[int]:
        schedule: list[int] = []
        for idx, (_, _, weight) in enumerate(self.providers):
            if idx in active:
                schedule.extend([idx] * weight)
        return schedule

    def _refresh_stats(self) -> None:
        self.total_retries = sum(getattr(gen, "total_retries", 0) for _, gen, _ in self.providers)
        self.total_wait = sum(getattr(gen, "total_wait", 0.0) for _, gen, _ in self.providers)

    def complete_batch(self, system_prompt: str, user_prompts: list[str]) -> list[str | None]:
        self.total_batches += 1
        self.total_rows += len(user_prompts)
        results: list[str | None] = [None] * len(user_prompts)
        pending = list(enumerate(user_prompts))
        active = set(range(len(self.providers))).difference(self._disabled)

        while pending and active:
            schedule = self._schedule(active)
            grouped: dict[int, list[tuple[int, str]]] = {idx: [] for idx in active}
            for item_idx, item in enumerate(pending):
                provider_idx = schedule[(self._cursor + item_idx) % len(schedule)]
                grouped[provider_idx].append(item)
            self._cursor = (self._cursor + len(pending)) % len(schedule)

            failed: list[tuple[int, str]] = []
            with ThreadPoolExecutor(max_workers=len(active)) as executor:
                futures = {
                    executor.submit(
                        self.providers[provider_idx][1].complete_batch,
                        system_prompt,
                        [prompt for _, prompt in items],
                    ): (provider_idx, items)
                    for provider_idx, items in grouped.items()
                    if items
                }
                for future, (provider_idx, items) in futures.items():
                    provider_name = self.providers[provider_idx][0]
                    try:
                        outputs = future.result()
                    except HostedProviderError as exc:
                        active.discard(provider_idx)
                        self._disabled.add(provider_idx)
                        failed.extend(items)
                        print(f"[summary] Provider {provider_name} paused: {exc}")
                        continue
                    for (row_idx, _), raw in zip(items, outputs, strict=True):
                        results[row_idx] = raw

            pending = failed

        self._refresh_stats()
        if pending:
            raise HostedProviderExhaustedError("all hosted summary providers failed before this batch completed")
        return results


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
            temperature=float(options.get("temperature", 0.2)),
            requests_per_minute=int(options.get("requests_per_minute", 30)),
            max_retries=int(options.get("max_retries", 5)),
            retry_base_delay=float(options.get("retry_base_delay", 2.0)),
            retry_max_delay=float(options.get("retry_max_delay", 60.0)),
            batch_size=int(options.get("batch_size", 1)),
            max_input_chars=int(options.get("max_input_chars", 2000)),
            max_concurrency=int(options.get("max_concurrency", options.get("batch_size", 1))),
        )
    if provider == "deepseek":
        return DeepSeekSummaryGenerator(
            model=options.get("model", "deepseek-v4-flash"),
            base_url=options.get("base_url", "https://api.deepseek.com"),
            max_tokens=int(options.get("max_tokens", options.get("max_new_tokens", 300))),
            temperature=float(options.get("temperature", 0.2)),
            thinking=options.get("thinking", "disabled"),
            requests_per_minute=int(options.get("requests_per_minute", 0)),
            max_retries=int(options.get("max_retries", 5)),
            retry_base_delay=float(options.get("retry_base_delay", 2.0)),
            retry_max_delay=float(options.get("retry_max_delay", 60.0)),
            batch_size=int(options.get("batch_size", 1)),
            max_input_chars=int(options.get("max_input_chars", 2000)),
            timeout_seconds=float(options.get("timeout_seconds", 120)),
            max_concurrency=int(options.get("max_concurrency", options.get("batch_size", 1))),
        )
    if provider == "nvidia":
        return NVIDIASummaryGenerator(
            model=options.get("model", "nvidia/nemotron-3-nano-30b-a3b"),
            base_url=options.get("base_url", "https://integrate.api.nvidia.com/v1"),
            max_tokens=int(options.get("max_tokens", options.get("max_new_tokens", 300))),
            temperature=float(options.get("temperature", 0.2)),
            requests_per_minute=int(options.get("requests_per_minute", 40)),
            max_retries=int(options.get("max_retries", 5)),
            retry_base_delay=float(options.get("retry_base_delay", 2.0)),
            retry_max_delay=float(options.get("retry_max_delay", 60.0)),
            batch_size=int(options.get("batch_size", 1)),
            max_input_chars=int(options.get("max_input_chars", 2000)),
            timeout_seconds=float(options.get("timeout_seconds", 120)),
            max_concurrency=int(options.get("max_concurrency", options.get("batch_size", 1))),
        )
    if provider == "multi":
        return MultiProviderSummaryGenerator(summary_config)
    raise ValueError(f"unsupported summary provider: {provider}")


def _tagged_user_prompt(template: str, text: str) -> str:
    prompt = template.format(text=text)
    if SUMMARY_RESPONSE_OPEN in prompt and SUMMARY_RESPONSE_CLOSE in prompt:
        return prompt
    return prompt.rstrip() + SUMMARY_PROMPT_SCHEMA


def extract_and_clean_explanation(raw: str | None) -> str | None:
    """Extract explanation tags, strip list markers, require at least two features."""
    if raw is None:
        return None
    match = SUMMARY_RESPONSE_RE.search(raw) or EXPLANATION_RE.search(raw)
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


def summary_prompt_fingerprint(system_prompt: str, user_prompt_template: str) -> str:
    payload = json.dumps(
        {
            "system_prompt": system_prompt,
            "user_prompt_template": user_prompt_template,
            "response_open": SUMMARY_RESPONSE_OPEN,
            "response_close": SUMMARY_RESPONSE_CLOSE,
            "prompt_schema": SUMMARY_PROMPT_SCHEMA,
        },
        ensure_ascii=True,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _default_checkpoint_dir(output_parquet: Path, prompt_fingerprint: str) -> Path:
    return output_parquet.with_suffix(f".summary.{prompt_fingerprint[:8]}.chunks")


def _checkpoint_metadata_path(chunks_dir: Path) -> Path:
    return chunks_dir / "summary_checkpoint.json"


def _checkpoint_has_chunks(chunks_dir: Path) -> bool:
    return any(chunks_dir.glob("chunk_*.parquet"))


def _write_checkpoint_metadata(meta_path: Path, meta: dict) -> None:
    tmp_path = meta_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp_path.rename(meta_path)


def _checkpoint_compatible(
    chunks_dir: Path,
    *,
    input_key: str,
    chunk_size: int,
    prompt_fingerprint: str,
) -> bool:
    meta_path = _checkpoint_metadata_path(chunks_dir)
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (
        meta.get("input_parquet") == input_key
        and int(meta.get("chunk_size", -1)) == int(chunk_size)
        and meta.get("prompt_fingerprint") == prompt_fingerprint
    )


def _legacy_checkpoint_dirs(
    output_parquet: Path,
    summary_config: dict,
    target_dir: Path,
    *,
    input_parquet: Path,
    chunk_size: int,
    prompt_fingerprint: str,
) -> list[Path]:
    provider_dir = output_parquet.with_suffix(f".{summary_cache_key(summary_config)}.chunks")
    candidates = [provider_dir, *output_parquet.parent.glob(f"{output_parquet.stem}.*.chunks")]
    input_key = str(input_parquet.resolve())
    dirs: list[Path] = []
    for candidate in candidates:
        if candidate == target_dir or not candidate.is_dir() or candidate in dirs:
            continue
        if not _checkpoint_compatible(
            candidate,
            input_key=input_key,
            chunk_size=chunk_size,
            prompt_fingerprint=prompt_fingerprint,
        ):
            continue
        dirs.append(candidate)
    return dirs


def _ensure_checkpoint_metadata(
    chunks_dir: Path,
    *,
    input_parquet: Path,
    chunk_size: int,
    prompt_fingerprint: str,
) -> None:
    meta_path = _checkpoint_metadata_path(chunks_dir)
    input_key = str(input_parquet.resolve())
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("input_parquet") != input_key:
            raise RuntimeError(
                f"checkpoint dir {chunks_dir} belongs to another input parquet; use a different checkpoint dir"
            )
        if int(meta.get("chunk_size", chunk_size)) != chunk_size:
            raise RuntimeError(
                f"checkpoint dir {chunks_dir} was created with chunk_size={meta.get('chunk_size')}; "
                f"rerun with --chunk-size {meta.get('chunk_size')} or use a new checkpoint dir"
            )
        existing_fingerprint = meta.get("prompt_fingerprint")
        if existing_fingerprint is None:
            if _checkpoint_has_chunks(chunks_dir):
                raise RuntimeError(
                    f"checkpoint dir {chunks_dir} has chunks from an older prompt contract; "
                    f"use a new checkpoint dir or finish that run with the old prompt"
                )
            meta["schema_version"] = 2
            meta["prompt_fingerprint"] = prompt_fingerprint
            _write_checkpoint_metadata(meta_path, meta)
            return
        if existing_fingerprint != prompt_fingerprint:
            raise RuntimeError(
                f"checkpoint dir {chunks_dir} was created with another summary prompt; "
                f"use a new checkpoint dir to avoid mixing teacher labels"
            )
        return

    meta = {
        "schema_version": 2,
        "input_parquet": input_key,
        "chunk_size": chunk_size,
        "prompt_fingerprint": prompt_fingerprint,
    }
    _write_checkpoint_metadata(meta_path, meta)


def _reuse_legacy_chunk_if_available(chunk_path: Path, legacy_dirs: list[Path]) -> bool:
    for legacy_dir in legacy_dirs:
        legacy_path = legacy_dir / chunk_path.name
        if not legacy_path.exists():
            continue
        shutil.copy2(legacy_path, chunk_path)
        print(f"[summaries] reused checkpoint chunk {legacy_path} -> {chunk_path}")
        return True
    return False


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
    max_new_rows: int | None = None,
    target_rows: int | None = None,
) -> None:
    input_parquet = Path(input_parquet)
    output_parquet = Path(output_parquet)
    table = pq.read_table(str(input_parquet))

    provider = str(summary_config.get("provider", "local")).lower()
    provider_cfg = provider_options(summary_config, provider)
    chunk_size = int(provider_cfg.get("chunk_size", summary_config.get("chunk_size", 128)))
    target_input_rows = table.num_rows if target_rows is None else min(table.num_rows, max(0, int(target_rows)))
    chunk_starts = list(range(0, target_input_rows, chunk_size))
    prompt_fingerprint = summary_prompt_fingerprint(system_prompt, user_prompt_template)
    chunks_dir = (
        Path(checkpoint_dir)
        if checkpoint_dir is not None
        else _default_checkpoint_dir(output_parquet, prompt_fingerprint)
    )
    chunks_dir.mkdir(parents=True, exist_ok=True)
    _ensure_checkpoint_metadata(
        chunks_dir,
        input_parquet=input_parquet,
        chunk_size=chunk_size,
        prompt_fingerprint=prompt_fingerprint,
    )
    legacy_dirs = _legacy_checkpoint_dirs(
        output_parquet,
        summary_config,
        chunks_dir,
        input_parquet=input_parquet,
        chunk_size=chunk_size,
        prompt_fingerprint=prompt_fingerprint,
    )

    generator = build_summary_generator(summary_config)
    print(f"[summary] checkpoint_dir={chunks_dir} prompt={prompt_fingerprint[:8]}")

    dropped_total = 0
    new_input_rows = 0
    for start in tqdm(chunk_starts, desc=f"chunks {input_parquet.name}"):
        chunk_path = chunks_dir / f"chunk_{start:08d}.parquet"
        if chunk_path.exists():
            continue
        chunk_input_rows = min(chunk_size, table.num_rows - start)
        if max_new_rows is not None and new_input_rows >= max(0, int(max_new_rows)):
            break
        if _reuse_legacy_chunk_if_available(chunk_path, legacy_dirs):
            new_input_rows += chunk_input_rows
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
        new_input_rows += chunk_input_rows

    chunk_paths = [chunks_dir / f"chunk_{start:08d}.parquet" for start in chunk_starts]
    completed_chunk_paths = [path for path in chunk_paths if path.exists()]
    completed_input_rows = sum(
        min(chunk_size, target_input_rows - int(path.stem.split("_")[1]))
        for path in completed_chunk_paths
    )

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    writer = None
    try:
        for chunk_path in completed_chunk_paths:
            chunk = pq.read_table(chunk_path)
            if writer is None:
                writer = pq.ParquetWriter(str(output_parquet), chunk.schema)
            writer.write_table(chunk)
            row_count += chunk.num_rows
    finally:
        if writer is not None:
            writer.close()

    if row_count == 0:
        raise RuntimeError("all summary explanations were dropped; check prompt format or token limits")

    retries = getattr(generator, "total_retries", 0)
    wait_seconds = getattr(generator, "total_wait", 0.0)
    print(
        f"[summaries] wrote {row_count} rows to {output_parquet}; "
        f"dropped={dropped_total}, generated_rows={generator.total_rows}, "
        f"batches={generator.total_batches}, retries={retries}, rate_wait={wait_seconds:.1f}s, "
        f"checkpoint_input_rows={completed_input_rows}/{table.num_rows}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NLA warm-start explanations")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--input", default=None, help="Override input parquet path")
    parser.add_argument("--output", default=None, help="Override output parquet path")
    parser.add_argument("--split", default=None, help="Split to process: av_sft, ar_sft, or both")
    parser.add_argument(
        "--provider",
        choices=["local", "groq", "deepseek", "nvidia", "multi"],
        default=None,
        help="Override summary provider",
    )
    parser.add_argument("--max-concurrency", type=int, default=None, help="Hosted provider parallel request count")
    parser.add_argument(
        "--requests-per-minute",
        type=int,
        default=None,
        help="Hosted provider RPM cap; 0 disables limiting",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Stage-2 prompt batch size")
    parser.add_argument("--chunk-size", type=int, default=None, help="Checkpoint chunk size")
    parser.add_argument("--timeout-seconds", type=float, default=None, help="Hosted provider HTTP timeout")
    parser.add_argument("--max-new-rows", type=int, default=None, help="Generate at most this many new input rows")
    parser.add_argument(
        "--target-rows",
        type=int,
        default=None,
        help="Build summaries only up to this cumulative input row target",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Shared compatible summary checkpoint dir; use it to continue with another provider",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    datagen_cfg = config["datagen"]
    prompts = config["prompts"]
    output_dir = Path(datagen_cfg["output_dir"])
    summary_cfg = resolve_summary_config(config, args.config)
    summary_cfg = apply_summary_provider_override(summary_cfg, args.provider)
    summary_cfg = apply_summary_runtime_overrides(
        summary_cfg,
        max_concurrency=args.max_concurrency,
        requests_per_minute=args.requests_per_minute,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        timeout_seconds=args.timeout_seconds,
    )

    try:
        if args.input and args.output:
            generate_summaries(
                args.input,
                args.output,
                prompts["summary_system"],
                prompts["summary_user"],
                summary_cfg,
                checkpoint_dir=args.checkpoint_dir,
                max_new_rows=args.max_new_rows,
                target_rows=args.target_rows,
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
                    checkpoint_dir=args.checkpoint_dir,
                    max_new_rows=args.max_new_rows,
                    target_rows=args.target_rows,
                )
    except HostedProviderError as exc:
        raise SystemExit(f"[summaries] stopped safely before writing a bad chunk: {exc}") from exc


if __name__ == "__main__":
    main()
