"""Shared training utilities for Nano-NLA scripts."""

from __future__ import annotations

import ast
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from nano_nla.models import last_token_values
from nano_nla.schema import ACTIVATION_COLUMN, NLAConfig, normalize_activation


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_pad_token(tokenizer: Any) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


def load_parquet_rows(
    path: str | Path,
    max_rows: int | None = None,
    row_offset: int = 0,
) -> list[dict[str, Any]]:
    row_offset = max(0, int(row_offset))
    if max_rows is None and row_offset == 0:
        table = pq.read_table(str(path))
    else:
        parquet_file = pq.ParquetFile(str(path))
        batches = []
        remaining = None if max_rows is None else max(0, int(max_rows))
        skipped = 0
        if remaining == 0:
            return []
        for batch in parquet_file.iter_batches(batch_size=8192):
            if skipped + batch.num_rows <= row_offset:
                skipped += batch.num_rows
                continue
            start = max(0, row_offset - skipped)
            available = batch.num_rows - start
            take = available if remaining is None else min(available, remaining)
            if take <= 0:
                break
            batches.append(batch.slice(start, take))
            skipped += batch.num_rows
            if remaining is not None:
                remaining -= take
            if remaining == 0:
                break
        if not batches:
            return []
        table = pa.Table.from_batches(batches, schema=parquet_file.schema_arrow)
    rows = table.to_pylist()
    for row in rows:
        row[ACTIVATION_COLUMN] = list(row[ACTIVATION_COLUMN])
    return rows


def parse_messages(value: Any) -> list[dict[str, str]]:
    if isinstance(value, list):
        return [{"role": str(m["role"]), "content": str(m["content"])} for m in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(value)
        return parse_messages(parsed)
    raise TypeError(f"unsupported prompt message value: {type(value).__name__}")


def as_tensor_activation(value: Any, device: str | torch.device = "cpu") -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float32, device=device)


def pad_1d(seqs: list[list[int]], pad_id: int) -> torch.Tensor:
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(seqs):
        out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out


def pad_labels(labels: list[list[int]], ignore_index: int = -100) -> torch.Tensor:
    max_len = max(len(s) for s in labels)
    out = torch.full((len(labels), max_len), ignore_index, dtype=torch.long)
    for i, seq in enumerate(labels):
        out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out


def build_av_sft_features(tokenizer: Any, messages: list[dict[str, str]], response: str) -> tuple[list[int], list[int]]:
    prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    full_ids = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": response}],
        tokenize=True,
        add_generation_prompt=False,
    )
    if hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids["input_ids"]
    if hasattr(full_ids, "input_ids"):
        full_ids = full_ids["input_ids"]
    if isinstance(prompt_ids, torch.Tensor):
        prompt_ids = prompt_ids[0].tolist() if prompt_ids.ndim == 2 else prompt_ids.tolist()
    if isinstance(full_ids, torch.Tensor):
        full_ids = full_ids[0].tolist() if full_ids.ndim == 2 else full_ids.tolist()
    if prompt_ids and isinstance(prompt_ids[0], list):
        prompt_ids = prompt_ids[0]
    if full_ids and isinstance(full_ids[0], list):
        full_ids = full_ids[0]
    labels = [-100] * len(full_ids)
    for idx in range(min(len(prompt_ids), len(full_ids)), len(full_ids)):
        labels[idx] = int(full_ids[idx])
    return list(map(int, full_ids)), labels


def tokenize_text_batch(tokenizer: Any, texts: list[str], device: str | torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer(texts, add_special_tokens=True, padding=True, return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items()}


def critic_mse_loss(
    critic: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_activations: torch.Tensor,
    cfg: NLAConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = critic(input_ids=input_ids, attention_mask=attention_mask)
    pred = last_token_values(out.values, attention_mask)
    target = target_activations.to(pred.device, pred.dtype)
    loss = F.mse_loss(
        normalize_activation(pred, cfg.mse_scale),
        normalize_activation(target, cfg.mse_scale),
        reduction="mean",
    )
    return loss, pred


def cosine_lr(base_lr: float, step: int, total_steps: int, warmup_steps: int) -> float:
    if total_steps <= 0:
        return base_lr
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr
