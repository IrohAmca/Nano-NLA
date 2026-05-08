"""Evaluation metrics for Nano-NLA."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from nano_nla.schema import ACTIVATION_COLUMN, compute_predict_mean_baselines, normalize_activation


def normalized_mse(pred: torch.Tensor, gold: torch.Tensor, mse_scale: float | None) -> torch.Tensor:
    return ((normalize_activation(pred.float(), mse_scale) - normalize_activation(gold.float(), mse_scale)) ** 2).mean(dim=-1)


def cosine_similarity(pred: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(pred.float(), gold.float(), dim=-1)


def fraction_variance_explained(
    pred: torch.Tensor,
    gold: torch.Tensor,
    *,
    mse_scale: float | None,
    baseline_mse: float | None = None,
) -> float:
    mse = normalized_mse(pred, gold, mse_scale).mean().item()
    if baseline_mse is None:
        _, baseline_mse = compute_predict_mean_baselines(gold.float(), mse_scale)
    if baseline_mse <= 0:
        raise ValueError(f"baseline_mse must be positive, got {baseline_mse}")
    return 1.0 - mse / baseline_mse


def load_activation_column(parquet_path: str | Path, max_rows: int | None = None) -> torch.Tensor:
    table = pq.read_table(str(parquet_path), columns=[ACTIVATION_COLUMN])
    values = table.column(ACTIVATION_COLUMN).to_pylist()
    if max_rows is not None:
        values = values[:max_rows]
    return torch.tensor(values, dtype=torch.float32)


def metric_summary(pred: torch.Tensor, gold: torch.Tensor, mse_scale: float | None) -> dict[str, float]:
    mse = normalized_mse(pred, gold, mse_scale)
    cos = cosine_similarity(pred, gold)
    mean_norm_baseline, rawvar_baseline = compute_predict_mean_baselines(gold.float(), mse_scale)
    return {
        "mse_nrm_mean": float(mse.mean().item()),
        "mse_nrm_std": float(mse.std(unbiased=False).item()),
        "cosine_mean": float(cos.mean().item()),
        "cosine_std": float(cos.std(unbiased=False).item()),
        "fve_nrm_meannorm": float(1.0 - mse.mean().item() / mean_norm_baseline),
        "fve_nrm": float(1.0 - mse.mean().item() / rawvar_baseline),
        "baseline_meannorm": float(mean_norm_baseline),
        "baseline_rawvar": float(rawvar_baseline),
    }


def shuffle_texts(texts: list[str], seed: int = 42) -> list[str]:
    rng = np.random.default_rng(seed)
    out = list(texts)
    rng.shuffle(out)
    return out
