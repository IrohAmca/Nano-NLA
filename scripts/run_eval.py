"""Comprehensive Nano-NLA evaluation runner.

Usage:
    # Basic FVE metrics from saved tensors
    uv run python scripts/run_eval.py --config configs/qwen05b.yaml --mode metrics \
        --gold-parquet data/generated/rl.parquet --pred-tensor results/preds.pt

    # Steganography detection (requires AV + AR checkpoints)
    uv run python scripts/run_eval.py --config configs/qwen05b.yaml --mode steganography \
        --av-checkpoint checkpoints/rl/av --ar-checkpoint checkpoints/rl/ar \
        --gold-parquet data/generated/rl.parquet

    # Confabulation analysis
    uv run python scripts/run_eval.py --config configs/qwen05b.yaml --mode confabulation \
        --av-checkpoint checkpoints/rl/av --ar-checkpoint checkpoints/rl/ar \
        --gold-parquet data/generated/rl.parquet

    # All evaluations
    uv run python scripts/run_eval.py --config configs/qwen05b.yaml --mode all \
        --av-checkpoint checkpoints/rl/av --ar-checkpoint checkpoints/rl/ar \
        --gold-parquet data/generated/rl.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nano_nla.eval.evaluations import load_activation_column, metric_summary
from nano_nla.schema import build_nla_config_from_yaml, load_config, merge_sidecar_into_config


def run_metrics(args, config, nla_cfg) -> dict:
    """Run basic FVE / MSE / cosine metrics."""
    gold = load_activation_column(args.gold_parquet, args.max_rows)
    pred = torch.load(args.pred_tensor, map_location="cpu", weights_only=True)
    if args.max_rows is not None:
        pred = pred[: args.max_rows]
    return metric_summary(pred.float(), gold.float(), nla_cfg.mse_scale)


def _load_client(args, config, nla_cfg, device):
    """Load NLAClient with AV and optionally AR checkpoints."""
    from nano_nla.inference import NLAClient
    return NLAClient(
        args.config,
        av_checkpoint=args.av_checkpoint,
        ar_checkpoint=args.ar_checkpoint,
        device=str(device),
    )


def _load_activations(args, nla_cfg, max_rows=None):
    """Load activation vectors from the gold parquet."""
    gold = load_activation_column(args.gold_parquet, max_rows)
    return [gold[i] for i in range(gold.shape[0])]


def run_steganography(args, config, nla_cfg) -> dict:
    """Run steganography detection tests."""
    from nano_nla.eval.steganography import measure_steganography
    from nano_nla.models import resolve_torch_device

    device = resolve_torch_device("auto", require_cuda=False)
    client = _load_client(args, config, nla_cfg, device)
    eval_cfg = config.get("eval", {}).get("steganography", {})
    num_samples = int(eval_cfg.get("num_samples", args.max_rows or 50))
    transforms = eval_cfg.get("transformations", ["shuffle_bullets", "paragraph_summary", "coherence"])

    activations = _load_activations(args, nla_cfg, max_rows=num_samples)
    results = measure_steganography(
        client,
        activations,
        transformations=transforms,
    )

    return {
        "steganography": [
            {
                "transformation": r.transformation,
                "fve_original": round(r.fve_original, 4),
                "fve_transformed": round(r.fve_transformed, 4),
                "fve_delta": round(r.fve_delta, 4),
                "num_samples": r.num_samples,
            }
            for r in results
        ]
    }


def run_confabulation(args, config, nla_cfg) -> dict:
    """Run confabulation analysis."""
    from nano_nla.eval.confabulation import analyze_confabulations
    from nano_nla.models import resolve_torch_device

    import pyarrow.parquet as pq

    device = resolve_torch_device("auto", require_cuda=False)
    client = _load_client(args, config, nla_cfg, device)
    eval_cfg = config.get("eval", {}).get("confabulation", {})
    num_samples = int(eval_cfg.get("num_samples", args.max_rows or 50))

    activations = _load_activations(args, nla_cfg, max_rows=num_samples)

    # Load contexts if available
    table = pq.read_table(str(args.gold_parquet))
    if "detokenized_text_truncated" in table.column_names:
        contexts = table.column("detokenized_text_truncated").to_pylist()[:num_samples]
    else:
        contexts = [""] * num_samples

    report = analyze_confabulations(
        client,
        activations,
        contexts,
        num_tokens_per_sample=int(eval_cfg.get("num_tokens_per_sample", 5)),
    )

    return {
        "confabulation": {
            "total_claims": report.total_claims,
            "supported_claims": report.supported_claims,
            "unsupported_claims": report.unsupported_claims,
            "unknown_claims": report.unknown_claims,
            "support_rate": round(report.support_rate, 4),
            "thematic_support_rate": round(report.thematic_support_rate, 4),
            "entity_support_rate": round(report.entity_support_rate, 4),
            "detail_support_rate": round(report.detail_support_rate, 4),
            "recurring_claim_support_rate": round(report.recurring_claim_support_rate, 4),
            "avg_mse_delta_true_claims": round(report.avg_mse_delta_true, 6),
            "avg_mse_delta_false_claims": round(report.avg_mse_delta_false, 6),
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive Nano-NLA evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode",
        choices=["metrics", "steganography", "confabulation", "all"],
        default="metrics",
    )
    parser.add_argument("--gold-parquet", required=True, help="Parquet with activation_vector column")
    parser.add_argument("--pred-tensor", default=None, help="torch .pt tensor (metrics mode)")
    parser.add_argument("--av-checkpoint", default=None)
    parser.add_argument("--ar-checkpoint", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    config = merge_sidecar_into_config(config, args.gold_parquet)
    nla_cfg = build_nla_config_from_yaml(config)
    results: dict = {}

    modes = ["metrics", "steganography", "confabulation"] if args.mode == "all" else [args.mode]

    for mode in modes:
        print(f"\n{'='*40} {mode.upper()} {'='*40}")
        if mode == "metrics":
            if args.pred_tensor is None:
                print("[skip] --pred-tensor required for metrics mode")
                continue
            results.update(run_metrics(args, config, nla_cfg))
        elif mode == "steganography":
            if args.av_checkpoint is None or args.ar_checkpoint is None:
                print("[skip] --av-checkpoint and --ar-checkpoint required")
                continue
            results.update(run_steganography(args, config, nla_cfg))
        elif mode == "confabulation":
            if args.av_checkpoint is None or args.ar_checkpoint is None:
                print("[skip] --av-checkpoint and --ar-checkpoint required")
                continue
            results.update(run_confabulation(args, config, nla_cfg))

    text = json.dumps(results, indent=2, ensure_ascii=False)
    print(f"\n{'='*40} RESULTS {'='*40}")
    print(text)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"\n[saved] {args.output}")


if __name__ == "__main__":
    main()
