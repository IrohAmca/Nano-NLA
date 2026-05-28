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

    # Steering
    uv run python scripts/run_eval.py --config configs/qwen05b.yaml --mode steering \
        --av-checkpoint checkpoints/rl/step_800/av --ar-checkpoint checkpoints/rl/step_800/ar \
        --gold-parquet data/generated/rl.parquet \
        --steering-replacements '{"reward":"penalty"}'

    # All evaluations
    uv run python scripts/run_eval.py --config configs/qwen05b.yaml --mode all \
        --av-checkpoint checkpoints/rl/av --ar-checkpoint checkpoints/rl/ar \
        --gold-parquet data/generated/rl.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch

from nano_nla.eval.evaluations import load_activation_column, metric_summary
from nano_nla.schema import ACTIVATION_COLUMN
from nano_nla.schema import build_nla_config_from_yaml, load_config, merge_sidecar_into_config


def _parse_steering_replacements(raw: str | None) -> dict[str, str]:
    """Parse steering replacements from JSON or comma-separated old=new pairs."""
    if raw is None or not raw.strip():
        return {}
    text = raw.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if parsed is not None:
        if not isinstance(parsed, dict):
            raise ValueError("--steering-replacements JSON must be an object")
        return {str(k): str(v) for k, v in parsed.items()}

    replacements: dict[str, str] = {}
    for item in text.split(","):
        if "=" not in item:
            raise ValueError(
                "--steering-replacements must be JSON or comma-separated old=new pairs"
            )
        old, new = item.split("=", 1)
        old = old.strip()
        new = new.strip()
        if not old:
            raise ValueError("--steering-replacements contains an empty source string")
        replacements[old] = new
    return replacements


def _parse_float_list(raw: str | None, default: list[float]) -> list[float]:
    if raw is None or not raw.strip():
        return default
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one comma-separated float")
    return values


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


def _load_steering_rows(
    parquet_path: str | Path,
    *,
    prompt_column: str,
    max_rows: int,
    prompt_override: str | None = None,
) -> list[dict[str, Any]]:
    """Load a bounded set of activation/prompt rows for steering eval."""
    pf = pq.ParquetFile(str(parquet_path))
    columns = [ACTIVATION_COLUMN]
    has_prompt_column = prompt_column in pf.schema_arrow.names
    if prompt_override is None:
        if not has_prompt_column:
            raise ValueError(
                f"prompt column {prompt_column!r} is missing; provide --steering-prompt"
            )
        columns.append(prompt_column)

    rows: list[dict[str, Any]] = []
    offset = 0
    for batch in pf.iter_batches(batch_size=min(max_rows, 128), columns=columns):
        table = batch.to_pydict()
        activations = table[ACTIVATION_COLUMN]
        prompts = (
            [prompt_override] * len(activations)
            if prompt_override is not None
            else table[prompt_column]
        )
        for activation, prompt in zip(activations, prompts, strict=True):
            rows.append({
                "row_index": offset,
                "activation": torch.tensor(activation, dtype=torch.float32),
                "prompt_text": str(prompt or ""),
            })
            offset += 1
            if len(rows) >= max_rows:
                return rows
    return rows


def _load_steering_target_model(args, config, client, device):
    """Load the model whose generation will be steered."""
    target_source = args.steering_target_model or args.av_checkpoint
    if target_source is None or str(target_source) == str(args.av_checkpoint):
        return client.av, client.tokenizer, str(args.av_checkpoint)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nano_nla.models import resolve_torch_dtype
    from nano_nla.training.common import ensure_pad_token

    inference_cfg = config.get("inference", {})
    dtype = resolve_torch_dtype(inference_cfg.get("dtype", "auto"), device=device)
    tokenizer = AutoTokenizer.from_pretrained(target_source, trust_remote_code=True)
    ensure_pad_token(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        target_source,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return model, tokenizer, str(target_source)


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


def run_steering(args, config, nla_cfg) -> dict:
    """Run NLA causal steering experiments."""
    from nano_nla.eval.steering import run_steering_experiment
    from nano_nla.models import resolve_torch_device

    replacements = _parse_steering_replacements(args.steering_replacements)
    if not replacements:
        raise ValueError("--steering-replacements is required for steering mode")

    device = resolve_torch_device("auto", require_cuda=False)
    client = _load_client(args, config, nla_cfg, device)
    target_model, target_tokenizer, target_source = _load_steering_target_model(
        args,
        config,
        client,
        device,
    )

    eval_cfg = config.get("eval", {}).get("steering", {})
    num_samples = int(args.max_rows or eval_cfg.get("num_samples", 20))
    alpha_values = _parse_float_list(
        args.steering_alpha_values,
        [float(v) for v in eval_cfg.get("alpha_range", [0.05, 0.1, 0.2, 0.3, 0.5])],
    )
    rows = _load_steering_rows(
        args.gold_parquet,
        prompt_column=args.steering_prompt_column,
        max_rows=num_samples,
        prompt_override=args.steering_prompt,
    )
    if not rows:
        raise ValueError("no steering rows loaded")

    sample_results = []
    for row in rows:
        prompt_text = row["prompt_text"]
        if not prompt_text:
            raise ValueError(f"empty steering prompt at row {row['row_index']}")

        results = run_steering_experiment(
            client,
            target_model,
            target_tokenizer,
            prompt_text=prompt_text,
            activation=row["activation"],
            token_position=args.steering_token_position,
            replacements=replacements,
            alpha_values=alpha_values,
            num_rollouts=int(args.steering_num_rollouts or eval_cfg.get("num_rollouts", 5)),
            max_new_tokens=int(args.steering_max_new_tokens or eval_cfg.get("max_new_tokens", 100)),
            temperature=float(
                args.steering_temperature
                if args.steering_temperature is not None
                else eval_cfg.get("temperature", config.get("inference", {}).get("temperature", 1.0))
            ),
            device=device,
        )
        sv = results[0].steering_vector
        sample_results.append({
            "row_index": row["row_index"],
            "prompt_preview": prompt_text[:240],
            "original_explanation": sv.original_explanation,
            "edited_explanation": sv.edited_explanation,
            "reconstruction_delta_norm": float(
                (sv.edited_reconstruction.float() - sv.original_reconstruction.float()).norm().item()
            ),
            "direction_norm": float(sv.direction.float().norm().item()),
            "generations": [
                {
                    "alpha": float(result.alpha),
                    "original_completion": result.original_completion,
                    "steered_completion": result.steered_completion,
                }
                for result in results
            ],
        })

    return {
        "steering": {
            "target_model": target_source,
            "replacements": replacements,
            "alpha_values": alpha_values,
            "token_position": args.steering_token_position,
            "num_samples": len(sample_results),
            "samples": sample_results,
        }
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
        choices=["metrics", "steganography", "confabulation", "steering", "all"],
        default="metrics",
    )
    parser.add_argument("--gold-parquet", required=True, help="Parquet with activation_vector column")
    parser.add_argument("--pred-tensor", default=None, help="torch .pt tensor (metrics mode)")
    parser.add_argument("--av-checkpoint", default=None)
    parser.add_argument("--ar-checkpoint", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--steering-replacements",
        default=None,
        help='Steering edit as JSON object or comma pairs, e.g. \'{"reward":"penalty"}\' or reward=penalty',
    )
    parser.add_argument(
        "--steering-alpha-values",
        default=None,
        help="Comma-separated alpha values; defaults to eval.steering.alpha_range",
    )
    parser.add_argument("--steering-token-position", type=int, default=-1)
    parser.add_argument("--steering-num-rollouts", type=int, default=None)
    parser.add_argument("--steering-max-new-tokens", type=int, default=None)
    parser.add_argument("--steering-temperature", type=float, default=None)
    parser.add_argument(
        "--steering-prompt-column",
        default="detokenized_text_truncated",
        help="Parquet prompt/context column used for target-model generation",
    )
    parser.add_argument(
        "--steering-prompt",
        default=None,
        help="Use one explicit prompt for every loaded activation instead of reading a parquet prompt column",
    )
    parser.add_argument(
        "--steering-target-model",
        default=None,
        help="Target model/checkpoint to steer; defaults to --av-checkpoint to avoid loading a third model",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config = merge_sidecar_into_config(config, args.gold_parquet)
    nla_cfg = build_nla_config_from_yaml(config)
    results: dict = {}

    modes = ["metrics", "steganography", "confabulation", "steering"] if args.mode == "all" else [args.mode]

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
        elif mode == "steering":
            if args.av_checkpoint is None or args.ar_checkpoint is None:
                print("[skip] --av-checkpoint and --ar-checkpoint required")
                continue
            if args.steering_replacements is None:
                print("[skip] --steering-replacements required")
                continue
            results.update(run_steering(args, config, nla_cfg))

    text = json.dumps(results, indent=2, ensure_ascii=False)
    print(f"\n{'='*40} RESULTS {'='*40}")
    print(text)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"\n[saved] {args.output}")


if __name__ == "__main__":
    main()
