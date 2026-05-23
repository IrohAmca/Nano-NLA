"""Generate example NLA explanations for sampled activation vectors.

Loads an RL-trained AV + AR model pair and produces human-readable explanations
together with cosine-similarity and normalized-MSE scores.

Usage (local):
    uv run python scripts/demo_explanations.py \
        --config configs/qwen05b.yaml \
        --av-checkpoint checkpoints/rl/av \
        --ar-checkpoint checkpoints/rl/ar \
        --gold-parquet data/generated/rl.parquet \
        --num-samples 5

    # Save results as a markdown table
    uv run python scripts/demo_explanations.py \
        --config configs/qwen05b.yaml \
        --av-checkpoint checkpoints/rl/av \
        --ar-checkpoint checkpoints/rl/ar \
        --gold-parquet data/generated/rl.parquet \
        --num-samples 10 \
        --output results/demo_explanations.md

Usage (Colab):
    !python scripts/demo_explanations.py \
        --config configs/qwen05b.yaml \
        --av-checkpoint checkpoints/rl/av \
        --ar-checkpoint checkpoints/rl/ar \
        --gold-parquet data/generated/rl.parquet \
        --num-samples 5 --device cuda:0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from nano_nla.eval.evaluations import load_activation_column
from nano_nla.inference import NLAClient


def _sample_indices(total: int, num_samples: int, seed: int = 42) -> list[int]:
    """Randomly sample *num_samples* unique indices from ``[0, total)``."""
    rng = np.random.default_rng(seed)
    k = min(num_samples, total)
    return sorted(rng.choice(total, size=k, replace=False).tolist())


def _format_markdown_table(rows: list[dict]) -> str:
    """Render results as a GitHub-flavoured markdown table."""
    lines = [
        "| Sample | Cosine | MSE (nrm) | Explanation |",
        "|-------:|-------:|----------:|:------------|",
    ]
    for r in rows:
        expl = r["explanation"].replace("\n", " ").replace("|", "\\|")
        lines.append(
            f"| {r['index']:>6d} "
            f"| {r['cosine']:>6.4f} "
            f"| {r['mse_nrm']:>9.6f} "
            f"| {expl} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate example NLA explanations for activation vectors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--av-checkpoint", required=True, help="AV (actor) checkpoint directory")
    parser.add_argument("--ar-checkpoint", required=True, help="AR (critic) checkpoint directory")
    parser.add_argument("--gold-parquet", required=True, help="Parquet with activation_vector column")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to explain")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--device", default=None, help="Torch device override (e.g. cuda:0)")
    parser.add_argument("--output", default=None, help="Save results as a markdown file")
    args = parser.parse_args()

    # --- load activations ------------------------------------------------
    activations = load_activation_column(args.gold_parquet)
    total = activations.shape[0]
    indices = _sample_indices(total, args.num_samples, seed=args.seed)
    print(f"Loaded {total} activations from {args.gold_parquet}")
    print(f"Sampled {len(indices)} indices: {indices}\n")

    # --- load NLAClient --------------------------------------------------
    client = NLAClient(
        args.config,
        av_checkpoint=args.av_checkpoint,
        ar_checkpoint=args.ar_checkpoint,
        device=args.device or "auto",
    )

    # --- generate explanations -------------------------------------------
    rows: list[dict] = []
    for rank, idx in enumerate(indices, 1):
        result = client.explain_and_score(activations[idx])
        row = {
            "index": idx,
            "explanation": result.explanation,
            "cosine": result.cosine if result.cosine is not None else float("nan"),
            "mse_nrm": result.mse_nrm if result.mse_nrm is not None else float("nan"),
        }
        rows.append(row)
        print(f"[{rank}/{len(indices)}] Sample {idx}")
        print(f"  Explanation : {result.explanation}")
        print(f"  Cosine sim  : {row['cosine']:.4f}")
        print(f"  MSE (nrm)   : {row['mse_nrm']:.6f}")
        print()

    # --- summary ---------------------------------------------------------
    cosines = [r["cosine"] for r in rows]
    mses = [r["mse_nrm"] for r in rows]
    print("=" * 60)
    print(f"  Mean cosine similarity : {np.mean(cosines):.4f}")
    print(f"  Mean normalized MSE    : {np.mean(mses):.6f}")
    print("=" * 60)

    # --- optional markdown output ----------------------------------------
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            f"# NLA Demo Explanations\n\n"
            f"- Config: `{args.config}`\n"
            f"- AV checkpoint: `{args.av_checkpoint}`\n"
            f"- AR checkpoint: `{args.ar_checkpoint}`\n"
            f"- Parquet: `{args.gold_parquet}`\n"
            f"- Samples: {len(rows)} (seed={args.seed})\n"
            f"- Mean cosine: {np.mean(cosines):.4f}\n"
            f"- Mean MSE (nrm): {np.mean(mses):.6f}\n\n"
        )
        out_path.write_text(header + _format_markdown_table(rows), encoding="utf-8")
        print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
