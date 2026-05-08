"""Full datagen pipeline orchestrator — run stages 0→3 from config.

Usage:
    # Full pipeline (extraction + split + local model summaries + build)
    python scripts/run_datagen.py --config configs/qwen05b.yaml

    # Individual stages
    python scripts/run_datagen.py --config configs/qwen05b.yaml --stage 0      # Extract activations
    python scripts/run_datagen.py --config configs/qwen05b.yaml --stage 1      # Split
    python scripts/run_datagen.py --config configs/qwen05b.yaml --stage 2      # Local summaries
    python scripts/run_datagen.py --config configs/qwen05b.yaml --stage 3      # Build final datasets
    python scripts/run_datagen.py --config configs/qwen05b.yaml --stage 2 --split av_sft  # Only AV summaries

Pipeline stages:
    0: extract_activations — Target modelden residual stream vektörleri çıkar
    1: split_dataset      — base.parquet → av_sft / ar_sft / rl splits
    2: generate_summaries — Yerel öğretmen model ile warm-start açıklamaları üret
    3: build_datasets     — Final parquet dosyalarını oluştur (prompt formatting)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nano_nla.schema import load_config


def computed_config_path(config: dict) -> Path:
    """Path written by stage 0 after token IDs and scales are measured."""
    current = Path(config["_config_path"])
    stem = current.stem if current.stem.endswith("_computed") else f"{current.stem}_computed"
    return Path(config["datagen"]["output_dir"]) / f"{stem}.yaml"


def reload_computed_config_if_present(config: dict) -> dict:
    """Prefer the stage-0 computed config for downstream stages."""
    path = computed_config_path(config)
    if not path.exists() or Path(config["_config_path"]).resolve() == path.resolve():
        return config
    source_config = config
    updated = load_config(path)
    if "summary_model" not in updated.get("datagen", {}) and "summary_model" in source_config.get("datagen", {}):
        updated.setdefault("datagen", {})["summary_model"] = source_config["datagen"]["summary_model"]
    updated["_config_path"] = str(path)
    print(f"[config] Using computed config for downstream stages: {path}")
    return updated


def run_stage_0(
    config: dict,
    device: str | None = None,
    devices: str | None = None,
    restart: bool = False,
) -> None:
    """Stage 0: Extract activations."""
    from nano_nla.datagen.extract_activations import main as extract_main
    args = ["--config", config["_config_path"]]
    if devices:
        args += ["--devices", devices]
    if device:
        args += ["--device", device]
    if restart:
        args += ["--restart"]
    sys.argv = ["extract_activations"] + args
    extract_main()


def run_stage_1(config: dict) -> None:
    """Stage 1: Split dataset."""
    from nano_nla.datagen.prepare_datasets import main as prepare_main
    sys.argv = ["prepare_datasets", "--config", config["_config_path"], "--stage", "split"]
    prepare_main()


def run_stage_2(config: dict, split: str | None = None) -> None:
    """Stage 2: Generate summaries with the local teacher model."""
    from nano_nla.datagen.generate_summaries import main as summary_main
    args = ["--config", config["_config_path"]]
    if split:
        args += ["--split", split]
    sys.argv = ["generate_summaries"] + args
    summary_main()


def run_stage_3(config: dict) -> None:
    """Stage 3: Build final datasets."""
    from nano_nla.datagen.prepare_datasets import main as prepare_main
    sys.argv = ["prepare_datasets", "--config", config["_config_path"], "--stage", "build"]
    prepare_main()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--stage", type=str, default=None,
                        help="Specific stage to run (0/1/2/3). Default: all")
    parser.add_argument("--device", default=None,
                        help="Single stage-0 device override (auto/cuda:0)")
    parser.add_argument("--devices", default=None,
                        help="Comma-separated stage-0 worker devices, e.g. auto,cuda:0,cuda:all")
    parser.add_argument("--restart-stage0", action="store_true",
                        help="Delete existing stage-0 shards and start extraction from scratch")
    parser.add_argument("--split", default=None,
                        help="Split for stage 2 (av_sft/ar_sft)")
    args = parser.parse_args()

    config = load_config(args.config)
    config["_config_path"] = args.config

    output_dir = Path(config["datagen"]["output_dir"])
    model_name = config["model"]["name"]

    print("=" * 70)
    print("Nano-NLA Data Generation Pipeline")
    print("=" * 70)
    print(f"  Model:      {model_name}")
    print(f"  Layer:      {config['model']['target_layer']}")
    print(f"  Output:     {output_dir}")
    print(f"  Docs:       {config['datagen']['corpus']['length']}")
    print(f"  Pos/doc:    {config['datagen']['extraction']['positions_per_doc']}")
    summary_model = config["datagen"].get("summary_model", {})
    print(f"  Summary:    {summary_model.get('name', '<missing datagen.summary_model>')}")
    print("=" * 70)

    stages = [0, 1, 2, 3] if args.stage is None else [int(args.stage)]
    if 0 not in stages:
        config = reload_computed_config_if_present(config)

    for stage in stages:
        print(f"\n{'='*20} STAGE {stage} {'='*20}")

        if stage == 0:
            run_stage_0(config, args.device, args.devices, args.restart_stage0)
            config = reload_computed_config_if_present(config)
        elif stage == 1:
            run_stage_1(config)
        elif stage == 2:
            run_stage_2(config, args.split)
        elif stage == 3:
            run_stage_3(config)
        else:
            print(f"Unknown stage: {stage}")
            sys.exit(1)

    print(f"\n{'='*70}")
    print("Pipeline complete!")
    print(f"{'='*70}")

    # Print final file listing
    if output_dir.exists():
        print("\nGenerated files:")
        for f in sorted(output_dir.rglob("*.parquet")):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.relative_to(output_dir)} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
