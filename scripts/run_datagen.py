"""Full datagen pipeline orchestrator - run stages 0 to 3 from config.

Usage:
    # Full pipeline (extraction + split + summary generation + build)
    python scripts/run_datagen.py --config configs/qwen05b.yaml

    # Individual stages
    python scripts/run_datagen.py --config configs/qwen05b.yaml --stage 0
    python scripts/run_datagen.py --config configs/qwen05b.yaml --stage 1
    python scripts/run_datagen.py --config configs/qwen05b.yaml --stage 2
    python scripts/run_datagen.py --config configs/qwen05b.yaml --stage 2 --summary-provider deepseek
    python scripts/run_datagen.py --config configs/qwen05b.yaml --stage 2 --summary-concurrency 64
    python scripts/run_datagen.py --config configs/qwen05b.yaml --stage 3
    python scripts/run_datagen.py --config configs/qwen05b.yaml --stage 2 --split av_sft

Pipeline stages:
    0: extract_activations - extract target-model residual-stream vectors
    1: split_dataset      - base.parquet to av_sft / ar_sft / rl splits
    2: generate_summaries - generate warm-start explanations with selected provider
    3: build_datasets     - build final prompt-formatted parquet files
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from nano_nla.schema import load_config


def computed_config_path(config: dict) -> Path:
    """Path written by stage 0 after token IDs and scales are measured."""
    current = Path(config["_config_path"])
    stem = current.stem if current.stem.endswith("_computed") else f"{current.stem}_computed"
    return Path(config["datagen"]["output_dir"]) / f"{stem}.yaml"


def apply_summary_provider_override(config: dict, provider: str | None) -> dict:
    if provider is None:
        return config
    value = provider.lower()
    if value not in {"local", "groq", "deepseek", "nvidia", "multi"}:
        raise ValueError(f"unsupported summary provider: {provider}")
    summary = dict(config.get("datagen", {}).get("summary_model", {}))
    summary["provider"] = value
    config.setdefault("datagen", {})["summary_model"] = summary
    print(f"[config] Overriding summary provider from CLI: {value}")
    return config


def apply_summary_runtime_overrides(
    config: dict,
    *,
    max_concurrency: int | None = None,
    requests_per_minute: int | None = None,
    batch_size: int | None = None,
    chunk_size: int | None = None,
    timeout_seconds: float | None = None,
) -> dict:
    summary = dict(config.get("datagen", {}).get("summary_model", {}))
    provider = str(summary.get("provider", "local")).lower()
    provider_section = dict(summary.get(provider, {}))
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
        summary[provider] = provider_section
        config.setdefault("datagen", {})["summary_model"] = summary
        rendered = ", ".join(f"{key}={value}" for key, value in changed.items())
        print(f"[config] Stage-2 runtime overrides for {provider}: {rendered}")
    return config


def reload_computed_config_if_present(config: dict) -> dict:
    """Prefer the stage-0 computed config for downstream stages."""
    path = computed_config_path(config)
    if not path.exists() or Path(config["_config_path"]).resolve() == path.resolve():
        return config
    source_config = config
    updated = load_config(path)
    patched = False
    source_summary = source_config.get("datagen", {}).get("summary_model")
    if source_summary is not None and updated.setdefault("datagen", {}).get("summary_model") != source_summary:
        updated["datagen"]["summary_model"] = source_summary
        patched = True
    if patched:
        path.write_text(yaml.safe_dump(updated, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"[config] Synced computed config with current summary_model: {path}")
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


def run_stage_2(
    config: dict,
    split: str | None = None,
    summary_provider: str | None = None,
    summary_concurrency: int | None = None,
    summary_rpm: int | None = None,
    summary_batch_size: int | None = None,
    summary_chunk_size: int | None = None,
    summary_timeout_seconds: float | None = None,
    summary_checkpoint_dir: str | None = None,
    summary_max_new_rows: int | None = None,
    summary_target_rows: int | None = None,
) -> None:
    """Stage 2: Generate summaries with the configured provider."""
    from nano_nla.datagen.generate_summaries import main as summary_main
    args = ["--config", config["_config_path"]]
    if split:
        args += ["--split", split]
    if summary_provider:
        args += ["--provider", summary_provider]
    if summary_concurrency is not None:
        args += ["--max-concurrency", str(summary_concurrency)]
    if summary_rpm is not None:
        args += ["--requests-per-minute", str(summary_rpm)]
    if summary_batch_size is not None:
        args += ["--batch-size", str(summary_batch_size)]
    if summary_chunk_size is not None:
        args += ["--chunk-size", str(summary_chunk_size)]
    if summary_timeout_seconds is not None:
        args += ["--timeout-seconds", str(summary_timeout_seconds)]
    if summary_checkpoint_dir is not None:
        args += ["--checkpoint-dir", summary_checkpoint_dir]
    if summary_max_new_rows is not None:
        args += ["--max-new-rows", str(summary_max_new_rows)]
    if summary_target_rows is not None:
        args += ["--target-rows", str(summary_target_rows)]
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
    parser.add_argument("--summary-provider", choices=["local", "groq", "deepseek", "nvidia", "multi"], default=None,
                        help="Override stage-2 summary provider without editing YAML")
    parser.add_argument("--summary-concurrency", type=int, default=None,
                        help="Override stage-2 hosted provider parallel request count")
    parser.add_argument("--summary-rpm", type=int, default=None,
                        help="Override stage-2 hosted provider requests-per-minute cap; 0 disables limiting")
    parser.add_argument("--summary-batch-size", type=int, default=None,
                        help="Override stage-2 prompt batch size")
    parser.add_argument("--summary-chunk-size", type=int, default=None,
                        help="Override stage-2 checkpoint chunk size")
    parser.add_argument("--summary-timeout-seconds", type=float, default=None,
                        help="Override stage-2 hosted provider HTTP timeout")
    parser.add_argument("--summary-checkpoint-dir", default=None,
                        help="Shared compatible stage-2 checkpoint dir for continuing with another provider")
    parser.add_argument("--summary-max-new-rows", type=int, default=None,
                        help="Generate at most this many new stage-2 input rows per split")
    parser.add_argument("--summary-target-rows", type=int, default=None,
                        help="Build stage-2 outputs only up to this cumulative input row target")
    args = parser.parse_args()

    config = load_config(args.config)
    config["_config_path"] = args.config
    config = apply_summary_provider_override(config, args.summary_provider)
    config = apply_summary_runtime_overrides(
        config,
        max_concurrency=args.summary_concurrency,
        requests_per_minute=args.summary_rpm,
        batch_size=args.summary_batch_size,
        chunk_size=args.summary_chunk_size,
        timeout_seconds=args.summary_timeout_seconds,
    )

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
    provider = str(summary_model.get("provider", "local")).lower()
    provider_cfg = {**summary_model, **summary_model.get(provider, {})}
    summary_name = provider_cfg.get("model", provider_cfg.get("name", f"<missing {provider} model>"))
    print(f"  Summary:    {provider}:{summary_name}")
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
            run_stage_2(
                config,
                args.split,
                args.summary_provider,
                args.summary_concurrency,
                args.summary_rpm,
                args.summary_batch_size,
                args.summary_chunk_size,
                args.summary_timeout_seconds,
                args.summary_checkpoint_dir,
                args.summary_max_new_rows,
                args.summary_target_rows,
            )
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
