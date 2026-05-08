"""Smoke tests for Nano-NLA.

Default mode is model-free and fast. It validates parquet shaping, sidecar
propagation, prompt building, explanation tags, and injection math without
loading Qwen or calling an API.

Use ``--mode model`` for the slower real Qwen/FineWeb extraction smoke.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from nano_nla.datagen.prepare_datasets import (
    build_ar_sft_dataset,
    build_av_sft_dataset,
    build_rl_dataset,
    merge_base_sidecar_if_available,
    split_dataset,
)
from nano_nla.schema import (
    ACTIVATION_COLUMN,
    NLAConfig,
    build_nla_config_from_yaml,
    load_config,
    merge_sidecar_into_config,
    read_sidecar,
    write_dataset_sidecar,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


AV_TEMPLATE = "Describe the vector in <concept>{injection_char}</concept>."
AR_TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def synthetic_cfg() -> NLAConfig:
    return NLAConfig(
        d_model=3,
        injection_char="X",
        injection_token_id=99,
        injection_left_neighbor_id=2,
        injection_right_neighbor_id=3,
        av_prompt_template=AV_TEMPLATE,
        ar_prompt_template=AR_TEMPLATE,
        injection_scale=7.5,
        mse_scale=3.0,
        target_layer=2,
        ar_num_layers=3,
        critic_suffix_ids=[10, 11],
    )


def write_synthetic_base(output_dir: Path, cfg: NLAConfig) -> Path:
    base_path = output_dir / "base.parquet"
    rows = {
        ACTIVATION_COLUMN: [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0],
        ],
        "detokenized_text_truncated": [
            "alpha context",
            "beta context",
            "gamma context",
            "delta context",
            "epsilon context",
            "zeta context",
            "eta context",
            "theta context",
        ],
        "n_raw_tokens": [8, 9, 10, 11, 12, 13, 14, 15],
        "activation_layer": [cfg.target_layer] * 8,
        "doc_id": [0, 0, 1, 1, 2, 2, 3, 3],
    }
    schema = pa.schema(
        [
            (ACTIVATION_COLUMN, pa.list_(pa.float32(), cfg.d_model)),
            ("detokenized_text_truncated", pa.string()),
            ("n_raw_tokens", pa.int64()),
            ("activation_layer", pa.int64()),
            ("doc_id", pa.int64()),
        ]
    )
    pq.write_table(pa.Table.from_pydict(rows, schema=schema), base_path)
    write_dataset_sidecar(base_path, cfg, base_model="synthetic/nano", stage="base")
    return base_path


def write_explained(input_path: Path, output_path: Path) -> None:
    table = pq.read_table(input_path)
    explanations = [
        f"feature {i}\n\nsecond feature {i}"
        for i in range(table.num_rows)
    ]
    pq.write_table(table.append_column("api_explanation", pa.array(explanations, type=pa.string())), output_path)


def validate_parquet(path: Path, *, min_rows: int, required_cols: set[str]) -> pa.Table:
    assert_true(path.exists(), f"missing parquet: {path}")
    table = pq.read_table(path)
    assert_true(table.num_rows >= min_rows, f"{path} has {table.num_rows} rows, expected >= {min_rows}")
    missing = required_cols.difference(table.column_names)
    assert_true(not missing, f"{path} missing columns: {sorted(missing)}")
    return table


def run_unit_smoke(output_dir: Path) -> None:
    reset_dir(output_dir)
    cfg = synthetic_cfg()
    print(f"[unit] output={output_dir}")

    print("[unit] writing synthetic base parquet + sidecar")
    base_path = write_synthetic_base(output_dir, cfg)

    bare_config: dict[str, Any] = {
        "model": {"d_model": cfg.d_model, "target_layer": cfg.target_layer},
        "injection": {
            "injection_char": cfg.injection_char,
            "injection_token_id": None,
            "injection_left_neighbor_id": None,
            "injection_right_neighbor_id": None,
            "injection_scale": None,
            "mse_scale": "sqrt_d_model",
        },
        "prompts": {"av": AV_TEMPLATE, "ar": AR_TEMPLATE},
    }
    merged = merge_sidecar_into_config(bare_config, base_path)
    assert_true(merged["injection"]["injection_scale"] == cfg.injection_scale, "base sidecar injection_scale did not merge")
    merged = merge_base_sidecar_if_available({**bare_config, "datagen": {"output_dir": str(output_dir)}}, output_dir)
    assert_true(merged["injection"]["injection_scale"] == cfg.injection_scale, "prepare_datasets sidecar merge failed")

    print("[unit] splitting by doc_id")
    paths = split_dataset(base_path, output_dir / "splits", av_sft_frac=0.25, ar_sft_frac=0.25, rl_frac=0.50)
    for name, path in paths.items():
        validate_parquet(path, min_rows=1, required_cols={ACTIVATION_COLUMN, "doc_id"})
        print(f"  {name}: {path}")

    print("[unit] creating synthetic API explanations")
    write_explained(paths["av_sft_raw"], output_dir / "splits" / "av_sft_explained.parquet")
    write_explained(paths["ar_sft_raw"], output_dir / "splits" / "ar_sft_explained.parquet")

    print("[unit] building final training parquets")
    build_av_sft_dataset(
        output_dir / "splits" / "av_sft_explained.parquet",
        output_dir / "av_sft.parquet",
        cfg.av_prompt_template,
        cfg.injection_char,
    )
    write_dataset_sidecar(output_dir / "av_sft.parquet", cfg, base_model="synthetic/nano", stage="av_sft")
    build_ar_sft_dataset(
        output_dir / "splits" / "ar_sft_explained.parquet",
        output_dir / "ar_sft.parquet",
        cfg.ar_prompt_template,
    )
    write_dataset_sidecar(output_dir / "ar_sft.parquet", cfg, base_model="synthetic/nano", stage="ar_sft")
    build_rl_dataset(paths["rl_raw"], output_dir / "rl.parquet", cfg.av_prompt_template, cfg.injection_char)
    write_dataset_sidecar(output_dir / "rl.parquet", cfg, base_model="synthetic/nano", stage="rl")

    av_table = validate_parquet(output_dir / "av_sft.parquet", min_rows=1, required_cols={"prompt", "response", ACTIVATION_COLUMN})
    ar_table = validate_parquet(output_dir / "ar_sft.parquet", min_rows=1, required_cols={"prompt", ACTIVATION_COLUMN})
    rl_table = validate_parquet(output_dir / "rl.parquet", min_rows=1, required_cols={"prompt", ACTIVATION_COLUMN})

    assert_true("<explanation>" in av_table.column("response")[0].as_py(), "AV response is not explanation-tagged")
    assert_true(cfg.injection_char in av_table.column("prompt")[0].as_py()[0]["content"], "AV prompt missing injection char")
    assert_true("<summary>" in ar_table.column("prompt")[0].as_py(), "AR prompt missing summary suffix")
    assert_true(cfg.injection_char in rl_table.column("prompt")[0].as_py()[0]["content"], "RL prompt missing injection char")

    for stage in ("base", "av_sft", "ar_sft", "rl"):
        parquet = base_path if stage == "base" else output_dir / f"{stage}.parquet"
        meta = read_sidecar(parquet)
        assert_true(meta["extraction"]["injection_scale"] == cfg.injection_scale, f"{stage} sidecar lost injection_scale")

    print("[unit] validating injection math")
    try:
        import torch

        from nano_nla.injection import inject_at_marked_positions

        input_ids = torch.tensor([[1, cfg.injection_left_neighbor_id, cfg.injection_token_id, cfg.injection_right_neighbor_id]])
        inputs_embeds = torch.zeros((1, 4, cfg.d_model), dtype=torch.float32)
        activation = torch.tensor([[1.0, 2.0, 2.0]], dtype=torch.float32)
        injected = inject_at_marked_positions(input_ids, inputs_embeds, activation, cfg)
        injected_norm = float(injected[0, 2].norm().item())
        assert_true(abs(injected_norm - cfg.injection_scale) < 1e-5, f"injected norm {injected_norm} != {cfg.injection_scale}")
    except OSError as exc:
        print(f"[unit] SKIP torch injection check: {exc}")

    manifest = {
        "base": str(base_path),
        "av_sft": str(output_dir / "av_sft.parquet"),
        "ar_sft": str(output_dir / "ar_sft.parquet"),
        "rl": str(output_dir / "rl.parquet"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[unit] PASS")


def print_windows_memory_hint() -> None:
    if os.name != "nt":
        return
    print(
        "[model] If this fails with Windows os error 1455, enable a system-managed "
        "pagefile or raise virtual memory before loading Qwen."
    )


def run_model_smoke(output_dir: Path, docs: int, positions_per_doc: int, max_length: int, device: str) -> None:
    reset_dir(output_dir)
    print_windows_memory_hint()

    from nano_nla.datagen.extract_activations import compute_injection_scale, extract_activations, save_to_parquet

    config = load_config("configs/qwen05b.yaml")
    config["datagen"]["corpus"]["length"] = docs
    config["datagen"]["extraction"]["positions_per_doc"] = positions_per_doc
    config["datagen"]["output_dir"] = str(output_dir)

    print(f"[model] loading {docs} FineWeb docs")
    from datasets import load_dataset
    from transformers import AutoTokenizer

    corpus = config["datagen"]["corpus"]
    ds = load_dataset(corpus["name"], corpus["config"], split=corpus["split"], streaming=True)
    texts = []
    for i, row in enumerate(ds):
        if i >= docs:
            break
        texts.append(row[corpus.get("text_column", "text")])

    print("[model] extracting activations")
    vectors, trunc_texts, n_tokens, doc_ids, norms = extract_activations(
        model_name=config["model"]["name"],
        layer_index=config["model"]["target_layer"],
        texts=texts,
        positions_per_doc=positions_per_doc,
        max_length=max_length,
        batch_size=1,
        seed=config["datagen"]["extraction"]["seed"],
        device=device,
        min_position=config["datagen"]["extraction"].get("min_position", 50),
    )
    assert_true(vectors, "model smoke extracted no vectors")
    assert_true(len(vectors[0]) == config["model"]["d_model"], "d_model mismatch")

    injection_scale = compute_injection_scale(norms)
    config["injection"]["injection_scale"] = injection_scale
    base_path = output_dir / "base.parquet"
    save_to_parquet(base_path, vectors, trunc_texts, n_tokens, doc_ids, config["model"]["target_layer"])

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"], trust_remote_code=True)
    nla_cfg = build_nla_config_from_yaml(config, tokenizer)
    write_dataset_sidecar(base_path, nla_cfg, base_model=config["model"]["name"], stage="base")
    print(f"[model] PASS vectors={len(vectors)} token={nla_cfg.injection_token_id} injection_scale={injection_scale}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["unit", "model"], default="unit")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--docs", type=int, default=5, help="Real-model smoke document count")
    parser.add_argument("--positions-per-doc", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir or ("data/smoke_unit" if args.mode == "unit" else "data/smoke_model"))
    if args.mode == "unit":
        run_unit_smoke(output_dir)
    else:
        run_model_smoke(output_dir, args.docs, args.positions_per_doc, args.max_length, args.device)


if __name__ == "__main__":
    main()
