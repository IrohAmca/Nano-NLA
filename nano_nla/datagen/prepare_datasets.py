"""Stages 1 & 3: Split base parquet into train splits and build final datasets.

Stage 1: Split base.parquet → av_sft_raw / ar_sft_raw / rl_raw
Stage 3: Build final training parquets with proper prompt formatting

Adapted from:
  - https://github.com/kitft/natural_language_autoencoders/blob/main/nla/datagen/stage1_split.py
  - https://github.com/kitft/natural_language_autoencoders/blob/main/nla/datagen/stage3_build.py

Output formats:
  av_sft.parquet: prompt (messages), response (<explanation>...</explanation>), activation_vector
  ar_sft.parquet: prompt (formatted AR string), activation_vector
  rl.parquet:     prompt (messages), activation_vector
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from nano_nla.schema import (
    ACTIVATION_COLUMN,
    build_nla_config_from_yaml,
    load_config,
    merge_sidecar_into_config,
    wrap_explanation,
    write_dataset_sidecar,
)

MESSAGE_TYPE = pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))


def merge_base_sidecar_if_available(config: dict, output_dir: Path) -> dict:
    """Load stage-0 computed token/scale values when running later stages."""
    base_path = output_dir / "base.parquet"
    merged = merge_sidecar_into_config(config, base_path)
    before = config.get("injection", {}).get("injection_scale")
    after = merged.get("injection", {}).get("injection_scale")
    if after != before:
        print(f"[config] Loaded computed NLA metadata from {base_path}.nla_meta.yaml")
    return merged


# ─── Stage 1: Split ────────────────────────────────────────────────────────


def split_dataset(
    base_parquet: str | Path,
    output_dir: str | Path,
    av_sft_frac: float = 0.25,
    ar_sft_frac: float = 0.25,
    rl_frac: float = 0.50,
    seed: int = 42,
) -> dict[str, Path]:
    """Split base parquet into training splits by document ID.

    Splits by doc_id to prevent data leakage — all positions from a
    single document go into the same split.
    """
    base_parquet = Path(base_parquet)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assert abs(av_sft_frac + ar_sft_frac + rl_frac - 1.0) < 1e-6, (
        f"fractions must sum to 1.0, got {av_sft_frac + ar_sft_frac + rl_frac}"
    )

    parquet_file = pq.ParquetFile(str(base_parquet))
    n_total = parquet_file.metadata.num_rows

    # Get unique doc IDs and shuffle
    doc_ids = pq.read_table(str(base_parquet), columns=["doc_id"]).column("doc_id").to_pylist()
    unique_docs = sorted(set(doc_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_docs)

    n_docs = len(unique_docs)
    av_end = int(n_docs * av_sft_frac)
    ar_end = av_end + int(n_docs * ar_sft_frac)

    av_docs = set(unique_docs[:av_end])
    ar_docs = set(unique_docs[av_end:ar_end])
    rl_docs = set(unique_docs[ar_end:])

    split_docs = {
        "av_sft_raw": av_docs,
        "ar_sft_raw": ar_docs,
        "rl_raw": rl_docs,
    }
    paths = {name: output_dir / f"{name}.parquet" for name in split_docs}
    tmp_paths = {name: path.with_suffix(".tmp") for name, path in paths.items()}
    writers: dict[str, pq.ParquetWriter | None] = {name: None for name in split_docs}
    counts = {name: 0 for name in split_docs}
    success = False

    try:
        for row_group_idx in range(parquet_file.metadata.num_row_groups):
            row_group = parquet_file.read_row_group(row_group_idx)
            row_doc_ids = row_group.column("doc_id").to_pylist()
            for name, docs in split_docs.items():
                mask = pa.array([doc_id in docs for doc_id in row_doc_ids], type=pa.bool_())
                subset = row_group.filter(mask)
                if subset.num_rows == 0:
                    continue
                if writers[name] is None:
                    writers[name] = pq.ParquetWriter(str(tmp_paths[name]), subset.schema)
                writers[name].write_table(subset)
                counts[name] += subset.num_rows
        success = True
    finally:
        for writer in writers.values():
            if writer is not None:
                writer.close()
        if not success:
            for tmp_path in tmp_paths.values():
                if tmp_path.exists():
                    tmp_path.unlink()

    for name, path in paths.items():
        tmp_paths[name].replace(path)
        print(f"[split] {name}: {counts[name]} rows ({counts[name]/n_total*100:.1f}%)")

    return paths


# ─── Stage 3: Build final datasets ─────────────────────────────────────────


def build_av_sft_dataset(
    explained_parquet: str | Path,
    output_parquet: str | Path,
    av_prompt_template: str,
    injection_char: str,
    keep_debug: bool = True,
) -> None:
    """Build AV-SFT dataset with proper prompt formatting.

    Input columns: activation_vector, api_explanation, detokenized_text_truncated, ...
    Output columns:
      - prompt: list[dict] messages with injection_char in the canonical slot
      - response: <explanation>\\n{api_explanation}\\n</explanation>
      - activation_vector: raw float list
      - (debug) detokenized_text_truncated
    """
    table = pq.read_table(str(explained_parquet))
    explanations = table.column("api_explanation").to_pylist()
    vectors = table.column(ACTIVATION_COLUMN).to_pylist()

    # Build the user content with injection placeholder
    # The actual injection happens at training time — here we just set up the prompt
    user_content = av_prompt_template.format(injection_char=injection_char)

    prompts = []
    responses = []
    valid_vectors = []
    valid_texts = []

    texts = table.column("detokenized_text_truncated").to_pylist() if keep_debug else [None] * len(vectors)

    for i, (explanation, vector) in enumerate(zip(explanations, vectors)):
        if not explanation or explanation.startswith("["):
            continue  # Skip failed/empty

        # Prompt: the AV template as a user message
        prompt_msgs = [{"role": "user", "content": user_content}]
        prompts.append(prompt_msgs)

        # Response: wrapped in explanation tags
        responses.append(wrap_explanation(explanation))
        valid_vectors.append(vector)
        if keep_debug:
            valid_texts.append(texts[i])

    if not valid_vectors:
        raise RuntimeError(f"no valid AV-SFT rows in {explained_parquet}")

    # Build output table
    columns = [
        pa.array(prompts, type=MESSAGE_TYPE),
        pa.array(responses, type=pa.string()),
        pa.array(valid_vectors, type=pa.list_(pa.float32(), len(valid_vectors[0]))),
    ]
    names = ["prompt", "response", ACTIVATION_COLUMN]
    if keep_debug:
        columns.append(pa.array(valid_texts, type=pa.string()))
        names.append("detokenized_text_truncated")

    out_table = pa.Table.from_arrays(columns, names=names)
    output_path = Path(output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, str(output_path))
    print(f"[build] AV-SFT: {len(out_table)} rows -> {output_path}")


def build_ar_sft_dataset(
    explained_parquet: str | Path,
    output_parquet: str | Path,
    ar_prompt_template: str,
    keep_debug: bool = True,
) -> None:
    """Build AR-SFT dataset with proper prompt formatting.

    Input columns: activation_vector, api_explanation, ...
    Output columns:
      - prompt: formatted AR string "Summary of the following text: <text>{explanation}</text> <summary>"
      - activation_vector: raw float list (the regression target)
    """
    table = pq.read_table(str(explained_parquet))
    explanations = table.column("api_explanation").to_pylist()
    vectors = table.column(ACTIVATION_COLUMN).to_pylist()

    prompts = []
    valid_vectors = []
    valid_texts = []

    texts = table.column("detokenized_text_truncated").to_pylist() if keep_debug else [None] * len(vectors)

    for i, (explanation, vector) in enumerate(zip(explanations, vectors)):
        if not explanation or explanation.startswith("["):
            continue

        # Format the AR prompt with the explanation
        prompt = ar_prompt_template.format(explanation=explanation)
        prompts.append(prompt)
        valid_vectors.append(vector)
        if keep_debug:
            valid_texts.append(texts[i])

    if not valid_vectors:
        raise RuntimeError(f"no valid AR-SFT rows in {explained_parquet}")

    columns = [
        pa.array(prompts, type=pa.string()),
        pa.array(valid_vectors, type=pa.list_(pa.float32(), len(valid_vectors[0]))),
    ]
    names = ["prompt", ACTIVATION_COLUMN]
    if keep_debug:
        columns.append(pa.array(valid_texts, type=pa.string()))
        names.append("detokenized_text_truncated")

    out_table = pa.Table.from_arrays(columns, names=names)
    output_path = Path(output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, str(output_path))
    print(f"[build] AR-SFT: {len(out_table)} rows -> {output_path}")


def build_rl_dataset(
    raw_parquet: str | Path,
    output_parquet: str | Path,
    av_prompt_template: str,
    injection_char: str,
    keep_debug: bool = True,
) -> None:
    """Build RL dataset — just prompt + activation_vector, no explanation.

    The AV generates explanations during RL rollouts; the AR scores them.
    """
    table = pq.read_table(str(raw_parquet))
    vectors = table.column(ACTIVATION_COLUMN).to_pylist()
    if not vectors:
        raise RuntimeError(f"no RL rows in {raw_parquet}")

    user_content = av_prompt_template.format(injection_char=injection_char)

    prompts = [[{"role": "user", "content": user_content}] for _ in vectors]
    columns = [
        pa.array(prompts, type=MESSAGE_TYPE),
        pa.array(vectors, type=pa.list_(pa.float32(), len(vectors[0]))),
    ]
    names = ["prompt", ACTIVATION_COLUMN]
    if keep_debug and "detokenized_text_truncated" in table.column_names:
        columns.append(pa.array(table.column("detokenized_text_truncated").to_pylist(), type=pa.string()))
        names.append("detokenized_text_truncated")

    out_table = pa.Table.from_arrays(columns, names=names)
    output_path = Path(output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, str(output_path))
    print(f"[build] RL: {len(out_table)} rows -> {output_path}")


# ─── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Split and build NLA datasets")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--stage", choices=["split", "build", "all"], default="all",
                        help="Which stage to run")
    args = parser.parse_args()

    config = load_config(args.config)
    datagen_cfg = config["datagen"]
    split_cfg = datagen_cfg["split"]
    output_dir = Path(datagen_cfg["output_dir"])
    config = merge_base_sidecar_if_available(config, output_dir)
    prompts = config["prompts"]
    injection_char = config["injection"]["injection_char"]

    if args.stage in ("split", "all"):
        print("\n" + "=" * 60)
        print("STAGE 1: Splitting base.parquet")
        print("=" * 60)

        base_path = output_dir / "base.parquet"
        assert base_path.exists(), f"base.parquet not found at {base_path}. Run extract_activations.py first."

        split_dataset(
            base_path,
            output_dir / "splits",
            av_sft_frac=split_cfg["av_sft_frac"],
            ar_sft_frac=split_cfg["ar_sft_frac"],
            rl_frac=split_cfg["rl_frac"],
            seed=split_cfg["seed"],
        )

    if args.stage in ("build", "all"):
        print("\n" + "=" * 60)
        print("STAGE 3: Building final training datasets")
        print("=" * 60)

        splits_dir = output_dir / "splits"

        # AV-SFT
        av_explained = splits_dir / "av_sft_explained.parquet"
        if av_explained.exists():
            build_av_sft_dataset(
                av_explained,
                output_dir / "av_sft.parquet",
                prompts["av"],
                injection_char,
            )

            # Write sidecar for AV-SFT
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                config["model"]["name"], trust_remote_code=True
            )
            nla_cfg = build_nla_config_from_yaml(config, tokenizer)
            write_dataset_sidecar(
                output_dir / "av_sft.parquet", nla_cfg,
                base_model=config["model"]["name"],
                stage="av_sft",
            )
        else:
            print(f"[skip] {av_explained} not found; run generate_summaries.py first")

        # AR-SFT
        ar_explained = splits_dir / "ar_sft_explained.parquet"
        if ar_explained.exists():
            build_ar_sft_dataset(
                ar_explained,
                output_dir / "ar_sft.parquet",
                prompts["ar"],
            )

            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                config["model"]["name"], trust_remote_code=True
            )
            nla_cfg = build_nla_config_from_yaml(config, tokenizer)
            write_dataset_sidecar(
                output_dir / "ar_sft.parquet", nla_cfg,
                base_model=config["model"]["name"],
                stage="ar_sft",
            )
        else:
            print(f"[skip] {ar_explained} not found; run generate_summaries.py first")

        # RL
        rl_raw = splits_dir / "rl_raw.parquet"
        if rl_raw.exists():
            build_rl_dataset(
                rl_raw,
                output_dir / "rl.parquet",
                prompts["av"],
                injection_char,
            )

            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                config["model"]["name"], trust_remote_code=True
            )
            nla_cfg = build_nla_config_from_yaml(config, tokenizer)
            write_dataset_sidecar(
                output_dir / "rl.parquet", nla_cfg,
                base_model=config["model"]["name"],
                stage="rl",
            )

    print("\nDataset preparation complete.")


if __name__ == "__main__":
    main()
