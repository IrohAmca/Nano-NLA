"""Stage 0: Extract residual stream activations from the target model.

Adapted from: https://github.com/kitft/natural_language_autoencoders/blob/main/nla/datagen/stage0_extract.py

Pipeline:
  1. Load corpus (HuggingFace FineWeb subset)
  2. For each document, randomly truncate and extract layer-K hidden states
  3. Save as parquet: (activation_vector, detokenized_text_truncated, n_raw_tokens, doc_id)
  4. Measure injection_scale from activation norm distribution
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import random
import re
import shutil
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from nano_nla.models import enable_cuda_performance, resolve_torch_device
from nano_nla.schema import (
    ACTIVATION_COLUMN,
    build_nla_config_from_yaml,
    load_config,
    write_dataset_sidecar,
)

MIN_POSITION = 50
DEFAULT_SHARD_FLUSH_ROWS = 1000
DEFAULT_SHARD_FLUSH_DOCS = 200
DONE_BASENAME = "completed_docs"
SHARD_RE = re.compile(r"worker_(\d+)_chunk_(\d+)\.parquet$")


def resolve_torch_dtype(name: str | None, *, device: str) -> torch.dtype:
    """Resolve an extraction dtype, defaulting to bf16/fp16 on CUDA."""
    if name is None or name == "auto":
        if device.startswith("cuda"):
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    value = str(name).lower()
    if value in {"float32", "fp32"}:
        return torch.float32
    if value in {"float16", "fp16"}:
        return torch.float16
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def load_extraction_model(model_name: str, device: str, dtype: torch.dtype):
    """Load target model for activation extraction."""
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
        enable_cuda_performance()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return model


def sample_positions(seq_len: int, positions_per_doc: int, min_position: int, seed: int, doc_idx: int) -> list[int]:
    """Sample truncation lengths deterministically per document."""
    if seq_len <= min_position:
        return []
    max_pos = seq_len + 1
    min_pos = min_position + 1
    rng = random.Random(seed + doc_idx * 1_000_003)
    return sorted(rng.sample(range(min_pos, max_pos), k=min(positions_per_doc, max_pos - min_pos)))


def layer_hidden_at_index(outputs: Any, layer_index: int) -> torch.Tensor:
    """Return hidden states after layer_index, falling back to final hidden state."""
    if layer_index + 1 >= len(outputs.hidden_states):
        return outputs.hidden_states[-1]
    return outputs.hidden_states[layer_index + 1]


def extract_doc_activations(
    *,
    tokenizer: Any,
    model: torch.nn.Module,
    device: str,
    layer_index: int,
    doc_idx: int,
    text: str,
    positions_per_doc: int,
    max_length: int,
    batch_size: int,
    seed: int,
    min_position: int,
) -> tuple[list[list[float]], list[str], list[int], list[int], list[float]]:
    """Extract all sampled activation positions for one document."""
    if not text or not text.strip():
        return [], [], [], [], []

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"]
    seq_len = input_ids.shape[1]
    positions = sample_positions(seq_len, positions_per_doc, min_position, seed, doc_idx)
    if not positions:
        return [], [], [], [], []

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    activation_vectors: list[list[float]] = []
    truncated_texts: list[str] = []
    n_raw_tokens_list: list[int] = []
    doc_ids: list[int] = []
    norms: list[float] = []

    microbatch = max(1, int(batch_size))
    for start in range(0, len(positions), microbatch):
        chunk_positions = positions[start : start + microbatch]
        max_pos = max(chunk_positions)
        batch_ids = torch.full((len(chunk_positions), max_pos), pad_id, dtype=torch.long)
        attention_mask = torch.zeros_like(batch_ids)
        for row_idx, pos in enumerate(chunk_positions):
            batch_ids[row_idx, :pos] = input_ids[0, :pos]
            attention_mask[row_idx, :pos] = 1

        with torch.no_grad():
            outputs = model(
                batch_ids.to(device),
                attention_mask=attention_mask.to(device),
                output_hidden_states=True,
                use_cache=False,
            )

        hidden = layer_hidden_at_index(outputs, layer_index)
        batch_idx = torch.arange(len(chunk_positions), device=hidden.device)
        last_idx = torch.tensor([pos - 1 for pos in chunk_positions], device=hidden.device)
        h_batch = hidden[batch_idx, last_idx].float().cpu()

        for row_idx, pos in enumerate(chunk_positions):
            h_cpu = h_batch[row_idx]
            activation_vectors.append(h_cpu.tolist())
            norms.append(h_cpu.norm().item())
            truncated_texts.append(tokenizer.decode(input_ids[0, :pos], skip_special_tokens=True))
            n_raw_tokens_list.append(pos)
            doc_ids.append(doc_idx)

    return activation_vectors, truncated_texts, n_raw_tokens_list, doc_ids, norms


def extract_activations(
    model_name: str,
    layer_index: int,
    texts: list[str],
    positions_per_doc: int,
    max_length: int,
    batch_size: int,
    seed: int,
    device: str = "auto",
    min_position: int = MIN_POSITION,
) -> tuple[list[list[float]], list[str], list[int], list[int], list[float]]:
    """Extract residual stream activations from the target model.

    For each document:
      1. Tokenize to at most max_length tokens
      2. Pick `positions_per_doc` random truncation points with enough left context
      3. Run forward pass, extract layer_index hidden state at the last token
      4. Store the raw activation vector (NO normalization — training normalizes)

    Returns:
      (activation_vectors, truncated_texts, n_raw_tokens, doc_ids, norms)
    """
    device = str(resolve_torch_device(device, require_cuda=str(device).lower() in {"auto", "gpu"}))
    print(f"[extract] Loading model {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = load_extraction_model(model_name, device, resolve_torch_dtype(None, device=device))

    activation_vectors: list[list[float]] = []
    truncated_texts: list[str] = []
    n_raw_tokens_list: list[int] = []
    doc_ids: list[int] = []
    norms: list[float] = []

    for doc_idx, text in enumerate(tqdm(texts, desc="Extracting activations")):
        if not text or not text.strip():
            continue

        vectors, texts_out, n_tokens, ids_out, norms_out = extract_doc_activations(
            tokenizer=tokenizer,
            model=model,
            device=device,
            layer_index=layer_index,
            doc_idx=doc_idx,
            text=text,
            positions_per_doc=positions_per_doc,
            max_length=max_length,
            batch_size=batch_size,
            seed=seed,
            min_position=min_position,
        )
        activation_vectors.extend(vectors)
        truncated_texts.extend(texts_out)
        n_raw_tokens_list.extend(n_tokens)
        doc_ids.extend(ids_out)
        norms.extend(norms_out)

    print(f"[extract] Extracted {len(activation_vectors)} vectors from {len(texts)} docs")
    return activation_vectors, truncated_texts, n_raw_tokens_list, doc_ids, norms


def compute_injection_scale(norms: list[float]) -> float:
    """Compute injection_scale from the activation norm distribution.

    Following the paper's heuristic:
      injection_scale ≈ 75th percentile of activation norms

    The AV was trained with vectors at this exact L2-norm;
    raw-magnitude vectors are out-of-distribution.
    """
    if not norms:
        raise ValueError("no activation norms collected; lower min_position or provide longer documents")
    arr = np.array(norms)
    p75 = float(np.percentile(arr, 75))
    print(f"[injection_scale] norm stats: "
          f"mean={arr.mean():.1f} std={arr.std():.1f} "
          f"min={arr.min():.1f} max={arr.max():.1f} "
          f"p25={np.percentile(arr, 25):.1f} p50={np.percentile(arr, 50):.1f} "
          f"p75={p75:.1f}")
    # Round to a clean number
    scale = round(p75 / 5) * 5  # Round to nearest 5
    if scale < 10:
        scale = round(p75, 1)
    print(f"[injection_scale] chosen: {scale}")
    return float(scale)


def save_to_parquet(
    output_path: str | Path,
    activation_vectors: list[list[float]],
    truncated_texts: list[str],
    n_raw_tokens: list[int],
    doc_ids: list[int],
    layer_index: int,
) -> None:
    """Save extracted data as parquet — the standard NLA data format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not activation_vectors:
        raise ValueError("cannot write empty activation table")
    d_model = len(activation_vectors[0])
    schema = pa.schema([
        (ACTIVATION_COLUMN, pa.list_(pa.float32(), d_model)),
        ("detokenized_text_truncated", pa.string()),
        ("n_raw_tokens", pa.int64()),
        ("activation_layer", pa.int64()),
        ("doc_id", pa.int64()),
    ])

    table = pa.Table.from_pydict({
        ACTIVATION_COLUMN: activation_vectors,
        "detokenized_text_truncated": truncated_texts,
        "n_raw_tokens": n_raw_tokens,
        "activation_layer": [layer_index] * len(activation_vectors),
        "doc_id": doc_ids,
    }, schema=schema)
    pq.write_table(table, str(output_path))
    print(f"[save] Wrote {len(activation_vectors)} rows to {output_path}")


def _flush_rows_to_shard(
    shard_dir: Path,
    worker_id: int,
    chunk_id: int,
    vectors: list[list[float]],
    texts: list[str],
    n_tokens: list[int],
    doc_ids: list[int],
    layer_index: int,
) -> Path | None:
    if not vectors:
        return None
    shard_path = shard_dir / f"worker_{worker_id:02d}_chunk_{chunk_id:05d}.parquet"
    tmp_path = shard_path.with_suffix(f"{shard_path.suffix}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    save_to_parquet(tmp_path, vectors, texts, n_tokens, doc_ids, layer_index)
    tmp_path.replace(shard_path)
    return shard_path


def done_path_for(shard_dir: Path, worker_id: int) -> Path:
    return shard_dir / f"{DONE_BASENAME}_worker_{worker_id:02d}.jsonl"


def append_done_records(shard_dir: Path, worker_id: int, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path = done_path_for(shard_dir, worker_id)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()


def read_completed_doc_ids(shard_dir: Path) -> set[int]:
    """Read completed doc IDs from done logs and existing shard files."""
    completed: set[int] = set()
    for path in shard_dir.glob(f"{DONE_BASENAME}_worker_*.jsonl"):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    completed.add(int(json.loads(line)["doc_id"]))
                except (KeyError, ValueError, json.JSONDecodeError):
                    continue

    # Backward-compatible resume for shards written before done logs existed.
    for path in shard_dir.glob("worker_*_chunk_*.parquet"):
        try:
            table = pq.read_table(str(path), columns=["doc_id"])
        except Exception:
            continue
        completed.update(int(value) for value in table.column("doc_id").to_pylist())
    return completed


def next_chunk_ids_by_worker(shard_dir: Path, num_workers: int) -> dict[int, int]:
    next_ids = {worker_id: 0 for worker_id in range(num_workers)}
    for path in shard_dir.glob("worker_*_chunk_*.parquet"):
        match = SHARD_RE.match(path.name)
        if match is None:
            continue
        worker_id = int(match.group(1))
        chunk_id = int(match.group(2))
        next_ids[worker_id] = max(next_ids.get(worker_id, 0), chunk_id + 1)
    return next_ids


def _worker_main(
    *,
    worker_id: int,
    device: str,
    model_name: str,
    layer_index: int,
    positions_per_doc: int,
    max_length: int,
    batch_size: int,
    seed: int,
    min_position: int,
    shard_dir: str,
    flush_rows: int,
    flush_docs: int,
    start_chunk_id: int,
    dtype_name: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    shard_paths: list[str] = []
    norms: list[float] = []
    rows_written = 0
    docs_seen = 0
    chunk_id = start_chunk_id
    vectors: list[list[float]] = []
    texts: list[str] = []
    n_tokens: list[int] = []
    doc_ids: list[int] = []
    done_records: list[dict[str, Any]] = []
    docs_since_flush = 0
    try:
        dtype = resolve_torch_dtype(dtype_name, device=device)
        print(f"[worker {worker_id}] loading {model_name} on {device} ({dtype})")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = load_extraction_model(model_name, device, dtype)

        while True:
            item = task_queue.get()
            if item is None:
                break
            doc_idx, text = item
            docs_seen += 1
            out = extract_doc_activations(
                tokenizer=tokenizer,
                model=model,
                device=device,
                layer_index=layer_index,
                doc_idx=int(doc_idx),
                text=str(text),
                positions_per_doc=positions_per_doc,
                max_length=max_length,
                batch_size=batch_size,
                seed=seed,
                min_position=min_position,
            )
            v, t, nt, did, ns = out
            vectors.extend(v)
            texts.extend(t)
            n_tokens.extend(nt)
            doc_ids.extend(did)
            norms.extend(ns)
            done_records.append({"doc_id": int(doc_idx), "rows": len(v)})
            docs_since_flush += 1
            if len(vectors) >= flush_rows or docs_since_flush >= flush_docs:
                path = _flush_rows_to_shard(
                    Path(shard_dir), worker_id, chunk_id, vectors, texts, n_tokens, doc_ids, layer_index
                )
                if path is not None:
                    shard_paths.append(str(path))
                    rows_written += len(vectors)
                    chunk_id += 1
                append_done_records(Path(shard_dir), worker_id, done_records)
                vectors, texts, n_tokens, doc_ids = [], [], [], []
                done_records = []
                docs_since_flush = 0

        path = _flush_rows_to_shard(
            Path(shard_dir), worker_id, chunk_id, vectors, texts, n_tokens, doc_ids, layer_index
        )
        if path is not None:
            shard_paths.append(str(path))
            rows_written += len(vectors)
        append_done_records(Path(shard_dir), worker_id, done_records)
        result_queue.put(
            {
                "worker_id": worker_id,
                "device": device,
                "docs_seen": docs_seen,
                "rows_written": rows_written,
                "shard_paths": shard_paths,
                "norms": norms,
                "error": None,
            }
        )
    except Exception:
        result_queue.put(
            {
                "worker_id": worker_id,
                "device": device,
                "docs_seen": docs_seen,
                "rows_written": rows_written,
                "shard_paths": shard_paths,
                "norms": norms,
                "error": traceback.format_exc(),
            }
        )


def parse_device_list(raw: str | list[str] | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",")]
    else:
        values = [str(part).strip() for part in raw]
    return [value for value in values if value]


def resolve_worker_devices(raw: str | list[str] | None, fallback_device: str | None = None) -> list[str]:
    requested = parse_device_list(raw)
    if not requested and fallback_device:
        requested = [fallback_device]
    if not requested:
        requested = ["auto"]

    devices: list[str] = []
    for item in requested:
        lowered = item.lower()
        if lowered in {"auto", "gpu", "cuda:all"}:
            if not torch.cuda.is_available():
                raise RuntimeError(f"{item} requested, but torch.cuda.is_available() is false")
            devices.extend([f"cuda:{i}" for i in range(torch.cuda.device_count())])
        elif lowered == "cuda":
            devices.append("cuda:0")
        else:
            devices.append(item)

    cuda_count = torch.cuda.device_count()
    for device in devices:
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(f"{device} requested, but CUDA is not available")
            index = int(device.split(":", 1)[1]) if ":" in device else 0
            if index >= cuda_count:
                raise RuntimeError(f"{device} requested, but only {cuda_count} CUDA device(s) are visible")
    return devices


def merge_parquet_shards(shard_paths: list[Path], output_path: Path) -> None:
    if not shard_paths:
        raise RuntimeError("no activation shards were produced")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    row_count = 0
    writer: pq.ParquetWriter | None = None
    try:
        for path in shard_paths:
            table = pq.read_table(str(path))
            if writer is None:
                writer = pq.ParquetWriter(str(tmp_path), table.schema)
            writer.write_table(table)
            row_count += table.num_rows
    finally:
        if writer is not None:
            writer.close()

    if row_count == 0:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError("activation shards were empty")
    tmp_path.replace(output_path)
    print(f"[merge] Wrote {row_count} rows from {len(shard_paths)} shard(s) to {output_path}")


def load_activation_norms(parquet_path: Path) -> list[float]:
    table = pq.read_table(str(parquet_path), columns=[ACTIVATION_COLUMN])
    return [float(np.linalg.norm(np.asarray(v, dtype=np.float32))) for v in table.column(ACTIVATION_COLUMN).to_pylist()]


def terminate_processes(processes: list[mp.Process]) -> None:
    for proc in processes:
        if proc.is_alive():
            proc.terminate()
    for proc in processes:
        proc.join(timeout=5)


def enqueue_with_health_check(task_queue: mp.Queue, result_queue: mp.Queue, processes: list[mp.Process], item: Any) -> None:
    while True:
        try:
            task_queue.put(item, timeout=1)
            return
        except queue.Full:
            pass
        for proc in processes:
            if not proc.is_alive() and proc.exitcode not in (0, None):
                terminate_processes(processes)
                raise RuntimeError(f"extraction worker pid={proc.pid} exited with code {proc.exitcode}")
        try:
            result = result_queue.get_nowait()
        except queue.Empty:
            continue
        if result.get("error"):
            terminate_processes(processes)
            raise RuntimeError(f"worker {result['worker_id']} failed:\n{result['error']}")


def extract_activations_parallel(
    *,
    model_name: str,
    layer_index: int,
    corpus_cfg: dict,
    positions_per_doc: int,
    max_length: int,
    batch_size: int,
    seed: int,
    output_dir: Path,
    devices: list[str],
    min_position: int = MIN_POSITION,
    flush_rows: int = DEFAULT_SHARD_FLUSH_ROWS,
    flush_docs: int = DEFAULT_SHARD_FLUSH_DOCS,
    dtype_name: str = "auto",
    resume: bool = True,
) -> tuple[Path, list[float], dict[str, Any]]:
    """Stream corpus rows through CUDA workers and merge shards."""
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "stage0_shards"
    if shard_dir.exists() and not resume:
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    completed_doc_ids = read_completed_doc_ids(shard_dir) if resume else set()
    next_chunk_ids = next_chunk_ids_by_worker(shard_dir, len(devices))
    flush_docs = max(1, int(flush_docs))
    flush_rows = max(1, int(flush_rows))
    if completed_doc_ids:
        print(f"[resume] Found {len(completed_doc_ids)} completed doc(s) in {shard_dir}; skipping them")

    corpus_name = corpus_cfg["name"]
    corpus_config = corpus_cfg.get("config")
    text_col = corpus_cfg.get("text_column", "text")
    start = int(corpus_cfg.get("start", 0))
    length = int(corpus_cfg.get("length", 10000))
    expected_doc_ids = set(range(start, start + length))
    if resume and expected_doc_ids and expected_doc_ids.issubset(completed_doc_ids):
        shard_paths = sorted(shard_dir.glob("worker_*_chunk_*.parquet"))
        base_path = output_dir / "base.parquet"
        merge_parquet_shards(shard_paths, base_path)
        return base_path, load_activation_norms(base_path), {
            "docs_enqueued": 0,
            "docs_skipped_resume": len(expected_doc_ids),
            "worker_devices": devices,
            "workers": [],
            "num_shards": len(shard_paths),
            "new_vectors": 0,
            "resume_complete": True,
        }

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue(maxsize=max(4, len(devices) * 4))
    result_queue = ctx.Queue()
    processes: list[mp.Process] = []

    for worker_id, device in enumerate(devices):
        proc = ctx.Process(
            target=_worker_main,
            kwargs={
                "worker_id": worker_id,
                "device": device,
                "model_name": model_name,
                "layer_index": layer_index,
                "positions_per_doc": positions_per_doc,
                "max_length": max_length,
                "batch_size": batch_size,
                "seed": seed,
                "min_position": min_position,
                "shard_dir": str(shard_dir),
                "flush_rows": flush_rows,
                "flush_docs": flush_docs,
                "start_chunk_id": next_chunk_ids.get(worker_id, 0),
                "dtype_name": dtype_name,
                "task_queue": task_queue,
                "result_queue": result_queue,
            },
        )
        proc.start()
        processes.append(proc)

    print(f"[corpus] Streaming {corpus_name} ({corpus_config}) to {len(devices)} worker(s): {devices}")
    ds = load_dataset(corpus_name, corpus_config, split=corpus_cfg["split"], streaming=True)

    docs_enqueued = 0
    docs_skipped = 0
    try:
        progress = tqdm(total=length, desc="Queueing docs")
        for i, row in enumerate(ds):
            if i < start:
                continue
            if i >= start + length:
                break
            if i in completed_doc_ids:
                docs_skipped += 1
                progress.update(1)
                continue
            enqueue_with_health_check(task_queue, result_queue, processes, (i, row[text_col]))
            docs_enqueued += 1
            progress.update(1)
        progress.close()

        for _ in processes:
            enqueue_with_health_check(task_queue, result_queue, processes, None)

        results: list[dict[str, Any]] = []
        while len(results) < len(processes):
            result = result_queue.get()
            if result.get("error"):
                terminate_processes(processes)
                raise RuntimeError(f"worker {result['worker_id']} failed:\n{result['error']}")
            results.append(result)

        for proc in processes:
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError(f"extraction worker pid={proc.pid} exited with code {proc.exitcode}")
    except Exception:
        terminate_processes(processes)
        raise

    shard_paths: list[Path] = sorted(shard_dir.glob("worker_*_chunk_*.parquet"))
    norms: list[float] = []
    worker_stats: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda r: r["worker_id"]):
        norms.extend(result["norms"])
        worker_stats.append(
            {
                "worker_id": result["worker_id"],
                "device": result["device"],
                "docs_seen": result["docs_seen"],
                "rows_written": result["rows_written"],
                "num_shards": len(result["shard_paths"]),
            }
        )
        print(
            f"[worker {result['worker_id']}] device={result['device']} "
            f"docs={result['docs_seen']} rows={result['rows_written']} shards={len(result['shard_paths'])}"
        )

    base_path = output_dir / "base.parquet"
    merge_parquet_shards(shard_paths, base_path)
    all_norms = load_activation_norms(base_path)
    return base_path, all_norms, {
        "docs_enqueued": docs_enqueued,
        "docs_skipped_resume": docs_skipped,
        "worker_devices": devices,
        "workers": worker_stats,
        "num_shards": len(shard_paths),
        "new_vectors": len(norms),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract activations from target model")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--device", default=None, help="Override device (auto/cuda:0)")
    parser.add_argument(
        "--devices",
        default=None,
        help="Comma-separated worker devices, e.g. auto,cuda:0,cuda:all. Overrides config worker_devices.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Delete existing stage0 shards and start extraction from scratch.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config["model"]
    datagen_cfg = config["datagen"]
    ext_cfg = datagen_cfg["extraction"]
    corpus_cfg = datagen_cfg["corpus"]
    output_dir = Path(datagen_cfg["output_dir"])

    worker_devices = resolve_worker_devices(
        args.devices if args.devices is not None else ext_cfg.get("worker_devices"),
        fallback_device=args.device,
    )
    print(f"[extract] Worker devices: {worker_devices}")

    base_path, norms, parallel_stats = extract_activations_parallel(
        model_name=model_cfg["name"],
        layer_index=model_cfg["target_layer"],
        corpus_cfg=corpus_cfg,
        positions_per_doc=ext_cfg["positions_per_doc"],
        max_length=ext_cfg["max_length"],
        batch_size=ext_cfg["batch_size"],
        seed=ext_cfg["seed"],
        output_dir=output_dir,
        devices=worker_devices,
        min_position=ext_cfg.get("min_position", MIN_POSITION),
        flush_rows=int(ext_cfg.get("shard_flush_rows", DEFAULT_SHARD_FLUSH_ROWS)),
        flush_docs=int(ext_cfg.get("shard_flush_docs", DEFAULT_SHARD_FLUSH_DOCS)),
        dtype_name=str(ext_cfg.get("dtype", "auto")),
        resume=bool(ext_cfg.get("resume", True)) and not args.restart,
    )

    injection_scale = compute_injection_scale(norms)
    stats = {
        "injection_scale": injection_scale,
        "norm_mean": float(np.mean(norms)),
        "norm_std": float(np.std(norms)),
        "norm_p75": float(np.percentile(norms, 75)),
        "num_vectors": len(norms),
        "num_docs": parallel_stats["docs_enqueued"],
        "model": model_cfg["name"],
        "layer": model_cfg["target_layer"],
        "parallel": parallel_stats,
    }
    stats_path = output_dir / "extraction_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[stats] Saved to {stats_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], trust_remote_code=True)
    config["injection"]["injection_scale"] = injection_scale
    nla_cfg = build_nla_config_from_yaml(config, tokenizer)
    write_dataset_sidecar(base_path, nla_cfg, base_model=model_cfg["name"], stage="base")

    config["injection"]["injection_token_id"] = nla_cfg.injection_token_id
    config["injection"]["injection_left_neighbor_id"] = nla_cfg.injection_left_neighbor_id
    config["injection"]["injection_right_neighbor_id"] = nla_cfg.injection_right_neighbor_id
    config["injection"]["injection_scale"] = injection_scale

    source_config_path = Path(args.config)
    computed_stem = (
        source_config_path.stem
        if source_config_path.stem.endswith("_computed")
        else f"{source_config_path.stem}_computed"
    )
    updated_config_path = output_dir / f"{computed_stem}.yaml"
    updated_config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"[config] Updated config saved to {updated_config_path}")
    print(f"[config] injection_token_id={nla_cfg.injection_token_id}")
    print(f"[config] injection_scale={injection_scale}")
    print("\nStage 0 complete. Next: run generate_summaries.py")


if __name__ == "__main__":
    mp.freeze_support()
    main()
