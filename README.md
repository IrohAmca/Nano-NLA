# Nano-NLA

Natural Language Autoencoders for `Qwen/Qwen2.5-0.5B-Instruct`, adapted from the
NLA research and reference training repo:

- https://transformer-circuits.pub/2026/nla/index.html
- https://github.com/kitft/natural_language_autoencoders

This project is a small CPU-first implementation. It follows the AV/AR contract,
but it is not a drop-in replacement for the upstream Miles + SGLang training
stack.

## Smoke Tests

Fast model-free smoke test:

```bash
uv run python scripts/smoke_test.py
```

Real Qwen extraction smoke test:

```bash
uv run python scripts/smoke_test.py --mode model --docs 5 --positions-per-doc 2
```

On Windows, the real-model smoke needs enough virtual memory/pagefile. If it
fails with `os error 1455`, enable a system-managed pagefile or raise the
pagefile size before loading Qwen.

## Datagen

```bash
uv sync
$env:GROQ_API_KEY = "your-key-here"  # PowerShell
uv run python scripts/run_datagen.py --config configs/qwen05b.yaml
```

CUDA is configured through the `pytorch-cu124` uv index. The expected check is:

```powershell
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

Stage 0 supports mixed extraction workers. The config defaults to `cuda:0` plus
one CPU worker. Override it from the CLI when needed:

```powershell
uv run python scripts\run_datagen.py --config configs\qwen05b.yaml --stage 0 --devices cuda:0
uv run python scripts\run_datagen.py --config configs\qwen05b.yaml --stage 0 --devices cuda:0,cpu,cpu
```

Workers pull the next document from a shared queue, write independent parquet
shards under `stage0_shards`, and the main process merges them into
`base.parquet`.

Stage 0 writes a computed config under `data/generated/*_computed.yaml` with
the measured `injection_scale` and tokenizer IDs. Downstream stages now pick
that config up automatically, and direct stage runs also read the base parquet
sidecar when available.

## License

Apache-2.0
