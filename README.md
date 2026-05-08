# Nano-NLA

Natural Language Autoencoders for `Qwen/Qwen2.5-0.5B-Instruct`, adapted from the
NLA research and reference training repo:

- https://transformer-circuits.pub/2026/nla/index.html
- https://github.com/kitft/natural_language_autoencoders

This project targets Colab-style CUDA runs. The base AV/AR model remains
`Qwen/Qwen2.5-0.5B-Instruct`; warm-start explanations default to Groq
`qwen/qwen3-32b`, with local `Qwen/Qwen2.5-7B-Instruct` available as a fallback.

## Smoke Tests

Fast model-free smoke test:

```bash
uv run python scripts/smoke_test.py
```

Small real-model extraction smoke test:

```bash
uv run python scripts/smoke_test.py --mode model --device cuda:0 --docs 5 --positions-per-doc 2
```

## Datagen

```bash
uv sync
uv run python scripts/run_datagen.py --config configs/qwen05b.yaml
```

CUDA is configured through the `pytorch-cu124` uv index. The expected check is:

```powershell
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

Stage 0 supports CUDA worker selection. `auto` expands to all visible CUDA
devices, and direct overrides are available when needed:

```powershell
uv run python scripts\run_datagen.py --config configs\qwen05b.yaml --stage 0 --devices auto
uv run python scripts\run_datagen.py --config configs\qwen05b.yaml --stage 0 --devices cuda:0
```

Workers pull the next document from a shared queue, write independent parquet
shards under `stage0_shards`, and the main process merges them into
`base.parquet`. Stage 0 writes a computed config under
`data/generated/*_computed.yaml` with the measured `injection_scale` and
tokenizer IDs; downstream stages pick it up automatically.

Stage 2 reads `datagen.summary_model.provider`. The default config uses Groq.
Install the optional dependency and set `GROQ_API_KEY` before running Stage 2:

```powershell
uv sync --extra groq
$env:GROQ_API_KEY = "your-key-here"
uv run python scripts\run_datagen.py --config configs\qwen05b.yaml --stage 2
```

To run the local fallback instead, set `datagen.summary_model.provider: local`.
Both providers write crash-safe chunks under `*_explained.chunks` before
rebuilding the final explained parquet.

## License

Apache-2.0
