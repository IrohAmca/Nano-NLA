# Nano-NLA

Natural Language Autoencoders for `Qwen/Qwen2.5-0.5B-Instruct`, adapted from the
NLA research and reference training repo:

- https://transformer-circuits.pub/2026/nla/index.html
- https://github.com/kitft/natural_language_autoencoders

This project targets Colab-style CUDA runs. The base AV/AR model remains
`Qwen/Qwen2.5-0.5B-Instruct`; warm-start explanations default to DeepSeek
`deepseek-v4-flash`, with Groq and local providers available as fallbacks.

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
When copying a Colab Stage-0 run back locally, copy the contents of Colab's
`generated` directory directly into `data/generated` so `base.parquet` and
`splits/` sit at that level, not under `data/generated/generated`.

Stage 2 reads `datagen.summary_model.provider`. The default config uses
DeepSeek. Set `DEEPSEEK_API_KEY` before running Stage 2:

```powershell
$env:DEEPSEEK_API_KEY = "your-key-here"
uv run python scripts\run_datagen.py --config configs\qwen05b.yaml --stage 2 --summary-provider deepseek
```

DeepSeek Stage 2 is I/O-bound and runs API calls concurrently. The default
config uses 64 parallel requests with provider-neutral crash-safe checkpoint
chunks. Checkpoints include a summary-prompt fingerprint, so rows from different
teacher prompt contracts are not mixed accidentally. If DeepSeek quota runs out,
the current chunk is not written as a bad partial result; rerun with another
provider and completed compatible chunks are reused.
Tune this without editing YAML when you hit provider limits:

```powershell
uv run python scripts\run_datagen.py --config configs\qwen05b.yaml --stage 2 --summary-provider deepseek --summary-concurrency 64 --summary-batch-size 64 --summary-chunk-size 512

# If the API returns 429/rate-limit errors, cap throughput instead of disabling resume:
uv run python scripts\run_datagen.py --config configs\qwen05b.yaml --stage 2 --summary-provider deepseek --summary-concurrency 32 --summary-rpm 1200

# Continue the same checkpoint stream with Groq if DeepSeek quota is exhausted:
uv run python scripts\run_datagen.py --config configs\qwen05b.yaml --stage 2 --summary-provider groq
```

To use every configured hosted provider in parallel, install the Groq extra if
you want Groq included, set the keys you have, and use `multi`. Missing
providers are skipped, while each available provider keeps its own configured
concurrency and RPM cap:

```powershell
uv sync --extra groq
$env:DEEPSEEK_API_KEY = "..."
$env:GROQ_API_KEY = "..."
$env:NVIDIA_API_KEY = "..."
uv run python scripts\run_datagen.py --config configs\qwen05b.yaml --stage 2 --summary-provider multi --summary-max-new-rows 20000
```

For iterative runs, generate a bounded amount of new summary data, build partial
SFT datasets, train, then come back later for another increment:

```powershell
# Adds up to 20k new input rows per split, preserving compatible prompt chunks.
uv run python scripts\run_datagen.py --config configs\qwen05b.yaml --stage 2 --summary-provider deepseek --summary-max-new-rows 20000
uv run python scripts\run_datagen.py --config configs\qwen05b.yaml --stage 3
uv run python scripts\run_sft.py --config configs\qwen05b.yaml --stage both
```

## Colab 20k Windows

The Colab notebooks keep generated data, summary chunks, checkpoints, logs, and
results under a Google Drive root and symlink those artifact directories into a
fresh `/content/Nano-NLA` checkout:

- `notebooks/colab_stage2_cpu.ipynb`: CPU/API Stage 2 increments.
- `notebooks/colab_rl_gpu_windows.ipynb`: GPU GRPO windows and resume checks.

The Stage 2 notebook uses the hosted `multi` provider and this bounded command:

```bash
uv run python scripts/run_datagen.py \
  --config configs/qwen05b.yaml \
  --stage 2 \
  --summary-provider multi \
  --summary-max-new-rows 20000
```

That limit is per Stage 2 SFT split, so one increment can add up to 20k AV-SFT
input rows and up to 20k AR-SFT input rows. Re-run the same command after a
provider or Colab interruption. The prompt-fingerprinted chunk directories under
`data/generated/splits` are the resume source; do not share one Stage 2
checkpoint directory between AV and AR splits.

The RL notebook uses one output directory per RL row window. The first window
starts from SFT checkpoints; later windows start from the previous window's
final `av` and `ar` directories:

```bash
uv run python scripts/run_rl.py \
  --config configs/qwen05b.yaml \
  --dataset data/generated/rl.parquet \
  --actor-checkpoint checkpoints/av_sft \
  --critic-checkpoint checkpoints/ar_sft \
  --output-dir checkpoints/rl/windows/rows_000000_019999 \
  --row-offset 0 \
  --max-rows 20000 \
  --end-step 800 \
  --resume-latest
```

`--resume-latest` uses the newest complete `step_N/av` and `step_N/ar` pair in
that window directory. `--end-step` bounds the resumed window, so a run resumed
from `step_400` with `--end-step 800` executes the remaining 400 RL steps. A
printed log line after the latest saved step is not itself resumable. RL resume
loads model weights from the selected checkpoint; optimizer state is rebuilt for
the resumed process.

To use Groq instead, install the optional dependency and set `GROQ_API_KEY`:

```powershell
uv sync --extra groq
$env:GROQ_API_KEY = "your-key-here"
uv run python scripts\run_datagen.py --config configs\qwen05b.yaml --stage 2 --summary-provider groq
```

To run the local fallback without editing YAML:

```powershell
uv run python scripts\run_datagen.py --config configs\qwen05b.yaml --stage 2 --summary-provider local
```

Hosted providers write crash-safe chunks under
`*_explained.summary.<prompt-hash>.chunks` before rebuilding the final explained
parquet. Use `--summary-checkpoint-dir` only when you intentionally want to pin
a specific compatible checkpoint directory across machines.

## License

Apache-2.0
