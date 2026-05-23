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

## Evaluation Results (Qwen 0.5B, 20K rows, 800 RL steps)

Results from `scripts/run_eval.py --mode all` on the final RL checkpoint:

### Reconstruction

| Metric | Value |
|---|---|
| FVE (Fraction of Variance Explained) | 0.90 |

### Steganography

The AV does not encode hidden signals in text formatting. Semantic
transformations leave reconstruction quality essentially unchanged:

| Transformation | FVE Original | FVE Transformed | Delta |
|---|---|---|---|
| shuffle_bullets | 0.9000 | 0.8996 | −0.0004 |
| coherence | 0.9000 | 0.9000 | 0.0000 |
| paragraph_summary | 0.9000 | 0.8994 | −0.0006 |

### Confabulation

| Metric | Value |
|---|---|
| Total claims | 254 |
| Support rate | 16.9% |
| Entity support rate | 100% |
| Detail support rate | 3.5% |
| Thematic support rate | 0% |
| MSE delta (true claims) | 0.0108 |
| MSE delta (false claims) | 0.0056 |

The high entity support rate (100%) confirms the model identifies *what* an
activation is about, but detail and thematic accuracy are low.  This is expected
at 0.5B scale: the AV learns to produce explanations that the AR can round-trip
(0.90 FVE) rather than explanations that match the original context verbatim.
True claims carry ≈2× the reconstructive information of false claims
(MSE delta 0.0108 vs 0.0056), which indicates the model is not purely
confabulating.

See the [NLA paper](https://transformer-circuits.pub/2026/nla/index.html)
for a discussion of why reconstruction quality and explanation faithfulness
are distinct properties.

## Known Warnings

### `model.norm.weight MISSING` on AR checkpoint load

The AR (critic) model is a truncated Qwen2 backbone that keeps only layers
0 through `target_layer`.  The constructor replaces the final LayerNorm with
`nn.Identity` via `_strip_final_norm` in `nano_nla/models.py`, so the weight
is unused at runtime.  When HuggingFace loads the backbone from the saved
checkpoint, it notices the config still declares `model.norm.weight` but no
saved tensor exists for it.  The newly initialized weight is immediately
replaced by `nn.Identity` and never participates in the forward pass.
**This warning is harmless.**

## Example Explanations

Generate sample AV explanations with reconstruction scores using the demo
script:

```bash
uv run python scripts/demo_explanations.py \
    --config configs/qwen05b.yaml \
    --av-checkpoint checkpoints/rl/av \
    --ar-checkpoint checkpoints/rl/ar \
    --gold-parquet data/generated/rl.parquet \
    --num-samples 5
```

Save results as a markdown table:

```bash
uv run python scripts/demo_explanations.py \
    --config configs/qwen05b.yaml \
    --av-checkpoint checkpoints/rl/av \
    --ar-checkpoint checkpoints/rl/ar \
    --gold-parquet data/generated/rl.parquet \
    --num-samples 10 \
    --output results/demo_explanations.md
```

### Real Demo Outputs (800 RL Steps Checkpoint)

Below are the actual generated explanations and reconstruction metrics from a successful Colab RL run (`configs/colab_rl_a100.yaml`) on 5 random activation samples:

- **Mean Cosine Similarity:** `0.6711`
- **Mean MSE (Normalized):** `0.657891`

| Sample | Cosine | MSE (nrm) | Explanation |
|-------:|-------:|----------:|:------------|
|  44598 | 0.4966 |  1.006774 | `<explanation> The text is a medical or visual description of a person's head and appearance, so the model predicts a word or phrase about a person's scalp, hair, or head anatomy.  The word "head" is a noun that strongly suggests a temporal or anatomical meaning, much like "head" in "head spikes" or "head of a head."  The final word "head" is part of a word that must be followed by a common descriptor (e.g., "head" + anatomical term) or a verb describing the bridle's role over a person's head. </explanation>` |
| 216377 | 0.7165 |  0.567032 | `<explanation> The text is a training or conference announcement for a logical thinking or public speaking series, so the model predicts a date or day label for a main event or conference.  The event name "Tue. Tuesday" indicates a date or event type with specific time and location details, likely continuing with a conference name or a networking discussion.  The final phrase " Mondays Tuesday" describes a formal or educational schedule, suggesting a typical Friday or Wednesday event with a call to action for meetings or discussion. </explanation>` |
| 219306 | 0.5248 |  0.950319 | `<explanation> The text is a marketing or pricing page for a database or financial data source, so the model predicts a monetary amount or a fiscal threshold value for a donation or purchase.  The value $[amount] $ indicates a fiscal or transaction processing threshold that will be applied for subscription, payment, or investment fees.  The final token "$" is a dollar sign followed by a currency code (e.g., $1000) that requires a numeric value to complete the economic reference for the source. </explanation>` |
| 327087 | 0.7880 |  0.424053 | `<explanation> The text is a travel, shopping, and planning guide for purchase or preparation before an event, so the model predicts a recommendation or planning step for purchasing additional items.  The phrase "and" begins a temporal clause indicating continued consideration or accompaniment, likely continuing with a reason for buying or planning something specific.  The final word "and" is a conjunction that strongly expects a continuation about personal needs, conditions, or actions in relation to preparing for the trip or upcoming experience. </explanation>` |
| 386742 | 0.8294 |  0.341275 | `<explanation> The text is a review or description of a setting or testimonial, so the model predicts a neutral or personal benefit or description of a positive outcome for readers or participants.  The phrase "or" begins a contrast or example that requires a causal or exemplary reason for mentioning a specific or common element in experiences or observations.  The final word "or" signals a parallel or context between a given example and a broader, general or conditional statement, likely emphasizing a particular value, credibility, or human aspect of the chosen image. </explanation>` |

## License

Apache-2.0
