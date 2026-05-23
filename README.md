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

- **Mean Cosine Similarity:** `0.6683`
- **Mean MSE (Normalized):** `0.663444`

| Sample | Cosine | MSE (nrm) | Explanation (Truncated) |
|-------:|-------:|----------:|:------------------------|
|  44598 | 0.5019 |  0.996282 | `<explanation> The text is describing a physiological or psychological concept related to personal appearance, specifi...` |
| 216377 | 0.7002 |  0.599660 | `<explanation> The text is a calendar or event warning schedule for a meeting or conference, so the model predicts a f...` |
| 219306 | 0.5212 |  0.957641 | `<explanation> The text is a pricing or button reference for a bill or payment amount, so the model predicts a monetar...` |
| 327087 | 0.7893 |  0.421444 | `<explanation> The text is a buying或旅行-related advice context for planning a trip or preparedness, suggesting future c...` |
| 386742 | 0.8289 |  0.342194 | `<explanation> The text is a review or assessment of a product or experience, so the model predict a review element or...` |

## Per-Sample Details (Q&A Format)

### Sample 44598 (Cosine: 0.5019, MSE: 0.996282)

#### 1. Target Model Input (Original Context Prefix)
```text
... Austin, TX, Unknown
When: August 19 and 20.
Where: The City Theatre.
Time: Saturday, 11 am – 2 pm. Sunday, 6:30 – 9:30 pm. Ten-minute slots by appointment. Callbacks if needed. Show dates November 3rd – December 3rd, with rehearsals beginning in September.
Needed: No prepared monologue is needed. Readings from the show will be performed so be familiar with the script. Accents preferred. Sides will be available on the website a few days before auditions. You may be asked to stay beyond the ten-minute appt. time. Please bring head ...
```

#### 2. NLA Question to Autoencoder Actor
> *"What semantic concept or feature does the Layer-16 activation vector represent?"*

#### 3. Autoencoder Actor's Answer (Explanation)
```xml
<explanation>
The text is describing a physiological or psychological concept related to personal appearance, specifically the head as a sign of a person's physical condition or mood.

The word "head" is a noun referring to the brain or face, and the next word must continue this anatomical term to complete a common phrase like "head" or "head" in a specific context.

The final word "head" is a common adjective-noun pair that strongly expects a noun or word ending in "head" to describe a specific part of the head or a related item.
</explanation>
```
---

### Sample 216377 (Cosine: 0.7002, MSE: 0.599660)

#### 1. Target Model Input (Original Context Prefix)
```text
... I was out of the Ed Tech world for a brief 8 weeks and now that I have returned it feels like coming home. I left Ed Tech and my area of expertise because I wasn’t sure I was in the right field anymore. I was bored, tired, and not challenged. I equated all of that with a need for a career change. As I look back on that time I realize (the hard way) that I didn’t need a career change, I needed a work place change.
I was reminded of this at the ETUG (Educational Technology Users’ Group) Conference on Monday and Tuesday ...
```

#### 2. NLA Question to Autoencoder Actor
> *"What semantic concept or feature does the Layer-16 activation vector represent?"*

#### 3. Autoencoder Actor's Answer (Explanation)
```xml
<explanation>
The text is a calendar or event warning schedule for a meeting or conference, so the model predicts a formal general-purpose announcement about a meeting day or day-time event.

The date "Tuesday, October 10" provides a specific day and time for the event ("Monday, October 9" would be the missing day), which is a formal subject for a notice of a larger gathering.

The final phrase "Tuesday, Monday, Wednesday, Thursday, and Friday, October 10" is a common guide for a general academic or professional day on the campus, indicating the next text will list a parallel date and time format for the conference.
</explanation>
```
---

### Sample 219306 (Cosine: 0.5212, MSE: 0.957641)

#### 1. Target Model Input (Original Context Prefix)
```text
... Final month, GM introduced a $ 24 million funding in its meeting plant in Fort Wayne, Indiana, to extend the manufacturing of the Chevrolet Silverado 1500 and GMC 1500 pickup vehicles. Automotive anticipates robust demand for its redesigned heavyweights, injecting $ 150 million into its Flint meeting plant in Michigan.
The Flint Manufacturing facility delivers the primary batch of Chevrolet Silverado autos and GMC Sierra redesigned obligation vehicles to dealers this week. Launched earlier this yr, these vehicles promise a most towing capability of 52% larger than earlier variations, for a complete of 35,500 kilos. The fashions additionally characteristic powertrain upgrades, together with a brand new Allison 10-speed automated transmission for the 6.6-liter diesel engine, in addition to extra superior cab applied sciences.
The automaker has invested greater than $ ...
```

#### 2. NLA Question to Autoencoder Actor
> *"What semantic concept or feature does the Layer-16 activation vector represent?"*

#### 3. Autoencoder Actor's Answer (Explanation)
```xml
<explanation>
The text is a pricing or button reference for a bill or payment amount, so the model predicts a monetary value or a transaction processing currency "$" followed by a dollar amount.

The number $ indicates a financial transaction fee or amount is to be displayed, with the value exceeding $0.10 after the decimal point.

The final token "$$ $" is a formatted currency symbol that requires a dollar amount (e.g., "USD $" or "¥$") to complete the monetary value for the fiscal or interactive payment.
</explanation>
```
---

### Sample 327087 (Cosine: 0.7893, MSE: 0.421444)

#### 1. Target Model Input (Original Context Prefix)
```text
... Get ready for gay travel destinations as a great way to treat yourself to the break you deserve. Whether you travel alone or with your partner, you can travel to exotic locations and give yourself time away from the daily grind. You will feel rejuvenated when you come back from these travel destinations because they give you time to be in an open and nonjudgmental gay environment.
To start to get ready for gay travel destinations, talk to a travel agent that specializes in gay travel. Whether you want to get away from it all without going too far away or if you prefer exploring far corners of the world, there will be destination packages that will make your travel dreams come true. If you are traveling single, let your travel agent know you want travel destinations that cater to singles. If you are with your partner, let your travel agent know what sort of romantic or adventuresome experience you want.
When you know where you are heading in your gay travel destinations adventure, you can begin to pack. Knowing what to pack will be easy when you have an agenda from your travel agent. You will know if you need dressier outfits or casual active clothing. Make sure and ...
```

#### 2. NLA Question to Autoencoder Actor
> *"What semantic concept or feature does the Layer-16 activation vector represent?"*

#### 3. Autoencoder Actor's Answer (Explanation)
```xml
<explanation>
The text is a buying或旅行-related advice context for planning a trip or preparedness, suggesting future considerations or additional necessities for clothing or travel accessories.

The phrase "and" initiates a temporal or conditional clause that requires a present particula context, likely describing a need for more planned, informed choices or purchase info.

The final word "and" is a conjunction that strongly expects a temporal consequence, "and" followed by a descriptive clause like "and you" or "and your" indicating additional details or adjustments for the upcoming trip or purchase.
</explanation>
```
---

### Sample 386742 (Cosine: 0.8289, MSE: 0.342194)

#### 1. Target Model Input (Original Context Prefix)
```text
... We have a rather ambivalent attitude towards awards because its what our guests actually think that matters, not the view of an inspector or an awarding body.
However, they serve to endorse any claims that we make for ourselves and help re-assure a prospective guest that we are ‘an ok place’.
They also make us feel better and lift morale. All to often satisfaction is silent (fair do’s, we are here to provide a service and you pay for it) and only dissatisfaction finds a voice. So it’s a good feeling to be recognised by professionals who have hundreds, no thousands of hotels and restaurant to select from.
Over the years we’ve been lucky enough to catch the eye on quite a few occasions and I’ve listed below the ones I regard as the more significant. Either for their prestige factors or ...
```

#### 2. NLA Question to Autoencoder Actor
> *"What semantic concept or feature does the Layer-16 activation vector represent?"*

#### 3. Autoencoder Actor's Answer (Explanation)
```xml
<explanation>
The text is a review or assessment of a product or experience, so the model predict a review element or a critical note about a particular outcome or benefit.

The phrase "or" starts a transitional clause that strongly expects a specific example or reason for noting a user's perspective or approval.

The final phrase "or" is a conjunction that signals an alternative or additional factor for a link or reference, requiring a phrase that explains what or why the users' judgment is inspired by or relates to.
</explanation>
```
---

## License

Apache-2.0
