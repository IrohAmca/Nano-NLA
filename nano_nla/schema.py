"""Shared NLA schema — config loading, normalization, sidecar I/O.

Adapted from: https://github.com/kitft/natural_language_autoencoders/blob/main/nla/schema.py

Single source of truth for:
  - NLAConfig dataclass
  - Activation normalization
  - Explanation tag handling
  - Sidecar YAML read/write
  - Injection token neighbor computation
"""

from __future__ import annotations

import math
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# ─── Constants ──────────────────────────────────────────────────────────────

SIDECAR_BASENAME = "nla_meta.yaml"

EXPLANATION_OPEN = "<explanation>"
EXPLANATION_CLOSE = "</explanation>"
EXPLANATION_RE = re.compile(
    f"{re.escape(EXPLANATION_OPEN)}(.*?){re.escape(EXPLANATION_CLOSE)}",
    re.DOTALL,
)

ACTIVATION_COLUMN = "activation_vector"
INJECT_PLACEHOLDER = "<INJECT>"
MM_ACTIVATION_KEY = "nla_activation"
MM_CRITIC_TOKENS_KEY = "nla_critic_tokens"
MM_MSE_SCALE_KEY = "nla_mse_scale"
SCALE_SQRT_D = "sqrt_d_model"


# ─── Tag helpers ────────────────────────────────────────────────────────────

def wrap_explanation(text: str) -> str:
    """Wrap text in explanation tags — used for AV-SFT response column."""
    return f"{EXPLANATION_OPEN}\n{text}\n{EXPLANATION_CLOSE}"


def extract_explanation(response: str) -> str | None:
    """Extract payload between explanation tags; None on miss.

    Strips whitespace so the result matches the critic template format.
    """
    m = EXPLANATION_RE.search(response)
    return m.group(1).strip() if m else None


# ─── Scale resolution ──────────────────────────────────────────────────────

def resolve_target_scale(raw: float | str | None, d_model: int) -> float | None:
    """Turn a scale value (from sidecar or config) into a concrete float or None.

    Accepts:
      - None / "raw" / "none"  → None → no normalization
      - "sqrt_d_model"         → sqrt(d_model)
      - a float or float-string → that exact L2 norm
    """
    if raw is None or raw in ("raw", "none"):
        return None
    if raw == SCALE_SQRT_D:
        return math.sqrt(d_model)
    if isinstance(raw, str):
        return float(raw)
    assert isinstance(raw, (int, float)), (
        f"scale must be None/'raw', {SCALE_SQRT_D!r}, or a number; got {raw!r}"
    )
    return float(raw)


# ─── Activation normalization ───────────────────────────────────────────────

def normalize_activation(v, target_scale: float | None):
    """Scale vectors to target_scale L2-norm, or pass through if None.

    Two purposes:
      - Actor injection: scale = injection_scale (hyperparameter)
      - Critic MSE: scale = mse_scale (numerical stability)

    Idempotent. Zero vectors stay zero. Norm computed in fp32 for precision.
    """
    if target_scale is None:
        return v
    norm_fp32 = v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v / (norm_fp32 / target_scale).to(v.dtype)


# ─── Predict-the-mean baselines ────────────────────────────────────────────

def compute_predict_mean_baselines(
    vectors, mse_scale: float | None
) -> tuple[float, float]:
    """Two predict-the-mean baseline MSEs for FVE logging.

    Returns (meannorm_baseline, raw_variance_baseline).
    """
    v_norm = normalize_activation(vectors.float(), mse_scale)
    mu = v_norm.mean(dim=0, keepdim=True)
    mu_normed = normalize_activation(mu, mse_scale)
    mse_meannorm = ((v_norm - mu_normed) ** 2).mean().item()
    mse_rawvar = ((v_norm - mu) ** 2).mean().item()
    return mse_meannorm, mse_rawvar


# ─── NLAConfig ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NLAConfig:
    """Complete NLA configuration loaded from YAML config + computed values."""
    d_model: int
    injection_char: str
    injection_token_id: int
    injection_left_neighbor_id: int
    injection_right_neighbor_id: int
    av_prompt_template: str
    ar_prompt_template: str
    injection_scale: float | None = None
    mse_scale: float | None = None
    target_layer: int = 16
    ar_num_layers: int | None = None  # K+1 for truncated AR model
    critic_suffix_ids: list[int] | None = None

    @property
    def sqrt_d(self) -> float:
        return math.sqrt(self.d_model)

    @property
    def actor_prompt_template(self) -> str:
        return self.av_prompt_template

    @property
    def critic_prompt_template(self) -> str:
        return self.ar_prompt_template


# ─── Sidecar I/O ───────────────────────────────────────────────────────────

def sidecar_path_for(path: str | Path) -> Path:
    """Resolve sidecar path for parquet file or checkpoint directory."""
    p = Path(str(path).split("@[")[0])
    if p.is_dir() or (not p.exists() and p.suffix == ""):
        return p / SIDECAR_BASENAME
    return p.with_name(p.name + f".{SIDECAR_BASENAME}")


def write_dataset_sidecar(
    parquet_path: str | Path,
    cfg: NLAConfig,
    *,
    base_model: str,
    stage: str,
) -> None:
    """Write {parquet}.nla_meta.yaml for an NLA dataset."""
    meta: dict[str, Any] = {
        "kind": "nla_dataset",
        "schema_version": 2,
        "stage": stage,
        "base_model": base_model,
        "extraction": {
            "d_model": cfg.d_model,
            "layer_index": cfg.target_layer,
            "injection_scale": cfg.injection_scale,
            "mse_scale": cfg.mse_scale,
        },
        "tokens": {
            "injection_char": cfg.injection_char,
            "injection_token_id": cfg.injection_token_id,
            "injection_left_neighbor_id": cfg.injection_left_neighbor_id,
            "injection_right_neighbor_id": cfg.injection_right_neighbor_id,
            "critic_suffix_ids": cfg.critic_suffix_ids,
        },
        "prompt_templates": {
            "av": cfg.av_prompt_template,
            "ar": cfg.ar_prompt_template,
            "actor": cfg.actor_prompt_template,
            "critic": cfg.critic_prompt_template,
        },
    }
    out_path = sidecar_path_for(parquet_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_model_sidecar(
    checkpoint_dir: str | Path,
    cfg: NLAConfig,
    *,
    role: str,   # "av" or "ar"
    stage: str,
    base_model: str,
) -> None:
    """Write {checkpoint_dir}/nla_meta.yaml for an NLA model."""
    meta: dict[str, Any] = {
        "kind": "nla_model",
        "schema_version": 2,
        "role": role,
        "stage": stage,
        "base_model": base_model,
        "d_model": cfg.d_model,
        "extraction": {
            "injection_scale": cfg.injection_scale,
            "mse_scale": cfg.mse_scale,
        },
        "tokens": {
            "injection_char": cfg.injection_char,
            "injection_token_id": cfg.injection_token_id,
            "injection_left_neighbor_id": cfg.injection_left_neighbor_id,
            "injection_right_neighbor_id": cfg.injection_right_neighbor_id,
            "critic_suffix_ids": cfg.critic_suffix_ids,
        },
        "prompt_templates": {
            "av": cfg.av_prompt_template,
            "ar": cfg.ar_prompt_template,
            "actor": cfg.actor_prompt_template,
            "critic": cfg.critic_prompt_template,
        },
    }
    if cfg.ar_num_layers is not None:
        meta["ar"] = {"num_hidden_layers": cfg.ar_num_layers}
    out_path = Path(checkpoint_dir) / SIDECAR_BASENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True), encoding="utf-8")


# ─── Token neighbor computation ────────────────────────────────────────────

def compute_canonical_neighbors(
    tokenizer: Any,
    av_template: str,
    injection_char: str,
    injection_token_id: int,
) -> tuple[int, int]:
    """Tokenize the canonical AV prompt, return token IDs at inj_pos ± 1.

    Used both to POPULATE neighbor fields (datagen) and VERIFY them (training).
    Both must use identical tokenization — this function is that contract.
    """
    content = av_template.format(injection_char=injection_char)
    result = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
    )
    # transformers >=5.x returns BatchEncoding; older returns list[int]
    if hasattr(result, "input_ids"):
        ids = result["input_ids"]
    elif isinstance(result, list):
        ids = result
    else:
        ids = list(result)
    if hasattr(ids, "ndim") and hasattr(ids, "tolist"):
        ids = ids[0].tolist() if ids.ndim == 2 else ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]

    matches = [i for i, tid in enumerate(ids) if tid == injection_token_id]
    assert len(matches) == 1, (
        f"injection token id {injection_token_id} ({injection_char!r}) appears "
        f"{len(matches)}× in canonical AV prompt (expected 1). Template: {content!r}"
    )
    p = matches[0]
    assert 0 < p < len(ids) - 1, (
        f"injection token at position {p} is at edge of sequence (len={len(ids)})"
    )
    return ids[p - 1], ids[p + 1]


def compute_critic_suffix_ids(
    tokenizer: Any,
    critic_template: str,
    *,
    suffix_len: int = 8,
) -> list[int]:
    """Token IDs for the final suffix of the canonical critic prompt.

    AR loss/reward extracts the final real token. Storing a short suffix in the
    sidecar lets loaders detect accidental prompt-template drift early.
    """
    prompt = critic_template.format(explanation="")
    encoded = tokenizer(prompt, add_special_tokens=True)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if hasattr(ids, "ndim") and hasattr(ids, "tolist"):
        ids = ids[0].tolist() if ids.ndim == 2 else ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids[-suffix_len:])


# ─── Config loading from YAML ──────────────────────────────────────────────

def load_config(config_path: str | Path) -> dict:
    """Load the master YAML config file."""
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def read_sidecar(path: str | Path) -> dict[str, Any]:
    """Read the sidecar associated with a parquet file or checkpoint dir."""
    return yaml.safe_load(sidecar_path_for(path).read_text(encoding="utf-8"))


def merge_sidecar_into_config(config: dict, sidecar_source: str | Path | None) -> dict:
    """Return config with dataset/model sidecar values taking precedence.

    This keeps scale factors, token IDs, and prompt templates pinned to the data
    or checkpoint that was actually produced, matching the original NLA design.
    """
    if sidecar_source is None:
        return config
    path = sidecar_path_for(sidecar_source)
    if not path.exists():
        return config
    meta = yaml.safe_load(path.read_text(encoding="utf-8"))
    out = deepcopy(config)
    extraction = meta.get("extraction", {})
    tokens = meta.get("tokens", {})
    prompts = meta.get("prompt_templates", {})
    out.setdefault("injection", {})
    out["injection"]["injection_scale"] = extraction.get("injection_scale", out["injection"].get("injection_scale"))
    out["injection"]["mse_scale"] = extraction.get("mse_scale", out["injection"].get("mse_scale"))
    for key in (
        "injection_char",
        "injection_token_id",
        "injection_left_neighbor_id",
        "injection_right_neighbor_id",
        "critic_suffix_ids",
    ):
        if key in tokens:
            out["injection"][key] = tokens[key]
    out.setdefault("prompts", {})
    if "av" in prompts or "actor" in prompts:
        out["prompts"]["av"] = prompts.get("av") or prompts["actor"]
    if "ar" in prompts or "critic" in prompts:
        out["prompts"]["ar"] = prompts.get("ar") or prompts["critic"]
    return out


def build_nla_config_from_yaml(
    config: dict,
    tokenizer: Any | None = None,
) -> NLAConfig:
    """Build NLAConfig from the master YAML config, optionally computing token IDs.

    If tokenizer is provided, auto-computes injection_token_id and neighbors.
    Otherwise uses values from config (which may be null for first run).
    """
    model_cfg = config["model"]
    inj_cfg = config["injection"]
    prompts = config["prompts"]
    d_model = model_cfg["d_model"]
    av_prompt = prompts.get("av") or prompts["actor"]
    ar_prompt = prompts.get("ar") or prompts["critic"]

    injection_char = inj_cfg["injection_char"]
    injection_token_id = inj_cfg.get("injection_token_id")
    left_neighbor = inj_cfg.get("injection_left_neighbor_id")
    right_neighbor = inj_cfg.get("injection_right_neighbor_id")
    critic_suffix_ids = inj_cfg.get("critic_suffix_ids")

    # Auto-compute token IDs from tokenizer
    if tokenizer is not None:
        live_ids = tokenizer.encode(injection_char, add_special_tokens=False)
        assert len(live_ids) == 1, (
            f"injection char {injection_char!r} tokenizes to {len(live_ids)} tokens "
            f"(expected 1). Pick a different character."
        )
        injection_token_id = live_ids[0]
        assert injection_token_id != tokenizer.unk_token_id, (
            f"{injection_char!r} maps to UNK — pick a different marker"
        )
        left_neighbor, right_neighbor = compute_canonical_neighbors(
            tokenizer, av_prompt, injection_char, injection_token_id
        )
        critic_suffix_ids = compute_critic_suffix_ids(tokenizer, ar_prompt)

    injection_scale = resolve_target_scale(inj_cfg.get("injection_scale"), d_model)
    mse_scale = resolve_target_scale(inj_cfg.get("mse_scale", SCALE_SQRT_D), d_model)
    target_layer = model_cfg["target_layer"]

    return NLAConfig(
        d_model=d_model,
        injection_char=injection_char,
        injection_token_id=injection_token_id or 0,
        injection_left_neighbor_id=left_neighbor or 0,
        injection_right_neighbor_id=right_neighbor or 0,
        av_prompt_template=av_prompt,
        ar_prompt_template=ar_prompt,
        injection_scale=injection_scale,
        mse_scale=mse_scale,
        target_layer=target_layer,
        ar_num_layers=target_layer + 1,
        critic_suffix_ids=critic_suffix_ids,
    )
