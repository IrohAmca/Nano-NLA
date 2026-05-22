"""NLA-based causal steering.

Implements the paper's steering methodology:
  1. Generate an NLA explanation for an activation at a specific token
  2. Edit the explanation (e.g., replace a concept with an alternative)
  3. Reconstruct both original and edited explanations via the AR
  4. Compute steering vector: Δ = AR(edited) - AR(original)
  5. Inject: h_orig → h_orig + α·‖h_orig‖·(Δ / ‖Δ‖)

Reference: Sections "Planning in Poetry" and "Reasoning about Rewards"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM

from nano_nla.inference import NLAClient


@dataclass
class SteeringVector:
    """A directional edit derived from NLA explanation modification."""
    direction: torch.Tensor          # unit-norm steering direction
    original_reconstruction: torch.Tensor
    edited_reconstruction: torch.Tensor
    original_explanation: str
    edited_explanation: str
    edit_description: str


@dataclass
class SteeringResult:
    """Result of applying a steering vector to the target model."""
    original_completion: str
    steered_completion: str
    alpha: float
    steering_vector: SteeringVector


def edit_explanation(
    explanation: str,
    replacements: dict[str, str],
) -> str:
    """Apply text replacements to an NLA explanation.

    Each key→value pair in replacements is applied case-insensitively.
    This mirrors the paper's methodology: e.g., "rabbit"→"mouse" or
    "reward"→"penalty".
    """
    result = explanation
    for old, new in replacements.items():
        result = re.sub(re.escape(old), new, result, flags=re.IGNORECASE)
    return result


@torch.no_grad()
def compute_steering_vector(
    client: NLAClient,
    activation: torch.Tensor,
    replacements: dict[str, str],
    *,
    edit_description: str = "",
    num_rollouts: int = 1,
) -> SteeringVector:
    """Generate an NLA explanation, edit it, and compute the steering direction.

    If num_rollouts > 1, averages the steering vector across multiple
    AV samples (as done in the "Reasoning about Rewards" case study).
    """
    if client.ar is None:
        raise RuntimeError("AR checkpoint required for steering")

    directions: list[torch.Tensor] = []
    last_orig_expl = ""
    last_edit_expl = ""
    last_orig_recon = torch.zeros(1)
    last_edit_recon = torch.zeros(1)

    for _ in range(num_rollouts):
        orig_explanation = client.generate_explanation(activation)
        edited_explanation = edit_explanation(orig_explanation, replacements)

        orig_recon = client.reconstruct(orig_explanation)
        edit_recon = client.reconstruct(edited_explanation)

        delta = edit_recon - orig_recon
        norm = delta.float().norm().clamp_min(1e-8)
        directions.append(delta.float() / norm)

        last_orig_expl = orig_explanation
        last_edit_expl = edited_explanation
        last_orig_recon = orig_recon
        last_edit_recon = edit_recon

    avg_direction = torch.stack(directions).mean(dim=0)
    avg_direction = avg_direction / avg_direction.norm().clamp_min(1e-8)

    return SteeringVector(
        direction=avg_direction,
        original_reconstruction=last_orig_recon,
        edited_reconstruction=last_edit_recon,
        original_explanation=last_orig_expl,
        edited_explanation=last_edit_expl,
        edit_description=edit_description,
    )


def apply_steering_hook(
    model: AutoModelForCausalLM,
    target_layer: int,
    token_position: int,
    steering_direction: torch.Tensor,
    alpha: float,
) -> list[Any]:
    """Register a forward hook that adds the steering vector at one position.

    Returns the list of hook handles (call .remove() to clean up).

    The injection formula from the paper:
      h_orig → h_orig + α·‖h_orig‖·(Δ / ‖Δ‖)
    """
    handles: list[Any] = []
    device = next(model.parameters()).device
    direction = steering_direction.to(device=device, dtype=torch.float32)

    def _hook(module: Any, _input: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3:
            return output

        h_orig = hidden[:, token_position, :].float()
        orig_norm = h_orig.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        delta = alpha * orig_norm * direction.unsqueeze(0)
        hidden = hidden.clone()
        hidden[:, token_position, :] = hidden[:, token_position, :] + delta.to(hidden.dtype)

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    # Attach to the target transformer layer
    inner = model.model if hasattr(model, "model") else model.transformer
    layers = inner.layers if hasattr(inner, "layers") else inner.h
    handle = layers[target_layer].register_forward_hook(_hook)
    handles.append(handle)
    return handles


@torch.no_grad()
def steer_and_generate(
    model: AutoModelForCausalLM,
    tokenizer: Any,
    prompt_text: str,
    target_layer: int,
    token_position: int,
    steering_direction: torch.Tensor,
    alpha: float,
    *,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    device: torch.device | str = "cpu",
) -> str:
    """Generate a completion with NLA-based steering applied at one token."""
    device = torch.device(device) if isinstance(device, str) else device
    handles = apply_steering_hook(model, target_layer, token_position, steering_direction, alpha)
    try:
        tok = tokenizer(prompt_text, return_tensors="pt").to(device)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        output_ids = model.generate(**tok, **gen_kwargs)
        generated = output_ids[:, tok["input_ids"].shape[1]:]
        return tokenizer.decode(generated[0], skip_special_tokens=True)
    finally:
        for handle in handles:
            handle.remove()


def run_steering_experiment(
    client: NLAClient,
    target_model: AutoModelForCausalLM,
    tokenizer: Any,
    *,
    prompt_text: str,
    activation: torch.Tensor,
    token_position: int,
    replacements: dict[str, str],
    alpha_values: list[float],
    num_rollouts: int = 5,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    device: torch.device | str = "cpu",
) -> list[SteeringResult]:
    """Run a complete steering experiment across multiple α values."""
    sv = compute_steering_vector(
        client,
        activation,
        replacements,
        edit_description=str(replacements),
        num_rollouts=num_rollouts,
    )

    device_obj = torch.device(device) if isinstance(device, str) else device

    # Baseline (no steering)
    tok = tokenizer(prompt_text, return_tensors="pt").to(device_obj)
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
    with torch.no_grad():
        baseline_ids = target_model.generate(**tok, **gen_kwargs)
    baseline = tokenizer.decode(baseline_ids[0, tok["input_ids"].shape[1]:], skip_special_tokens=True)

    results: list[SteeringResult] = []
    for alpha in alpha_values:
        steered = steer_and_generate(
            target_model,
            tokenizer,
            prompt_text,
            client.nla_cfg.target_layer,
            token_position,
            sv.direction,
            alpha,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device_obj,
        )
        results.append(SteeringResult(
            original_completion=baseline,
            steered_completion=steered,
            alpha=alpha,
            steering_vector=sv,
        ))

    return results
