"""Reward helpers for Nano-NLA GRPO."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from nano_nla.models import NLACriticModel, last_token_values
from nano_nla.schema import NLAConfig, extract_explanation, normalize_activation

MSE_EPS = 1e-8


def failed_extraction_reward(*, log_transform: bool = False) -> float:
    return -math.log(2.0) if log_transform else -2.0


def mse_nrm(pred: torch.Tensor, gold: torch.Tensor, scale: float | None) -> torch.Tensor:
    pn = normalize_activation(pred.float(), scale)
    gn = normalize_activation(gold.float(), scale)
    return ((pn - gn) ** 2).mean(dim=-1)


def mse_to_reward(mse: torch.Tensor, *, log_transform: bool = False) -> torch.Tensor:
    if log_transform:
        return -torch.log(torch.clamp(mse, min=MSE_EPS))
    return -mse


@dataclass
class RewardResult:
    reward: float
    explanation: str | None
    mse: float | None


@torch.no_grad()
def score_response_text(
    response_text: str,
    gold_activation: torch.Tensor,
    *,
    critic: NLACriticModel,
    tokenizer: Any,
    cfg: NLAConfig,
    device: str | torch.device = "cpu",
    log_transform: bool = False,
) -> RewardResult:
    explanation = extract_explanation(response_text)
    if explanation is None:
        return RewardResult(failed_extraction_reward(log_transform=log_transform), None, None)

    prompt = cfg.critic_prompt_template.format(explanation=explanation)
    tok = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").to(device)
    out = critic(input_ids=tok["input_ids"], attention_mask=tok.get("attention_mask"))
    pred = last_token_values(out.values, tok.get("attention_mask"))
    gold = gold_activation.to(pred.device).view(1, -1)
    mse = mse_nrm(pred, gold, cfg.mse_scale)
    reward = mse_to_reward(mse, log_transform=log_transform)
    return RewardResult(float(reward.item()), explanation, float(mse.item()))
