"""Activation injection helpers for AV prompts.

The AV model receives a normal chat prompt that contains one marker token. At
runtime we replace the embedding at that marker position with the activation
vector, optionally rescaled to the configured injection norm.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from nano_nla.schema import NLAConfig, normalize_activation


def _as_2d_input_ids(input_ids: torch.Tensor) -> torch.Tensor:
    if input_ids.ndim == 1:
        return input_ids.unsqueeze(0)
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be 1D or 2D, got shape {tuple(input_ids.shape)}")
    return input_ids


def find_injection_positions(
    input_ids: torch.Tensor,
    cfg: NLAConfig,
    *,
    verify_neighbors: bool = True,
) -> torch.Tensor:
    """Return one injection position per row.

    Raises if a row has zero or multiple marker tokens. If canonical neighbor
    IDs are present in the config, also verifies the surrounding tokens.
    """
    ids = _as_2d_input_ids(input_ids)
    positions: list[int] = []

    for row_idx, row in enumerate(ids):
        matches = (row == cfg.injection_token_id).nonzero(as_tuple=False).flatten()
        if matches.numel() != 1:
            raise ValueError(
                f"expected exactly one injection token id {cfg.injection_token_id} "
                f"in row {row_idx}, found {matches.numel()}"
            )

        pos = int(matches.item())
        if verify_neighbors and cfg.injection_left_neighbor_id and cfg.injection_right_neighbor_id:
            if pos == 0 or pos == row.numel() - 1:
                raise ValueError(f"injection token in row {row_idx} is at sequence edge")
            left = int(row[pos - 1].item())
            right = int(row[pos + 1].item())
            if left != cfg.injection_left_neighbor_id or right != cfg.injection_right_neighbor_id:
                raise ValueError(
                    "injection marker neighbor mismatch in row "
                    f"{row_idx}: got ({left}, {right}), expected "
                    f"({cfg.injection_left_neighbor_id}, {cfg.injection_right_neighbor_id})"
                )

        positions.append(pos)

    return torch.tensor(positions, device=ids.device, dtype=torch.long)


def inject_at_marked_positions(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    activation_vectors: torch.Tensor,
    cfg: NLAConfig,
    *,
    verify_neighbors: bool = True,
) -> torch.Tensor:
    """Return a copy of ``inputs_embeds`` with marker positions replaced."""
    ids = _as_2d_input_ids(input_ids)
    if inputs_embeds.ndim != 3:
        raise ValueError(f"inputs_embeds must be [batch, seq, d], got {tuple(inputs_embeds.shape)}")
    if ids.shape[:2] != inputs_embeds.shape[:2]:
        raise ValueError("input_ids and inputs_embeds batch/sequence dimensions differ")

    acts = activation_vectors
    if acts.ndim == 1:
        acts = acts.unsqueeze(0)
    if acts.shape[0] != ids.shape[0]:
        raise ValueError(f"activation batch {acts.shape[0]} does not match input batch {ids.shape[0]}")
    if acts.shape[-1] != inputs_embeds.shape[-1]:
        raise ValueError(f"activation dim {acts.shape[-1]} does not match embedding dim {inputs_embeds.shape[-1]}")

    positions = find_injection_positions(ids, cfg, verify_neighbors=verify_neighbors)
    injected = inputs_embeds.clone()
    scaled = normalize_activation(acts.to(device=injected.device, dtype=injected.dtype), cfg.injection_scale)
    batch_idx = torch.arange(ids.shape[0], device=ids.device)
    injected[batch_idx, positions] = scaled
    return injected


def prepare_injected_inputs_embeds(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    activation_vectors: torch.Tensor,
    cfg: NLAConfig,
    *,
    verify_neighbors: bool = True,
) -> torch.Tensor:
    """Embed ``input_ids`` with ``model`` and inject activations at the marker."""
    embedding = model.get_input_embeddings()
    inputs_embeds = embedding(input_ids.to(embedding.weight.device))
    return inject_at_marked_positions(
        input_ids.to(inputs_embeds.device),
        inputs_embeds,
        activation_vectors.to(inputs_embeds.device),
        cfg,
        verify_neighbors=verify_neighbors,
    )


def encode_chat_messages(
    tokenizer: Any,
    messages: Sequence[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
) -> list[int]:
    """Apply a tokenizer chat template and normalize the return type to IDs."""
    encoded = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if hasattr(encoded, "input_ids"):
        value = encoded["input_ids"]
        if isinstance(value, torch.Tensor):
            return value[0].tolist() if value.ndim == 2 else value.tolist()
        return list(value[0] if value and isinstance(value[0], list) else value)
    if isinstance(encoded, torch.Tensor):
        return encoded[0].tolist() if encoded.ndim == 2 else encoded.tolist()
    if isinstance(encoded, list) and encoded and isinstance(encoded[0], list):
        return list(encoded[0])
    return list(encoded)


def build_av_prompt_ids(tokenizer: Any, cfg: NLAConfig) -> list[int]:
    """Token IDs for the canonical AV prompt."""
    content = cfg.av_prompt_template.format(injection_char=cfg.injection_char)
    return encode_chat_messages(tokenizer, [{"role": "user", "content": content}], add_generation_prompt=True)
