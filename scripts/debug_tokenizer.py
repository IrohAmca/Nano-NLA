"""Debug injection-marker tokenization for the Qwen-0.5B config."""

from __future__ import annotations

import sys
from typing import Any

import torch
from transformers import AutoTokenizer

from nano_nla.schema import build_nla_config_from_yaml, load_config


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def flat_input_ids(value: Any) -> list[int]:
    """Normalize tokenizer/chat-template outputs to a flat list of token IDs."""
    if hasattr(value, "input_ids"):
        value = value["input_ids"]
    elif isinstance(value, dict) and "input_ids" in value:
        value = value["input_ids"]

    if isinstance(value, torch.Tensor):
        value = value[0] if value.ndim == 2 else value
        return [int(x) for x in value.tolist()]

    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    return [int(x) for x in value]


def main() -> None:
    config = load_config("configs/qwen05b.yaml")
    model_name = config["model"]["name"]
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    char = config["injection"]["injection_char"]
    ids = tok.encode(char, add_special_tokens=False)
    print(f"Model: {model_name}")
    print(f"Injection char: {char!r}")
    print(f"Standalone encode: {ids}")
    print(f"Single token: {len(ids) == 1}")

    nla_cfg = build_nla_config_from_yaml(config, tok)
    print(
        "Canonical config: "
        f"token={nla_cfg.injection_token_id}, "
        f"neighbors=({nla_cfg.injection_left_neighbor_id}, {nla_cfg.injection_right_neighbor_id}), "
        f"mse_scale={nla_cfg.mse_scale:.4f}"
    )

    content = config["prompts"]["av"].format(injection_char=char)
    messages = [{"role": "user", "content": content}]
    result = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    full_ids = flat_input_ids(result)

    print(f"\nFull token IDs ({len(full_ids)} total):")
    target_id = ids[0] if len(ids) == 1 else -1
    found_count = 0
    for i, tid in enumerate(full_ids):
        decoded = tok.decode(tid)
        marker = ""
        if tid == target_id:
            marker = " <--- INJECTION"
            found_count += 1
        print(f"  [{i:3d}] {tid:6d} -> {decoded!r}{marker}")

    print(f"\nInjection token found {found_count} times")

    print("\nAlternative marker candidates:")
    for candidate, label in [
        ("\u320e", "U+320E"),
        ("\u321c", "U+321C"),
        ("\u3297", "U+3297"),
        ("\u3299", "U+3299"),
        ("\u3200", "U+3200"),
    ]:
        enc = tok.encode(candidate, add_special_tokens=False)
        print(f"  {label}: tokens={enc} count={len(enc)}")


if __name__ == "__main__":
    main()
