"""AV SFT warm-start training.

Input parquet columns: prompt (chat messages), response, activation_vector.
The activation is injected at the marker token; CE loss is applied only to the
assistant response tokens.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from nano_nla.injection import prepare_injected_inputs_embeds
from nano_nla.models import resolve_torch_device, resolve_torch_dtype
from nano_nla.schema import (
    ACTIVATION_COLUMN,
    build_nla_config_from_yaml,
    load_config,
    merge_sidecar_into_config,
    write_model_sidecar,
)
from nano_nla.training.common import (
    build_av_sft_features,
    cosine_lr,
    ensure_pad_token,
    load_parquet_rows,
    pad_1d,
    pad_labels,
    parse_messages,
    set_optimizer_lr,
    set_seed,
)


def train_av_sft(
    config: dict,
    dataset_path: str | Path | None = None,
    init_checkpoint: str | Path | None = None,
) -> Path:
    model_cfg = config["model"]
    sft_cfg = config["training"]["sft"]
    av_cfg = sft_cfg["av"]
    device_name = sft_cfg.get("device", "auto")
    device = resolve_torch_device(device_name, require_cuda=str(device_name).lower() in {"auto", "gpu"})
    dtype = resolve_torch_dtype(sft_cfg.get("dtype", "auto"), device=device)
    dataset_path = Path(dataset_path or Path(config["datagen"]["output_dir"]) / "av_sft.parquet")
    config = merge_sidecar_into_config(config, dataset_path)
    model_cfg = config["model"]
    sft_cfg = config["training"]["sft"]
    av_cfg = sft_cfg["av"]
    output_dir = Path(av_cfg["output_dir"])

    set_seed(int(sft_cfg.get("seed", 42)))
    tokenizer = AutoTokenizer.from_pretrained(init_checkpoint or model_cfg["name"], trust_remote_code=True)
    ensure_pad_token(tokenizer)
    nla_cfg = build_nla_config_from_yaml(config, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        init_checkpoint or model_cfg["name"],
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.train()
    rows = load_parquet_rows(dataset_path)
    if not rows:
        raise RuntimeError(f"empty AV-SFT dataset: {dataset_path}")

    batch_size = int(av_cfg.get("batch_size", 1))
    grad_accum = int(av_cfg.get("gradient_accumulation_steps", 1))
    epochs = int(av_cfg.get("num_epochs", 1))
    total_micro_steps = epochs * ((len(rows) + batch_size - 1) // batch_size)
    total_updates = max(1, total_micro_steps // grad_accum)
    warmup_steps = int(total_updates * float(sft_cfg.get("warmup_ratio", 0.0)))

    optimizer = AdamW(
        model.parameters(),
        lr=float(sft_cfg["learning_rate"]),
        weight_decay=float(sft_cfg.get("weight_decay", 0.0)),
    )

    update_step = 0
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        progress = tqdm(range(0, len(rows), batch_size), desc=f"AV SFT epoch {epoch + 1}/{epochs}")
        for start in progress:
            batch = rows[start : start + batch_size]
            ids_list: list[list[int]] = []
            labels_list: list[list[int]] = []
            activations: list[list[float]] = []

            for row in batch:
                ids, labels = build_av_sft_features(tokenizer, parse_messages(row["prompt"]), row["response"])
                ids_list.append(ids)
                labels_list.append(labels)
                activations.append(row[ACTIVATION_COLUMN])

            input_ids = pad_1d(ids_list, tokenizer.pad_token_id).to(device)
            labels = pad_labels(labels_list).to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
            activation_tensor = torch.tensor(activations, dtype=torch.float32, device=device)
            inputs_embeds = prepare_injected_inputs_embeds(model, input_ids, activation_tensor, nla_cfg)

            out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
            loss = out.loss / grad_accum
            loss.backward()
            micro_step += 1
            progress.set_postfix(loss=float(out.loss.detach().cpu()))

            if micro_step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(sft_cfg.get("max_grad_norm", 1.0)))
                lr = cosine_lr(float(sft_cfg["learning_rate"]), update_step, total_updates, warmup_steps)
                set_optimizer_lr(optimizer, lr)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1

                if update_step % int(av_cfg.get("logging_steps", 10)) == 0:
                    print(f"[av-sft] step={update_step} loss={float(out.loss.detach().cpu()):.4f} lr={lr:.2e}")
                if update_step % int(av_cfg.get("save_steps", 500)) == 0:
                    ckpt = output_dir / f"step_{update_step}"
                    ckpt.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)
                    write_model_sidecar(ckpt, nla_cfg, role="av", stage="sft", base_model=model_cfg["name"])

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    write_model_sidecar(output_dir, nla_cfg, role="av", stage="sft", base_model=model_cfg["name"])
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AV SFT warm-start")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--init-checkpoint", default=None)
    args = parser.parse_args()
    train_av_sft(load_config(args.config), args.dataset, args.init_checkpoint)


if __name__ == "__main__":
    main()
