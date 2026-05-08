"""AR SFT warm-start training.

Input parquet columns: prompt (critic text), activation_vector. The critic is
the truncated K+1-layer model and learns normalized MSE at the final real token.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer

from nano_nla.models import NLACriticModel, resolve_torch_device, resolve_torch_dtype, save_critic_checkpoint
from nano_nla.schema import (
    ACTIVATION_COLUMN,
    build_nla_config_from_yaml,
    load_config,
    merge_sidecar_into_config,
    write_model_sidecar,
)
from nano_nla.training.common import (
    cosine_lr,
    critic_mse_loss,
    ensure_pad_token,
    load_parquet_rows,
    set_optimizer_lr,
    set_seed,
    tokenize_text_batch,
)


def train_ar_sft(config: dict, dataset_path: str | Path | None = None, init_checkpoint: str | Path | None = None) -> Path:
    model_cfg = config["model"]
    sft_cfg = config["training"]["sft"]
    ar_cfg = sft_cfg["ar"]
    device_name = sft_cfg.get("device", "auto")
    device = resolve_torch_device(device_name, require_cuda=str(device_name).lower() in {"auto", "gpu"})
    dtype = resolve_torch_dtype(sft_cfg.get("dtype", "auto"), device=device)
    dataset_path = Path(dataset_path or Path(config["datagen"]["output_dir"]) / "ar_sft.parquet")
    config = merge_sidecar_into_config(config, dataset_path)
    model_cfg = config["model"]
    sft_cfg = config["training"]["sft"]
    ar_cfg = sft_cfg["ar"]
    output_dir = Path(ar_cfg["output_dir"])

    set_seed(int(sft_cfg.get("seed", 42)))
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], trust_remote_code=True)
    ensure_pad_token(tokenizer)
    nla_cfg = build_nla_config_from_yaml(config, tokenizer)

    if init_checkpoint:
        critic = NLACriticModel.from_pretrained(init_checkpoint, dtype=dtype, device=device)
    else:
        critic = NLACriticModel.from_pretrained_base(
            model_cfg["name"],
            d_model=nla_cfg.d_model,
            num_hidden_layers=nla_cfg.ar_num_layers or model_cfg["target_layer"] + 1,
            dtype=dtype,
            device=device,
        )
    critic.train()

    rows = load_parquet_rows(dataset_path)
    if not rows:
        raise RuntimeError(f"empty AR-SFT dataset: {dataset_path}")

    batch_size = int(ar_cfg.get("batch_size", 1))
    grad_accum = int(ar_cfg.get("gradient_accumulation_steps", 1))
    epochs = int(ar_cfg.get("num_epochs", 1))
    total_micro_steps = epochs * ((len(rows) + batch_size - 1) // batch_size)
    total_updates = max(1, total_micro_steps // grad_accum)
    warmup_steps = int(total_updates * float(sft_cfg.get("warmup_ratio", 0.0)))

    optimizer = AdamW(
        critic.parameters(),
        lr=float(sft_cfg["learning_rate"]),
        weight_decay=float(sft_cfg.get("weight_decay", 0.0)),
    )

    update_step = 0
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        progress = tqdm(range(0, len(rows), batch_size), desc=f"AR SFT epoch {epoch + 1}/{epochs}")
        for start in progress:
            batch = rows[start : start + batch_size]
            prompts = [row["prompt"] for row in batch]
            targets = torch.tensor([row[ACTIVATION_COLUMN] for row in batch], dtype=torch.float32, device=device)
            tok = tokenize_text_batch(tokenizer, prompts, device)

            loss, _ = critic_mse_loss(critic, tok["input_ids"], tok["attention_mask"], targets, nla_cfg)
            (loss / grad_accum).backward()
            micro_step += 1
            progress.set_postfix(loss=float(loss.detach().cpu()))

            if micro_step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(critic.parameters(), float(sft_cfg.get("max_grad_norm", 1.0)))
                lr = cosine_lr(float(sft_cfg["learning_rate"]), update_step, total_updates, warmup_steps)
                set_optimizer_lr(optimizer, lr)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1

                if update_step % int(ar_cfg.get("logging_steps", 10)) == 0:
                    print(f"[ar-sft] step={update_step} loss={float(loss.detach().cpu()):.4f} lr={lr:.2e}")
                if update_step % int(ar_cfg.get("save_steps", 500)) == 0:
                    ckpt = output_dir / f"step_{update_step}"
                    save_critic_checkpoint(critic, ckpt, base_model_name=model_cfg["name"], mse_scale=nla_cfg.mse_scale)
                    tokenizer.save_pretrained(ckpt)
                    write_model_sidecar(ckpt, nla_cfg, role="ar", stage="sft", base_model=model_cfg["name"])

    save_critic_checkpoint(critic, output_dir, base_model_name=model_cfg["name"], mse_scale=nla_cfg.mse_scale)
    tokenizer.save_pretrained(output_dir)
    write_model_sidecar(output_dir, nla_cfg, role="ar", stage="sft", base_model=model_cfg["name"])
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AR SFT warm-start")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--init-checkpoint", default=None)
    args = parser.parse_args()
    train_ar_sft(load_config(args.config), args.dataset, args.init_checkpoint)


if __name__ == "__main__":
    main()
