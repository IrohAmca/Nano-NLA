"""Minimal standalone GRPO for Nano-NLA.

This follows the NLA loop: generate G AV explanations per activation, score
them with the AR, train the AV with group-normalized rewards plus a reference
KL term, and train the AR on valid generated explanations.
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

from nano_nla.injection import build_av_prompt_ids, prepare_injected_inputs_embeds
from nano_nla.models import (
    NLACriticModel,
    last_token_values,
    resolve_torch_device,
    resolve_torch_dtype,
    save_critic_checkpoint,
)
from nano_nla.schema import (
    ACTIVATION_COLUMN,
    build_nla_config_from_yaml,
    load_config,
    merge_sidecar_into_config,
    normalize_activation,
    write_model_sidecar,
)
from nano_nla.training.common import ensure_pad_token, load_parquet_rows, set_seed, tokenize_text_batch
from nano_nla.training.reward import score_response_text


def _sample_next(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def generate_batch_with_injection(
    model: torch.nn.Module,
    tokenizer,
    prompt_ids: list[int],
    activations: torch.Tensor,
    cfg,
    *,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> list[tuple[list[int], str]]:
    batch_size = activations.shape[0]
    input_ids = torch.tensor([prompt_ids] * batch_size, dtype=torch.long, device=device)
    inputs_embeds = prepare_injected_inputs_embeds(model, input_ids, activations, cfg)
    attention_mask = torch.ones_like(input_ids)
    out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=True)
    past = out.past_key_values
    generated: list[list[int]] = [[] for _ in range(batch_size)]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    if eos_id is None:
        eos_id = 0
    eos = torch.full((batch_size,), int(eos_id), dtype=torch.long, device=device)
    next_id = _sample_next(out.logits[:, -1, :], temperature)

    for _ in range(max_new_tokens):
        next_id = torch.where(finished, eos, next_id)
        for row_idx, token in enumerate(next_id.tolist()):
            if finished[row_idx]:
                continue
            if token == eos_id:
                finished[row_idx] = True
                continue
            generated[row_idx].append(int(token))
        if bool(finished.all()):
            break
        out = model(input_ids=next_id.view(batch_size, 1), past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_id = _sample_next(out.logits[:, -1, :], temperature)

    return [(ids, tokenizer.decode(ids, skip_special_tokens=True)) for ids in generated]


@torch.no_grad()
def generate_with_injection(
    model: torch.nn.Module,
    tokenizer,
    prompt_ids: list[int],
    activation: torch.Tensor,
    cfg,
    *,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> tuple[list[int], str]:
    return generate_batch_with_injection(
        model,
        tokenizer,
        prompt_ids,
        activation.view(1, -1),
        cfg,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        device=device,
    )[0]


def _response_logprobs(
    model: torch.nn.Module,
    prompt_ids: list[int],
    response_ids: list[int],
    activation: torch.Tensor,
    cfg,
    *,
    device: torch.device,
) -> torch.Tensor:
    if not response_ids:
        return torch.zeros((), device=device, requires_grad=True)
    full_ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device=device)
    inputs_embeds = prepare_injected_inputs_embeds(model, full_ids, activation.view(1, -1), cfg)
    attention_mask = torch.ones_like(full_ids)
    logits = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits
    start = len(prompt_ids) - 1
    end = len(prompt_ids) + len(response_ids) - 1
    token_logits = logits[0, start:end, :]
    targets = torch.tensor(response_ids, dtype=torch.long, device=device)
    log_probs = F.log_softmax(token_logits, dim=-1)
    return log_probs.gather(1, targets.view(-1, 1)).squeeze(1).mean()


def _group_advantages(rewards: list[float], group_size: int) -> list[float]:
    advantages: list[float] = []
    for start in range(0, len(rewards), group_size):
        group = torch.tensor(rewards[start : start + group_size], dtype=torch.float32)
        if group.numel() == 1:
            advantages.append(0.0)
            continue
        normed = (group - group.mean()) / group.std(unbiased=False).clamp_min(1e-6)
        advantages.extend(normed.tolist())
    return advantages


def train_rl_grpo(
    config: dict,
    dataset_path: str | Path | None = None,
    actor_checkpoint: str | Path | None = None,
    critic_checkpoint: str | Path | None = None,
) -> Path:
    model_cfg = config["model"]
    rl_cfg = config["training"]["rl"]
    device_name = rl_cfg.get("device", "auto")
    device = resolve_torch_device(device_name, require_cuda=str(device_name).lower() in {"auto", "gpu"})
    dtype = resolve_torch_dtype(rl_cfg.get("dtype", "auto"), device=device)
    dataset_path = Path(dataset_path or Path(config["datagen"]["output_dir"]) / "rl.parquet")
    config = merge_sidecar_into_config(config, dataset_path)
    model_cfg = config["model"]
    rl_cfg = config["training"]["rl"]
    output_dir = Path(rl_cfg["output_dir"])

    set_seed(int(config["training"]["sft"].get("seed", 42)))
    tokenizer = AutoTokenizer.from_pretrained(actor_checkpoint or model_cfg["name"], trust_remote_code=True)
    ensure_pad_token(tokenizer)
    nla_cfg = build_nla_config_from_yaml(config, tokenizer)
    prompt_ids = build_av_prompt_ids(tokenizer, nla_cfg)

    actor = AutoModelForCausalLM.from_pretrained(
        actor_checkpoint or model_cfg["name"],
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    actor.train()
    ref_actor = copy.deepcopy(actor).to(device)
    ref_actor.eval()
    for p in ref_actor.parameters():
        p.requires_grad_(False)

    if critic_checkpoint:
        critic = NLACriticModel.from_pretrained(critic_checkpoint, dtype=dtype, device=device)
    else:
        critic = NLACriticModel.from_pretrained_base(
            model_cfg["name"],
            d_model=nla_cfg.d_model,
            num_hidden_layers=nla_cfg.ar_num_layers or model_cfg["target_layer"] + 1,
            dtype=dtype,
            device=device,
        )
    critic.train()

    actor_optim = AdamW(actor.parameters(), lr=float(rl_cfg["actor_lr"]))
    critic_optim = AdamW(critic.parameters(), lr=float(rl_cfg["critic_lr"]))
    rows = load_parquet_rows(dataset_path)
    if not rows:
        raise RuntimeError(f"empty RL dataset: {dataset_path}")

    group_size = int(rl_cfg.get("grpo_group_size", 4))
    batch_size = int(rl_cfg.get("batch_size", 1))
    max_new = int(rl_cfg.get("rollout_max_length", 150))
    beta = float(rl_cfg.get("kl_coeff", 0.05))
    log_reward = bool(rl_cfg.get("reward_log_transform", False))
    rng = random.Random(int(config["training"]["sft"].get("seed", 42)))

    for step in trange(int(rl_cfg.get("num_steps", 1000)), desc="RL GRPO"):
        batch = [rows[rng.randrange(len(rows))] for _ in range(batch_size)]
        samples: list[dict] = []
        rewards: list[float] = []

        actor.eval()
        critic.eval()
        rollout_activations = torch.stack(
            [
                torch.tensor(row[ACTIVATION_COLUMN], dtype=torch.float32, device=device)
                for row in batch
                for _ in range(group_size)
            ]
        )
        generated_outputs = generate_batch_with_injection(
            actor,
            tokenizer,
            prompt_ids,
            rollout_activations,
            nla_cfg,
            max_new_tokens=max_new,
            temperature=float(config.get("inference", {}).get("temperature", 1.0)),
            device=device,
        )
        for activation, (response_ids, response_text) in zip(rollout_activations, generated_outputs, strict=True):
            result = score_response_text(
                response_text,
                activation,
                critic=critic,
                tokenizer=tokenizer,
                cfg=nla_cfg,
                device=device,
                log_transform=log_reward,
            )
            samples.append(
                {
                    "activation": activation,
                    "response_ids": response_ids,
                    "response_text": response_text,
                    "explanation": result.explanation,
                }
            )
            rewards.append(result.reward)

        advantages = _group_advantages(rewards, group_size)

        actor.train()
        actor_optim.zero_grad(set_to_none=True)
        policy_losses: list[torch.Tensor] = []
        for sample, adv in zip(samples, advantages, strict=True):
            logp = _response_logprobs(actor, prompt_ids, sample["response_ids"], sample["activation"], nla_cfg, device=device)
            with torch.no_grad():
                ref_logp = _response_logprobs(
                    ref_actor,
                    prompt_ids,
                    sample["response_ids"],
                    sample["activation"],
                    nla_cfg,
                    device=device,
                )
            policy_losses.append(-(torch.tensor(adv, device=device) * logp) + beta * (logp - ref_logp).pow(2))
        actor_loss = torch.stack(policy_losses).mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), float(config["training"]["sft"].get("max_grad_norm", 1.0)))
        actor_optim.step()

        valid = [s for s in samples if s["explanation"] is not None]
        critic_loss = torch.zeros((), device=device)
        if valid:
            critic.train()
            critic_optim.zero_grad(set_to_none=True)
            prompts = [nla_cfg.critic_prompt_template.format(explanation=s["explanation"]) for s in valid]
            targets = torch.stack([s["activation"].float() for s in valid]).to(device)
            tok = tokenize_text_batch(tokenizer, prompts, device)
            out = critic(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
            pred = last_token_values(out.values, tok["attention_mask"])
            critic_loss = F.mse_loss(
                normalize_activation(pred.float(), nla_cfg.mse_scale),
                normalize_activation(targets.float(), nla_cfg.mse_scale),
            )
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), float(config["training"]["sft"].get("max_grad_norm", 1.0)))
            critic_optim.step()

        if step % int(rl_cfg.get("logging_steps", 5)) == 0:
            mean_reward = sum(rewards) / max(1, len(rewards))
            print(
                f"[rl] step={step} reward={mean_reward:.4f} "
                f"actor_loss={float(actor_loss.detach().cpu()):.4f} "
                f"critic_loss={float(critic_loss.detach().cpu()):.4f} valid={len(valid)}/{len(samples)}"
            )

        if (step + 1) % int(rl_cfg.get("save_interval", 100)) == 0:
            ckpt = output_dir / f"step_{step + 1}"
            ckpt.mkdir(parents=True, exist_ok=True)
            actor.save_pretrained(ckpt / "av")
            tokenizer.save_pretrained(ckpt / "av")
            save_critic_checkpoint(critic, ckpt / "ar", base_model_name=model_cfg["name"], mse_scale=nla_cfg.mse_scale)
            write_model_sidecar(ckpt / "av", nla_cfg, role="av", stage="rl", base_model=model_cfg["name"])
            write_model_sidecar(ckpt / "ar", nla_cfg, role="ar", stage="rl", base_model=model_cfg["name"])

    output_dir.mkdir(parents=True, exist_ok=True)
    actor.save_pretrained(output_dir / "av")
    tokenizer.save_pretrained(output_dir / "av")
    save_critic_checkpoint(critic, output_dir / "ar", base_model_name=model_cfg["name"], mse_scale=nla_cfg.mse_scale)
    write_model_sidecar(output_dir / "av", nla_cfg, role="av", stage="rl", base_model=model_cfg["name"])
    write_model_sidecar(output_dir / "ar", nla_cfg, role="ar", stage="rl", base_model=model_cfg["name"])
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Nano-NLA with minimal GRPO")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--actor-checkpoint", default=None)
    parser.add_argument("--critic-checkpoint", default=None)
    args = parser.parse_args()
    train_rl_grpo(load_config(args.config), args.dataset, args.actor_checkpoint, args.critic_checkpoint)


if __name__ == "__main__":
    main()
