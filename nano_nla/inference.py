"""Pure torch Nano-NLA inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from nano_nla.injection import build_av_prompt_ids, prepare_injected_inputs_embeds
from nano_nla.models import NLACriticModel, last_token_values, resolve_torch_dtype
from nano_nla.schema import NLAConfig, build_nla_config_from_yaml, load_config, merge_sidecar_into_config, normalize_activation
from nano_nla.training.common import ensure_pad_token


@dataclass
class NLAInferenceResult:
    explanation: str
    reconstruction: torch.Tensor | None = None
    mse_nrm: float | None = None
    cosine: float | None = None


def _sample_next(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)


class NLAClient:
    def __init__(
        self,
        config_path: str | Path,
        *,
        av_checkpoint: str | Path | None = None,
        ar_checkpoint: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        self.config = load_config(config_path)
        self.config = merge_sidecar_into_config(self.config, av_checkpoint or ar_checkpoint)
        model_name = self.config["model"]["name"]
        inference_cfg = self.config.get("inference", {})
        self.device = torch.device(device or inference_cfg.get("device", "cpu"))
        dtype = resolve_torch_dtype(inference_cfg.get("dtype", "float32"))

        self.tokenizer = AutoTokenizer.from_pretrained(av_checkpoint or model_name, trust_remote_code=True)
        ensure_pad_token(self.tokenizer)
        self.nla_cfg: NLAConfig = build_nla_config_from_yaml(self.config, self.tokenizer)
        self.prompt_ids = build_av_prompt_ids(self.tokenizer, self.nla_cfg)

        self.av = AutoModelForCausalLM.from_pretrained(
            av_checkpoint or model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.av.eval()
        self.ar = None
        if ar_checkpoint is not None:
            self.ar = NLACriticModel.from_pretrained(ar_checkpoint, dtype=dtype, device=self.device)
            self.ar.eval()

    @torch.no_grad()
    def generate_explanation(
        self,
        activation: torch.Tensor,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        max_new_tokens = int(max_new_tokens or self.config.get("inference", {}).get("max_new_tokens", 200))
        temperature = float(temperature if temperature is not None else self.config.get("inference", {}).get("temperature", 1.0))
        activation = activation.to(self.device, dtype=torch.float32)
        input_ids = torch.tensor([self.prompt_ids], dtype=torch.long, device=self.device)
        inputs_embeds = prepare_injected_inputs_embeds(self.av, input_ids, activation.view(1, -1), self.nla_cfg)
        out = self.av(inputs_embeds=inputs_embeds, attention_mask=torch.ones_like(input_ids), use_cache=True)
        past = out.past_key_values
        next_id = _sample_next(out.logits[:, -1, :], temperature)
        generated: list[int] = []
        for _ in range(max_new_tokens):
            token = int(next_id.item())
            if token == self.tokenizer.eos_token_id:
                break
            generated.append(token)
            out = self.av(input_ids=next_id.view(1, 1), past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_id = _sample_next(out.logits[:, -1, :], temperature)
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    @torch.no_grad()
    def reconstruct(self, explanation: str) -> torch.Tensor:
        if self.ar is None:
            raise RuntimeError("AR checkpoint was not provided")
        prompt = self.nla_cfg.critic_prompt_template.format(explanation=explanation)
        tok = self.tokenizer(prompt, add_special_tokens=True, return_tensors="pt").to(self.device)
        out = self.ar(input_ids=tok["input_ids"], attention_mask=tok.get("attention_mask"))
        return last_token_values(out.values, tok.get("attention_mask"))[0].float().cpu()

    @torch.no_grad()
    def explain_and_score(self, activation: torch.Tensor) -> NLAInferenceResult:
        explanation = self.generate_explanation(activation)
        if self.ar is None:
            return NLAInferenceResult(explanation=explanation)
        reconstruction = self.reconstruct(explanation)
        gold = activation.detach().float().cpu().view(1, -1)
        pred = reconstruction.view(1, -1)
        mse = ((normalize_activation(pred, self.nla_cfg.mse_scale) - normalize_activation(gold, self.nla_cfg.mse_scale)) ** 2).mean()
        cosine = F.cosine_similarity(pred, gold).item()
        return NLAInferenceResult(
            explanation=explanation,
            reconstruction=reconstruction,
            mse_nrm=float(mse.item()),
            cosine=float(cosine),
        )
