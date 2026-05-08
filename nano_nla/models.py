"""NLA model wrappers.

The AV is a standard causal LM trained with embedding injection. The AR
("critic") is a truncated causal LM plus a bias-free vector head. It returns
raw residual-space predictions; losses and rewards perform L2 normalization.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM


VALUE_HEAD_NAME = "value_head.safetensors"
CRITIC_CONFIG_NAME = "critic_config.json"


@dataclass
class NLACriticOutput:
    values: torch.Tensor
    backbone_last_hidden: torch.Tensor


@dataclass
class CriticCheckpointConfig:
    base_model_name: str
    d_model: int
    num_hidden_layers: int
    mse_scale: float | None = None


def enable_cuda_performance() -> None:
    """Enable GPU math modes that are useful for Colab-class CUDA runs."""
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def resolve_torch_device(
    device: str | torch.device | None = "auto",
    *,
    require_cuda: bool = False,
) -> torch.device:
    """Resolve repo config device names, including `auto` and legacy `gpu`."""
    if isinstance(device, torch.device):
        resolved = device
    else:
        value = "auto" if device is None else str(device).lower()
        if value in {"auto", "gpu"}:
            if torch.cuda.is_available():
                resolved = torch.device("cuda:0")
            elif require_cuda:
                raise RuntimeError("CUDA is required for device=auto/gpu, but torch.cuda.is_available() is false")
            else:
                resolved = torch.device("cpu")
        elif value == "cuda":
            resolved = torch.device("cuda:0")
        else:
            resolved = torch.device(value)

    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"{resolved} requested, but CUDA is not available")
        if resolved.index is None:
            resolved = torch.device("cuda:0")
        index = resolved.index
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"{resolved} requested, but only {torch.cuda.device_count()} CUDA device(s) are visible")
        torch.cuda.set_device(resolved)
        enable_cuda_performance()
    elif require_cuda:
        raise RuntimeError(f"CUDA is required, but resolved device is {resolved}")
    return resolved


def resolve_torch_dtype(
    dtype: str | torch.dtype | None,
    *,
    device: str | torch.device | None = None,
) -> torch.dtype | None:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    value = dtype.lower()
    if value == "auto":
        resolved_device = resolve_torch_device(device, require_cuda=False) if device is not None else None
        if resolved_device is not None and resolved_device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if resolved_device is None and torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    if value in {"float32", "fp32"}:
        return torch.float32
    if value in {"float16", "fp16"}:
        return torch.float16
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {dtype}")


def _text_config(config: Any) -> Any:
    return getattr(config, "text_config", config)


def _truncate_config_layers(config: Any, num_layers: int) -> None:
    text_cfg = _text_config(config)
    text_cfg.num_hidden_layers = num_layers
    for attr in ("layer_types", "sliding_window_pattern", "no_rope_layers"):
        value = getattr(text_cfg, attr, None)
        if isinstance(value, (list, tuple)) and len(value) > num_layers:
            setattr(text_cfg, attr, type(value)(value[:num_layers]))


def _inner_transformer(backbone: nn.Module) -> nn.Module:
    if hasattr(backbone, "model"):
        return backbone.model
    if hasattr(backbone, "transformer"):
        return backbone.transformer
    raise TypeError(f"{type(backbone).__name__} has no .model or .transformer inner module")


def _strip_final_norm(inner: nn.Module) -> None:
    for attr in ("norm", "final_layernorm", "ln_f"):
        if hasattr(inner, attr):
            setattr(inner, attr, nn.Identity())
            return
    raise TypeError(f"could not find final layernorm on {type(inner).__name__}")


class NLACriticModel(nn.Module):
    """Truncated transformer + bias-free value head for activation recovery."""

    def __init__(self, config: Any, backbone: nn.Module) -> None:
        super().__init__()
        text_cfg = _text_config(config)
        self.config = text_cfg
        self.backbone = backbone
        self.d_model = int(text_cfg.hidden_size)
        self.num_hidden_layers = int(text_cfg.num_hidden_layers)
        self.value_head = nn.Linear(self.d_model, self.d_model, bias=False)

        inner = _inner_transformer(self.backbone)
        self._no_split_modules = getattr(backbone, "_no_split_modules", [])
        last_param = next(inner.parameters(), None)
        if last_param is not None:
            self.value_head.to(device=last_param.device, dtype=last_param.dtype)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        nla_num_layers: int | None = None,
        dtype: str | torch.dtype | None = torch.float32,
        device: str | torch.device = "cpu",
        trust_remote_code: bool = True,
        **kwargs: Any,
    ) -> "NLACriticModel":
        """Load a critic from a base model or a saved critic checkpoint.

        ``nla_num_layers`` is the extraction layer K. The critic keeps blocks
        0..K inclusive, so config.num_hidden_layers becomes K+1.
        """
        model_path = str(pretrained_model_name_or_path)
        resolved_device = resolve_torch_device(device, require_cuda=False)
        torch_dtype = resolve_torch_dtype(dtype, device=resolved_device)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        if nla_num_layers is not None:
            needed = nla_num_layers + 1
            assert needed <= _text_config(config).num_hidden_layers, (
                f"nla_num_layers={nla_num_layers} requires {needed} layers, "
                f"but model only has {_text_config(config).num_hidden_layers}"
            )
            _truncate_config_layers(config, needed)

        backbone = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        if hasattr(backbone, "lm_head"):
            backbone.lm_head = nn.Identity()
        _strip_final_norm(_inner_transformer(backbone))

        model = cls(config, backbone)
        head_path = Path(model_path) / VALUE_HEAD_NAME
        if head_path.exists():
            model.value_head.load_state_dict(load_file(str(head_path)))
        return model.to(resolved_device)

    @classmethod
    def from_pretrained_base(
        cls,
        model_name: str,
        *,
        d_model: int | None = None,
        num_hidden_layers: int,
        dtype: str | torch.dtype | None = torch.float32,
        device: str | torch.device = "cpu",
        trust_remote_code: bool = True,
    ) -> "NLACriticModel":
        model = cls.from_pretrained(
            model_name,
            nla_num_layers=num_hidden_layers - 1,
            dtype=dtype,
            device=device,
            trust_remote_code=trust_remote_code,
        )
        if d_model is not None and model.d_model != d_model:
            raise ValueError(f"critic d_model={model.d_model}, expected {d_model}")
        return model

    def get_input_embeddings(self) -> nn.Module:
        return self.backbone.get_input_embeddings()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None, **kwargs: Any):
        out = _inner_transformer(self.backbone)(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            return_dict=True,
            **kwargs,
        )
        h = out.last_hidden_state
        return NLACriticOutput(values=self.value_head(h), backbone_last_hidden=h)

    def save_pretrained(self, save_directory: str | Path, state_dict=None, **kwargs: Any) -> None:
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        if state_dict is None:
            state_dict = self.state_dict()
        backbone_sd = {
            key.removeprefix("backbone."): value
            for key, value in state_dict.items()
            if key.startswith("backbone.")
        }
        head_sd = {
            key.removeprefix("value_head."): value
            for key, value in state_dict.items()
            if key.startswith("value_head.")
        }
        self.backbone.save_pretrained(str(save_dir), state_dict=backbone_sd, **kwargs)
        save_file(head_sd, str(save_dir / VALUE_HEAD_NAME))

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self) -> None:
        if hasattr(self.backbone, "gradient_checkpointing_disable"):
            self.backbone.gradient_checkpointing_disable()


def last_token_values(values: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """Extract value-head outputs at each row's final real token."""
    if attention_mask is None:
        return values[:, -1, :]
    lengths = attention_mask.long().sum(dim=-1).clamp_min(1) - 1
    batch_idx = torch.arange(values.shape[0], device=values.device)
    return values[batch_idx, lengths]


def save_critic_checkpoint(
    model: NLACriticModel,
    output_dir: str | Path,
    *,
    base_model_name: str,
    mse_scale: float | None = None,
) -> None:
    model.save_pretrained(output_dir)
    cfg = CriticCheckpointConfig(
        base_model_name=base_model_name,
        d_model=model.d_model,
        num_hidden_layers=model.num_hidden_layers,
        mse_scale=mse_scale,
    )
    Path(output_dir, CRITIC_CONFIG_NAME).write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")


def load_critic_checkpoint(
    checkpoint_dir: str | Path,
    *,
    dtype: str | torch.dtype | None = torch.float32,
    device: str | torch.device = "cpu",
    trust_remote_code: bool = True,
) -> NLACriticModel:
    return NLACriticModel.from_pretrained(
        checkpoint_dir,
        dtype=dtype,
        device=device,
        trust_remote_code=trust_remote_code,
    )


def prepare_critic_checkpoint(
    model_name: str,
    output_dir: str | Path,
    *,
    d_model: int,
    num_hidden_layers: int,
    mse_scale: float | None = None,
    dtype: str | torch.dtype | None = torch.float32,
    device: str | torch.device = "cpu",
) -> NLACriticModel:
    model = NLACriticModel.from_pretrained_base(
        model_name,
        d_model=d_model,
        num_hidden_layers=num_hidden_layers,
        dtype=dtype,
        device=device,
    )
    save_critic_checkpoint(model, output_dir, base_model_name=model_name, mse_scale=mse_scale)
    return model


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad_(False)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
