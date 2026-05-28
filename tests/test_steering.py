import torch
from torch import nn

from nano_nla.eval.steering import apply_steering_hook, compute_steering_vector


class _DummyInner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity()])


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.zeros(()))
        self.model = _DummyInner()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model.layers[0](hidden)


def test_steering_hook_applies_once_to_prompt_position() -> None:
    model = _DummyModel()
    direction = torch.tensor([1.0, 0.0, 0.0])
    prompt_hidden = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 3.0, 4.0], [0.0, 0.0, 2.0]]])

    handles = apply_steering_hook(
        model,
        target_layer=0,
        token_position=1,
        steering_direction=direction,
        alpha=0.5,
        min_sequence_length=prompt_hidden.shape[1],
        apply_once=True,
    )
    try:
        steered = model(prompt_hidden)
        expected = prompt_hidden.clone()
        expected[:, 1, 0] += 2.5
        assert torch.allclose(steered, expected)

        assert torch.allclose(model(prompt_hidden), prompt_hidden)
    finally:
        for handle in handles:
            handle.remove()


def test_steering_hook_skips_decode_step_shorter_than_prompt() -> None:
    model = _DummyModel()
    decode_hidden = torch.tensor([[[0.0, 3.0, 4.0]]])

    handles = apply_steering_hook(
        model,
        target_layer=0,
        token_position=0,
        steering_direction=torch.tensor([1.0, 0.0, 0.0]),
        alpha=1.0,
        min_sequence_length=3,
        apply_once=True,
    )
    try:
        assert torch.allclose(model(decode_hidden), decode_hidden)
    finally:
        for handle in handles:
            handle.remove()


class _FakeClient:
    ar = object()

    def __init__(self) -> None:
        self.reconstruct_calls: list[str] = []

    def generate_explanation(self, activation: torch.Tensor) -> str:
        return "<explanation>rabbit reward</explanation>"

    def reconstruct(self, explanation: str) -> torch.Tensor:
        self.reconstruct_calls.append(explanation)
        if "mouse" in explanation:
            return torch.tensor([1.0, 0.0])
        return torch.tensor([0.0, 0.0])


def test_compute_steering_vector_reconstructs_extracted_explanation_payload() -> None:
    client = _FakeClient()

    result = compute_steering_vector(
        client,
        torch.zeros(2),
        {"rabbit": "mouse"},
    )

    assert client.reconstruct_calls == ["rabbit reward", "mouse reward"]
    assert result.original_explanation == "rabbit reward"
    assert result.edited_explanation == "mouse reward"
    assert torch.allclose(result.direction, torch.tensor([1.0, 0.0]))
