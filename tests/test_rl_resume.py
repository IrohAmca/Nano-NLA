from pathlib import Path

from nano_nla.training.rl_grpo import (
    latest_complete_rl_checkpoint,
    resolve_rl_resume_checkpoints,
    rl_steps_until_end,
)


def _write_step_dir(root: Path, step: int, *, actor: bool = True, critic: bool = True) -> Path:
    step_dir = root / f"step_{step}"
    if actor:
        (step_dir / "av").mkdir(parents=True)
        (step_dir / "av" / "nla_meta.yaml").write_text("", encoding="utf-8")
    if critic:
        (step_dir / "ar").mkdir(parents=True)
        (step_dir / "ar" / "nla_meta.yaml").write_text("", encoding="utf-8")
    return step_dir


def test_latest_complete_checkpoint_uses_newest_pair(tmp_path: Path) -> None:
    _write_step_dir(tmp_path, 200)
    _write_step_dir(tmp_path, 400)

    step, actor_dir, critic_dir = latest_complete_rl_checkpoint(tmp_path)

    assert step == 400
    assert actor_dir == tmp_path / "step_400" / "av"
    assert critic_dir == tmp_path / "step_400" / "ar"


def test_latest_complete_checkpoint_skips_incomplete_newer_step(tmp_path: Path) -> None:
    _write_step_dir(tmp_path, 400)
    _write_step_dir(tmp_path, 600, critic=False)

    assert latest_complete_rl_checkpoint(tmp_path) == (
        400,
        tmp_path / "step_400" / "av",
        tmp_path / "step_400" / "ar",
    )


def test_resume_without_saved_pair_keeps_initialization_checkpoints(tmp_path: Path) -> None:
    actor_init = tmp_path / "init" / "av"
    critic_init = tmp_path / "init" / "ar"

    actor_checkpoint, critic_checkpoint, start_step = resolve_rl_resume_checkpoints(
        tmp_path / "window",
        actor_init,
        critic_init,
        resume_latest=True,
    )

    assert actor_checkpoint == actor_init
    assert critic_checkpoint == critic_init
    assert start_step == 0


def test_end_step_runs_only_remaining_window_steps() -> None:
    assert rl_steps_until_end(configured_steps=3000, start_step=400, end_step=800) == 400
