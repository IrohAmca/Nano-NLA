"""Run Nano-NLA standalone GRPO."""

from __future__ import annotations

import argparse

from nano_nla.schema import load_config
from nano_nla.training.rl_grpo import train_rl_grpo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal Nano-NLA GRPO")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--actor-checkpoint", default=None)
    parser.add_argument("--critic-checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--start-step", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--row-offset", type=int, default=0)
    args = parser.parse_args()
    train_rl_grpo(
        load_config(args.config),
        args.dataset,
        args.actor_checkpoint,
        args.critic_checkpoint,
        args.output_dir,
        args.start_step,
        args.max_rows,
        args.row_offset,
    )


if __name__ == "__main__":
    main()
