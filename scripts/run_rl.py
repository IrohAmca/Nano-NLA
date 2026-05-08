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
    args = parser.parse_args()
    train_rl_grpo(load_config(args.config), args.dataset, args.actor_checkpoint, args.critic_checkpoint)


if __name__ == "__main__":
    main()
