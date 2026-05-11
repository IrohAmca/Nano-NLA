"""Run Nano-NLA SFT stages."""

from __future__ import annotations

import argparse

from nano_nla.schema import load_config
from nano_nla.training.sft_ar import train_ar_sft
from nano_nla.training.sft_av import train_av_sft


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AV/AR SFT")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["av", "ar", "both"], default="both")
    parser.add_argument("--av-dataset", default=None)
    parser.add_argument("--ar-dataset", default=None)
    parser.add_argument("--av-init-checkpoint", default=None)
    parser.add_argument("--ar-init-checkpoint", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.stage in ("av", "both"):
        train_av_sft(config, args.av_dataset, args.av_init_checkpoint)
    if args.stage in ("ar", "both"):
        train_ar_sft(config, args.ar_dataset, args.ar_init_checkpoint)


if __name__ == "__main__":
    main()
