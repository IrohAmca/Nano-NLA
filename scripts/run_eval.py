"""Run simple Nano-NLA metric summaries from saved tensors/parquets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nano_nla.eval.evaluations import load_activation_column, metric_summary
from nano_nla.schema import build_nla_config_from_yaml, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate reconstruction tensors")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gold-parquet", required=True, help="Parquet with activation_vector column")
    parser.add_argument("--pred-tensor", required=True, help="torch .pt tensor shaped [N, d_model]")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    cfg = build_nla_config_from_yaml(config)
    gold = load_activation_column(args.gold_parquet, args.max_rows)
    pred = torch.load(args.pred_tensor, map_location="cpu")
    if args.max_rows is not None:
        pred = pred[: args.max_rows]
    summary = metric_summary(pred.float(), gold.float(), cfg.mse_scale)
    text = json.dumps(summary, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
