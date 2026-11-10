#!/usr/bin/env python3
"""Run all five component ablations with the same fold protocol."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("breakhis", "bach"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for ablation in (
        "full",
        "without_cyto_features",
        "without_temporal_graph",
        "without_noise_filter",
        "without_weighted_kd",
    ):
        command = [
            sys.executable,
            str(Path(__file__).with_name("train_eval_ssl.py")),
            "--dataset",
            args.dataset,
            "--data-root",
            str(args.data_root),
            "--config",
            str(args.config),
            "--output",
            str(args.output / ablation),
            "--ablation",
            ablation,
        ]
        if args.epochs is not None:
            command.extend(("--epochs", str(args.epochs)))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
