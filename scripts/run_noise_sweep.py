#!/usr/bin/env python3
"""Run five-fold training at 0%-50% train-label corruption."""

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
    parser.add_argument(
        "--noise-mode", choices=("symmetric", "asymmetric"), default="symmetric"
    )
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for rate in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
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
            str(args.output / f"noise_{rate:.1f}"),
            "--noise-rate",
            str(rate),
            "--noise-mode",
            args.noise_mode,
        ]
        if args.epochs is not None:
            command.extend(("--epochs", str(args.epochs)))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
