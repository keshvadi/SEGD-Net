#!/usr/bin/env python3
"""Plot the training-time values already reported in manuscript Table 10."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

REPORTED_MINUTES = {
    "Standard CNN": 315,
    "M2G-HBoost": 284,
    "Distillation\nEnsemble": 228,
    "EffNetV2-ViT": 198,
    "HyperCLSA": 175,
    "SEGD-Net": 118,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("../images/Fig9.png"))
    args = parser.parse_args()
    names = list(REPORTED_MINUTES)
    values = list(REPORTED_MINUTES.values())
    colors = ["#9da8b4"] * (len(names) - 1) + ["#2474a6"]
    figure, axis = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    bars = axis.bar(names, values, color=colors, edgecolor="#34495e", linewidth=0.6)
    axis.set_ylabel("Reported training time (min)")
    axis.set_title("Training time values reported in manuscript Table 10")
    axis.set_ylim(0, max(values) * 1.16)
    axis.grid(axis="y", alpha=0.25, linestyle="--")
    axis.set_axisbelow(True)
    axis.tick_params(axis="x", rotation=15)
    for bar, value in zip(bars, values):
        axis.annotate(
            str(value),
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote plot of manuscript-reported times: {args.output.resolve()}")


if __name__ == "__main__":
    main()
