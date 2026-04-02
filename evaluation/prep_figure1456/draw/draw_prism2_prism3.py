"""Plot PRISM-2 vs PRISM-3 module ablation (matching figure5 style)."""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


# =========================
# 1) Config & data
# =========================
CONFIG = {
    "PRISM-2": {"color": "#d62728", "marker": "D"},
    "PRISM-3": {"color": "#7B2D8E", "marker": "p"},
}

XTICKS = ["100k", "200k", "400k", "600k", "800k"]

MEAN_DATA = {
    "PRISM-2": [
        [5.30, 5.47, 5.64, 5.71, 5.77],   # t=0
        [5.11, 5.25, 5.42, 5.50, 5.55],   # t=1
    ],
    "PRISM-3": [
        [5.34, 5.49, 5.62, 5.67, 5.68],   # t=0
        [5.14, 5.29, 5.41, 5.49, 5.46],   # t=1
    ],
}


# =========================
# 2) CLI arguments
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot PRISM-2 vs PRISM-3 module ablation"
    )
    parser.add_argument(
        "--save-path", type=str, default="figure_module_ablation.pdf",
        help="Output figure path"
    )
    return parser.parse_args()


# =========================
# 3) Main: plot
# =========================
def main():
    args = parse_args()

    fig, ax = plt.subplots(figsize=(9, 7))

    values = []
    legends = []
    x_range = range(1, len(XTICKS) + 1)

    for model, series in MEAN_DATA.items():
        ax.plot(
            x_range,
            series[0],
            color=CONFIG[model]["color"],
            marker=CONFIG[model]["marker"],
            linestyle="-",
            markersize=10,
        )
        values.extend(series[0])
        legends.append(f"{model} (t=0)")

        ax.plot(
            x_range,
            series[1],
            color=CONFIG[model]["color"],
            marker=CONFIG[model]["marker"],
            linestyle=":",
            markersize=10,
        )
        values.extend(series[1])
        legends.append(f"{model} (t=1)")

    ax.grid()
    ax.legend(legends, loc="lower right", fontsize=24)
    ax.set_xticks(x_range)
    ax.set_xticklabels(XTICKS, fontsize=24)

    y_min = round(min(values), 1)
    y_max = round(max(values), 1)
    ax.set_yticks(np.arange(y_min, y_max, 0.1))
    ax.tick_params(axis="y", labelsize=24)
    ax.set_ylabel("Acceptance Length", fontsize=26)
    ax.set_xlabel("Train Data Volume", fontsize=26)

    fig.tight_layout()

    save_path = Path(args.save_path)
    plt.savefig(save_path, bbox_inches="tight")
    print(f"[DONE] Figure saved to: {save_path.resolve()}")
    plt.close(fig)


if __name__ == "__main__":
    main()