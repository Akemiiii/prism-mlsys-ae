"""Plot PRISM-2 vs PRISM-3 module ablation (matching figure5 style)."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FormatStrFormatter


# =========================
# 1) Config & data
# =========================
CONFIG = {
    "PRISM-2": {"color": "#d62728", "marker": "D"},
    "PRISM-3": {"color": "#7B2D8E", "marker": "p"},
}

XTICKS = ["1", "2", "4", "6", "8"]

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

    x_range = range(1, len(XTICKS) + 1)

    for model, series in MEAN_DATA.items():
        ax.plot(
            x_range,
            series[0],
            color=CONFIG[model]["color"],
            marker=CONFIG[model]["marker"],
            linestyle="-",
            linewidth=2.5,
            markersize=10,
        )

        ax.plot(
            x_range,
            series[1],
            color=CONFIG[model]["color"],
            marker=CONFIG[model]["marker"],
            linestyle=":",
            linewidth=2.5,
            markersize=10,
        )

    ax.grid(True, linestyle='--', alpha=0.6)

    ax.set_xticks(x_range)
    ax.set_xticklabels(XTICKS, fontsize=34)
    ax.set_xlabel(r"Train Data Volume ($10^{2}$K)", fontsize=36)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune='both'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.tick_params(axis="y", labelsize=34)
    ax.set_ylabel("Acceptance Length", fontsize=36)

    handles = []
    for model_label, style in CONFIG.items():
        handles.append(Line2D(
            [0], [0],
            color=style['color'],
            marker=style['marker'],
            linestyle='-',
            linewidth=2.5,
            markersize=10,
            label=f'{model_label} (t=0)'
        ))
        handles.append(Line2D(
            [0], [0],
            color=style['color'],
            marker=style['marker'],
            linestyle=':',
            linewidth=2.5,
            markersize=10,
            label=f'{model_label} (t=1)'
        ))

    fig.legend(
        handles=handles,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),
        bbox_transform=fig.transFigure,
        ncol=2,
        fontsize=32,
        frameon=False
    )

    fig.tight_layout()

    save_path = Path(args.save_path)
    plt.savefig(save_path, bbox_inches="tight")
    print(f"[DONE] Figure saved to: {save_path.resolve()}")
    plt.close(fig)


if __name__ == "__main__":
    main()
