#!/usr/bin/env python3
"""Plot tree/verify sweep figure from CSV files."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import ScaledTranslation

PREFERRED_CONFIG_ORDER = [
    (3, 2, 6),
    (3, 4, 12),
    (4, 4, 16),
    (5, 4, 16),
    (3, 8, 16),
    (6, 4, 16),
    (4, 8, 16),
    (5, 8, 32),
    (6, 8, 32),
    (7, 8, 32),
    (7, 10, 64),
    (8, 8, 64),
    (8, 10, 64),
]


def canonical_algorithm(name: str) -> str:
    key = (name or "").strip().upper()
    if key in {"PRISM", "LD"}:
        return "PRISM"
    if key in {"NONE", "NO", "OFF", "DISABLED"}:
        return "NONE"
    return key


def load_avg_tps(avg_csv: Path) -> dict[tuple[int, int, int], float]:
    tps_by_cfg: dict[tuple[int, int, int], float] = {}
    with avg_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {
            "algorithm",
            "speculative_num_steps",
            "speculative_eagle_topk",
            "speculative_num_draft_tokens",
            "avg_throughput",
        }
        if not required_cols.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV missing required columns: {required_cols}")

        for row in reader:
            if (row.get("status") or "").strip().lower() == "failed":
                continue
            if canonical_algorithm(row.get("algorithm", "")) != "PRISM":
                continue
            try:
                cfg = (
                    int(row["speculative_num_steps"]),
                    int(row["speculative_eagle_topk"]),
                    int(row["speculative_num_draft_tokens"]),
                )
                tps = float(row["avg_throughput"])
            except (KeyError, TypeError, ValueError):
                continue
            tps_by_cfg[cfg] = tps
    return tps_by_cfg


def load_detail_al(detail_csv: Path) -> dict[tuple[int, int, int], float]:
    al_buckets: dict[tuple[int, int, int], list[float]] = {}
    with detail_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {
            "algorithm",
            "speculative_num_steps",
            "speculative_eagle_topk",
            "speculative_num_draft_tokens",
            "accept_length",
        }
        if not required_cols.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV missing required columns: {required_cols}")

        for row in reader:
            if (row.get("status") or "").strip().lower() == "failed":
                continue
            if canonical_algorithm(row.get("algorithm", "")) != "PRISM":
                continue
            try:
                cfg = (
                    int(row["speculative_num_steps"]),
                    int(row["speculative_eagle_topk"]),
                    int(row["speculative_num_draft_tokens"]),
                )
                al = float(row["accept_length"])
            except (KeyError, TypeError, ValueError):
                continue
            al_buckets.setdefault(cfg, []).append(al)

    return {cfg: sum(vals) / len(vals) for cfg, vals in al_buckets.items() if vals}


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot tree/verify sweep (AL + TPS) from CSV.")
    parser.add_argument("--input-avg-csv", default="e2e_llama2_tree_verify_sweep_avg.csv")
    parser.add_argument("--input-detail-csv", default="e2e_llama2_tree_verify_sweep_detail.csv")
    parser.add_argument("--output", default="figure8.pdf")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    avg_csv = Path(args.input_avg_csv)
    if not avg_csv.is_absolute():
        avg_csv = script_dir / avg_csv
    detail_csv = Path(args.input_detail_csv)
    if not detail_csv.is_absolute():
        detail_csv = script_dir / detail_csv
    # Always save under script directory, independent of current cwd.
    output_pdf = script_dir / Path(args.output).name

    tps_by_cfg = load_avg_tps(avg_csv)
    al_by_cfg = load_detail_al(detail_csv)

    common_set = set(tps_by_cfg) & set(al_by_cfg)
    common_cfgs = [cfg for cfg in PREFERRED_CONFIG_ORDER if cfg in common_set]
    # Keep deterministic behavior for any extra configs not in preferred order.
    common_cfgs.extend(sorted(common_set - set(common_cfgs)))
    if not common_cfgs:
        raise ValueError("No shared PRISM configs found between avg/detail CSV.")

    x_labels = [f"{s},{k},{d}" for s, k, d in common_cfgs]
    acceptance_length_data = [al_by_cfg[cfg] for cfg in common_cfgs]
    tps_data = [tps_by_cfg[cfg] for cfg in common_cfgs]

    if len(acceptance_length_data) != len(x_labels) or len(tps_data) != len(x_labels):
        raise ValueError("Data length does not match x-label length.")

    fig, ax1 = plt.subplots(figsize=(8, 6))

    bar_color = "#82b0d2"
    line_color = "#fa7f6f"

    bar_width = 0.4
    x_positions = np.arange(len(x_labels))

    ax1.bar(x_positions, acceptance_length_data, width=bar_width, color=bar_color, alpha=0.7, label="AL")
    ax1.set_ylabel("Acceptance Length", color="black", fontsize=24)
    ax1.tick_params(axis="y", labelcolor="black", labelsize=22)
    ax1.set_ylim(min(acceptance_length_data) * 0.9, max(acceptance_length_data) * 1.1)

    ax2 = ax1.twinx()
    ax2.plot(x_positions, tps_data, color=line_color, marker="o", linestyle="--", linewidth=2, label="TPS")
    ax2.set_ylabel("TPS", color="black", fontsize=24)
    ax2.tick_params(axis="y", labelcolor="black", labelsize=22)
    ax2.set_ylim(min(tps_data) * 0.9, max(tps_data) * 1.1)

    ax1.set_xlabel("Configuration (d, w, v)", fontsize=24, labelpad=15)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=24)

    label_dx_points = 20
    label_transform = ScaledTranslation(label_dx_points / 72, 0, fig.dpi_scale_trans)
    for tick_label in ax1.get_xticklabels():
        tick_label.set_transform(tick_label.get_transform() + label_transform)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=23)

    fig.tight_layout()
    plt.savefig(output_pdf, dpi=300)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
