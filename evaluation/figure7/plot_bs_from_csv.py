#!/usr/bin/env python3
"""Plot batch-size acceleration bars from sweep CSV."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def canonical_algorithm(name: str) -> str:
    key = (name or "").strip().upper()
    if key in {"PRISM", "LD"}:
        return "PRISM"
    if key in {"EAGLE", "EAGLE2"}:
        return "EAGLE"
    if key == "NONE":
        return "NONE"
    return key


def load_acceleration_from_csv(csv_path: Path):
    throughput_by_algo_bs: dict[str, dict[int, float]] = {"NONE": {}, "EAGLE": {}, "PRISM": {}}

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"algorithm", "max_running_requests", "avg_throughput"}
        if not required_cols.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV missing required columns: {required_cols}")

        for row in reader:
            algo = canonical_algorithm(row.get("algorithm", ""))
            if algo not in throughput_by_algo_bs:
                continue
            try:
                bs = int(row["max_running_requests"])
                throughput = float(row["avg_throughput"])
            except (KeyError, TypeError, ValueError):
                continue
            throughput_by_algo_bs[algo][bs] = throughput

    available_bs = sorted(
        set(throughput_by_algo_bs["NONE"])
        & set(throughput_by_algo_bs["EAGLE"])
        & set(throughput_by_algo_bs["PRISM"])
    )
    if not available_bs:
        raise ValueError("CSV does not contain shared batch sizes for NONE/EAGLE/PRISM.")

    prism_data = []
    eagle_data = []
    for bs in available_bs:
        none_tp = throughput_by_algo_bs["NONE"][bs]
        if none_tp <= 0:
            raise ValueError(f"NONE throughput must be > 0 for batch size {bs}, got {none_tp}.")
        prism_data.append(throughput_by_algo_bs["PRISM"][bs] / none_tp)
        eagle_data.append(throughput_by_algo_bs["EAGLE"][bs] / none_tp)

    return available_bs, prism_data, eagle_data


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot PRISM/EAGLE acceleration ratio vs batch size.")
    parser.add_argument("--input-csv", default="e2e_llama2_bs_sweep_avg.csv")
    parser.add_argument("--output", default="figure7.pdf")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    input_csv = Path(args.input_csv)
    if not input_csv.is_absolute():
        input_csv = script_dir / input_csv
    # Always save figure under this script's directory, regardless of current cwd.
    output_pdf = script_dir / Path(args.output).name

    batch_sizes, prism_data, eagle_data = load_acceleration_from_csv(input_csv)

    if len(prism_data) != len(batch_sizes) or len(eagle_data) != len(batch_sizes):
        raise ValueError("Data length does not match batch size length.")

    fig, ax = plt.subplots(figsize=(7, 5))

    prism_color = "#82b0d2"
    eagle2_color = "#fa7f6f"

    x_positions = np.arange(len(batch_sizes))
    bar_width = 0.35

    ax.bar(
        x_positions - bar_width / 2,
        prism_data,
        width=bar_width,
        color=prism_color,
        alpha=0.8,
        label="PRISM",
    )
    ax.bar(
        x_positions + bar_width / 2,
        eagle_data,
        width=bar_width,
        color=eagle2_color,
        alpha=0.8,
        label="EAGLE2",
    )

    ax.set_ylabel("Acceleration Ratio", fontsize=24)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    min_val = min(min(prism_data), min(eagle_data))
    max_val = max(max(prism_data), max(eagle_data))
    padding = (max_val - min_val) * 0.1
    ax.set_ylim(0.5, max_val + padding)

    ax.axhline(y=1.0, color="#4c4c4c", linestyle="--", linewidth=1.5, label="Vanilla")
    ax.tick_params(axis="y", labelsize=24)

    ax.set_xlabel("Batch Size", fontsize=24, labelpad=15)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(batch_sizes, fontsize=24)
    ax.legend(loc="upper right", fontsize=18)

    fig.tight_layout()
    plt.savefig(output_pdf, dpi=300)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
