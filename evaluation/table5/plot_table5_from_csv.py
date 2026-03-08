#!/usr/bin/env python3
"""Render Table 5 style PDF from LLaMA-2/3 CSV results."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

DATASET_ORDER = ["mt_bench", "humaneval", "gsm8k", "alpaca", "sum", "qa"]
DATASET_LABELS = {
    "mt_bench": "MT-bench",
    "humaneval": "HumanEval",
    "gsm8k": "GSM8K",
    "alpaca": "Alpaca",
    "sum": "CNN/DM",
    "qa": "Natural Ques.",
}
METHOD_ORDER = ["NONE", "STANDALONE", "EAGLE", "HASS", "PRISM"]
METHOD_LABELS = {
    "NONE": "Vanilla",
    "STANDALONE": "Standard",
    "EAGLE": "EAGLE-2",
    "HASS": "HASS",
    "PRISM": "PRISM (ours)",
}
TEMP_ORDER = [0.0, 1.0]


def load_csv_rows(csv_path: Path) -> dict[tuple[str, str, float], tuple[float, float]]:
    out: dict[tuple[str, str, float], tuple[float, float]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"algorithm", "dataset", "temperature", "throughput", "accept_length", "status"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{csv_path} missing required columns: {required}")

        for row in reader:
            if (row.get("status") or "").strip().lower() != "ok":
                continue
            try:
                algo = (row["algorithm"] or "").strip().upper()
                dataset = (row["dataset"] or "").strip()
                temp = float(row["temperature"])
                tps = float(row["throughput"])
                al = float(row["accept_length"])
            except (KeyError, TypeError, ValueError):
                continue
            out[(algo, dataset, temp)] = (al, tps)
    return out


def format_num(x: float) -> str:
    return f"{x:.2f}"


def build_rows(model_name: str, values: dict[tuple[str, str, float], tuple[float, float]]) -> tuple[list[str], list[list[str]]]:
    headers = ["Model", "Method", "Temp"]
    for ds in DATASET_ORDER:
        ds_name = DATASET_LABELS[ds]
        headers.append(f"{ds_name}\nAL")
        headers.append(f"{ds_name}\nTPS")

    rows: list[list[str]] = []
    for method in METHOD_ORDER:
        for temp in TEMP_ORDER:
            temp_label = "T = 0" if temp == 0.0 else "T = 1"
            row = [model_name, METHOD_LABELS[method], temp_label]
            for ds in DATASET_ORDER:
                key = (method, ds, temp)
                if key not in values:
                    row.extend(["-", "-"])
                    continue
                al, tps = values[key]
                if method == "NONE":
                    row.extend(["N/A", format_num(tps)])
                else:
                    row.extend([format_num(al), format_num(tps)])
            rows.append(row)
    return headers, rows


def merge_vertical_cells(table, col: int, start_row: int, end_row: int) -> None:
    """Visually merge a vertical cell range [start_row, end_row] (inclusive)."""
    if end_row <= start_row:
        return
    for r in range(start_row, end_row + 1):
        cell = table[(r, col)]
        if r != start_row:
            cell.get_text().set_text("")
        if r == start_row:
            cell.visible_edges = "TLR"
        elif r == end_row:
            cell.visible_edges = "BLR"
        else:
            cell.visible_edges = "LR"


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw Table 5 PDF using matplotlib table.")
    parser.add_argument("--llama2-csv", default="e2e_llama2.csv")
    parser.add_argument("--llama3-csv", default="e2e_llama3.csv")
    parser.add_argument("--output", default="table5.pdf")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    llama2_csv = Path(args.llama2_csv)
    if not llama2_csv.is_absolute():
        llama2_csv = script_dir / llama2_csv
    llama3_csv = Path(args.llama3_csv)
    if not llama3_csv.is_absolute():
        llama3_csv = script_dir / llama3_csv
    output_pdf = script_dir / Path(args.output).name

    llama2_values = load_csv_rows(llama2_csv)
    llama3_values = load_csv_rows(llama3_csv)

    headers, llama2_rows = build_rows("LLaMA-2", llama2_values)
    _, llama3_rows = build_rows("LLaMA-3", llama3_values)
    all_rows = llama2_rows + llama3_rows

    fig, ax = plt.subplots(figsize=(24, 7))
    ax.axis("off")
    title = "Acceptance lengths (AL) and throughput (TPS, tokens per second) on NVIDIA A800 GPU"
    fig.text(0.5, 0.98, title, ha="center", va="top", fontsize=13)

    col_widths = [0.06, 0.08, 0.05] + [0.055] * (len(headers) - 3)
    table = ax.table(
        cellText=all_rows,
        colLabels=headers,
        colWidths=col_widths,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(0.95, 1.5)

    # Header styling.
    for c in range(len(headers)):
        cell = table[(0, c)]
        cell.set_facecolor("#e6e6e6")
        cell.get_text().set_fontweight("bold")

    # Shade and bold PRISM rows, and keep model column visually grouped.
    n_rows = len(all_rows)
    for r in range(1, n_rows + 1):
        method = all_rows[r - 1][1]
        model = all_rows[r - 1][0]
        if method == "PRISM (ours)":
            for c in range(len(headers)):
                cell = table[(r, c)]
                cell.get_text().set_fontweight("bold")
                cell.set_facecolor("#f2f7ff")
        if model == "LLaMA-3":
            table[(r, 0)].set_facecolor("#f7f7f7")

    # Visual merge for the "Model" column.
    merge_vertical_cells(table, col=0, start_row=1, end_row=10)
    merge_vertical_cells(table, col=0, start_row=11, end_row=20)

    # Visual merge for the "Method" column (each method has two temperature rows).
    for start in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]:
        merge_vertical_cells(table, col=1, start_row=start, end_row=start + 1)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_pdf, dpi=300)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
