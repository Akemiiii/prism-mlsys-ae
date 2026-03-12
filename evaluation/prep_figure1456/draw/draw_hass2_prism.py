"""Plot mean comparison of HASS vs PRISM with log parsing support."""

import re
import copy
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend for headless servers
from matplotlib import pyplot as plt


# =========================
# 1) Hardcoded fallback data
# =========================
CONFIG = {
    "HASS": {"color": "#2ca02c", "marker": "s"},
    "PRISM": {"color": "#d62728", "marker": "D"},
}

XTICKS = ["100k", "200k", "400k", "600k", "800k"]

# Hardcoded mean acceptance lengths (averaged over 6 benchmarks).
# Used as fallback when log files are unavailable.
MEAN_FALLBACK = {
    "HASS": [
        [4.82782, 5.11311, 5.30529, 5.39198, 5.40402],
        [4.64747, 4.92876, 5.0943, 5.20222, 5.18965],
    ],
    "PRISM": [
        [5.15345, 5.33306, 5.46842, 5.52751, 5.58582],
        [4.94495, 5.13315, 5.275, 5.28516, 5.34297],
    ],
}


# =========================
# 2) CLI arguments
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot mean acceptance length comparison of HASS vs PRISM"
    )
    parser.add_argument(
        "--outputs-root", type=str, default="./outputs",
        help="Root directory of outputs, e.g. ./outputs"
    )
    parser.add_argument(
        "--project", type=str, default="Llama-3-8B",
        help="Project name under outputs root"
    )
    parser.add_argument(
        "--save-path", type=str, default="hass_vs_ld.pdf",
        help="Output figure path"
    )
    return parser.parse_args()


# =========================
# 3) Log parsing config
# =========================
SUFFIXES = ["100k", "200k", "400k", "600k", "800k"]
TEMPS = [0.0, 1.0]
TEMP_TO_IDX = {0.0: 0, 1.0: 1}
SUFFIX_TO_IDX = {s: i for i, s in enumerate(SUFFIXES)}

# Mapping from display benchmark names to log benchmark keys
BENCH_TO_LOGKEY = {
    "MT-bench": "mt_bench",
    "HumanEval": "humaneval",
    "GSM8K": "gsm8k",
    "Alpaca": "alpaca",
    "CNN/DM": "sum",
    "Natural Ques.": "qa",
}

# Candidate model directory names under outputs/<project>/
# Display label "HASS" maps to directory name "HASS-2" for log parsing
MODEL_DIR_CANDIDATES = {
    "HASS": ["HASS-2"],
    "PRISM": ["PRISM"],
}

# Regex for extracting average acceptance length
AVG_RE = re.compile(r"average acceptance length\s*=\s*([0-9]*\.?[0-9]+)")


def extract_avg_from_log(log_path: Path):
    """Read log file and extract the last matched average acceptance length."""
    text = log_path.read_text(errors="ignore")
    matches = AVG_RE.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def find_model_logs_dir(outputs_root: Path, project: str, candidates):
    """Locate the logs directory for a given model.

    Priority 1: outputs/<project>/<model>/latest/logs
    Priority 2: outputs/<project>/<model>/*/logs (latest by mtime)
    """
    for c in candidates:
        d = outputs_root / project / c / "latest" / "logs"
        if d.is_dir():
            return d, c

    for c in candidates:
        d = outputs_root / project / c
        if d.is_dir():
            all_logs = sorted(
                d.glob("*/logs"),
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            if all_logs:
                return all_logs[0], c

    return None, None


def find_best_log(logs_dir: Path, model_dir_name: str, suffix: str,
                  bench_key: str, temp: float):
    """Find the most recent matching log file.

    Expected filename format:
        <MODEL>-<SUFFIX>__<BENCH>__temp-<TEMP>__gpu-<ID>.log
    """
    pattern = f"{model_dir_name}-{suffix}__{bench_key}__temp-{temp:.1f}__gpu-*.log"
    candidates = list(logs_dir.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def compute_means_from_logs(outputs_root: Path, project: str):
    """Parse logs for all benchmarks and compute per-model mean acceptance lengths.

    For each (model, temperature, suffix) slot, the mean is computed over all
    benchmarks for which a valid log was found.  If no logs are found at all
    for a slot, the hardcoded fallback value is kept.

    Returns:
        means:   dict {model_label: [[t0 vals], [t1 vals]]}
        updated: list of successfully parsed entries
        missing: list of entries that could not be parsed
    """
    benchmarks = list(BENCH_TO_LOGKEY.keys())
    models = list(MODEL_DIR_CANDIDATES.keys())

    # Collect per-benchmark acceptance lengths
    # per_bench[model][temp_idx][suffix_idx] -> list of floats (one per benchmark)
    per_bench = {
        m: [[[] for _ in SUFFIXES] for _ in range(len(TEMPS))]
        for m in models
    }

    updated = []
    missing = []

    for bench_display in benchmarks:
        bench_key = BENCH_TO_LOGKEY[bench_display]

        for model_label in models:
            candidates = MODEL_DIR_CANDIDATES[model_label]
            logs_dir, chosen_model_dir = find_model_logs_dir(
                outputs_root, project, candidates
            )

            if logs_dir is None:
                for t in TEMPS:
                    for s in SUFFIXES:
                        missing.append((
                            project, bench_display, model_label, t, s,
                            "model logs dir not found"
                        ))
                continue

            for t in TEMPS:
                t_idx = TEMP_TO_IDX[t]
                for s in SUFFIXES:
                    s_idx = SUFFIX_TO_IDX[s]
                    log_file = find_best_log(
                        logs_dir, chosen_model_dir, s, bench_key, t
                    )

                    if log_file is None:
                        missing.append((
                            project, bench_display, model_label, t, s,
                            "log file missing"
                        ))
                        continue

                    avg = extract_avg_from_log(log_file)
                    if avg is None:
                        missing.append((
                            project, bench_display, model_label, t, s,
                            f"avg not found in {log_file.name}"
                        ))
                        continue

                    per_bench[model_label][t_idx][s_idx].append(avg)
                    updated.append((
                        project, bench_display, model_label, t, s,
                        avg, log_file.name
                    ))

    # Compute means from collected per-benchmark data
    means = copy.deepcopy(MEAN_FALLBACK)
    n_bench = len(benchmarks)

    for model_label in models:
        for t_idx in range(len(TEMPS)):
            for s_idx in range(len(SUFFIXES)):
                vals = per_bench[model_label][t_idx][s_idx]
                if len(vals) == n_bench:
                    # All benchmarks available from logs; use exact mean
                    means[model_label][t_idx][s_idx] = float(np.mean(vals))
                elif len(vals) > 0:
                    # Partial data; compute mean from available benchmarks
                    means[model_label][t_idx][s_idx] = float(np.mean(vals))
                # Otherwise keep the hardcoded fallback value

    return means, updated, missing


# =========================
# 4) Main: parse logs + plot
# =========================
def main():
    args = parse_args()
    outputs_root = Path(args.outputs_root).resolve()

    means, updated, missing = compute_means_from_logs(outputs_root, args.project)

    # Print update summary
    print(f"[INFO] Number of points parsed from logs: {len(updated)}")
    for row in updated[:20]:
        project, bench, model, t, s, avg, fn = row
        print(f"[UPDATE] {project} | {bench} | {model} | t={t} | {s}: {avg:.5f} ({fn})")
    if len(updated) > 20:
        print(f"[INFO] ... {len(updated) - 20} more updates omitted")

    print(f"[INFO] Number of missing points (fallback to hardcoded): {len(missing)}")
    for row in missing:
        project, bench, model, t, s, reason = row
        print(f"[MISSING] {project} | {bench} | {model} | t={t} | {s} | {reason}")

    # Build experiment data for plotting
    experiments = {"Mean": means}

    # Plot
    for benchmark, models in experiments.items():
        fig, ax = plt.subplots(figsize=(9, 7))

        values = []
        legends = []
        x_range = range(1, len(XTICKS) + 1)

        for model, series in models.items():
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
        ax.legend(legends, loc="lower right", fontsize=26)
        ax.set_xticks(x_range)
        ax.set_xticklabels(XTICKS, fontsize=26)

        y_min = round(min(values), 1)
        y_max = round(max(values), 1)
        ax.set_yticks(np.arange(y_min, y_max, 0.1))
        ax.tick_params(axis="y", labelsize=26)
        ax.set_ylabel("Acceptance Length", fontsize=28)
        ax.set_xlabel("Train Data Volume", fontsize=28)

        fig.tight_layout()

    save_path = Path(args.save_path)
    plt.savefig(save_path, bbox_inches="tight")
    print(f"[DONE] Figure saved to: {save_path.resolve()}")
    plt.close(fig)


if __name__ == "__main__":
    main()
