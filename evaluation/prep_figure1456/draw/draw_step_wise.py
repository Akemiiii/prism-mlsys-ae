import re
import copy
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend for headless environments
from matplotlib import pyplot as plt

# =========================
# 1) Hardcoded fallback data
# =========================
config = {
    "EAGLE-2": {"color": "#1f77b4", "marker": "o"},
    "HASS": {"color": "#ff7f0e", "marker": "s"},
    "EAGLE-3": {"color": "#2ca02c", "marker": "^"},
    "PRISM": {"color": "#d62728", "marker": "D"},
}

xticks = ["1", "2", "3", "4"]

exps = {
    "MT-bench": {
        "EAGLE-2": np.array([
            [0.95168, 0.85820, 0.78740, 0.74881, 0.73402, 0.70774],
            [0.91741, 0.81940, 0.74222, 0.71887, 0.68023, 0.68187]
        ]),
        "HASS": np.array([
            [0.95035, 0.88873, 0.83105, 0.80298, 0.78187, 0.76980],
            [0.91983, 0.84843, 0.79176, 0.74740, 0.73715, 0.73363]
        ]),
        "EAGLE-3": np.array([
            [0.95365, 0.92527, 0.88856, 0.86207, 0.84251, 0.83703],
            [0.92429, 0.88602, 0.83245, 0.81271, 0.79941, 0.79377]
        ]),
        "PRISM": np.array([
            [0.97464, 0.93783, 0.89291, 0.85264, 0.83015, 0.81685],
            [0.93804, 0.88541, 0.83029, 0.80393, 0.78147, 0.78456]
        ])
    },
    "HumanEval": {
        "EAGLE-2": np.array([
            [0.99025, 0.93371, 0.87170, 0.85806, 0.82002, 0.77479],
            [0.97426, 0.91214, 0.85389, 0.82546, 0.80194, 0.77260]
        ]),
        "HASS": np.array([
            [0.98817, 0.96844, 0.91049, 0.91403, 0.89514, 0.86928],
            [0.97378, 0.94023, 0.90080, 0.89027, 0.86150, 0.83266]
        ]),
        "EAGLE-3": np.array([
            [0.97377, 0.96995, 0.95775, 0.94832, 0.93753, 0.92817],
            [0.97434, 0.95010, 0.93982, 0.92317, 0.91635, 0.91422]
        ]),
        "PRISM": np.array([
            [0.99646, 0.98659, 0.97361, 0.94949, 0.93901, 0.91893],
            [0.98723, 0.97203, 0.95719, 0.93198, 0.90968, 0.88830]
        ])
    },
    "GSM8K": {
        "EAGLE-2": np.array([
            [0.97174, 0.91753, 0.86075, 0.81445, 0.76932, 0.73289],
            [0.96259, 0.88544, 0.84213, 0.78235, 0.73758, 0.71474]
        ]),
        "HASS": np.array([
            [0.97097, 0.94545, 0.90513, 0.87158, 0.85049, 0.83631],
            [0.95853, 0.92410, 0.90185, 0.84699, 0.83280, 0.79987]
        ]),
        "EAGLE-3": np.array([
            [0.97911, 0.95690, 0.94631, 0.91971, 0.92473, 0.89429],
            [0.96768, 0.94592, 0.91971, 0.90402, 0.87816, 0.89177]
        ]),
        "PRISM": np.array([
            [0.99087, 0.97632, 0.95597, 0.91541, 0.91992, 0.88839],
            [0.97932, 0.96115, 0.93366, 0.90447, 0.89698, 0.87645]
        ])
    },
    "Alpaca": {
        "EAGLE-2": np.array([
            [0.95423, 0.86624, 0.78345, 0.71888, 0.67395, 0.66045],
            [0.93811, 0.83625, 0.75179, 0.71008, 0.66706, 0.60771]
        ]),
        "HASS": np.array([
            [0.95497, 0.89586, 0.82047, 0.77568, 0.74480, 0.72548],
            [0.93469, 0.85495, 0.79350, 0.74756, 0.71555, 0.71845]
        ]),
        "EAGLE-3": np.array([
            [0.96875, 0.94070, 0.89816, 0.85461, 0.83032, 0.82772],
            [0.94916, 0.90777, 0.86068, 0.83732, 0.79990, 0.79952]
        ]),
        "PRISM": np.array([
            [0.97727, 0.94521, 0.90630, 0.83525, 0.80232, 0.77303],
            [0.95685, 0.91132, 0.86018, 0.80888, 0.77422, 0.75154]
        ])
    },
    "CNN/DM": {
        "EAGLE-2": np.array([
            [0.95597, 0.85579, 0.76169, 0.69762, 0.62905, 0.59982],
            [0.93032, 0.81991, 0.71124, 0.65568, 0.62113, 0.55602]
        ]),
        "HASS": np.array([
            [0.95311, 0.89956, 0.80744, 0.76313, 0.74160, 0.71530],
            [0.93633, 0.85248, 0.75748, 0.71720, 0.72173, 0.67925]
        ]),
        "EAGLE-3": np.array([
            [0.94309, 0.88978, 0.85329, 0.82238, 0.78156, 0.77413],
            [0.92640, 0.85462, 0.80923, 0.76546, 0.75195, 0.71379]
        ]),
        "PRISM": np.array([
            [0.98191, 0.94373, 0.88982, 0.85462, 0.82256, 0.78484],
            [0.95637, 0.89720, 0.85323, 0.79706, 0.78514, 0.75769]
        ])
    },
    "Natural Ques.": {
        "EAGLE-2": np.array([
            [0.90445, 0.81951, 0.74575, 0.66649, 0.62609, 0.62643],
            [0.88670, 0.78172, 0.68846, 0.63184, 0.61450, 0.56691]
        ]),
        "HASS": np.array([
            [0.89935, 0.86010, 0.76948, 0.70877, 0.67526, 0.65649],
            [0.87069, 0.81699, 0.73769, 0.67727, 0.65493, 0.61051]
        ]),
        "EAGLE-3": np.array([
            [0.89080, 0.87317, 0.83841, 0.78252, 0.76985, 0.72378],
            [0.87089, 0.84822, 0.78319, 0.75452, 0.71197, 0.69337]
        ]),
        "PRISM": np.array([
            [0.92480, 0.90507, 0.84329, 0.76759, 0.72821, 0.70922],
            [0.89576, 0.86548, 0.79235, 0.72444, 0.69413, 0.68215]
        ])
    }
}

# =========================
# 2) CLI arguments
# =========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=str, default="./outputs")
    parser.add_argument("--project", type=str, default="Llama-3-8B")
    parser.add_argument(
        "--suffix",
        type=str,
        default="800k",
        help="Log filename suffix, e.g. 800k; set to '*' for wildcard matching"
    )
    parser.add_argument("--save-path", type=str, default="step-wise.pdf")
    return parser.parse_args()

# =========================
# 3) Log parsing setup
# =========================
TEMPS = [0.0, 1.0]
TEMP_TO_IDX = {0.0: 0, 1.0: 1}

BENCH_TO_LOGKEY = {
    "MT-bench": "mt_bench",
    "HumanEval": "humaneval",
    "GSM8K": "gsm8k",
    "Alpaca": "alpaca",
    "CNN/DM": "sum",
    "Natural Ques.": "qa",
}

MODEL_DIR_CANDIDATES = {
    "EAGLE-2": ["Eagle2", "EAGLE-2", "EAGLE2"],
    "HASS": ["HASS", "HASS-1"],
    "EAGLE-3": ["Eagle3", "EAGLE-3", "EAGLE3"],
    "PRISM": ["PRISM"],
}

FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

# Common patterns for array-style acceptance rate fields
DIST_PATTERNS = [
    re.compile(r"distribution acceptance rate[s]?\s*[:=]\s*\[([^\]]+)\]", re.IGNORECASE | re.DOTALL),
    re.compile(r"distribution_acceptance_rate[s]?\s*[:=]\s*\[([^\]]+)\]", re.IGNORECASE | re.DOTALL),
    re.compile(r"acceptance rate distribution[s]?\s*[:=]\s*\[([^\]]+)\]", re.IGNORECASE | re.DOTALL),
    re.compile(r"acceptance_rate_dist(?:ribution)?\s*[:=]\s*\[([^\]]+)\]", re.IGNORECASE | re.DOTALL),
    re.compile(r"step[_\-\s]*accept(?:ance)?[_\-\s]*rate[s]?\s*[:=]\s*\[([^\]]+)\]", re.IGNORECASE | re.DOTALL),
]

def parse_float_list(s: str):
    return [float(x) for x in FLOAT_RE.findall(s)]

def _parse_position_table(text: str):
    # Match blocks like:
    # Position  Acceptance Rate
    #   1       0.97464
    #   2       0.93783
    block_pat = re.compile(
        r"Position\s+Acceptance\s+Rate\s*\n(?P<body>(?:\s*\d+\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*\n)+)",
        re.IGNORECASE
    )
    blocks = list(block_pat.finditer(text))
    if not blocks:
        return None

    body = blocks[-1].group("body")
    row_pat = re.compile(r"^\s*(\d+)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$", re.MULTILINE)

    rows = []
    for m in row_pat.finditer(body):
        pos = int(m.group(1))
        val = float(m.group(2))
        rows.append((pos, val))

    if not rows:
        return None

    rows.sort(key=lambda x: x[0])
    vals = [v for p, v in rows if 1 <= p <= 6]
    return vals if len(vals) >= 6 else None

def _parse_from_cum_counter(text: str):
    # Parse line like: cum_counter = [8714, 8493, 7965, ...]
    m = re.findall(r"cum_counter\s*=\s*\[([^\]]+)\]", text, flags=re.IGNORECASE)
    if not m:
        return None

    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", m[-1])]
    # Need at least 7 numbers to derive 6 rates: rate_i = cum[i+1] / cum[i]
    if len(nums) < 7:
        return None

    rates = []
    for i in range(6):
        denom = nums[i]
        numer = nums[i + 1]
        if denom <= 0:
            return None
        rates.append(numer / denom)

    return rates

def extract_distribution_from_log(log_path: Path):
    text = log_path.read_text(errors="ignore")

    # 1) Parse explicit array-style formats
    for pat in DIST_PATTERNS:
        matches = pat.findall(text)
        if matches:
            nums = parse_float_list(matches[-1])
            if len(nums) >= 6:
                return nums[:6]

    # 2) Parse "Position Acceptance Rate" table format
    vals = _parse_position_table(text)
    if vals is not None:
        return vals[:6]

    # 3) Parse single-line fallback with related keywords
    candidate_lines = []
    for line in text.splitlines():
        low = line.lower()
        if ("distribution" in low and "acceptance" in low and "rate" in low):
            candidate_lines.append(line)

    if candidate_lines:
        nums = parse_float_list(candidate_lines[-1])
        if len(nums) >= 6:
            return nums[:6]

    # 4) Parse from cum_counter as a final fallback
    vals = _parse_from_cum_counter(text)
    if vals is not None:
        return vals[:6]

    return None

def find_model_logs_dir(outputs_root: Path, project: str, candidates):
    # Priority 1: outputs/<project>/<model>/latest/logs
    for c in candidates:
        d = outputs_root / project / c / "latest" / "logs"
        if d.is_dir():
            return d, c

    # Priority 2: outputs/<project>/<model>/*/logs, pick the latest by mtime
    for c in candidates:
        d = outputs_root / project / c
        if d.is_dir():
            all_logs = sorted(d.glob("*/logs"), key=lambda x: x.stat().st_mtime, reverse=True)
            if all_logs:
                return all_logs[0], c

    return None, None

def find_best_log(logs_dir: Path, model_dir_name: str, suffix: str, bench_key: str, temp: float):
    # Expected filename:
    # <MODEL>-<SUFFIX>__<BENCH>__temp-<TEMP>__gpu-<ID>.log
    if suffix == "*":
        pattern = f"{model_dir_name}-*__{bench_key}__temp-{temp:.1f}__gpu-*.log"
    else:
        pattern = f"{model_dir_name}-{suffix}__{bench_key}__temp-{temp:.1f}__gpu-*.log"

    candidates = list(logs_dir.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

def overlay_from_logs(outputs_root: Path, project: str, suffix: str, exps_default: dict):
    # Start from hardcoded data and overwrite with parsed log values when available
    exps_runtime = copy.deepcopy(exps_default)
    updated, missing = [], []

    for bench_display in exps_runtime.keys():
        bench_key = BENCH_TO_LOGKEY[bench_display]

        for model_label in exps_runtime[bench_display].keys():
            candidates = MODEL_DIR_CANDIDATES.get(model_label, [model_label])
            logs_dir, chosen_model_dir = find_model_logs_dir(outputs_root, project, candidates)

            if logs_dir is None:
                for t in TEMPS:
                    missing.append((project, bench_display, model_label, t, "model logs dir not found"))
                continue

            for t in TEMPS:
                t_idx = TEMP_TO_IDX[t]
                log_file = find_best_log(logs_dir, chosen_model_dir, suffix, bench_key, t)

                if log_file is None:
                    missing.append((project, bench_display, model_label, t, "log file missing"))
                    continue

                dist = extract_distribution_from_log(log_file)
                if dist is None:
                    missing.append((project, bench_display, model_label, t, f"distribution not found in {log_file.name}"))
                    continue

                old_vals = exps_runtime[bench_display][model_label][t_idx].copy()
                exps_runtime[bench_display][model_label][t_idx, :6] = np.array(dist[:6], dtype=float)
                updated.append((project, bench_display, model_label, t, old_vals, dist, log_file.name))

    return exps_runtime, updated, missing

# =========================
# 4) Main
# =========================
def main():
    args = parse_args()
    outputs_root = Path(args.outputs_root).resolve()

    exps_runtime, updated, missing = overlay_from_logs(
        outputs_root=outputs_root,
        project=args.project,
        suffix=args.suffix,
        exps_default=exps
    )

    print(f"[INFO] Number of points overwritten from logs: {len(updated)}")
    for row in updated[:20]:
        project, bench, model, t, old_vals, new_vals, fn = row
        print(f"[UPDATE] {project} | {bench} | {model} | t={t} | {fn}")
        print(f"         old={np.array2string(old_vals, precision=5)}")
        print(f"         new={np.array2string(np.array(new_vals), precision=5)}")
    if len(updated) > 20:
        print(f"[INFO] ... {len(updated) - 20} more updates omitted")

    print(f"[INFO] Number of missing points (fallback to hardcoded): {len(missing)}")
    for row in missing:
        project, bench, model, t, reason = row
        print(f"[MISSING] {project} | {bench} | {model} | t={t} | {reason}")

    # Compute mean curves by averaging over all benchmarks
    models = list(config.keys())
    means = {m: np.zeros((2, 6), dtype=float) for m in models}
    n_bench = len(exps_runtime)

    for results in exps_runtime.values():
        for model in models:
            means[model] += results[model] / float(n_bench)

    # Plot two subplots with independent y-axes to avoid shared-label side effects
    fig, ax = plt.subplots(1, 2, figsize=(12, 8), sharey=False)
    temperatures = [0, 1]
    titles = ["Temperature = 0", "Temperature = 1"]
    y_ticks = np.arange(0.7, 1.01, 0.05)

    lines, labels = [], []
    for i, temperature in enumerate(temperatures):
        for model in models:
            values = means[model]
            line = ax[i].plot(
                range(1, 5),
                values[temperature][:4],
                color=config[model]["color"],
                marker=config[model]["marker"],
                linestyle="-",
                markersize=10
            )
            if i == 0:
                lines.append(line[0])
                labels.append(model)

        ax[i].grid(True, linestyle="--", alpha=0.6)
        ax[i].set_xticks(range(1, 5))
        ax[i].set_xticklabels(xticks, fontsize=24)
        ax[i].set_xlabel("Position No.", fontsize=24)

        # Force identical y-scale and y-ticks on both panels
        ax[i].set_ylim(0.7, 1.0)
        ax[i].set_yticks(y_ticks)
        ax[i].set_title(titles[i], fontsize=24)

    # Left subplot: show y-axis labels
    ax[0].set_ylabel("Acceptance Rate", fontsize=24)
    ax[0].set_yticklabels([f"{y:.2f}" for y in y_ticks], fontsize=24)
    ax[0].tick_params(axis="y", which="both", left=True, labelleft=True, length=6, width=1.2)

    # Right subplot: keep y tick marks but hide y labels
    ax[1].tick_params(axis="y", which="both", left=True, labelleft=False, length=6, width=1.2)

    # Add a shared legend below subplots
    fig.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=4, fontsize=24)

    plt.tight_layout()
    save_path = Path(args.save_path)
    plt.savefig(save_path, bbox_inches="tight")
    print(f"[DONE] Figure saved to: {save_path.resolve()}")

if __name__ == "__main__":
    main()
