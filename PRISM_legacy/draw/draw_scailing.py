import re
import copy
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend for headless servers

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

# =========================
# 1) Hardcoded fallback data
# =========================
config_1 = {
    'EAGLE-2': {'color': '#1f77b4', 'marker': 'o'},
    'HASS-1': {'color': '#ff7f0e', 'marker': 's'},
    'PRISM': {'color': '#d62728', 'marker': 'D'}
}
exps_1 = {
    'MT-bench': {
        'EAGLE-2': [
            [4.73315, 4.89139, 4.98708, 5.04436, 5.02846],
            [4.51329, 4.70636, 4.77445, 4.84118, 4.82961]
        ],
        'HASS-1': [
            [5.09173, 5.23063, 5.3507, 5.33786, 5.34127],
            [4.86165, 4.94343, 5.04385, 5.06115, 5.09524]
        ],
        'PRISM': [
            [5.2074, 5.38151, 5.50813, 5.56517, 5.58304],
            [4.94805, 5.11929, 5.22097, 5.2853, 5.3318]
        ]
    },
    'HumanEval': {
        'EAGLE-2': [
            [5.35449, 5.53025, 5.61162, 5.64755, 5.62172],
            [5.07249, 5.25476, 5.24275, 5.29181, 5.30573]
        ],
        'HASS-1': [
            [5.78596, 5.86957, 5.9641, 5.99358, 5.99337],
            [5.37265, 5.587, 5.66884, 5.64579, 5.58696]
        ],
        'PRISM': [
            [5.86056, 6.00311, 6.09059, 6.15714, 6.17389],
            [5.45157, 5.70982, 5.81977, 5.78003, 5.86214]
        ]
    },
    'GSM8K': {
        'EAGLE-2': [
            [4.84016, 4.99998, 5.04102, 5.0628, 5.1267],
            [4.87757, 4.87929, 5.05543, 5.11579, 5.16248]
        ],
        'HASS-1': [
            [5.22653, 5.30258, 5.39044, 5.42064, 5.41632],
            [5.17258, 5.32093, 5.24608, 5.38297, 5.31655]
        ],
        'PRISM': [
            [5.31914, 5.45325, 5.59527, 5.58577, 5.69556],
            [5.20347, 5.41386, 5.50501, 5.5652, 5.52479]
        ]
    },
    'Alpaca': {
        'EAGLE-2': [
            [4.62961, 4.80094, 4.94934, 4.93946, 4.96793],
            [4.54291, 4.54484, 4.75235, 4.76099, 4.81447]
        ],
        'HASS-1': [
            [4.96695, 5.11803, 5.20702, 5.24013, 5.26022],
            [4.77251, 4.94477, 5.03474, 5.18521, 5.14378]
        ],
        'PRISM': [
            [5.082, 5.31743, 5.45703, 5.51971, 5.57422],
            [4.95387, 5.06893, 5.36129, 5.28095, 5.33552]
        ]
    },
    'CNN/DM': {
        'EAGLE-2': [
            [4.31285, 4.4856, 4.57852, 4.61813, 4.63727],
            [4.13297, 4.29954, 4.36809, 4.45527, 4.40932]
        ],
        'HASS-1': [
            [4.67958, 4.85362, 4.94466, 4.95558, 4.98692],
            [4.44409, 4.61786, 4.67893, 4.76209, 4.72477]
        ],
        'PRISM': [
            [4.80826, 4.97504, 5.13965, 5.2211, 5.2871],
            [4.61597, 4.76905, 4.92235, 4.9378, 4.98713]
        ]
    },
    'Natural Ques.': {
        'EAGLE-2': [
            [4.25697, 4.4412, 4.54477, 4.56902, 4.57432],
            [4.11488, 4.23569, 4.39021, 4.40386, 4.43694]
        ],
        'HASS-1': [
            [4.54263, 4.74907, 4.84555, 4.87466, 4.91995],
            [4.45015, 4.55391, 4.69417, 4.74354, 4.7457]
        ],
        'PRISM': [
            [4.64334, 4.86803, 5.01984, 5.11617, 5.20111],
            [4.49674, 4.71792, 4.82061, 4.86167, 5.01646]
        ]
    }
}

config_2 = {
    'EAGLE-2': {'color': '#1f77b4', 'marker': 'o'},
    'HASS': {'color': '#ff7f0e', 'marker': 's'},
    'PRISM': {'color': '#d62728', 'marker': 'D'}
}
exps_2 = {
    'MT-bench': {
        'EAGLE-2': [
            [4.30874, 4.41915, 4.54165, 4.57129, 4.60145],
            [3.94275, 4.11398, 4.15711, 4.18139, 4.25300]
        ],
        'HASS': [
            [4.65840, 4.78146, 4.88533, 4.92224, 4.94916],
            [4.32679, 4.39275, 4.47149, 4.58964, 4.53607]
        ],
        'PRISM': [
            [4.86140, 5.03638, 5.17766, 5.28763, 5.28886],
            [4.39011, 4.55351, 4.74018, 4.81152, 4.74776]
        ]
    },
    'HumanEval': {
        'EAGLE-2': [
            [5.05997, 5.18363, 5.30485, 5.33594, 5.37071],
            [4.75526, 5.01080, 5.07591, 5.16940, 5.15949]
        ],
        'HASS': [
            [5.62106, 5.78474, 5.84373, 5.90184, 5.89876],
            [5.35828, 5.45787, 5.59145, 5.55664, 5.59962]
        ],
        'PRISM': [
            [5.88361, 6.08987, 6.18428, 6.25062, 6.27796],
            [5.60824, 5.74910, 5.87758, 5.93852, 6.04268]
        ]
    },
    'GSM8K': {
        'EAGLE-2': [
            [4.82814, 4.87006, 5.04712, 5.09010, 5.14008],
            [4.68040, 4.73418, 4.86839, 4.89488, 4.85073]
        ],
        'HASS': [
            [5.34545, 5.43475, 5.52197, 5.57795, 5.62539],
            [5.14849, 5.21478, 5.31959, 5.38968, 5.40697]
        ],
        'PRISM': [
            [5.56964, 5.70499, 5.88651, 5.94577, 6.04517],
            [5.32615, 5.48946, 5.68885, 5.69401, 5.76251]
        ]
    },
    'Alpaca': {
        'EAGLE-2': [
            [4.36539, 4.59282, 4.67625, 4.73934, 4.65080],
            [4.16385, 4.36017, 4.40548, 4.46680, 4.39822]
        ],
        'HASS': [
            [4.80779, 4.98708, 5.13525, 5.09008, 5.09730],
            [4.46411, 4.71002, 4.72297, 4.87121, 4.74604]
        ],
        'PRISM': [
            [5.09857, 5.32155, 5.48346, 5.54811, 5.58146],
            [4.66989, 4.95478, 5.14559, 5.04535, 5.20449]
        ]
    },
    'CNN/DM': {
        'EAGLE-2': [
            [3.99287, 4.15549, 4.22826, 4.30444, 4.21119],
            [3.71950, 3.86962, 3.93289, 4.00122, 3.91945]
        ],
        'HASS': [
            [4.36982, 4.53280, 4.67901, 4.67512, 4.64654],
            [4.02909, 4.21652, 4.27908, 4.33923, 4.26335]
        ],
        'PRISM': [
            [4.50440, 4.78492, 4.93750, 5.01652, 5.01489],
            [4.14324, 4.31155, 4.44956, 4.57951, 4.61742]
        ]
    },
    'Natural Ques.': {
        'EAGLE-2': [
            [3.62111, 3.75216, 3.84230, 3.88938, 3.87716],
            [3.37224, 3.56317, 3.68102, 3.68866, 3.67726]
        ],
        'HASS': [
            [3.86946, 4.07623, 4.12755, 4.16449, 4.12784],
            [3.65147, 3.66539, 3.86571, 3.87656, 3.84909]
        ],
        'PRISM': [
            [4.04164, 4.28424, 4.39427, 4.54650, 4.53064],
            [3.75955, 4.00731, 4.04033, 4.14121, 4.12963]
        ]
    }
}

# =========================
# 2) CLI arguments
# =========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outputs-root",
        type=str,
        default="./outputs",
        help="Root directory of outputs, e.g. ./outputs"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="scaling_from_logs.pdf",
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

# Panel-level settings
PANEL_SPECS = [
    {
        "title": "LLaMA-2-7B",
        "project": "llama-2-7b",
        "config": config_1,
        "exps_default": exps_1,
        # Candidate model directory names under outputs/<project>/
        "model_dir_candidates": {
            "EAGLE-2": ["Eagle2", "EAGLE-2", "EAGLE2"],
            "HASS-1": ["HASS-1", "HASS"],
            "PRISM": ["PRISM"],
        },
    },
    {
        "title": "LLaMA-3-8B",
        "project": "llama-3-8b",
        "config": config_2,
        "exps_default": exps_2,
        "model_dir_candidates": {
            "EAGLE-2": ["Eagle2", "EAGLE-2", "EAGLE2"],
            "HASS": ["HASS", "HASS-1"],
            "PRISM": ["PRISM"],
        },
    },
]

# Regex for extracting average acceptance length
AVG_RE = re.compile(r"average acceptance length\s*=\s*([0-9]*\.?[0-9]+)")

def extract_avg_from_log(log_path: Path):
    # Read log content and extract the last matched average value
    text = log_path.read_text(errors="ignore")
    matches = AVG_RE.findall(text)
    if not matches:
        return None
    return float(matches[-1])

def find_model_logs_dir(outputs_root: Path, project: str, candidates):
    # Priority 1: outputs/<project>/<model>/latest/logs
    for c in candidates:
        d = outputs_root / project / c / "latest" / "logs"
        if d.is_dir():
            return d, c

    # Priority 2: outputs/<project>/<model>/*/logs (latest by mtime)
    for c in candidates:
        d = outputs_root / project / c
        if d.is_dir():
            all_logs = sorted(d.glob("*/logs"), key=lambda x: x.stat().st_mtime, reverse=True)
            if all_logs:
                return all_logs[0], c

    return None, None

def find_best_log(logs_dir: Path, model_dir_name: str, suffix: str, bench_key: str, temp: float):
    # Expected filename format:
    # <MODEL>-<SUFFIX>__<BENCH>__temp-<TEMP>__gpu-<ID>.log
    pattern = f"{model_dir_name}-{suffix}__{bench_key}__temp-{temp:.1f}__gpu-*.log"
    candidates = list(logs_dir.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

def overlay_from_logs(outputs_root: Path, panel_spec: dict):
    # Start from hardcoded data, then overwrite with parsed log values
    exps_runtime = copy.deepcopy(panel_spec["exps_default"])
    missing = []
    updated = []

    project = panel_spec["project"]
    for bench_display in exps_runtime.keys():
        bench_key = BENCH_TO_LOGKEY[bench_display]

        for model_label in exps_runtime[bench_display].keys():
            candidates = panel_spec["model_dir_candidates"].get(model_label, [model_label])
            logs_dir, chosen_model_dir = find_model_logs_dir(outputs_root, project, candidates)

            if logs_dir is None:
                # Missing entire model log directory; keep hardcoded values
                for t in TEMPS:
                    for s in SUFFIXES:
                        missing.append((project, bench_display, model_label, t, s, "model logs dir not found"))
                continue

            for t in TEMPS:
                t_idx = TEMP_TO_IDX[t]
                for s in SUFFIXES:
                    s_idx = SUFFIX_TO_IDX[s]
                    log_file = find_best_log(logs_dir, chosen_model_dir, s, bench_key, t)

                    if log_file is None:
                        missing.append((project, bench_display, model_label, t, s, "log file missing"))
                        continue

                    avg = extract_avg_from_log(log_file)
                    if avg is None:
                        missing.append((project, bench_display, model_label, t, s, f"avg not found in {log_file.name}"))
                        continue

                    old_val = exps_runtime[bench_display][model_label][t_idx][s_idx]
                    exps_runtime[bench_display][model_label][t_idx][s_idx] = avg
                    updated.append((project, bench_display, model_label, t, s, old_val, avg, log_file.name))

    return exps_runtime, updated, missing

# =========================
# 4) Main: parse logs + plot
# =========================
def main():
    args = parse_args()
    outputs_root = Path(args.outputs_root).resolve()

    all_exps_runtime = []
    all_updated = []
    all_missing = []

    # Parse logs and overwrite fallback values panel by panel
    for spec in PANEL_SPECS:
        exps_runtime, updated, missing = overlay_from_logs(outputs_root, spec)
        all_exps_runtime.append(exps_runtime)
        all_updated.extend(updated)
        all_missing.extend(missing)

    # Print update summary
    print(f"[INFO] Number of points overwritten from logs: {len(all_updated)}")
    for row in all_updated[:20]:
        project, bench, model, t, s, old_v, new_v, fn = row
        print(f"[UPDATE] {project} | {bench} | {model} | t={t} | {s}: {old_v:.5f} -> {new_v:.5f} ({fn})")
    if len(all_updated) > 20:
        print(f"[INFO] ... {len(all_updated) - 20} more updates omitted")

    # Print missing summary (hardcoded values are kept)
    print(f"[INFO] Number of missing points (fallback to hardcoded values): {len(all_missing)}")
    for row in all_missing:
        project, bench, model, t, s, reason = row
        print(f"[MISSING] {project} | {bench} | {model} | t={t} | {s} | {reason}")

    # Plot settings
    xticks = ['1', '2', '4', '6', '8']
    benchmarks = list(exps_1.keys())
    x_axis_range = range(1, 6)

    fig = plt.figure(figsize=(15, 15), constrained_layout=True)
    subfigs = fig.subfigures(1, 2, wspace=0.07)

    subfigs[0].suptitle(PANEL_SPECS[0]["title"], fontsize=26)
    subfigs[1].suptitle(PANEL_SPECS[1]["title"], fontsize=26)

    axs_left = subfigs[0].subplots(3, 2)
    axs_right = subfigs[1].subplots(3, 2)

    all_configs = [PANEL_SPECS[0]["config"], PANEL_SPECS[1]["config"]]
    all_axs_grids = [axs_left, axs_right]

    # Draw all subplots
    for i, benchmark in enumerate(benchmarks):
        row, col = divmod(i, 2)
        for j in range(2):
            ax = all_axs_grids[j][row, col]
            config = all_configs[j]
            exp_data = all_exps_runtime[j][benchmark]

            for model, values in exp_data.items():
                ax.plot(
                    x_axis_range, values[0],
                    color=config[model]['color'],
                    marker=config[model]['marker'],
                    linestyle='-',
                    linewidth=2.5,
                    markersize=8
                )
                ax.plot(
                    x_axis_range, values[1],
                    color=config[model]['color'],
                    marker=config[model]['marker'],
                    linestyle=':',
                    linewidth=2.5,
                    markersize=8
                )

            ax.set_title(benchmark, fontsize=24)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_xticks(x_axis_range)
            ax.set_xticklabels(xticks, fontsize=24)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune='both'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.tick_params(axis='y', labelsize=24)

    # Shared axis labels
    fig.supylabel('Acceptance Length', fontsize=24)
    fig.supxlabel('Train Data Volume (x 10^2 K)', fontsize=24)

    # Unified legend
    legend_info = {
        'EAGLE-2': {'color': '#1f77b4', 'marker': 'o'},
        'HASS': {'color': '#ff7f0e', 'marker': 's'},
        'PRISM': {'color': '#d62728', 'marker': 'D'},
    }
    handles = []
    for model_label, style in legend_info.items():
        handles.append(Line2D(
            [0], [0],
            color=style['color'],
            marker=style['marker'],
            linestyle='-',
            linewidth=2.5,
            markersize=8,
            label=f'{model_label} (t=0)'
        ))
        handles.append(Line2D(
            [0], [0],
            color=style['color'],
            marker=style['marker'],
            linestyle=':',
            linewidth=2.5,
            markersize=8,
            label=f'{model_label} (t=1)'
        ))

    fig.legend(
        handles=handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.09),
        bbox_transform=fig.transFigure,
        ncol=3,
        fontsize=24,
        frameon=False
    )

    save_path = Path(args.save_path)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[DONE] Figure saved to: {save_path.resolve()}")
    plt.close(fig)

if __name__ == "__main__":
    main()
