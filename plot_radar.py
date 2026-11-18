"""Generate BenchECG radar plots dynamically from `logs_lightning/` and optional paper baselines.

Examples:
  python plot_radar.py best_xlstm_lp --vs paper_xecg_lp
  python plot_radar.py my_model_lp best_xlstm_lp --vs paper_xecg_lp
  python plot_radar.py my_model_lp --out radar_mine.png
"""

import argparse
import glob
import math
import os
from pathlib import Path
import re
from typing import Optional

import matplotlib
if not os.environ.get("DISPLAY") and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ordered clockwise starting near top-right (matching the paper's radar layout)
TASK_SPECS = [
    ("CPSC2018", "train-cpsc2018-multilabel", "test_auroc"),
    ("PTB-XL", "train-ptbxl-diagnosis_superclass-multilabel", "test_auroc"),
    ("Survival", "train-survival", "test_ci"),
    ("Blood Test", "train-lab", "test_auroc_avg"),
    (
        "Age",
        "train-age",
        ["mimic/test_r2/dataloader_idx_1", "ptbxl/test_r2/dataloader_idx_0", "cpsc/test_r2/dataloader_idx_2"],
    ),
    ("R-peak", "train-mitbih-r_peaks", "test_f1_20"),
    ("Exercise\nR-Peak", "train-exercise-r-peak", "test_f1_20"),
    ("PPG AF", "train-deepbeat", "test_auroc"),
    ("Sleep Apnea", "train-sleep-apnea_bis", "test_auc"),
    ("MIT-BIH", "train-mitbih-5", "test_f1/mean"),
]

AGE_RSMAPE_COLS = [
    "mimic/test_rsmape/dataloader_idx_1",
    "ptbxl/test_rsmape/dataloader_idx_0",
    "cpsc/test_rsmape/dataloader_idx_2",
]

# Reference baselines from supplementary-information.pdf (Age = R^2)
PAPER_BASELINES: dict[str, tuple[str, dict[str, float]]] = {
    "paper_xecg_lp": (
        "Paper xECG LP",
        {
            "CPSC2018": 0.968, "PTB-XL": 0.915, "Survival": 0.704, "Blood Test": 0.733,
            "Age": 0.609, "R-peak": 0.591, "Exercise\nR-Peak": 0.892, "PPG AF": 0.751,
            "Sleep Apnea": 0.879, "MIT-BIH": 0.674,
        },
    ),
    "paper_xecg_ft": (
        "Paper xECG FT",
        {
            "CPSC2018": 0.981, "PTB-XL": 0.928, "Survival": 0.710, "Blood Test": 0.747,
            "Age": 0.636, "R-peak": 0.924, "Exercise\nR-Peak": 0.965, "PPG AF": 0.883,
            "Sleep Apnea": 0.928, "MIT-BIH": 0.677,
        },
    ),
    "paper_stmem_lp": (
        "Paper ST-MEM LP",
        {
            "CPSC2018": 0.922, "PTB-XL": 0.867, "Survival": 0.696, "Blood Test": 0.708,
            "Age": 0.410, "R-peak": 0.566, "Exercise\nR-Peak": 0.632, "PPG AF": 0.653,
            "Sleep Apnea": 0.599, "MIT-BIH": 0.436,
        },
    ),
    "paper_ecgjepa_lp": (
        "Paper ECG-JEPA LP",
        {
            "CPSC2018": 0.961, "PTB-XL": 0.902, "Survival": 0.706, "Blood Test": 0.722,
            "Age": 0.569, "R-peak": 0.444, "Exercise\nR-Peak": 0.526, "PPG AF": 0.477,
            "Sleep Apnea": 0.667, "MIT-BIH": 0.496,
        },
    ),
    "paper_ecgfounder_lp": (
        "Paper ECGFounder LP",
        {
            "CPSC2018": 0.962, "PTB-XL": 0.917, "Survival": 0.683, "Blood Test": 0.711,
            "Age": 0.487, "R-peak": 0.037, "Exercise\nR-Peak": 0.083, "PPG AF": 0.641,
            "Sleep Apnea": 0.777, "MIT-BIH": 0.206,
        },
    ),
    "paper_ecgcpc_lp": (
        "Paper ECG-CPC LP",
        {
            "CPSC2018": 0.951, "PTB-XL": 0.890, "Survival": 0.665, "Blood Test": 0.713,
            "Age": 0.245, "R-peak": 0.475, "Exercise\nR-Peak": 0.861, "PPG AF": 0.639,
            "Sleep Apnea": 0.734, "MIT-BIH": 0.381,
        },
    ),
}

PALETTE = [
    ("#2E9E5B", "o", "-"),
    ("#4E79A7", "s", "-"),
    ("#E15759", "^", "-"),
    ("#B07AA1", "P", "-"),
    ("#F28E2B", "D", "--"),
    ("#76B7B2", "v", "--"),
]


def get_sorted_versions(exp_path: Path) -> list[int]:
    versions = []
    regex = re.compile(r"version_(\d+)$")
    for p in glob.glob(str(exp_path / "version_*")):
        if os.path.isdir(p):
            m = regex.search(os.path.basename(p))
            if m:
                versions.append(int(m.group(1)))
    versions.sort(reverse=True)
    return versions


def extract_score(csv_path: Path, metric_cfg: str | list[str], invert: bool = False) -> Optional[float]:
    if not csv_path.is_file():
        return None
    try:
        df = pd.read_csv(csv_path)
        if isinstance(metric_cfg, list):
            if any(c not in df.columns for c in metric_cfg):
                return None
            s = df[metric_cfg].dropna(how="all").mean(axis=1)
            if s.empty:
                return None
            val = float(s.iloc[-1])
        else:
            if metric_cfg not in df.columns:
                return None
            s = df[metric_cfg].dropna()
            if s.empty:
                return None
            val = float(s.iloc[-1])
        return (1.0 - val) if invert else val
    except Exception:
        return None


def load_experiment_scores(
    logs_dir: Path, experiment: str, max_versions: int, age_metric: str
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for label, task_folder, metric_cfg in TASK_SPECS:
        invert = False
        if label == "Age" and age_metric == "rsmape":
            metric_cfg = AGE_RSMAPE_COLS
            invert = True

        exp_path = logs_dir / task_folder / experiment
        versions = get_sorted_versions(exp_path)
        vals = []
        for v in versions:
            val = extract_score(exp_path / f"version_{v}" / "metrics.csv", metric_cfg, invert=invert)
            if val is not None:
                vals.append(val)
                if len(vals) >= max_versions:
                    break
        if vals:
            mean_v = float(np.mean(vals))
            if label == "Age" and age_metric == "r2":
                mean_v = max(0.0, mean_v)
            scores[label] = mean_v
    return scores


def plot_radar(series: list[tuple[str, dict[str, float]]], out_path: Path, title: str, include_missing: bool) -> None:
    all_labels = [t[0] for t in TASK_SPECS]
    if include_missing:
        labels = [l for l in all_labels if any(l in s for _, s in series)]
    else:
        labels = [l for l in all_labels if all(l in s for _, s in series)]

    if len(labels) < 3:
        raise ValueError(f"Need at least 3 tasks in common to plot radar (found {len(labels)}: {labels})")

    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7.6, 6.6), facecolor="white")
    ax = fig.add_subplot(111, polar=True, facecolor="#EBF0F7")

    ax.set_theta_offset(math.radians(70))
    ax.set_theta_direction(-1)

    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1"], fontsize=9, color="#1F2937")
    ax.set_rlabel_position(340)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11, color="#111827")
    ax.tick_params(axis="x", pad=12)

    ax.xaxis.grid(True, color="white", linewidth=1.2)
    ax.yaxis.grid(True, color="white", linewidth=1.0)
    ax.spines["polar"].set_visible(False)

    for idx, (name, sc) in enumerate(series):
        vals = [sc.get(l, 0.0) for l in labels]
        mean_val = float(np.mean([sc[l] for l in labels if l in sc]))
        vals_closed = vals + vals[:1]
        color, marker, ls = PALETTE[idx % len(PALETTE)]
        ax.plot(
            angles, vals_closed,
            color=color, linewidth=2.2, linestyle=ls, marker=marker, markersize=6.0,
            label=f"{name} ({mean_val:.3f})",
        )
        ax.fill(angles, vals_closed, color=color, alpha=0.22 if len(series) > 2 else 0.30)

    ax.set_title(title, fontsize=13.5, fontweight="bold", color="#1E3A5F", pad=36)
    ax.legend(
        loc="upper right", bbox_to_anchor=(1.30, 1.00),
        frameon=True, facecolor="white", edgecolor="#D1D5DB", fontsize=9.5,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot BenchECG radar chart comparing experiments and paper baselines.")
    parser.add_argument("experiments", nargs="*", help="Experiment folder names under logs_lightning/<task>/<exp>")
    parser.add_argument(
        "--vs", nargs="*", default=[], choices=list(PAPER_BASELINES.keys()),
        help=f"Paper baselines to overlay ({', '.join(PAPER_BASELINES.keys())})",
    )
    parser.add_argument("--logs-dir", type=Path, default=Path("logs_lightning"), help="Path to logs_lightning")
    parser.add_argument("--max-versions", type=int, default=5, help="Max recent versions per task (default: 5)")
    parser.add_argument("--age-metric", choices=["r2", "rsmape"], default="r2", help="Age metric (default: r2)")
    parser.add_argument("--include-missing", action="store_true", help="Include tasks missing in some series (as 0)")
    parser.add_argument("--title", default="BenchECG Radar Comparison", help="Chart title")
    parser.add_argument("--out", type=Path, default=Path("radar_comparison.png"), help="Output PNG path")
    args = parser.parse_args()

    series: list[tuple[str, dict[str, float]]] = []
    for exp in args.experiments:
        sc = load_experiment_scores(args.logs_dir, exp, args.max_versions, args.age_metric)
        if not sc:
            print(f"[WARN] No scores found for experiment '{exp}' in {args.logs_dir}")
            continue
        series.append((exp, sc))
        print(f"{exp}: " + ", ".join(f"{k.replace(chr(10), ' ')}={v:.4f}" for k, v in sc.items()))

    for base_key in args.vs:
        label, sc = PAPER_BASELINES[base_key]
        series.append((label, sc))

    if not series:
        parser.error("Provide at least one experiment name or --vs baseline.")

    plot_radar(series, args.out, args.title, args.include_missing)
    print(f"Saved radar plot to {args.out}")


if __name__ == "__main__":
    main()
