"""Generate BenchECG radar plots and side-by-side comparison tables from `logs_lightning/` and paper baselines.

Examples:
  python scripts/plot_radar.py best_xlstm_lp --vs paper_xecg_lp
  python scripts/plot_radar.py best_xlstm_lp --vs all
  python scripts/plot_radar.py my_model_lp best_xlstm_lp --labels "My Model" "xLSTM LP" --vs paper_xecg_lp
  python scripts/plot_radar.py best_xlstm_ft --vs all_ft --out figs/radar_ft.png
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

# Reference baselines from results/scores_linear_probing.csv & results/scores_finetuning.csv (Age = R^2)
PAPER_BASELINES: dict[str, tuple[str, dict[str, float]]] = {
    "paper_xecg_lp": (
        "Paper xECG LP",
        {
            "CPSC2018": 0.9678, "PTB-XL": 0.9154, "Survival": 0.7038, "Blood Test": 0.7325,
            "Age": 0.6085, "R-peak": 0.5907, "Exercise\nR-Peak": 0.8921, "PPG AF": 0.7508,
            "Sleep Apnea": 0.8791, "MIT-BIH": 0.6740,
        },
    ),
    "paper_xecg_ft": (
        "Paper xECG FT",
        {
            "CPSC2018": 0.9813, "PTB-XL": 0.9276, "Survival": 0.7101, "Blood Test": 0.7475,
            "Age": 0.6364, "R-peak": 0.9235, "Exercise\nR-Peak": 0.9651, "PPG AF": 0.8835,
            "Sleep Apnea": 0.9278, "MIT-BIH": 0.6773,
        },
    ),
    "paper_stmem_lp": (
        "Paper ST-MEM LP",
        {
            "CPSC2018": 0.9221, "PTB-XL": 0.8670, "Survival": 0.6961, "Blood Test": 0.7082,
            "Age": 0.4105, "R-peak": 0.5656, "Exercise\nR-Peak": 0.6321, "PPG AF": 0.6532,
            "Sleep Apnea": 0.5987, "MIT-BIH": 0.4362,
        },
    ),
    "paper_stmem_ft": (
        "Paper ST-MEM FT",
        {
            "CPSC2018": 0.9582, "PTB-XL": 0.9299, "Survival": 0.6932, "Blood Test": 0.7444,
            "Age": 0.6914, "R-peak": 0.9382, "Exercise\nR-Peak": 0.9838, "PPG AF": 0.6427,
            "Sleep Apnea": 0.7003, "MIT-BIH": 0.6442,
        },
    ),
    "paper_ecgjepa_lp": (
        "Paper ECG-JEPA LP",
        {
            "CPSC2018": 0.9610, "PTB-XL": 0.9022, "Survival": 0.7055, "Blood Test": 0.7218,
            "Age": 0.5693, "R-peak": 0.4443, "Exercise\nR-Peak": 0.5262, "PPG AF": 0.4769,
            "Sleep Apnea": 0.6669, "MIT-BIH": 0.4962,
        },
    ),
    "paper_ecgjepa_ft": (
        "Paper ECG-JEPA FT",
        {
            "CPSC2018": 0.9652, "PTB-XL": 0.9233, "Survival": 0.7049, "Blood Test": 0.7351,
            "Age": 0.6347, "R-peak": 0.9103, "Exercise\nR-Peak": 0.8660, "PPG AF": 0.8184,
            "Sleep Apnea": 0.6904, "MIT-BIH": 0.5831,
        },
    ),
    "paper_ecgfounder_lp": (
        "Paper ECGFounder LP",
        {
            "CPSC2018": 0.9617, "PTB-XL": 0.9173, "Survival": 0.6830, "Blood Test": 0.7113,
            "Age": 0.4874, "R-peak": 0.0366, "Exercise\nR-Peak": 0.0832, "PPG AF": 0.6407,
            "Sleep Apnea": 0.7769, "MIT-BIH": 0.2065,
        },
    ),
    "paper_ecgfounder_ft": (
        "Paper ECGFounder FT",
        {
            "CPSC2018": 0.9694, "PTB-XL": 0.9304, "Survival": 0.6877, "Blood Test": 0.7145,
            "Age": 0.6305, "R-peak": 0.0367, "Exercise\nR-Peak": 0.0818, "PPG AF": 0.8456,
            "Sleep Apnea": 0.7784, "MIT-BIH": 0.2072,
        },
    ),
    "paper_ecgcpc_lp": (
        "Paper ECG-CPC LP",
        {
            "CPSC2018": 0.9512, "PTB-XL": 0.8896, "Survival": 0.6647, "Blood Test": 0.7126,
            "Age": 0.2448, "R-peak": 0.4754, "Exercise\nR-Peak": 0.8607, "PPG AF": 0.6388,
            "Sleep Apnea": 0.7344, "MIT-BIH": 0.3805,
        },
    ),
    "paper_ecgcpc_ft": (
        "Paper ECG-CPC FT",
        {
            "CPSC2018": 0.9665, "PTB-XL": 0.9288, "Survival": 0.6774, "Blood Test": 0.7394,
            "Age": 0.5810, "R-peak": 0.8787, "Exercise\nR-Peak": 0.9675, "PPG AF": 0.8321,
            "Sleep Apnea": 0.9350, "MIT-BIH": 0.4662,
        },
    ),
}

VS_SHORTCUTS: dict[str, list[str]] = {
    "all": [
        "paper_xecg_lp",
        "paper_stmem_lp",
        "paper_ecgjepa_lp",
        "paper_ecgfounder_lp",
        "paper_ecgcpc_lp",
    ],
    "all_lp": [
        "paper_xecg_lp",
        "paper_stmem_lp",
        "paper_ecgjepa_lp",
        "paper_ecgfounder_lp",
        "paper_ecgcpc_lp",
    ],
    "all_ft": [
        "paper_xecg_ft",
        "paper_stmem_ft",
        "paper_ecgjepa_ft",
        "paper_ecgfounder_ft",
        "paper_ecgcpc_ft",
    ],
    "all_paper": list(PAPER_BASELINES.keys()),
}

PALETTE = [
    ("#2E9E5B", "o", "-"),
    ("#4E79A7", "s", "-"),
    ("#E15759", "^", "-"),
    ("#B07AA1", "P", "-"),
    ("#F28E2B", "D", "--"),
    ("#76B7B2", "v", "--"),
    ("#9C755F", "X", "--"),
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
            s = df[metric_cfg].dropna(how="all").mean(axis=1).dropna()
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
) -> tuple[dict[str, float], dict[str, tuple[int, int]], Optional[float]]:
    """Load per-task scores, (valid_count, skipped_incomplete) counts, and legacy 1-rSMAPE score."""
    scores: dict[str, float] = {}
    counts: dict[str, tuple[int, int]] = {}
    legacy_scores: list[float] = []

    for label, task_folder, metric_cfg in TASK_SPECS:
        invert = False
        radar_cfg = metric_cfg
        if label == "Age" and age_metric == "rsmape":
            radar_cfg = AGE_RSMAPE_COLS
            invert = True

        exp_path = logs_dir / task_folder / experiment
        versions = get_sorted_versions(exp_path)
        vals: list[float] = []
        rsmape_vals: list[float] = []
        skipped = 0

        for v in versions:
            csv_path = exp_path / f"version_{v}" / "metrics.csv"
            val = extract_score(csv_path, radar_cfg, invert=invert)
            if val is None:
                skipped += 1
                continue
            if label == "Age" and age_metric == "r2":
                val = max(0.0, val)
            vals.append(val)
            if label == "Age":
                rsmape_val = val if age_metric == "rsmape" else extract_score(csv_path, AGE_RSMAPE_COLS, invert=True)
                if rsmape_val is not None:
                    rsmape_vals.append(rsmape_val)
            if len(vals) >= max_versions:
                break

        counts[label] = (len(vals), skipped)
        if vals:
            mean_v = float(np.mean(vals))
            scores[label] = mean_v
            if label == "Age":
                if rsmape_vals:
                    legacy_scores.append(float(np.mean(rsmape_vals)))
            else:
                legacy_scores.append(mean_v)

    legacy_agg = float(np.mean(legacy_scores)) if legacy_scores else None
    return scores, counts, legacy_agg


def print_comparison_table(
    series: list[tuple[str, dict[str, float]]],
    exp_meta: dict[str, tuple[dict[str, tuple[int, int]], Optional[float]]],
    max_versions: int,
    age_metric: str,
    plotted_labels: list[str],
) -> None:
    """Print a unified side-by-side comparison table of all experiments and --vs baselines."""
    show_delta = len(series) == 2
    col_names = [name for name, _ in series]
    col_widths = [max(len(name), 11) for name in col_names]
    task_w = 17
    runs_col = bool(exp_meta) and len(exp_meta) == 1
    first_exp = next(iter(exp_meta.keys())) if runs_col else None

    header_parts = [f"{'Task':<{task_w}}"]
    if runs_col:
        header_parts.append(f"{'Runs':<10}")
    for name, w in zip(col_names, col_widths):
        header_parts.append(f"{name:>{w}}")
    if show_delta:
        header_parts.append(f"{'Δ (1st-2nd)':>11}")

    header = " | ".join(header_parts)
    sep = "-" * len(header)
    print(f"\n{header}\n{sep}")

    for label, _, _ in TASK_SPECS:
        clean_label = label.replace("\n", " ")
        row_parts = [f"{clean_label:<{task_w}}"]
        if runs_col and first_exp is not None:
            valid_n, skipped = exp_meta[first_exp][0].get(label, (0, 0))
            runs_str = f"{valid_n}/{max_versions}" + (f" ({skipped}i)" if skipped > 0 else "")
            row_parts.append(f"{runs_str:<10}")

        vals_for_row: list[Optional[float]] = []
        for (_, sc), w in zip(series, col_widths):
            v = sc.get(label)
            vals_for_row.append(v)
            row_parts.append(f"{v:>{w}.4f}" if v is not None else f"{'—':>{w}}")

        if show_delta:
            if vals_for_row[0] is not None and vals_for_row[1] is not None:
                diff = vals_for_row[0] - vals_for_row[1]
                row_parts.append(f"{diff:>+11.4f}")
            else:
                row_parts.append(f"{'—':>11}")

        print(" | ".join(row_parts))

    print(sep)

    # Row 1: BenchECG score across common plotted tasks (apples-to-apples)
    common_label = f"BenchECG ({len(plotted_labels)}/10)"
    row_common = [f"{common_label:<{task_w}}"]
    if runs_col:
        row_common.append(f"{'common':<10}")
    common_means: list[Optional[float]] = []
    for (_, sc), w in zip(series, col_widths):
        present = [sc[l] for l in plotted_labels if l in sc]
        m = float(np.mean(present)) if present else None
        common_means.append(m)
        row_common.append(f"{m:>{w}.4f}" if m is not None else f"{'—':>{w}}")
    if show_delta:
        if common_means[0] is not None and common_means[1] is not None:
            row_common.append(f"{common_means[0] - common_means[1]:>+11.4f}")
        else:
            row_common.append(f"{'—':>11}")
    print(" | ".join(row_common))

    # Row 2: Full-available mean (e.g., paper 10/10 score if any series has more tasks than plotted_labels)
    if any(len(sc) != len(plotted_labels) for _, sc in series):
        row_all = [f"{'Full avail. mean':<{task_w}}"]
        if runs_col:
            row_all.append(f"{'all tasks':<10}")
        for (_, sc), w in zip(series, col_widths):
            if sc:
                m = float(np.mean(list(sc.values())))
                cell = f"{m:.4f} ({len(sc)})"
                row_all.append(f"{cell:>{w}}")
            else:
                row_all.append(f"{'—':>{w}}")
        if show_delta:
            row_all.append(f"{'':>11}")
        print(" | ".join(row_all))

    # Row 3: Legacy 1-rSMAPE BenchECG score for local experiments (when plotting R^2)
    if age_metric == "r2" and any(meta[1] is not None for meta in exp_meta.values()):
        row_leg = [f"{'Legacy (1-rSMAPE)':<{task_w}}"]
        if runs_col:
            row_leg.append(f"{'local':<10}")
        for (name, _), w in zip(series, col_widths):
            leg = exp_meta[name][1] if name in exp_meta else None
            row_leg.append(f"{leg:>{w}.4f}" if leg is not None else f"{'—':>{w}}")
        if show_delta:
            row_leg.append(f"{'':>11}")
        print(" | ".join(row_leg))


def select_radar_labels(series: list[tuple[str, dict[str, float]]], include_missing: bool) -> list[str]:
    all_labels = [t[0] for t in TASK_SPECS]
    if include_missing:
        labels = [l for l in all_labels if any(l in s for _, s in series)]
    else:
        labels = [l for l in all_labels if all(l in s for _, s in series)]
    if len(labels) < 3:
        raise ValueError(f"Need at least 3 tasks in common to plot radar (found {len(labels)}: {labels})")
    return labels


def plot_radar(series: list[tuple[str, dict[str, float]]], labels: list[str], out_path: Path, title: str) -> None:
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
        present = [sc[l] for l in labels if l in sc]
        mean_val = float(np.mean(present)) if present else 0.0
        vals_closed = vals + vals[:1]
        color, marker, ls = PALETTE[idx % len(PALETTE)]
        ax.plot(
            angles, vals_closed,
            color=color, linewidth=2.2, linestyle=ls, marker=marker, markersize=6.0,
            label=f"{name} ({mean_val:.3f})",
        )
        ax.fill(angles, vals_closed, color=color, alpha=0.16 if len(series) > 3 else (0.22 if len(series) > 2 else 0.30))

    ax.set_title(title, fontsize=13.5, fontweight="bold", color="#1E3A5F", pad=36)
    ax.legend(
        loc="upper right", bbox_to_anchor=(1.32, 1.00),
        frameon=True, facecolor="white", edgecolor="#D1D5DB", fontsize=9.5,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def expand_vs_keys(vs_args: list[str]) -> list[str]:
    expanded: list[str] = []
    for key in vs_args:
        targets = VS_SHORTCUTS.get(key, [key])
        for t in targets:
            if t not in expanded:
                expanded.append(t)
    return expanded


def main() -> None:
    vs_choices = list(PAPER_BASELINES.keys()) + list(VS_SHORTCUTS.keys())
    parser = argparse.ArgumentParser(description="Plot BenchECG radar chart comparing experiments and paper baselines.")
    parser.add_argument("experiments", nargs="*", help="Experiment folder names under logs_lightning/<task>/<exp>")
    parser.add_argument(
        "--labels", nargs="*", default=None,
        help="Optional custom legend labels matching the positional experiments",
    )
    parser.add_argument(
        "--vs", nargs="*", default=[], choices=vs_choices,
        help=f"Paper baselines to overlay ({', '.join(vs_choices)})",
    )
    parser.add_argument("--logs-dir", type=Path, default=Path("logs_lightning"), help="Path to logs_lightning")
    parser.add_argument("--max-versions", type=int, default=5, help="Max recent valid versions per task (default: 5)")
    parser.add_argument("--age-metric", choices=["r2", "rsmape"], default="r2", help="Age metric (default: r2)")
    parser.add_argument("--include-missing", action="store_true", help="Include tasks missing in some series (as 0)")
    parser.add_argument("--title", default="BenchECG Radar Comparison", help="Chart title")
    parser.add_argument("--out", type=Path, default=Path("radar_comparison.png"), help="Output image path (PNG/PDF/SVG)")
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.experiments):
        parser.error(f"--labels expects {len(args.experiments)} values (matching experiments), got {len(args.labels)}")

    series: list[tuple[str, dict[str, float]]] = []
    exp_meta: dict[str, tuple[dict[str, tuple[int, int]], Optional[float]]] = {}

    for idx, exp in enumerate(args.experiments):
        display_name = args.labels[idx] if args.labels is not None else exp
        sc, counts, legacy_agg = load_experiment_scores(args.logs_dir, exp, args.max_versions, args.age_metric)
        if not sc:
            print(f"[WARN] No scores found for experiment '{exp}' in {args.logs_dir}")
            continue
        series.append((display_name, sc))
        exp_meta[display_name] = (counts, legacy_agg)

    for base_key in expand_vs_keys(args.vs):
        label, sc = PAPER_BASELINES[base_key]
        series.append((label, sc))

    if not series:
        parser.error("Provide at least one experiment name or --vs baseline.")

    plotted_labels = select_radar_labels(series, args.include_missing)
    print_comparison_table(series, exp_meta, args.max_versions, args.age_metric, plotted_labels)
    plot_radar(series, plotted_labels, args.out, args.title)
    print(f"\nSaved radar plot to {args.out}")


if __name__ == "__main__":
    main()
