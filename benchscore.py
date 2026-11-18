import glob
import os
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import typer

app = typer.Typer()

# ================= CONFIGURATION =================
# Map folders to the specific metric column name in metrics.csv
# Published paper (npj Digital Medicine, 2026) uses clipped R^2 for Age.
AGE_R2_COLS = [
    "mimic/test_r2/dataloader_idx_1",
    "ptbxl/test_r2/dataloader_idx_0",
    "cpsc/test_r2/dataloader_idx_2",
]
AGE_RSMAPE_COLS = [
    "mimic/test_rsmape/dataloader_idx_1",
    "ptbxl/test_rsmape/dataloader_idx_0",
    "cpsc/test_rsmape/dataloader_idx_2",
]

TASK_METRIC_MAP: Dict[str, Union[str, List[str]]] = {
    "train-age": AGE_R2_COLS,
    "train-cpsc2018-multilabel": "test_auroc",
    "train-exercise-r-peak": "test_f1_20",
    "train-lab": "test_auroc_avg",
    "train-mitbih-5": "test_f1/mean",
    "train-mitbih-r_peaks": "test_f1_20",
    "train-ptbxl-diagnosis_superclass-multilabel": "test_auroc",
    "train-sleep-apnea_bis": "test_auc",
    "train-survival": "test_ci",
    "train-deepbeat": "test_auroc",
}
# =================================================


def get_sorted_versions(exp_path: str) -> List[int]:
    """Return all `version_*` folder numbers sorted descending (newest first)."""
    search_pattern = os.path.join(exp_path, "version_*")
    folder_paths = glob.glob(search_pattern)

    versions = []
    version_regex = re.compile(r"version_(\d+)$")

    for path in folder_paths:
        if os.path.isdir(path):
            folder_name = os.path.basename(path)
            match = version_regex.search(folder_name)
            if match:
                versions.append(int(match.group(1)))

    versions.sort(reverse=True)
    return versions


def extract_raw_score(file_path: str, metric_config: Union[str, List[str]]) -> Optional[float]:
    """Read `metrics.csv` and return the latest non-null value (or row-wise mean for composite tasks)."""
    if not os.path.isfile(file_path):
        return None
    try:
        df = pd.read_csv(file_path)

        if isinstance(metric_config, list):
            missing = [c for c in metric_config if c not in df.columns]
            if missing:
                return None

            valid_rows = df[metric_config].dropna(how="all")
            if valid_rows.empty:
                return None
            combined_series = valid_rows.mean(axis=1).dropna()
            if combined_series.empty:
                return None
            return float(combined_series.iloc[-1])

        if metric_config not in df.columns:
            return None
        valid_vals = df[metric_config].dropna()
        if valid_vals.empty:
            return None
        return float(valid_vals.iloc[-1])

    except Exception:
        return None


def bootstrap_bencheg_score(
    per_task_runs: List[List[float]],
    n_resamples: int = 10_000,
    seed: int = 42,
) -> Tuple[float, float, float, float]:
    """Paper Table 4 bootstrap: sample 1 run per task, average across tasks, repeat N times."""
    rng = np.random.default_rng(seed)
    n_tasks = len(per_task_runs)
    acc = np.zeros(n_resamples, dtype=np.float64)
    for runs in per_task_runs:
        arr = np.asarray(runs, dtype=np.float64)
        idx = rng.integers(0, len(arr), size=n_resamples)
        acc += arr[idx]
    boot = acc / n_tasks
    mean = float(np.mean(boot))
    std = float(np.std(boot, ddof=1)) if n_resamples > 1 else 0.0
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    return mean, std, float(ci_lo), float(ci_hi)


@app.command()
def main(
    base_folder: str = typer.Argument(..., help="Path to the logs folder (e.g. logs_lightning)"),
    experiment_name: str = typer.Argument(..., help="Name of the experiment (wandb_group)"),
    max_versions: int = typer.Option(5, help="Number of most recent valid versions to average"),
    age_metric: str = typer.Option(
        "r2",
        "--age-metric",
        help="Primary Age metric shown in table: 'r2' (paper default, clipped >= 0) or 'rsmape' (legacy 1-rSMAPE)",
    ),
    n_bootstrap: int = typer.Option(10_000, "--n-bootstrap", help="Bootstrap resamples for BenchECG CI (paper: 10000)"),
):
    age_metric = age_metric.lower()
    if age_metric not in {"r2", "rsmape"}:
        typer.secho(f"Error: --age-metric must be 'r2' or 'rsmape' (got '{age_metric}').", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    r2_task_runs: List[List[float]] = []
    rsmape_task_runs: List[List[float]] = []
    total_tasks = len(TASK_METRIC_MAP)

    header = f"{'Task':<45} | {'Metric':<11} | {'Run Status':<38} | {'Score (Mean ± Std)':<20}"
    typer.echo("\n" + header)
    typer.echo("-" * len(header))

    for task_folder, metric_config in TASK_METRIC_MAP.items():
        exp_path = os.path.join(base_folder, task_folder, experiment_name)
        found_versions = get_sorted_versions(exp_path)
        total_found = len(found_versions)

        if task_folder == "train-age":
            primary_cfg = AGE_R2_COLS if age_metric == "r2" else AGE_RSMAPE_COLS
        else:
            primary_cfg = metric_config

        primary_values: List[float] = []
        age_r2_values: List[float] = []
        age_rsmape_values: List[float] = []
        skipped_incomplete = 0

        for v in found_versions:
            csv_path = os.path.join(exp_path, f"version_{v}", "metrics.csv")
            val = extract_raw_score(csv_path, primary_cfg)
            if val is None:
                skipped_incomplete += 1
                continue

            if task_folder == "train-age":
                r2_raw = val if age_metric == "r2" else extract_raw_score(csv_path, AGE_R2_COLS)
                rsmape_raw = val if age_metric == "rsmape" else extract_raw_score(csv_path, AGE_RSMAPE_COLS)
                if r2_raw is not None:
                    age_r2_values.append(max(0.0, r2_raw))
                if rsmape_raw is not None:
                    age_rsmape_values.append(1.0 - rsmape_raw)
                primary_values.append(max(0.0, val) if age_metric == "r2" else (1.0 - val))
            else:
                primary_values.append(val)

            if len(primary_values) >= max_versions:
                break

        valid_count = len(primary_values)
        skip_note = f", {skipped_incomplete} incomplete" if skipped_incomplete > 0 else ""

        if valid_count == 0:
            status_msg = f"MISSING (0 valid{skip_note})"
            color = typer.colors.RED
        elif valid_count < max_versions:
            status_msg = f"WARNING: {valid_count}/{max_versions} valid{skip_note}"
            color = typer.colors.RED
        elif skipped_incomplete > 0:
            status_msg = f"WARN: used {valid_count}/{total_found}{skip_note}"
            color = typer.colors.YELLOW
        elif total_found > max_versions:
            status_msg = f"NOTE: {total_found} found, used top {max_versions}"
            color = typer.colors.YELLOW
        else:
            status_msg = f"OK: {valid_count}/{max_versions}"
            color = typer.colors.GREEN

        score_str = "N/A"
        if task_folder == "train-age":
            metric_label = "Age R²" if age_metric == "r2" else "Age 1-rSM"
        else:
            metric_label = "Single"

        if primary_values:
            avg_val = float(np.mean(primary_values))
            if len(primary_values) > 1:
                std_val = float(np.std(primary_values, ddof=1))
                score_str = f"{avg_val:.5f} ± {std_val:.5f}"
            else:
                score_str = f"{avg_val:.5f}"

            if task_folder == "train-age":
                if age_r2_values:
                    r2_task_runs.append(age_r2_values)
                if age_rsmape_values:
                    rsmape_task_runs.append(age_rsmape_values)
            else:
                r2_task_runs.append(primary_values)
                rsmape_task_runs.append(primary_values)

        typer.echo(f"{task_folder:<45} | {metric_label:<11} | ", nl=False)
        typer.secho(f"{status_msg:<38}", fg=color, nl=False)
        typer.echo(f" | {score_str}")

    typer.echo("-" * len(header))

    if r2_task_runs:
        n_scored = len(r2_task_runs)
        fg = typer.colors.BRIGHT_GREEN if n_scored == total_tasks else typer.colors.YELLOW
        mean_r2, std_r2, lo_r2, hi_r2 = bootstrap_bencheg_score(r2_task_runs, n_resamples=n_bootstrap)
        typer.secho(
            f"\nBENCHECG SCORE (Paper, Age=R², {n_scored}/{total_tasks} tasks):      "
            f"{mean_r2:.5f} ± {std_r2:.5f} [{lo_r2:.5f}, {hi_r2:.5f}]",
            fg=fg,
            bold=True,
        )
        if rsmape_task_runs:
            mean_rs, std_rs, lo_rs, hi_rs = bootstrap_bencheg_score(rsmape_task_runs, n_resamples=n_bootstrap)
            typer.echo(
                f"LEGACY SCORE   (Age=1-rSMAPE, {len(rsmape_task_runs)}/{total_tasks} tasks):      "
                f"{mean_rs:.5f} ± {std_rs:.5f} [{lo_rs:.5f}, {hi_rs:.5f}]"
            )
    else:
        typer.secho("\nNo scores calculated.", fg=typer.colors.RED)


if __name__ == "__main__":
    app()
