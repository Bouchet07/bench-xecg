"""Compare runtime and GPU utilization of `all_pipeline` (CephFS) vs `run_task` (memmap).

Auto-discovers the latest COMPLETED runs in `runs/` for both tags, prints a
Rich summary table with time saved and speedup factors, and saves a bar chart
to `speedup_comparison.png`.
"""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path

try:
    import matplotlib
    if not os.environ.get("DISPLAY") and os.name != "nt":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

TASKS = [
    "mit_bih",
    "r_peaks",
    "exercise",
    "apnea",
    "ppg_af",
    "ptb_xl",
    "cpsc2018",
    "lab",
    "survival",
    "age",
]


def discover_latest_runs(
    runs_dir: Path,
    pipeline_tag: str,
    require_substr: str | None = None,
    exclude_substr: str | None = None,
) -> dict[str, Path]:
    """Find the newest COMPLETED run directory in `runs_dir` for each task."""
    found: dict[str, Path] = {}
    if not runs_dir.exists():
        return found
    for task in TASKS:
        candidates = sorted(
            (
                p for p in runs_dir.iterdir()
                if p.is_dir()
                and f"_{pipeline_tag}_" in p.name
                and p.name.endswith(f"_{task}")
                and (require_substr is None or require_substr in p.name)
                and (exclude_substr is None or exclude_substr not in p.name)
            ),
            key=lambda p: p.name,
            reverse=True,
        )
        for cand in candidates:
            rj = cand / "run.json"
            if rj.exists() and json.loads(rj.read_text(encoding="utf-8")).get("status") == "COMPLETED":
                found[task] = cand
                break
    return found


def load_run_info(run_dir: Path) -> dict:
    """Extract total runtime (seconds), node name, and average GPU utilization."""
    data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    execs = data.get("executions", [])
    telem = execs[0].get("telemetry_summary", {}) if execs else {}
    return {
        "runtime": float(sum(ex.get("runtime_seconds", 0.0) for ex in execs)),
        "node": execs[0].get("hostname", "?").split(".")[0] if execs else "?",
        "gpu_util": telem.get("gpu_util_avg", float("nan")),
    }


def fmt(seconds: float) -> str:
    """Format duration in seconds as `Xd HHh MMm` or `HHh MMm SSs`."""
    td = timedelta(seconds=int(round(seconds)))
    h, rem = divmod(td.seconds, 3600)
    m, s = divmod(rem, 60)
    if td.days > 0:
        return f"{td.days}d {h:02d}h {m:02d}m"
    return f"{h:02d}h {m:02d}m {s:02d}s"


def plot_comparison(rows: list[dict], output_png: Path, slow_tag: str, fast_tag: str) -> bool:
    """Save a publication-style grouped horizontal bar chart comparing slow vs fast runtimes."""
    if not rows or not HAS_MPL:
        return False

    labels = [r["task"] for r in rows]
    slow_min = [r["slow_s"] / 60.0 for r in rows]
    fast_min = [r["fast_s"] / 60.0 for r in rows]
    total_slow = sum(r["slow_s"] for r in rows)
    total_fast = sum(r["fast_s"] for r in rows)
    overall = total_slow / total_fast if total_fast > 0 else float("inf")

    y = list(range(len(labels)))
    h = 0.35

    plt.rcParams.update({"font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"]})
    fig, ax = plt.subplots(figsize=(10.5, max(4.2, 0.82 * len(labels) + 1.8)), facecolor="white")
    ax.set_facecolor("#FAFBFC")

    ax.barh(
        [v - h / 2 for v in y],
        slow_min,
        height=h,
        color="#E05A5A",
        edgecolor="#B83B3B",
        linewidth=0.6,
        label=f"{slow_tag} (CephFS)",
        zorder=3,
    )
    ax.barh(
        [v + h / 2 for v in y],
        fast_min,
        height=h,
        color="#2E9E5B",
        edgecolor="#1E7340",
        linewidth=0.6,
        label=f"{fast_tag} (/dev/shm memmap)",
        zorder=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.5, fontweight="semibold", color="#1F2937")
    ax.invert_yaxis()
    ax.set_xlabel("Runtime (minutes)", fontsize=10.5, fontweight="semibold", color="#374151", labelpad=8)
    ax.tick_params(axis="x", colors="#4B5563", labelsize=9.5)
    ax.tick_params(axis="y", length=0)

    ax.set_axisbelow(True)
    ax.grid(axis="x", linestyle="--", linewidth=0.7, color="#D1D5DB", alpha=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#D1D5DB")

    ax.set_title(
        "Benchmark Runtime: CephFS vs /dev/shm Memmap",
        fontsize=13,
        fontweight="bold",
        color="#111827",
        loc="left",
        pad=22,
    )
    ax.text(
        0.0,
        1.02,
        f"Matched tasks: {len(rows)}  |  Total: {fmt(total_slow)} -> {fmt(total_fast)}  ({overall:.2f}x faster)",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#4B5563",
    )

    max_x = max(max(slow_min), max(fast_min))
    ax.set_xlim(0, max_x * 1.32)

    for i, r in enumerate(rows):
        ax.text(
            slow_min[i] + max_x * 0.015,
            y[i] - h / 2,
            f"{slow_min[i]:.1f}m",
            va="center",
            fontsize=9,
            color="#7F1D1D",
        )
        sp_color = "#14532D" if r["speedup"] >= 1.0 else "#991B1B"
        ax.text(
            fast_min[i] + max_x * 0.015,
            y[i] + h / 2,
            f"{fast_min[i]:.1f}m  ({r['speedup']:.2f}x)",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=sp_color,
        )

    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#E5E7EB",
        framealpha=0.95,
        fontsize=9.5,
    )

    fig.tight_layout()
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare slow vs fast Hermes runtimes in runs/")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="Path to runs directory (default: runs)")
    parser.add_argument("--model", choices=["mine", "xlstm"], default="mine", help="Model preset (default: mine)")
    parser.add_argument("--slow-tag", default=None, help="Override baseline pipeline tag")
    parser.add_argument("--fast-tag", default="run_task", help="Optimized pipeline tag (default: run_task)")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path")
    args = parser.parse_args()

    if args.model == "xlstm":
        slow_tag = args.slow_tag or "xlstm_lp"
        fast_req, fast_exc = "_xlstm_", None
        out_png = args.out or Path("speedup_comparison_xlstm.png")
    else:
        slow_tag = args.slow_tag or "all_pipeline"
        fast_req, fast_exc = None, "_xlstm_"
        out_png = args.out or Path("speedup_comparison.png")

    slow_runs = discover_latest_runs(args.runs_dir, slow_tag)
    fast_runs = discover_latest_runs(args.runs_dir, args.fast_tag, require_substr=fast_req, exclude_substr=fast_exc)

    console = Console()
    table = Table(
        title=f"Runtime Speedup ({args.model}): {slow_tag} vs {args.fast_tag}",
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    table.add_column("Task", style="bold")
    table.add_column(f"Slow ({slow_tag})", justify="right")
    table.add_column(f"Fast ({args.fast_tag})", justify="right")
    table.add_column("Saved", justify="right")
    table.add_column("Speedup", justify="right")
    table.add_column("GPU Util (Slow -> Fast)", justify="center")

    matched_rows: list[dict] = []
    total_slow = 0.0
    total_fast = 0.0

    for task in TASKS:
        has_slow = task in slow_runs
        has_fast = task in fast_runs
        if not (has_slow or has_fast):
            continue

        slow_info = load_run_info(slow_runs[task]) if has_slow else None
        fast_info = load_run_info(fast_runs[task]) if has_fast else None

        slow_str = fmt(slow_info["runtime"]) if slow_info else "-"
        fast_str = fmt(fast_info["runtime"]) if fast_info else "-"

        if slow_info and fast_info:
            slow_s = slow_info["runtime"]
            fast_s = fast_info["runtime"]
            saved_s = slow_s - fast_s
            speedup = slow_s / fast_s if fast_s > 0 else float("inf")
            total_slow += slow_s
            total_fast += fast_s
            matched_rows.append({"task": task, "slow_s": slow_s, "fast_s": fast_s, "speedup": speedup})

            saved_str = f"[green]+{fmt(saved_s)}[/green]" if saved_s >= 0 else f"[red]-{fmt(-saved_s)}[/red]"
            sp_color = "bold green" if speedup >= 1.2 else ("yellow" if speedup >= 0.95 else "red")
            sp_str = f"[{sp_color}]{speedup:.2f}x[/{sp_color}]"
            gpu_str = f"{slow_info['gpu_util']:.1f}% -> {fast_info['gpu_util']:.1f}%"
        else:
            saved_str = "-"
            sp_str = "-"
            gpu_val = slow_info["gpu_util"] if slow_info else fast_info["gpu_util"]
            gpu_str = f"{gpu_val:.1f}%"

        table.add_row(task, slow_str, fast_str, saved_str, sp_str, gpu_str)

    console.print(table)

    if matched_rows:
        overall_speedup = total_slow / total_fast if total_fast > 0 else float("inf")
        total_saved = total_slow - total_fast
        summary = (
            f"[bold]Matched tasks:[/bold] {len(matched_rows)}\n"
            f"[bold]Total Slow ({slow_tag}):[/bold] {fmt(total_slow)}\n"
            f"[bold]Total Fast ({args.fast_tag}):[/bold] {fmt(total_fast)}\n"
            f"[bold]Total Time Saved:[/bold] [green]{fmt(total_saved)}[/green]\n"
            f"[bold]Overall Speedup:[/bold] [bold green]{overall_speedup:.2f}x[/bold green]"
        )
        console.print(Panel(summary, title="Summary", expand=False))
        if plot_comparison(matched_rows, out_png, slow_tag, args.fast_tag):
            console.print(f"Saved chart to [cyan]{out_png}[/cyan]")
        else:
            console.print("[dim]matplotlib not installed; skipped PNG chart.[/dim]")
    else:
        console.print("[yellow]No matched tasks found with both tags.[/yellow]")


if __name__ == "__main__":
    main()
