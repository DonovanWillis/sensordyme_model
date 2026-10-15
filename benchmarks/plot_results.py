"""Renders FPS, latency, and thermal charts from benchmark JSON results as 300dpi PNGs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent / "results"
REALTIME_FPS_THRESHOLD = 15.0  # below this, a module isn't considered real-time on this hardware


def _is_benchmark_result(data: object) -> bool:
    """A run_benchmarks.py result, as opposed to the other JSON that shares the
    directory (the committed accuracy baseline, accuracy_state_farm runs,
    accuracy_protocol recordings and reports)."""
    return isinstance(data, dict) and {"module", "fps", "latency_ms"} <= data.keys()


def load_results(timestamp: Optional[str] = None, results_dir: Optional[Path] = None) -> list[dict]:
    """Load benchmark result JSON files, optionally filtered to one run's timestamp.

    Only run_benchmarks.py output is returned. benchmarks/results/ also holds
    the committed baseline_phase1.json and other harnesses' JSON; globbing all
    of it made every plot crash with KeyError: 'module' on a fresh checkout.
    """
    directory = results_dir or RESULTS_DIR
    pattern = f"*_{timestamp}.json" if timestamp else "*.json"
    results = []
    for path in sorted(directory.glob(pattern)):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if _is_benchmark_result(data):
            results.append(data)
    return results


def plot_fps_bar_chart(results: list[dict], out_path: Path) -> None:
    """Bar chart of FPS per module with a real-time threshold line."""
    fig, ax = plt.subplots(figsize=(7, 4))
    modules = [r["module"] for r in results]
    fps_values = [r["fps"] for r in results]
    ax.bar(modules, fps_values, color="#3b82f6")
    ax.axhline(
        REALTIME_FPS_THRESHOLD,
        color="red",
        linestyle="--",
        label=f"Real-time threshold ({REALTIME_FPS_THRESHOLD:.0f} FPS)",
    )
    ax.set_ylabel("FPS")
    ax.set_title("Frames per Second by Module")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_latency_boxplot(results: list[dict], out_path: Path) -> None:
    """Box plot summarizing p50/mean/p95/p99 latency per module."""
    fig, ax = plt.subplots(figsize=(7, 4))
    modules = [r["module"] for r in results]
    data = [
        [r["latency_ms"]["p50"], r["latency_ms"]["mean"], r["latency_ms"]["p95"], r["latency_ms"]["p99"]]
        for r in results
    ]
    ax.boxplot(data, tick_labels=modules)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Per-Frame Latency Distribution (p50/mean/p95/p99) by Module")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_thermal_over_time(results: list[dict], out_path: Path) -> None:
    """Line chart of CPU temperature over time per module (Pi only; notes when absent)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    has_data = False
    for r in results:
        samples = r.get("cpu_temp_c", [])
        if not samples:
            continue
        has_data = True
        xs = [s[0] for s in samples]
        ys = [s[1] for s in samples]
        ax.plot(xs, ys, label=r["module"])
    if not has_data:
        ax.text(
            0.5,
            0.5,
            "No CPU temperature data (not running on a Pi)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("CPU temp (C)")
    ax.set_title("CPU Temperature Over Time")
    if has_data:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entrypoint: render all benchmark charts as 300dpi PNGs."""
    parser = argparse.ArgumentParser(description="Plot SensorDyme benchmark results")
    parser.add_argument("--timestamp", default=None, help="Only plot results from this benchmark run")
    args = parser.parse_args(argv)

    results = load_results(args.timestamp)
    if not results:
        print(f"No benchmark results found in {RESULTS_DIR}. Run `make bench` first.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_fps_bar_chart(results, RESULTS_DIR / "fps_bar_chart.png")
    plot_latency_boxplot(results, RESULTS_DIR / "latency_boxplot.png")
    plot_thermal_over_time(results, RESULTS_DIR / "thermal_over_time.png")
    print(f"Charts written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
