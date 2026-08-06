"""Generates a weekly summary PDF: events per day, per type, and per hour-of-day."""

from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib.pagesizes import letter  # noqa: E402
from reportlab.lib.units import inch  # noqa: E402
from reportlab.pdfgen import canvas  # noqa: E402


def load_events(csv_path: Union[str, Path]) -> list[dict]:
    """Load events.csv rows as a list of dicts. Returns [] if the file doesn't exist yet."""
    path = Path(csv_path)
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def summarize(rows: list[dict]) -> dict[str, Counter]:
    """Aggregate raw event rows into per-day, per-type, and per-hour-of-day counters."""
    per_day: Counter = Counter()
    per_type: Counter = Counter()
    per_hour: Counter = Counter()
    for row in rows:
        ts = datetime.fromisoformat(row["timestamp"])
        per_day[ts.date().isoformat()] += 1
        per_type[row["type"]] += 1
        per_hour[ts.hour] += 1
    return {"per_day": per_day, "per_type": per_type, "per_hour": per_hour}


def _bar_chart(data: dict, title: str, xlabel: str, out_path: Path) -> None:
    """Render a single bar chart to `out_path` as a PNG."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    keys = list(data.keys())
    values = [data[k] for k in keys]
    ax.bar([str(k) for k in keys], values, color="#3b82f6")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Events")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_weekly_report(csv_path: Union[str, Path], output_dir: Union[str, Path]) -> Path:
    """Build a weekly summary PDF from events.csv and write it to `output_dir`.

    Args:
        csv_path: Path to the events CSV log.
        output_dir: Directory to write the report PDF into (created if missing).

    Returns:
        Path to the generated PDF.
    """
    rows = load_events(csv_path)
    stats = summarize(rows)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_tmp_charts"
    tmp_dir.mkdir(exist_ok=True)

    day_chart = tmp_dir / "per_day.png"
    type_chart = tmp_dir / "per_type.png"
    hour_chart = tmp_dir / "per_hour.png"
    _bar_chart(dict(sorted(stats["per_day"].items())), "Events per Day", "Date", day_chart)
    _bar_chart(dict(stats["per_type"]), "Events per Type", "Type", type_chart)
    _bar_chart(
        {h: stats["per_hour"].get(h, 0) for h in range(24)},
        "Events per Hour of Day",
        "Hour",
        hour_chart,
    )

    timestamp_str = rows[-1]["timestamp"].replace(":", "").replace("-", "") if rows else "empty"
    report_path = out_dir / f"weekly_report_{timestamp_str}.pdf"

    c = canvas.Canvas(str(report_path), pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 18)
    c.drawString(0.75 * inch, height - 1 * inch, "SensorDyme Weekly Report")
    c.setFont("Helvetica", 10)
    c.drawString(0.75 * inch, height - 1.3 * inch, f"Total events: {len(rows)}")
    c.drawString(0.75 * inch, height - 1.5 * inch, f"Event types: {dict(stats['per_type'])}")

    y = height - 4.0 * inch
    for chart in (day_chart, type_chart, hour_chart):
        c.drawImage(str(chart), 0.75 * inch, y, width=6 * inch, height=3.0 * inch, preserveAspectRatio=True)
        y -= 3.2 * inch
        if y < 1 * inch:
            c.showPage()
            y = height - 3.2 * inch
    c.save()

    for chart in (day_chart, type_chart, hour_chart):
        chart.unlink()
    tmp_dir.rmdir()

    return report_path
