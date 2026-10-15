"""Generates a weekly summary PDF: events per day, per type, and per hour-of-day.

Weekly means the last seven days, not the whole log: events.csv is append-only
for the life of the device. Days and hours are the reader's local time (events
are stored in UTC), and a row that cannot be read -- a torn last line from a
power cut, or one caught half-written while the device is logging -- is skipped
rather than aborting the report.
"""

from __future__ import annotations

import csv
from collections import Counter
from datetime import UTC, datetime, timedelta, tzinfo
from pathlib import Path
from typing import Optional, Union

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


def load_events(csv_path: Union[str, Path]) -> list[dict]:
    """Load events.csv rows as a list of dicts. Returns [] if the file doesn't exist yet."""
    path = Path(csv_path)
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _row_time(row: dict, tz: Optional[tzinfo]) -> Optional[datetime]:
    """The row's timestamp in `tz` (None = this machine's local zone), or None if unreadable.

    A naive timestamp is taken as UTC, which is what the device writes.
    """
    raw = row.get("timestamp")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        stamp = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return stamp.astimezone(tz)


def summarize(rows: list[dict], tz: Optional[tzinfo] = None) -> dict[str, Counter]:
    """Aggregate raw event rows into per-day, per-type, and per-hour-of-day counters.

    Days and hours are in `tz` (default: this machine's local zone). Unreadable
    rows are skipped.
    """
    per_day: Counter = Counter()
    per_type: Counter = Counter()
    per_hour: Counter = Counter()
    for row in rows:
        ts = _row_time(row, tz)
        if ts is None:
            continue
        per_day[ts.date().isoformat()] += 1
        per_type[row.get("type") or "unknown"] += 1
        per_hour[ts.hour] += 1
    return {"per_day": per_day, "per_type": per_type, "per_hour": per_hour}


def last_days(rows: list[dict], days: int = 7, now: Optional[datetime] = None) -> list[dict]:
    """The readable rows from the `days` days up to `now` (default: the current time)."""
    end = now or datetime.now(UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    start = end - timedelta(days=days)
    kept = []
    for row in rows:
        ts = _row_time(row, UTC)
        if ts is not None and start <= ts <= end:
            kept.append(row)
    return kept


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


def generate_weekly_report(
    csv_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    now: Optional[datetime] = None,
    days: int = 7,
    tz: Optional[tzinfo] = None,
) -> Path:
    """Build a weekly summary PDF from events.csv and write it to `output_dir`.

    Args:
        csv_path: Path to the events CSV log.
        output_dir: Directory to write the report PDF into (created if missing).
        now: End of the reporting window (default: the current time).
        days: Length of the reporting window.
        tz: Zone for day/hour buckets (default: this machine's local zone).

    Returns:
        Path to the generated PDF.
    """
    rows = last_days(load_events(csv_path), days=days, now=now)
    stats = summarize(rows, tz=tz)

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
    _width, height = letter

    c.setFont("Helvetica-Bold", 18)
    c.drawString(0.75 * inch, height - 1 * inch, "SensorDyme Weekly Report")
    c.setFont("Helvetica", 10)
    c.drawString(0.75 * inch, height - 1.3 * inch, f"Total events (last {days} days): {len(rows)}")
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
