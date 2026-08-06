#!/usr/bin/env python3
"""Generate docs/DETECTIONS.md from core/taxonomy.py — run after editing the taxonomy."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.taxonomy import TAXONOMY  # noqa: E402

BADGE = {"supported": "✅", "partial": "🟡", "planned": "🔵", "unsupported": "❌"}

lines = [
    "# SensorDyme Detection Capability Matrix",
    "",
    "*Generated from `core/taxonomy.py` by `scripts/generate_detections_doc.py` — edit the",
    "taxonomy, not this file.* This is the honest contract of what each mode can and cannot",
    "detect. ✅ supported today · 🟡 partial (caveats listed) · 🔵 planned · ❌ not possible",
    "with the current camera/models (what it would need is listed).",
    "",
]
for mode, detections in TAXONOMY.items():
    lines += [f"## {mode}", "", "| | Detection | How / Caveat / Requires |", "|---|---|---|"]
    for d in detections:
        detail = d.how if d.support in ("supported", "partial") else d.requires
        if d.caveat:
            detail += f" **Caveat:** {d.caveat}"
        lines.append(f"| {BADGE[d.support]} | {d.title} (`{d.id}`) | {detail} |")
    lines.append("")

out = Path(__file__).resolve().parent.parent / "docs" / "DETECTIONS.md"
out.write_text("\n".join(lines))
print(f"wrote {out} ({len(lines)} lines)")
