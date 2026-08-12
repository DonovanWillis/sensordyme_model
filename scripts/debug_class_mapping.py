"""Trace raw_class_id -> raw_model_label -> SensorDyme_label -> alert_state
for a handful of known COCO classes, through both detection paths that exist
in this repo:

  occupancy.py / custom AI rules   raw COCO name is the SensorDyme label
                                    directly (core/detections.py Detection.label)
  driver_attention.py              raw COCO name collapses through config.yaml's
                                    yolo_classes: map into a coarser event type

Run: .venv/bin/python scripts/debug_class_mapping.py
"""
from __future__ import annotations

import yaml

from core.accel import COCO_NAMES

TEST_CLASSES = ["bottle", "banana", "person", "cell phone", "chair", "car"]

name_to_id = {name: cid for cid, name in COCO_NAMES.items()}

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
# The real location: modules.driver_attention.yolo_classes. (Presets live in
# the separate presets.yaml, not under a "presets" key here, and the preset
# itself was renamed driver_distraction -> driver_safety a while back — this
# used to look in both wrong places before falling back to the search below.)
driver_yolo_classes = cfg.get("modules", {}).get("driver_attention", {}).get("yolo_classes", {})
if not driver_yolo_classes:
    def _find_yolo_classes(node):
        if isinstance(node, dict):
            if "yolo_classes" in node and isinstance(node["yolo_classes"], dict):
                return node["yolo_classes"]
            for v in node.values():
                found = _find_yolo_classes(v)
                if found:
                    return found
        return None
    driver_yolo_classes = _find_yolo_classes(cfg) or {}

rows = []
for name in TEST_CLASSES:
    class_id = name_to_id.get(name)
    occupancy_label = name  # occupancy.py / custom rules: COCO_NAMES.get(class_id) verbatim
    driver_event_type = driver_yolo_classes.get(name, "(not in driver_attention's requested classes)")
    rows.append((class_id, name, occupancy_label, driver_event_type))

col = "{:<8} {:<14} {:<28} {:<45}"
print(col.format("class_id", "raw_label", "occupancy/custom_rule label", "driver_attention event_type"))
print("-" * 100)
for class_id, name, occ, drv in rows:
    print(col.format(str(class_id), name, occ, drv))

print()
print("Alert state depends on the RULE, not the label itself — a detection is")
print("only ever an alert when a rule (preset alert_types or a custom AI rule)")
print("names that specific label/event_type. None of these six are alerts by")
print("default; each becomes one only if configured to be.")
