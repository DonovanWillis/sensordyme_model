"""What the screen shows must agree with what the notification says.

Someone who configures "alert me when you see a bottle" gets a notification
about a bottle. Two things then contradicted it: the box on the camera was
captioned "People", and the screen stayed calm because a configured rule
defaults to severity "warning".

Both are the same mistake in different places — deciding what something is, or
how serious it is, from something other than what the user actually asked for.
"""

import numpy as np
import pytest

from core.detections import Detection, DetectionFrame
from core.rules import RuleEngine

pytest.importorskip("cv2")


def a_frame(label: str, n: int = 1) -> DetectionFrame:
    return DetectionFrame(
        width=640, height=480,
        detections=tuple(Detection(label=label, confidence=0.9, box=(10, 10, 60, 60))
                         for _ in range(n)),
    )


def rule(target: str) -> dict:
    return {"target": target, "predicate": "present", "description": f"Alert on {target}",
            "persist_s": 1.0, "cooldown_s": 10, "notify": "push"}


# ------------------------------------------------------------------ alert state

def test_a_configured_rule_raises_the_alert_state():
    """The red screen. It used to come only from the module's own opinion, so a
    rule the user explicitly asked to be alerted about left the screen calm."""
    engine = RuleEngine(raw_rules=[rule("bottle")])
    assert engine.is_alert_active() is False
    for t in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
        engine.observe_frame(a_frame("bottle"), now=t)
    assert engine.is_alert_active() is True


def test_the_alert_state_clears_when_the_thing_goes_away():
    """An alert that never resolves is a red screen nobody can trust."""
    engine = RuleEngine(raw_rules=[rule("bottle")])
    for t in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
        engine.observe_frame(a_frame("bottle"), now=t)
    assert engine.is_alert_active() is True
    for t in range(3, 40):
        engine.observe_frame(DetectionFrame(width=640, height=480), now=float(t))
    assert engine.is_alert_active() is False


def test_it_works_for_any_object_not_only_bottles():
    for target in ("cell phone", "backpack", "dog", "car"):
        engine = RuleEngine(raw_rules=[rule(target)])
        for t in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
            engine.observe_frame(a_frame(target), now=t)
        assert engine.is_alert_active() is True, f"{target} never raised the alert state"


# ------------------------------------------------------------------ the overlay

def occupancy_showing(labels: list[str], requested: set[str]):
    """Build the module without running __init__.

    Drawing is a pure function of the last frame's results; constructing the
    real thing would load YOLO weights to test a caption.
    """
    from modules.occupancy import OccupancyModule

    module = object.__new__(OccupancyModule)
    module._last_labels = labels
    module._last_boxes = [(10 + i * 70, 10, 60 + i * 70, 60) for i in range(len(labels))]
    module._last_confidences = [0.9] * len(labels)
    module._extra_labels = requested
    module._zones = []
    return module


def test_a_bottle_is_captioned_bottle_not_people():
    """The regression. This module began as a person counter, so every box was
    drawn under a "People: N" header — a specific, confident, wrong claim about
    the picture underneath it."""
    module = occupancy_showing(["bottle"], {"bottle"})
    assert module._headline(["bottle"]) == "Bottle: 1"
    assert "People" not in module._headline(["bottle"])


def test_each_requested_object_is_counted_by_its_own_name():
    module = occupancy_showing(["cell phone", "cell phone", "backpack"],
                               {"cell phone", "backpack"})
    headline = module._headline(["cell phone", "cell phone", "backpack"])
    assert "Cell Phone: 2" in headline
    assert "Backpack: 1" in headline


def test_plain_person_counting_is_unchanged_when_nothing_was_requested():
    """The module's original job must keep working."""
    module = occupancy_showing(["person", "person"], set())
    assert module._headline(["person", "person"]) == "People: 2"


def test_the_overlay_draws_without_raising_for_any_label():
    module = occupancy_showing(["bottle", "dog"], {"bottle"})
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert module.annotate(frame) is not None


def test_the_overlay_survives_labels_and_boxes_disagreeing():
    """A dropped frame can leave them mismatched; a caption is not worth a crash."""
    module = occupancy_showing(["bottle"], {"bottle"})
    module._last_boxes = [(10, 10, 60, 60), (70, 70, 120, 120)]   # more boxes than labels
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert module.annotate(frame) is not None
