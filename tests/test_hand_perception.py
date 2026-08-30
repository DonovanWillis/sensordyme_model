"""Tests for core/hand_perception.py -- hand anchoring, dynamic ROI sizing,
hand-object association scoring, and the full per-frame engine.

Landmarks are synthetic MediaPipe Pose lists (33 points, everything hidden
except what a test places), the same pattern test_activity_gestures.py and
test_fall_detection_zones.py already use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.hand_perception import (
    UNKNOWN_OBJECT,
    HandAnchor,
    HandPerceptionEngine,
    association_score,
    detect_hand_anchors,
    hand_roi_box,
    torso_length_from_landmarks,
)
from core.tracking import BoxTracker

W, H = 640, 480
FRAME = np.zeros((H, W, 3), dtype=np.uint8)


@dataclass
class _Landmark:
    x: float
    y: float
    visibility: float = 1.0


def _landmarks(overrides: dict[int, tuple[float, float, float]]) -> list[_Landmark]:
    """33-point list, everything hidden (visibility=0) except overrides."""
    marks = [_Landmark(0.0, 0.0, visibility=0.0) for _ in range(33)]
    for idx, values in overrides.items():
        x, y = values[0], values[1]
        vis = values[2] if len(values) > 2 else 1.0
        marks[idx] = _Landmark(x, y, vis)
    return marks


_TORSO = {
    11: (0.45, 0.30), 12: (0.55, 0.30),  # shoulders
    23: (0.45, 0.55), 24: (0.55, 0.55),  # hips
}


def _standing_with_hands(left_wrist=None, right_wrist=None, left_elbow=None, right_elbow=None):
    overrides = dict(_TORSO)
    if left_wrist is not None:
        overrides[15] = left_wrist
    if right_wrist is not None:
        overrides[16] = right_wrist
    if left_elbow is not None:
        overrides[13] = left_elbow
    if right_elbow is not None:
        overrides[14] = right_elbow
    return _landmarks(overrides)


# ------------------------------------------------------------------ hand anchors
class TestDetectHandAnchors:
    def test_no_hands_visible_returns_empty(self):
        lm = _standing_with_hands()
        assert detect_hand_anchors(lm) == []

    def test_one_hand_visible(self):
        lm = _standing_with_hands(right_wrist=(0.65, 0.45))
        anchors = detect_hand_anchors(lm)
        assert len(anchors) == 1
        assert anchors[0].side == "right"
        assert anchors[0].wrist == (0.65, 0.45)

    def test_both_hands_visible_independently(self):
        lm = _standing_with_hands(left_wrist=(0.30, 0.45), right_wrist=(0.70, 0.45))
        anchors = detect_hand_anchors(lm)
        sides = {a.side for a in anchors}
        assert sides == {"left", "right"}

    def test_low_visibility_wrist_is_treated_as_absent(self):
        lm = _standing_with_hands(right_wrist=(0.65, 0.45, 0.1))  # below the visibility floor
        assert detect_hand_anchors(lm) == []

    def test_elbow_is_included_when_visible(self):
        lm = _standing_with_hands(right_wrist=(0.65, 0.45), right_elbow=(0.60, 0.40))
        anchors = detect_hand_anchors(lm)
        assert anchors[0].elbow == (0.60, 0.40)

    def test_elbow_is_none_when_not_visible(self):
        lm = _standing_with_hands(right_wrist=(0.65, 0.45))  # no elbow override -> hidden
        anchors = detect_hand_anchors(lm)
        assert anchors[0].elbow is None


class TestTorsoLength:
    def test_computes_a_positive_length_for_a_visible_torso(self):
        lm = _standing_with_hands()
        length = torso_length_from_landmarks(lm)
        assert length is not None and length > 0

    def test_none_when_shoulders_or_hips_not_visible(self):
        lm = _landmarks({})  # nothing visible at all
        assert torso_length_from_landmarks(lm) is None


# ------------------------------------------------------------------ hand ROI
class TestHandRoiBox:
    def _anchor(self, wx=0.5, wy=0.5):
        return HandAnchor(side="right", wrist=(wx, wy), elbow=None, visibility=1.0)

    def test_roi_is_centered_on_the_wrist(self):
        box = hand_roi_box(self._anchor(0.5, 0.5), torso_length_norm=0.3, frame_w=W, frame_h=H, aspect=1.0)
        x1, y1, x2, y2 = box
        assert abs((x1 + x2) / 2 - 0.5 * W) < 5
        assert abs((y1 + y2) / 2 - 0.5 * H) < 5

    def test_roi_grows_with_torso_length(self):
        small = hand_roi_box(self._anchor(), torso_length_norm=0.15, frame_w=W, frame_h=H, aspect=1.0)
        large = hand_roi_box(self._anchor(), torso_length_norm=0.45, frame_w=W, frame_h=H, aspect=1.0)
        area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
        assert area(large) > area(small)

    def test_roi_grows_with_hand_speed(self):
        still = hand_roi_box(self._anchor(), torso_length_norm=0.3, frame_w=W, frame_h=H, aspect=1.0,
                              hand_speed_diag_per_s=0.0)
        fast = hand_roi_box(self._anchor(), torso_length_norm=0.3, frame_w=W, frame_h=H, aspect=1.0,
                             hand_speed_diag_per_s=3.0)
        area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
        assert area(fast) > area(still)

    def test_roi_is_clamped_to_the_frame(self):
        box = hand_roi_box(self._anchor(0.02, 0.02), torso_length_norm=0.5, frame_w=W, frame_h=H, aspect=1.0)
        x1, y1, x2, y2 = box
        assert x1 >= 0 and y1 >= 0 and x2 <= W and y2 <= H

    def test_a_degenerate_torso_reading_still_produces_a_usable_roi(self):
        box = hand_roi_box(self._anchor(), torso_length_norm=0.0, frame_w=W, frame_h=H, aspect=1.0)
        x1, y1, x2, y2 = box
        assert x2 > x1 and y2 > y1


# ------------------------------------------------------------------ association scoring
def _tracked_box(tracker: BoxTracker, label: str, positions: list[tuple[float, float, float, float]],
                  dt: float = 0.1, conf: float = 0.9):
    """Feed a sequence of boxes for one label through a tracker; return the
    final TrackedBox (with a real trail, for velocity/direction)."""
    track = None
    t = 0.0
    for box in positions:
        track = tracker.update([(label, box, conf)], now=t)[0]
        t += dt
    return track


class TestAssociationScore:
    def test_overlapping_and_moving_together_scores_high(self):
        tracker = BoxTracker()
        hand = _tracked_box(tracker, "right_hand",
                             [(100 + 20 * i, 100, 180 + 20 * i, 180) for i in range(6)])
        obj = _tracked_box(tracker, "phone",
                            [(110 + 20 * i, 110, 150 + 20 * i, 150) for i in range(6)])
        assert association_score(hand, obj) > 0.7

    def test_hand_passing_a_stationary_object_scores_low(self):
        """The nearest-innocent-motion case this must reject: proximity alone
        (even a moment of overlap) is not enough if the object never moves
        with the hand."""
        tracker = BoxTracker()
        hand = _tracked_box(tracker, "right_hand",
                             [(50 + 30 * i, 100, 130 + 30 * i, 180) for i in range(6)])
        obj = _tracked_box(tracker, "cup", [(150, 120, 190, 160)] * 6)  # stationary the whole time
        assert association_score(hand, obj) < 0.5

    def test_far_apart_unrelated_boxes_score_low(self):
        tracker = BoxTracker()
        hand = _tracked_box(tracker, "right_hand", [(0, 0, 40, 40)] * 4)
        obj = _tracked_box(tracker, "bottle", [(500, 400, 540, 440)] * 4)
        assert association_score(hand, obj) < 0.3

    def test_both_stationary_and_close_scores_moderately_supportive(self):
        """A held object can be held still -- this must not score as low as
        the 'hand passed a stationary object' case above."""
        tracker = BoxTracker()
        hand = _tracked_box(tracker, "right_hand", [(100, 100, 180, 180)] * 6)
        obj = _tracked_box(tracker, "bottle", [(140, 130, 180, 170)] * 6)  # overlapping, both still
        score = association_score(hand, obj)
        assert score > 0.5

    def test_score_is_bounded_zero_to_one(self):
        tracker = BoxTracker()
        hand = _tracked_box(tracker, "right_hand", [(0, 0, 400, 400)] * 3)
        obj = _tracked_box(tracker, "bottle", [(0, 0, 400, 400)] * 3)
        assert 0.0 <= association_score(hand, obj) <= 1.0


# ------------------------------------------------------------------ the full engine
class _FakeYoloBox:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = [np.array(xyxy, dtype=np.float32)]
        self.cls = [np.float32(cls)]
        self.conf = [np.float32(conf)]


class _FakeYoloResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeYoloModel:
    """A scriptable stand-in for get_yolo_model()'s return value: each call to
    predict() consumes one entry from `responses` (a list of (label_xyxy_in_
    crop_coords, class_id, confidence) tuples), matching how predict_roi
    receives a CROP and expects crop-relative coordinates back."""

    def __init__(self):
        self.responses: list[list[tuple[tuple[int, int, int, int], int, float]]] = []
        self.calls = 0

    def predict(self, frame, **kwargs):
        boxes = []
        if self.calls < len(self.responses):
            for xyxy, cls, conf in self.responses[self.calls]:
                boxes.append(_FakeYoloBox(xyxy, cls, conf))
        self.calls += 1
        return [_FakeYoloResult(boxes)]


OBJECT_LABELS = {39: "bottle", 67: "cell phone"}


class TestHandPerceptionEngine:
    def test_no_hands_produces_an_empty_result(self):
        engine = HandPerceptionEngine(OBJECT_LABELS)
        model = _FakeYoloModel()
        lm = _standing_with_hands()
        result = engine.update(lm, FRAME, model, now=0.0, aspect=1.0)
        assert result.hands == []
        assert result.associations == {}

    def test_one_hand_with_no_nearby_object_has_no_association(self):
        engine = HandPerceptionEngine(OBJECT_LABELS)
        model = _FakeYoloModel()
        model.responses = [[]]  # ROI crop finds nothing
        lm = _standing_with_hands(right_wrist=(0.65, 0.45))
        result = engine.update(lm, FRAME, model, now=0.0, aspect=1.0)
        assert len(result.hands) == 1
        assert result.associations == {}

    def test_a_confidently_classified_object_moving_with_the_hand_associates(self):
        """The hand moves fast enough per step to cross adaptive scheduling's
        own re-trigger threshold (a real carry/pickup motion would), so the
        object gets multiple fresh detections -- not one frozen box the hand
        then drifts away from, which would understate motion agreement for
        reasons that have nothing to do with whether it's actually held."""
        engine = HandPerceptionEngine(OBJECT_LABELS, association_threshold=0.5)
        model = _FakeYoloModel()
        # Same crop-relative small box each stride tick (moves with the ROI,
        # which itself follows the wrist) -- classified as a phone at high confidence.
        model.responses = [[((5, 5, 25, 25), 67, 0.9)]] * 6
        wrist_positions = [(0.5 + 0.1 * i, 0.5) for i in range(6)]
        result = None
        for i, (wx, wy) in enumerate(wrist_positions):
            lm = _standing_with_hands(right_wrist=(wx, wy))
            result = engine.update(lm, FRAME, model, now=i * 0.1, aspect=1.0)
        assert model.calls > 1, "test setup: the hand must have moved enough to re-trigger ROI inference"
        assert "right_hand" in result.associations
        obj_track, score = result.associations["right_hand"]
        assert obj_track.label == "cell phone"
        assert score >= 0.5

    def test_an_unclassified_or_low_confidence_detection_becomes_unknown_object(self):
        engine = HandPerceptionEngine(OBJECT_LABELS, detection_confidence=0.4, unknown_object_confidence_floor=0.1)
        model = _FakeYoloModel()
        # class_id 99 is not in OBJECT_LABELS at all -- must not be dropped, must
        # not be hallucinated into "bottle"/"cell phone" either.
        model.responses = [[((5, 5, 25, 25), 99, 0.6)]] * 4
        lm = _standing_with_hands(right_wrist=(0.5, 0.5))
        result = None
        for i in range(4):
            result = engine.update(lm, FRAME, model, now=i * 0.1, aspect=1.0)
        tracked_labels = {t.label for t in result.objects}
        assert UNKNOWN_OBJECT in tracked_labels
        assert "bottle" not in tracked_labels and "cell phone" not in tracked_labels

    def test_a_low_confidence_known_class_still_reports_unknown_not_the_guessed_class(self):
        """Detector confidence below detection_confidence must not be silently
        promoted to a confident class label -- that would be exactly the
        'hallucinating an object class' the spec explicitly forbids."""
        engine = HandPerceptionEngine(OBJECT_LABELS, detection_confidence=0.6, unknown_object_confidence_floor=0.1)
        model = _FakeYoloModel()
        model.responses = [[((5, 5, 25, 25), 67, 0.25)]] * 4  # "cell phone" class, but low confidence
        lm = _standing_with_hands(right_wrist=(0.5, 0.5))
        result = None
        for i in range(4):
            result = engine.update(lm, FRAME, model, now=i * 0.1, aspect=1.0)
        tracked_labels = {t.label for t in result.objects}
        assert UNKNOWN_OBJECT in tracked_labels
        assert "cell phone" not in tracked_labels

    def test_a_detection_below_the_unknown_floor_is_dropped_entirely(self):
        engine = HandPerceptionEngine(OBJECT_LABELS, unknown_object_confidence_floor=0.3)
        model = _FakeYoloModel()
        model.responses = [[((5, 5, 25, 25), 67, 0.05)]]  # below even the unknown floor
        lm = _standing_with_hands(right_wrist=(0.5, 0.5))
        result = engine.update(lm, FRAME, model, now=0.0, aspect=1.0)
        assert result.objects == []

    def test_both_hands_tracked_and_associated_independently(self):
        engine = HandPerceptionEngine(OBJECT_LABELS, association_threshold=0.5)
        model = _FakeYoloModel()
        model.responses = [[((5, 5, 25, 25), 39, 0.9)], [((5, 5, 25, 25), 67, 0.9)]] * 6
        result = None
        for i in range(6):
            lm = _standing_with_hands(left_wrist=(0.3 + 0.005 * i, 0.5), right_wrist=(0.7 + 0.005 * i, 0.5))
            result = engine.update(lm, FRAME, model, now=i * 0.1, aspect=1.0)
        assert set(result.associations.keys()) <= {"left_hand", "right_hand"}

    def test_adaptive_roi_scheduling_skips_a_stationary_unchanged_hand(self):
        """Lightweight tracking runs every call; the (relatively) expensive
        ROI-crop inference should NOT re-run every single frame for a hand
        that hasn't moved -- see HandPerceptionEngine._should_run_roi."""
        engine = HandPerceptionEngine(OBJECT_LABELS, roi_min_move_diag=10.0, roi_max_interval_s=100.0)
        model = _FakeYoloModel()
        model.responses = [[]] * 10
        lm = _standing_with_hands(right_wrist=(0.5, 0.5))  # perfectly stationary across calls
        for i in range(5):
            engine.update(lm, FRAME, model, now=i * 0.1, aspect=1.0)
        # First call must run ROI (nothing cached yet); a stationary hand
        # well under both thresholds should not trigger it again.
        assert model.calls == 1

    def test_a_moving_hand_retriggers_roi_inference(self):
        engine = HandPerceptionEngine(OBJECT_LABELS, roi_min_move_diag=0.2, roi_max_interval_s=100.0)
        model = _FakeYoloModel()
        model.responses = [[]] * 10
        for i in range(5):
            lm = _standing_with_hands(right_wrist=(0.5 + 0.05 * i, 0.5))  # moves each frame
            engine.update(lm, FRAME, model, now=i * 0.1, aspect=1.0)
        assert model.calls > 1

    def test_reset_clears_hand_and_object_tracks(self):
        engine = HandPerceptionEngine(OBJECT_LABELS)
        model = _FakeYoloModel()
        model.responses = [[]]
        lm = _standing_with_hands(right_wrist=(0.5, 0.5))
        engine.update(lm, FRAME, model, now=0.0, aspect=1.0)
        engine.reset()
        assert engine._hand_tracker.tracks == []
        assert engine._object_tracker.tracks == []
