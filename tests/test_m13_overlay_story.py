"""M13: the driver module's overlay half says what the module is actually timing. SIMULATED.

Before this fix DriverAttentionModule.overlay_extras() (sd-overlay/1's
behaviours and head pose, docs/LIVE_OVERLAY.md):

(a) showed phone_use "developing" for ANY visible phone -- a mounted one, a
    passenger's -- though the phone_use tracker was never fed True for it;
(b) showed drowsiness "developing" for a downward look whose lowered lids the
    module itself attributes to looking down (no closure is being timed);
(c) sent the RAW Euler head angles under `calibrated: true` -- an upright head
    read pitch ~ +-180, a 30-degree look down +150 -- where the contract says
    calibrated angles are relative to this driver's forward pose.

The phases now come from the conditions the trackers are fed, and the head
pose from this frame's perception signals. No alert, threshold or decision
changed (the module's events are untouched; the M10 / M8.5 suites pin them).
Synthetic inputs through the real module.
"""

from __future__ import annotations

from m10_temporal_harness import LOWERED, Seg, face, run

from sim.perception import CabinScene, run_cabin
from sim.scenario import _cabin

PASSENGER = CabinScene(passenger="seated", phone="passenger", passenger_hand_on_phone=True)


def _phases(scenario, tmp_path) -> set[str]:
    seen: set[str] = set()

    def observe(_t, module, _events):
        seen.update(f"{b['type']}:{b['phase']}" for b in module.overlay_extras()["behaviours"])

    run_cabin(scenario, str(tmp_path), on_frame=observe)
    return seen


def test_a_passengers_or_mounted_phone_is_never_phone_use_developing(tmp_path):
    passenger = _cabin("passenger", "A passenger holds a phone; the driver watches the road.", 8.0, (0.0, PASSENGER))
    assert not {p for p in _phases(passenger, tmp_path / "p") if p.startswith("phone_use")}  # was developing
    mount = _cabin("mount", "A mounted phone, eyes on the road.", 8.0, (0.0, CabinScene(phone="mount")))
    assert not {p for p in _phases(mount, tmp_path / "m") if p.startswith("phone_use")}


def test_a_held_phone_looked_at_still_develops_then_confirms(tmp_path):
    held = _cabin("held", "A phone held in the lap, looked at for 4 s.", 8.0,
                  (0.0, CabinScene(phone="lap", hand_on_phone=True)),
                  (2.0, CabinScene(phone="lap", hand_on_phone=True, head_down_deg=30.0)))
    assert {"phone_use:developing", "phone_use:confirmed"} <= _phases(held, tmp_path)


def test_a_downward_look_is_eyes_down_not_drowsiness_developing(tmp_path, monkeypatch):
    r = run(tmp_path, monkeypatch, [Seg(2.0, face()), Seg(1.5, face(pitch=-30, lid=LOWERED))], fps=10.0)
    kinds = [b["type"] for b in r.module.overlay_extras()["behaviours"]]
    assert kinds == ["eyes_down"], kinds  # was ["eyes_down", "drowsiness"]


def test_a_real_closure_still_develops_drowsiness(tmp_path, monkeypatch):
    from m10_temporal_harness import SHUT

    r = run(tmp_path, monkeypatch, [Seg(2.0, face()), Seg(1.0, face(lid=SHUT))], fps=10.0)
    assert {"type": "drowsiness", "phase": "developing"}.items() <= next(
        b for b in r.module.overlay_extras()["behaviours"] if b["type"] == "drowsiness").items()


def test_head_pose_is_relative_to_the_drivers_forward_pose(tmp_path, monkeypatch):
    forward = run(tmp_path, monkeypatch, [Seg(2.0, face())], fps=10.0).module.overlay_extras()["head"]
    assert forward["calibrated"] is True and abs(forward["yaw_deg"]) < 2 and abs(forward["pitch_deg"]) < 2, forward
    down = run(tmp_path, monkeypatch, [Seg(1.0, face()), Seg(1.5, face(pitch=-30))], fps=10.0)
    head = down.module.overlay_extras()["head"]
    assert head["direction"] == "Down" and head["calibrated"] is True
    assert -35 < head["pitch_deg"] < -25, head  # was +150 (raw Euler)
    signals = down.module.perception_signals()["head_pose"]
    assert (head["yaw_deg"], head["pitch_deg"]) == (round(signals.yaw_deg, 1), round(signals.pitch_deg, 1))
