"""M10 deterministic temporal regression suite: named scenario timelines through
the PRODUCTION DriverAttentionModule, with explicit expected events and latencies.

SYNTHETIC SCENARIOS ONLY (tests/m10_temporal_harness.py): constructed per-frame
signals (head pose, eyelid gap, face present / lost, camera stalls, head-pose
failures, detector and hand boxes) on injected frame times, with the SHIPPED
configuration. Every scenario runs at 30, 10 and 5 fps and, at 30 and 10 fps,
again with +/-30 % frame-interval jitter. The result is what the temporal logic
does on constructed inputs -- not a measured accuracy on real drivers.

Only the alertable, non-experimental event types are checked (ALERTABLE). An
expectation is exhaustive over them: an alertable event a scenario does not name
must not fire. Latencies are from the start of the named segment (the onset of
the condition) to the frame the event fired on, bounded below by the configured
sustain time and above by it plus the signal median filter's lag (the median of
5 frames flips on the 3rd) and one frame of sampling, i.e. 4 frame intervals.

docs/TEMPORAL_REASONING.md has the same table measured on base 4dfd761 and here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

import pytest
from m10_temporal_harness import LOWERED, SHUT, Result, Seg, face, run
from test_behavior_integration import LAP_HAND, LAP_PHONE
from test_m8_driver_attention_defects import MOUNTED_IN_VIEW, WHEEL_HANDS

#: Alertable (main.ALERT_EVENT_TYPES_PI), minus the experimental, non-audible families.
ALERTABLE = frozenset({"drowsiness", "fatigue_warning", "distraction", "frequent_glancing", "eyes_down",
                       "eyes_up", "phone_use", "attention_lost"})

#: (fps, jitter) every scenario runs at unless it pins its own.
RATES: tuple[tuple[float, float], ...] = ((30.0, 0.0), (10.0, 0.0), (5.0, 0.0), (30.0, 0.3), (10.0, 0.3))


@dataclass(frozen=True)
class Expect:
    """`event` fires exactly `count` times (or at most `at_most`); its first firing
    is `latency` (lo, hi) seconds after segment `seg` starts, hi + the lag bound."""

    event: str
    count: Optional[int] = 1
    seg: int = 1
    latency: Optional[tuple[float, float]] = None
    at_most: Optional[int] = None
    rates: Optional[tuple[float, ...]] = None  # only at these fps (a known frame-rate dependence)


@dataclass(frozen=True)
class Scenario:
    name: str
    what: str
    segments: Callable[[float], list[Seg]]  # fps -> timeline (fps for per-frame patterns)
    expect: tuple[Expect, ...] = ()
    rates: tuple[tuple[float, float], ...] = RATES
    exhaustive: bool = True
    notes: str = ""
    kwargs: dict = field(default_factory=dict)


def _blinks(seconds: float, closed: float, period: float, lid: float = SHUT) -> list[Seg]:
    out: list[Seg] = []
    for _ in range(round(seconds / period)):
        out += [Seg(closed, face(lid=lid)), Seg(period - closed, face())]
    return out


def _glances(n: int, yaw: float, on: float, off: float) -> list[Seg]:
    out: list[Seg] = []
    for _ in range(n):
        out += [Seg(on, face(yaw=yaw)), Seg(off, face())]
    return out


def _per_frame(fps: float, seconds: float, a: dict, b: dict) -> list[Seg]:
    """Alternate two signals on every frame (threshold-boundary chatter)."""
    step = 1.0 / fps
    return [Seg(step, a if i % 2 == 0 else b) for i in range(round(seconds * fps))]


def _hover(seconds: float, lo: dict, hi: dict, period: float) -> list[Seg]:
    out: list[Seg] = []
    for _ in range(round(seconds / period)):
        out += [Seg(period / 2, hi), Seg(period / 2, lo)]
    return out


#: A closure of 2 s+ in these short sessions also lifts PERCLOS over 0.15: allowed, once.
PERCLOS_TOO = Expect("fatigue_warning", count=None, at_most=1)

ONE_FRAME = 1e-3  # a segment this short gets exactly the one frame at its start after a stall

SCENARIOS: tuple[Scenario, ...] = (
    # ------------------------------------------------------------ eyes
    Scenario("blinks_normal", "0.15 s blinks every 4 s for 24 s",
             lambda fps: [Seg(1.0, face()), *_blinks(24.0, 0.15, 4.0)]),
    Scenario("microsleep_1_2s", "one 1.2 s closure (below drowsy_seconds 2.0)",
             lambda fps: [Seg(2.0, face()), Seg(1.2, face(lid=SHUT)), Seg(3.0, face())],
             notes="A sub-2 s microsleep raises nothing by design: drowsy_seconds is 2.0."),
    Scenario("sustained_closure", "4 s closure after 10 s open",
             lambda fps: [Seg(10.0, face()), Seg(4.0, face(lid=SHUT)), Seg(2.0, face())],
             (Expect("drowsiness", latency=(2.0, 2.0)), Expect("fatigue_warning", latency=(2.0, 4.0)))),
    Scenario("heavy_long_blinks", "0.7 s closures every 2 s for 30 s (35 % closed)",
             lambda fps: [Seg(10.0, face()), *_blinks(30.0, 0.7, 2.0)],
             (Expect("fatigue_warning", latency=(0.0, 30.0), rates=(30.0, 10.0)),),
             notes="Frame-rate dependent (open item): the 5-frame EAR median erases most of a 0.7 s "
                   "closure at 5 fps (3.5 frames), so PERCLOS stays under 0.15 and nothing fires."),
    Scenario("look_down_lowered_lids", "30 deg look into the lap; lids lowered to 0.67 x open, iris visible",
             lambda fps: [Seg(2.0, face()), Seg(5.0, face(pitch=-30, lid=LOWERED)), Seg(2.0, face())],
             (Expect("eyes_down", latency=(2.5, 2.5)),)),
    Scenario("nod_off_head_down", "lids over the iris with the head pitched 15 deg down",
             lambda fps: [Seg(2.0, face()), Seg(4.0, face(pitch=-15, lid=SHUT)), Seg(2.0, face())],
             (Expect("drowsiness", latency=(2.0, 2.0)), PERCLOS_TOO)),
    Scenario("closure_with_one_frame_glitches", "4 s closure with one open (landmark-glitch) frame in 5",
             lambda fps: [Seg(2.0, face()),
                          *[s for _ in range(round(4.0 * fps / 5)) for s in (Seg(4.0 / fps, face(lid=SHUT)),
                                                                             Seg(1.0 / fps, face()))],
                          Seg(2.0, face())],
             (Expect("drowsiness", latency=(2.0, 2.6)), PERCLOS_TOO),
             notes="One episode, never reset; each glitch frame PAUSES the closure clock (the EAR median "
                   "hides it, the per-frame iris veto does not), so 2 s of closure takes ~2.5 s."),
    Scenario("ear_chatter_at_threshold", "eyelid alternating shut / open on every frame for 6 s",
             lambda fps: [Seg(2.0, face()), *_per_frame(fps, 6.0, face(lid=SHUT), face()), Seg(2.0, face())],
             (Expect("drowsiness", count=None, at_most=1), Expect("fatigue_warning", count=None, at_most=1))),
    # ------------------------------------------------------------ gaze
    Scenario("mirror_checks", "three 1 s mirror checks at 60 deg, 8 s apart",
             lambda fps: [Seg(2.0, face()), *_glances(3, 60.0, 1.0, 8.0)]),
    Scenario("blind_spot_check_4s", "a 4 s wide-angle (65 deg) turn: within rearview_tolerance_seconds 5",
             lambda fps: [Seg(2.0, face()), Seg(4.0, face(yaw=65)), Seg(2.0, face())]),
    Scenario("blind_spot_check_7s", "a 7 s wide-angle (65 deg) turn",
             lambda fps: [Seg(2.0, face()), Seg(7.0, face(yaw=65)), Seg(2.0, face())],
             (Expect("distraction", latency=(5.0, 5.0)),)),
    Scenario("distraction_sustained", "45 deg for 5 s",
             lambda fps: [Seg(2.0, face()), Seg(5.0, face(yaw=45)), Seg(2.0, face())],
             (Expect("distraction", latency=(3.0, 3.0)),)),
    Scenario("distraction_recovery", "2 s away, 1 s back on the road, 2 s away: no partial credit",
             lambda fps: [Seg(2.0, face()), Seg(2.0, face(yaw=45)), Seg(1.0, face()), Seg(2.0, face(yaw=45)),
                          Seg(2.0, face())]),
    Scenario("distraction_brief_return", "2 s away, a 0.2 s flick back, 1.5 s away (inside the release grace)",
             lambda fps: [Seg(2.0, face()), Seg(2.0, face(yaw=45)), Seg(0.2, face()), Seg(1.5, face(yaw=45)),
                          Seg(2.0, face())],
             (Expect("distraction", latency=(3.0, 3.0)),)),
    Scenario("frequent_glancing", "five 1.2 s glances at 45 deg, 3 s apart",
             lambda fps: [Seg(2.0, face()), *_glances(5, 45.0, 1.2, 3.0)],
             (Expect("frequent_glancing", seg=9, latency=(0.0, 0.0)),)),
    Scenario("look_up", "20+ deg up (visor) for 4 s",
             lambda fps: [Seg(2.0, face()), Seg(4.0, face(pitch=30)), Seg(2.0, face())],
             (Expect("eyes_up", latency=(2.5, 2.5)),)),
    Scenario("yaw_chatter_at_threshold", "yaw alternating 33 / 37 deg on every frame for 6 s",
             lambda fps: [Seg(2.0, face()), *_per_frame(fps, 6.0, face(yaw=33), face(yaw=37)), Seg(2.0, face())],
             (Expect("distraction", count=None, at_most=1), Expect("frequent_glancing", count=0))),
    Scenario("yaw_hover_at_threshold_1hz", "yaw 33 / 37 deg, 0.5 s each, for 10 s",
             lambda fps: [Seg(2.0, face()), *_hover(10.0, face(yaw=33), face(yaw=37), 1.0), Seg(2.0, face())],
             (Expect("distraction", count=None, at_most=1), Expect("frequent_glancing", count=None, at_most=1)),
             notes="Frame-rate dependent (open item): 0.5 s below the threshold is just past the 0.4 s "
                   "release grace, so 30 fps counts five 'glances' and 5-10 fps bridges them into one "
                   "distraction. Pinned only as: no event twice."),
    # ------------------------------------------------------------ phone
    Scenario("phone_handled", "a phone held in the lap, looked down at for 4 s",
             lambda fps: [Seg(2.0, face(), phones=[LAP_PHONE], hands=[LAP_HAND]), Seg(4.0, face(pitch=-30)),
                          Seg(2.0, face())],
             (Expect("phone_use", latency=(2.0, 2.0)), Expect("eyes_down", latency=(2.5, 2.5)))),
    Scenario("phone_mounted", "a still, mounted phone, hands on the wheel; a 4 s look down",
             lambda fps: [Seg(3.0, face(), phones=[MOUNTED_IN_VIEW], hands=WHEEL_HANDS),
                          Seg(4.0, face(pitch=-30)), Seg(2.0, face())],
             (Expect("eyes_down", latency=(2.5, 2.5)),)),
    Scenario("phone_use_through_pose_failure", "phone_use live, then 1 s unreadable head pose, then more",
             lambda fps: [Seg(2.0, face(), phones=[LAP_PHONE], hands=[LAP_HAND]), Seg(3.5, face(pitch=-30)),
                          Seg(1.0, face(pitch=-30), pose_fail=True), Seg(3.0, face(pitch=-30)), Seg(1.0, face())],
             (Expect("phone_use", latency=(2.0, 2.0)), Expect("eyes_down", latency=(2.5, 2.5)))),
    # ------------------------------------------------------------ missing evidence
    Scenario("face_lost_brief", "1 s with no face and no person",
             lambda fps: [Seg(2.0, face()), Seg(1.0, None, person=False), Seg(2.0, face(), person=True)]),
    Scenario("face_lost_long_no_body", "4 s with no face and no person",
             lambda fps: [Seg(2.0, face()), Seg(4.0, None, person=False), Seg(2.0, face(), person=True)],
             (Expect("attention_lost", latency=(2.5, 2.5)),)),
    Scenario("face_lost_long_with_body", "10 s with no face, a person still detected",
             lambda fps: [Seg(2.0, face()), Seg(10.0, None), Seg(2.0, face())],
             (Expect("attention_lost", latency=(8.0, 8.0)),)),
    Scenario("gaze_unknown_mid_turn", "1.5 s turned, 5 s head pose unreadable, then turned again",
             lambda fps: [Seg(1.0, face()), Seg(1.5, face(yaw=45)), Seg(5.0, face(yaw=45), pose_fail=True),
                          Seg(3.0, face(yaw=45)), Seg(1.0, face())],
             (Expect("distraction", seg=3, latency=(1.3, 1.5)),)),
    Scenario("gaze_unknown_look_back", "1.5 s turned, 5 s unreadable (facing the road), 0.4 s turned",
             lambda fps: [Seg(1.0, face()), Seg(1.5, face(yaw=45)), Seg(5.0, face(), pose_fail=True),
                          Seg(0.4, face(yaw=45)), Seg(2.0, face())]),
    Scenario("face_lost_brief_mid_closure", "1.5 s shut, face lost 0.3 s (body seen), 1.0 s shut",
             lambda fps: [Seg(2.0, face()), Seg(1.5, face(lid=SHUT)), Seg(0.3, None), Seg(1.0, face(lid=SHUT)),
                          Seg(2.0, face())],
             (Expect("drowsiness", count=None, at_most=1),),
             notes="Frame-rate dependent by design (open item): face loss feeds the closure tracker "
                   "'not closed', bridged for eye_closure_release_frames (2) frames -- 0.07 s at 30 fps, "
                   "0.4 s at 5 fps. The shadow engine coasts instead (core/behavior/eye_state.py)."),
    # ------------------------------------------------------------ stalls and frame rate
    Scenario("stall_three_closed_frames", "three closed frames, 3 s camera stalls between them",
             lambda fps: [Seg(ONE_FRAME, face(lid=SHUT)), Seg(3.0, stall=True), Seg(ONE_FRAME, face(lid=SHUT)),
                          Seg(3.0, stall=True), Seg(ONE_FRAME, face(lid=SHUT)), Seg(2.0, face())]),
    Scenario("stall_mid_closure_short", "1 s shut, 3 s stall, 0.5 s shut: 1.5 s observed",
             lambda fps: [Seg(2.0, face()), Seg(1.0, face(lid=SHUT)), Seg(3.0, stall=True),
                          Seg(0.5, face(lid=SHUT)), Seg(2.0, face())]),
    Scenario("stall_mid_closure_long", "1.5 s shut, 3 s stall, 1.5 s shut: fires on fresh frames",
             lambda fps: [Seg(2.0, face()), Seg(1.5, face(lid=SHUT)), Seg(3.0, stall=True),
                          Seg(1.5, face(lid=SHUT)), Seg(2.0, face())],
             (Expect("drowsiness", seg=3, latency=(0.3, 0.6)), PERCLOS_TOO)),
    Scenario("stall_mid_face_lost", "1.5 s no face / no person, 5 s stall, 0.5 s no face",
             lambda fps: [Seg(2.0, face()), Seg(1.5, None, person=False), Seg(5.0, stall=True),
                          Seg(0.5, None), Seg(2.0, face(), person=True)]),
    Scenario("stall_mid_distraction", "1.5 s turned, 5 s stall, 0.3 s turned",
             lambda fps: [Seg(2.0, face()), Seg(1.5, face(yaw=45)), Seg(5.0, stall=True), Seg(0.3, face(yaw=45)),
                          Seg(2.0, face())]),
    Scenario("frame_rate_drop_30_to_5", "closure starting at 30 fps, the camera dropping to 5 fps after 1 s",
             lambda fps: [Seg(2.0, face()), Seg(1.0, face(lid=SHUT)), Seg(3.0, face(lid=SHUT), fps=5.0),
                          Seg(2.0, face(), fps=5.0)],
             (Expect("drowsiness", latency=(2.0, 2.3)), PERCLOS_TOO), rates=((30.0, 0.0), (30.0, 0.3))),
    Scenario("frame_rate_drop_30_to_2", "the same, dropping to 2 fps (0.5 s frames, the gap bound)",
             lambda fps: [Seg(2.0, face()), Seg(1.0, face(lid=SHUT)), Seg(5.0, face(lid=SHUT), fps=2.0),
                          Seg(2.0, face(), fps=2.0)],
             (Expect("drowsiness", latency=(2.0, 3.0)), PERCLOS_TOO), rates=((30.0, 0.0),)),
)

BY_NAME = {s.name: s for s in SCENARIOS}


def lag(fps: float, jitter: float) -> float:
    return 4.0 * (1.0 + jitter) / fps + 0.02


def play(scenario: Scenario, tmp_path, monkeypatch, fps: float, jitter: float) -> Result:
    return run(tmp_path, monkeypatch, scenario.segments(fps), fps=fps, jitter=jitter, seed=17,
               **scenario.kwargs)


def check(scenario: Scenario, result: Result, fps: float, jitter: float) -> list[str]:
    """Every violated expectation, as text (empty: the scenario behaves as specified)."""
    problems: list[str] = []
    fired = [e for e in result.events if e.type in ALERTABLE]
    named = set()
    for exp in scenario.expect:
        if exp.rates is not None and fps not in exp.rates:
            named.add(exp.event)
            continue
        named.add(exp.event)
        hits = [e for e in fired if e.type == exp.event]
        if exp.count is not None and len(hits) != exp.count:
            problems.append(f"{exp.event}: fired {len(hits)}x, expected {exp.count}")
        if exp.at_most is not None and len(hits) > exp.at_most:
            problems.append(f"{exp.event}: fired {len(hits)}x, at most {exp.at_most}")
        if exp.latency is not None and hits:
            got = hits[0].t - result.seg_starts[exp.seg]
            lo, hi = exp.latency
            if not lo - 1e-6 <= got <= hi + lag(fps, jitter):
                problems.append(f"{exp.event}: latency {got:.2f}s outside [{lo:.2f}, {hi + lag(fps, jitter):.2f}]")
    if scenario.exhaustive:
        extra = sorted({e.type for e in fired} - named)
        if extra:
            problems.append(f"unexpected: {', '.join(extra)}")
    # No scenario here holds two real episodes of one condition: whatever the
    # frame rate, jitter, chatter or gaps, no alertable type fires twice.
    types = [e.type for e in fired]
    twice = sorted({t for t in types if types.count(t) > 1})
    if twice:
        problems.append(f"fired more than once: {', '.join(twice)}")
    return problems


def describe(scenario: Scenario, result: Result) -> str:
    """The alertable events fired, with latency from the first expectation's segment."""
    fired = [e for e in result.events if e.type in ALERTABLE]
    if not fired:
        return "none"
    seg_of = {exp.event: exp.seg for exp in scenario.expect}
    parts = []
    for e in fired:
        start = result.seg_starts[seg_of.get(e.type, 1)] if len(result.seg_starts) > 1 else 0.0
        parts.append(f"{e.type}@{e.t - start:+.2f}s")
    return ", ".join(parts)


CASES = [pytest.param(s, fps, jitter, id=f"{s.name}-{fps:g}fps" + (f"-jitter{jitter:g}" if jitter else ""))
         for s in SCENARIOS for fps, jitter in s.rates]


@pytest.mark.parametrize(("scenario", "fps", "jitter"), CASES)
def test_temporal_scenario(scenario: Scenario, fps: float, jitter: float, tmp_path, monkeypatch):
    result = play(scenario, tmp_path, monkeypatch, fps, jitter)
    problems = check(scenario, result, fps, jitter)
    where = f"{scenario.name} @ {fps:g} fps (jitter {jitter:g})"
    assert not problems, f"{where}: {problems}; fired: {describe(scenario, result)}"
