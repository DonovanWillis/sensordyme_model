"""M13 flagship distracted-driving demonstration: the 10 scenarios, run and judged.

SIMULATED, EVERY LINE. Scenarios 1-8 run the REAL DriverAttentionModule at the
SHIPPED config.yaml thresholds (sim.perception.run_cabin: scripted FaceMesh,
detector and hands; calibration, every tracker, phone handling, driver
association and event emission are the production code). Scenarios 9-10 run
the virtual device (sim.harness.SimStack: the real runtime, uplink and backend
on one SimClock) through a camera freeze and its recovery. Nothing here is a
measured accuracy on real cabins, faces, lenses or a Pi; it shows what the
shipped logic does on constructed inputs, stationary, never real driving.

Used by tests/test_m13_flagship_scenarios.py (the pins) and
scripts/demo/flagship_rehearsal.py (the table). docs/DEMO_SCENARIOS.md says how
to rehearse each one on the real Pi.

A requirement the current product does not meet is reported as a FAIL tagged
with a KNOWN_GAPS id -- never tuned away. Thresholds and policies are the
shipped ones; changing them to make a row pass is out of bounds.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from sim.perception import CabinScenario, CabinScene, run_cabin
from sim.scenario import _cabin

REPO = Path(__file__).resolve().parent.parent
MODULE_HARNESS = "SIMULATED (module harness)"
VIRTUAL_DEVICE = "SIMULATED (virtual device)"
#: The validated driver alerts (main.ALERT_EVENT_TYPES_PI minus the experimental
#: observations): what "an alert" means in every expectation below.
ALERTABLE = frozenset({"drowsiness", "fatigue_warning", "distraction", "frequent_glancing", "eyes_down",
                       "eyes_up", "phone_use", "attention_lost"})
#: Frame rate of the table (the M8.4/M8.5 cabin suites' rate); the test also runs 15 and 30.
FPS = 10.0


def lag(fps: float) -> float:
    """Upper latency slack over a sustain time: the 5-frame median flips on the
    3rd frame, plus one frame of sampling (tests/test_m10_temporal_scenarios.py)."""
    return 4.0 / fps + 0.02


# ====================================================================== known gaps
@dataclass(frozen=True)
class Gap:
    id: str
    title: str
    measured: str
    classification: str  # "defect" | "limitation"
    fix: str


KNOWN_GAPS: dict[str, Gap] = {g.id: g for g in (
    Gap("G1", "A passenger's phone carries no overlay role (expected role 'other')",
        "core/overlay.py OverlayBuilder sets a phone's role only when the driver has a phone of their own "
        "(`elif d.label == \"cell phone\" and driver_phones`). With the driver associated and only the "
        "passenger's phone in view, the phone box is sent with NO role: not mis-attributed to the driver, "
        "but not marked 'other' either, so the app cannot tell 'not the driver's' from 'undecided'.",
        "defect (overlay labelling; no alert or attribution decision is affected)",
        "core/overlay.py: `elif d.label == \"cell phone\" and (driver_phones or driver_box is not None):` "
        "-- a phone is 'other' whenever a driver was associated and it is not one of theirs. Not made here: "
        "core/overlay.py is outside this mission's write scope."),
    Gap("G2", "distraction / drowsiness / eyes_down events carry an empty contributing_signals",
        "The module writes contributing_signals only for the special paths (a wide-angle turn, a face-lost "
        "extrapolation, phone handling, an uncalibrated closure). The ordinary distraction, drowsiness and "
        "eyes_down events say why through their TYPE and DURATION (the sustained seconds) only; "
        "signals is \"\", so the overlay's 'why' line (last_event.signals) and the history's detail are blank.",
        "limitation (the alert is correct and explained by type + duration; the free-text reason is missing)",
        "modules/driver_attention.py: write a measured one-line reason on these events (e.g. 'head turned "
        "40 deg from your forward pose for 3.0 s', 'eyes closed 2.0 s'). Needs an owner decision on wording "
        "because jupiter/explain.py turns module signals into facts and the app shows them."),
)}


@dataclass
class Problem:
    text: str
    gap: Optional[str] = None  # a KNOWN_GAPS id: the requirement is not met, and that is recorded


# ====================================================================== scenarios 1-8
_ROAD = CabinScene(label="attentive")


@dataclass(frozen=True)
class Case:
    """One module-harness flagship scenario and what it must show."""

    number: int
    key: str
    title: str
    expected: str
    scenario: CabinScenario
    onset: float  # scenario time the behaviour under test starts (latency origin)
    #: (event type, lo, hi): must fire exactly once, lo <= latency <= hi + lag. Exhaustive over ALERTABLE.
    alerts: tuple[tuple[str, float, float], ...] = ()
    #: alertable types that may fire at most once besides (e.g. PERCLOS after a long closure).
    allowed: tuple[str, ...] = ()
    #: "type:phase" the overlay must show, in this order.
    story: tuple[str, ...] = ()
    #: behaviour types the overlay must never show.
    never: tuple[str, ...] = ()
    #: the overlay role every phone box must carry while visible (None: no phone expected).
    phone_role: Optional[str] = None
    #: (start, end) scenario seconds the face is hidden: the evidence must say so.
    occluded: Optional[tuple[float, float]] = None


CASES: tuple[Case, ...] = (
    Case(1, "normal_forward_attention", "Normal forward attention",
         "no alert; face, eyes and head pose measured on every frame; driver identified",
         _cabin("m13_normal", "Attentive driver, hands on the wheel, 20 s.", 20.0), onset=0.0,
         never=("distraction", "eyes_down", "eyes_up", "phone_use", "drowsiness", "face_not_visible")),
    Case(2, "short_ordinary_glance", "Short ordinary glance",
         "distraction developing for the glance, never confirmed; no alert",
         _cabin("m13_glance", "One 1.5 s glance 40 deg sideways (distraction needs 3.0 s).", 10.0,
                (3.0, CabinScene(yaw_deg=40.0)), (4.5, _ROAD)), onset=3.0,
         story=("distraction:developing",), never=("phone_use", "drowsiness")),
    Case(3, "prolonged_attention_away", "Prolonged attention away",
         "distraction developing -> confirmed; one loud distraction alert at ~3.0 s",
         _cabin("m13_away", "Head turned 40 deg away for 6 s.", 10.0,
                (2.0, CabinScene(yaw_deg=40.0)), (8.0, _ROAD)), onset=2.0,
         alerts=(("distraction", 3.0, 3.0),),
         story=("distraction:developing", "distraction:confirmed"), never=("phone_use",)),
    Case(4, "phone_visible_in_mount", "Phone visible in a mount",
         "no alert, no phone_use developing; the mounted phone is the driver's but not in use",
         _cabin("m13_mount", "A phone in a dash mount, hands on the wheel; eyes on the road with one 1.2 s "
                "glance down at it.", 15.0,
                (0.0, CabinScene(phone="mount")), (8.0, CabinScene(phone="mount", head_down_deg=25.0)),
                (9.2, CabinScene(phone="mount"))), onset=0.0,
         story=("eyes_down:developing",), never=("phone_use", "drowsiness"), phone_role="driver"),
    Case(5, "phone_held", "Phone held",
         "phone_use developing -> confirmed (hand on the phone) at ~2.0 s and eyes_down at ~2.5 s, both loud",
         _cabin("m13_held", "Holds a phone in the lap and looks down at it for 6 s.", 10.0,
                (0.0, CabinScene(phone="lap", hand_on_phone=True)),
                (2.0, CabinScene(phone="lap", hand_on_phone=True, head_down_deg=30.0)),
                (8.0, CabinScene(phone="lap", hand_on_phone=True))), onset=2.0,
         alerts=(("phone_use", 2.0, 2.0), ("eyes_down", 2.5, 2.5)),
         story=("phone_use:developing", "phone_use:confirmed"), never=("drowsiness",), phone_role="driver"),
    Case(6, "passenger_using_phone", "Passenger using a phone",
         "no alert; the passenger's phone is never the driver's (role 'other', no phone_use developing)",
         _cabin("m13_passenger", "The passenger holds a phone; the driver watches the road, with one 1 s "
                "glance across.", 15.0,
                (0.0, CabinScene(passenger="seated", phone="passenger", passenger_hand_on_phone=True)),
                (8.0, CabinScene(passenger="seated", phone="passenger", passenger_hand_on_phone=True,
                                 yaw_deg=40.0)),
                (9.0, CabinScene(passenger="seated", phone="passenger", passenger_hand_on_phone=True))),
         onset=0.0, never=("phone_use", "drowsiness"), phone_role="other"),
    Case(7, "eyes_closed", "Driver's eyes closed",
         "drowsiness developing -> confirmed; one loud (critical) drowsiness alert at ~2.0 s",
         _cabin("m13_eyes", "Eyes closed for 4 s.", 10.0, (2.0, CabinScene(eyes_closed=True)), (6.0, _ROAD)),
         onset=2.0, alerts=(("drowsiness", 2.0, 2.0),), allowed=("fatigue_warning",),
         story=("drowsiness:developing", "drowsiness:confirmed"), never=("phone_use", "eyes_down")),
    Case(8, "briefly_occluded", "Driver briefly occluded",
         "no alert (body still seen: attention_lost waits 8 s); the face is reported not visible and the "
         "face rules abstain while it is hidden",
         _cabin("m13_occluded", "An arm passes in front of the face for 3 s; the body stays in view.", 10.0,
                (2.0, CabinScene(face=False)), (5.0, _ROAD)), onset=2.0,
         never=("face_not_visible", "phone_use"), occluded=(2.0, 5.0)),
)
BY_NUMBER = {c.number: c for c in CASES}


@dataclass
class FrameView:
    """What the module showed after one frame: its overlay half, through the real OverlayBuilder."""

    t: float
    state: str
    behaviours: tuple[tuple[str, str, Optional[float]], ...]
    face_visible: Optional[bool]
    head_direction: Optional[str]
    phone_roles: tuple[Optional[str], ...]  # one per phone box the overlay sends
    driver_phones: int
    driver_identity: str
    abstaining: str


@dataclass
class CaseResult:
    case: Case
    fps: float
    events: list[dict]
    frames: list[FrameView]
    problems: list[Problem] = field(default_factory=list)

    @property
    def alerts(self) -> list[dict]:
        return [e for e in self.events if e["type"] in ALERTABLE]

    def latency(self, event_type: str) -> Optional[float]:
        hit = next((e for e in self.alerts if e["type"] == event_type), None)
        return None if hit is None else round(hit["t"] - self.case.onset, 2)


def shipped_alert_policy() -> dict:
    """config.yaml's alert severities and the driver_safety role (core/roles.py)."""
    from core.roles import config_violations, role_named

    config = yaml.safe_load((REPO / "config.yaml").read_text())
    role = role_named("driver_safety")
    alerts = config.get("alerts", {})
    return {"loud": frozenset(alerts.get("loud_types") or ()), "soft": frozenset(alerts.get("soft_types") or ()),
            "mandatory": frozenset(role.mandatory_alert_types), "critical": frozenset(role.critical_types),
            "floor_violations": config_violations({**config, "role": "driver_safety"})}


def severity(event_type: str, policy: dict) -> str:
    """services/runtime.py's severity tag for a non-experimental event."""
    if event_type in policy["loud"]:
        return "loud"
    return "soft" if event_type in policy["soft"] or event_type in ALERTABLE else "log"


def _observation(module, t: float, frame_id: int):
    """The shared perception engine's Observation for this frame, as the runtime builds
    it (core/perception.py end_frame): the module's own boxes, its face signals."""
    from core import perception as pc

    frame = module.get_detections()
    signals = module.perception_signals()
    face_found = module._last_face_proto is not None
    face = (pc.FaceObservation(468, eye_state=signals.get("eye_state"), head_pose=signals.get("head_pose"),
                               gaze=signals.get("gaze")) if face_found else None)
    ok = {name: pc.ComponentStatus(name=name, state="ok") for name in (pc.OBJECT_DETECTOR, pc.FACE_LANDMARKS)}
    detections = tuple(pc.ObjectDetection(d.label, float(d.confidence), tuple(float(v) for v in d.box))
                       for d in frame.detections)
    return pc.Observation(
        frame_id=frame_id, captured_at=t, source="sim", simulated=True, width=frame.width, height=frame.height,
        detections=detections, detections_age_s=0.0, fresh_detections=True, detection_consumers=frozenset(),
        face_visible=face_found, face=face, hands=None, components=ok)


def run_case(case: Case, tmp_dir: Optional[str] = None, fps: float = FPS) -> CaseResult:
    """Play one scenario through the real module and record what it did and showed."""
    from core.overlay import OverlayBuilder

    frames: list[FrameView] = []
    builder = OverlayBuilder(clock=lambda: 0.0)

    def observe(t: float, module, _produced) -> None:
        extras = module.overlay_extras()
        doc = builder.build(_observation(module, t, len(frames)), extras, module="driver_attention",
                            mode="driver_safety", captured_at=0.0)
        telemetry = module.get_telemetry_snapshot() or {}
        face = doc.get("face")
        frames.append(FrameView(
            t=t, state=str(doc.get("state")),
            behaviours=tuple((b["type"], b["phase"], b.get("for_s")) for b in doc.get("behaviours") or ()),
            face_visible=None if face is None else bool(face.get("visible")),
            head_direction=((face or {}).get("head_pose") or {}).get("direction"),
            phone_roles=tuple(d.get("role") for d in doc.get("det") or () if d["label"] == "phone"),
            driver_phones=len(extras.get("driver_phone_boxes") or ()),
            driver_identity=str(telemetry.get("driver_identity")),
            abstaining=str(telemetry.get("engine_abstaining") or "")))

    with tempfile.TemporaryDirectory() if tmp_dir is None else nullcontext(tmp_dir) as where:
        trace = run_cabin(case.scenario, where, fps=fps, on_frame=observe)
    result = CaseResult(case, fps, trace["events"], frames)
    result.problems = check_case(result)
    return result


def _story(frames: list[FrameView]) -> list[str]:
    """The overlay's behaviour phases as they changed: ["distraction:developing", ...]."""
    out: list[str] = []
    live: set[str] = set()
    for f in frames:
        now = {f"{kind}:{phase}" for kind, phase, _ in f.behaviours}
        out += sorted(now - live)
        live = now
    return out


def check_case(result: CaseResult) -> list[Problem]:
    """Every expectation of the case that the run did not meet (empty: PASS)."""
    case, fps = result.case, result.fps
    problems: list[Problem] = []
    policy = shipped_alert_policy()
    fired = result.alerts
    # 1. alerts only when the policy is satisfied: exactly the expected ones, at the shipped timing.
    named = {a[0] for a in case.alerts}
    for event_type, lo, hi in case.alerts:
        hits = [e for e in fired if e["type"] == event_type]
        if len(hits) != 1:
            problems.append(Problem(f"{event_type} fired {len(hits)}x, expected once"))
            continue
        got = hits[0]["t"] - case.onset
        if not lo - 1e-6 <= got <= hi + lag(fps):
            problems.append(Problem(f"{event_type} latency {got:.2f}s outside [{lo:.2f}, {hi + lag(fps):.2f}]"))
    for event_type in case.allowed:
        if sum(1 for e in fired if e["type"] == event_type) > 1:
            problems.append(Problem(f"{event_type} fired more than once"))
    extra = sorted({e["type"] for e in fired} - named - set(case.allowed))
    if extra:
        problems.append(Problem(f"unexpected alert(s): {', '.join(extra)}"))
    # 2. the role's mandatory safety policy: the shipped floor holds, mandatory alerts are loud.
    if policy["floor_violations"]:
        problems.append(Problem(f"shipped config breaks the driver_safety floor: {policy['floor_violations']}"))
    for e in fired:
        if e["type"] in policy["mandatory"] and severity(e["type"], policy) != "loud":
            problems.append(Problem(f"{e['type']} is mandatory for driver_safety but would not be loud"))
    # 3. why: the event's own contributing_signals.
    for e in fired:
        if e["type"] in named and not str(e.get("signals") or "").strip():
            problems.append(Problem(f"{e['type']} carries no contributing_signals", gap="G2"))
    # 4. the overlay tells the behaviour's story, and never one that is not happening.
    story = _story(result.frames)
    position = 0
    for wanted in case.story:
        try:
            position = story.index(wanted, position) + 1
        except ValueError:
            problems.append(Problem(f"overlay never showed {wanted} (in order); showed {story}"))
            break
    shown = {s.split(":")[0] for s in story}
    for kind in case.never:
        if kind in shown:
            problems.append(Problem(f"overlay showed {kind} ({[s for s in story if s.startswith(kind)]})"))
    # 5. attribution: whose phone it is.
    roles = {r for f in result.frames for r in f.phone_roles}
    if case.phone_role is not None:
        if not any(f.phone_roles for f in result.frames):
            problems.append(Problem("no phone box ever reached the overlay"))
        elif roles != {case.phone_role}:
            gap = "G1" if case.phone_role == "other" and roles == {None} else None
            problems.append(Problem(f"phone role(s) {sorted(roles, key=str)}, expected {case.phone_role!r}", gap=gap))
    if case.phone_role == "other" and any(f.driver_phones for f in result.frames):
        problems.append(Problem("the passenger's phone was counted as the driver's phone"))
    # 6. evidence: insufficient evidence is reported where it is insufficient.
    if case.occluded is not None:
        start, end = case.occluded
        hidden = [f for f in result.frames if start + 0.15 <= f.t < end]
        if not hidden or any(f.face_visible is not False for f in hidden):
            problems.append(Problem("the hidden face was not reported not visible on every frame"))
        if any(not {"distraction", "drowsiness", "eyes_down"} <= set(f.abstaining.split(";")) for f in hidden):
            problems.append(Problem("the face rules did not abstain while the face was hidden"))
        back = [f for f in result.frames if f.t >= end + 0.5]
        if not back or any(f.face_visible is not True for f in back):
            problems.append(Problem("the face was not reported visible again after the occlusion"))
    else:
        if any(f.face_visible is not True for f in result.frames):
            problems.append(Problem("the face was reported not visible in a scenario where it never is"))
    if any(f.driver_identity != "identified" for f in result.frames):
        problems.append(Problem(f"driver identity not 'identified': {sorted({f.driver_identity for f in result.frames})}"))
    return problems


def describe_case(result: CaseResult) -> dict:
    """The table row's columns (plain strings) plus the raw facts, JSON-ready."""
    case = result.case
    alerts = [f"{e['type']}@+{e['t'] - case.onset:.1f}s" for e in result.alerts]
    log_only = sorted({e["type"] for e in result.events if e["type"] not in ALERTABLE
                       and not e["type"].endswith("_risk")})
    story = _story(result.frames)
    roles = sorted({r or "none" for f in result.frames for r in f.phone_roles})
    hidden = [f for f in result.frames if f.face_visible is False]
    if hidden:
        abstained = sorted({a for f in hidden for a in f.abstaining.split(";") if a})
        evidence = (f"face not visible {len(hidden) / result.fps:.1f}s; face rules abstained "
                    f"({', '.join(abstained) or 'none'})")
    else:
        evidence = "face measured every frame"
    identities = sorted({f.driver_identity for f in result.frames})
    attribution = f"driver {'/'.join(identities)}"
    if roles:
        attribution += f"; phone role {'/'.join(roles)}; driver's phone {'yes' if any(f.driver_phones for f in result.frames) else 'no'}"
    first = result.alerts[0] if result.alerts else None
    why = [f"{e['type']}: {e['signals']!r}" for e in result.alerts if e["type"] in {a[0] for a in case.alerts}]
    return {
        "number": case.number, "scenario": case.title, "key": case.key, "harness": MODULE_HARNESS,
        "fps": result.fps, "expected": case.expected,
        "events": ", ".join(alerts) or "no alert",
        "log_only_events": log_only,
        "phases": " -> ".join(story) or "none",
        "attribution": attribution, "evidence": evidence,
        "why": why,
        "latency_s": None if first is None else round(first["t"] - case.onset, 2),
        "problems": [{"text": p.text, "gap": p.gap} for p in result.problems],
        "verdict": verdict(result.problems),
        "raw_events": [{k: e[k] for k in ("t", "type", "signals")} for e in result.events],
    }


def verdict(problems: list[Problem]) -> str:
    if not problems:
        return "PASS"
    gaps = sorted({p.gap for p in problems if p.gap})
    if all(p.gap for p in problems):
        return f"FAIL (known gap {', '.join(gaps)})"
    return "FAIL"


# ====================================================================== scenarios 9-10
#: Timeline on the device's clock (seconds since boot): calibrate, freeze, restore, a closure.
FREEZE_AT = 20.0
RESTORE_AT = 40.0
CLOSE_EYES_AT = 52.0
DEVICE_RUN_S = 64.0
#: Upper bounds the expectations allow (the production cadences, not tuned): a frame
#: is overdue after 2 s, the monitoring check runs every 2 s, the uplink every 2 s.
DETECT_BOUND_S = 4.5
APP_BOUND_S = 6.5
RESTORE_BOUND_S = 4.5


@dataclass
class DeviceSample:
    t: float
    device_state: Optional[str]
    device_protection: str
    app_state: str
    app_protection: str
    app_active: bool
    reasons: list[str]
    buzzer: int


@dataclass
class DeviceResult:
    samples: list[DeviceSample]
    events: list[dict]
    buzzer: list[tuple]
    local_events: list[dict]
    problems9: list[Problem] = field(default_factory=list)
    problems10: list[Problem] = field(default_factory=list)

    def first(self, predicate: Callable[[DeviceSample], bool], after: float) -> Optional[float]:
        return next((s.t for s in self.samples if s.t >= after and predicate(s)), None)


def device_scenario():
    from sim.scenario import ASLEEP, ATTENTIVE, Faults, Scenario, Segment

    return Scenario("m13_camera_unavailable_then_restored",
                    "Attentive; the camera stops delivering frames for 20 s; it recovers; the driver then "
                    "closes their eyes for 4 s.",
                    (Segment(0.0, ATTENTIVE), Segment(FREEZE_AT, ATTENTIVE, Faults(camera="frozen")),
                     Segment(RESTORE_AT, ATTENTIVE), Segment(CLOSE_EYES_AT, ASLEEP),
                     Segment(CLOSE_EYES_AT + 4.0, ATTENTIVE)))


def run_device(root: Optional[str] = None, step: float = 0.25) -> DeviceResult:
    """Scenarios 9 and 10 on the virtual device + the real backend, sampled every `step` s."""
    from sim.harness import SimStack

    with tempfile.TemporaryDirectory() if root is None else nullcontext(root) as where, \
            SimStack(Path(where)) as stack:
        device = stack.boot(device_scenario(), dashboard_port=0)
        stack.claim()
        t0 = stack.clock.monotonic()
        samples: list[DeviceSample] = []
        while stack.clock.monotonic() - t0 < DEVICE_RUN_S - 1e-6:
            device.advance(step)
            runtime = device.runtime
            status = stack.status()
            samples.append(DeviceSample(
                t=round(stack.clock.monotonic() - t0, 2), device_state=runtime._monitoring_state,
                device_protection=runtime.protection_state()["state"],
                app_state=status["monitoring_state"], app_protection=status["protection"],
                app_active=bool(status["monitoring_active"]), reasons=list(status.get("reasons") or []),
                buzzer=len(device.buzzer.played)))
        events = stack.events()
        page, _ = runtime.dashboard.events_response({"limit": ["200"]})
        result = DeviceResult(samples, events, list(device.buzzer.played), list(page["events"]))
    result.problems9, result.problems10 = check_device(result)
    return result


def check_device(r: DeviceResult) -> tuple[list[Problem], list[Problem]]:
    p9: list[Problem] = []
    p10: list[Problem] = []
    before = [s for s in r.samples if FREEZE_AT - 4.0 <= s.t < FREEZE_AT]
    if not before or any(s.app_state != "ready" or s.app_protection != "protecting" for s in before):
        p9.append(Problem("the device was not READY/protecting before the camera failed"))
    detected = r.first(lambda s: s.device_state == "failed", FREEZE_AT)
    app_told = r.first(lambda s: s.app_protection == "not_protecting", FREEZE_AT)
    if detected is None or detected - FREEZE_AT > DETECT_BOUND_S:
        p9.append(Problem(f"the device did not report FAILED within {DETECT_BOUND_S}s (got {detected})"))
    if app_told is None or app_told - FREEZE_AT > APP_BOUND_S:
        p9.append(Problem(f"the app was not told 'not protecting' within {APP_BOUND_S}s (got {app_told})"))
    during = [s for s in r.samples if FREEZE_AT + APP_BOUND_S <= s.t < RESTORE_AT]
    if not during or any(s.app_protection != "not_protecting" or s.app_active for s in during):
        p9.append(Problem("the app showed protecting / monitoring active while the camera was down"))
    if any(not any("camera" in reason for reason in s.reasons) for s in during):
        p9.append(Problem("the app's reasons did not name the camera"))
    # never a false "protecting": the app never says protecting while the device is not READY
    if any(s.app_protection == "protecting" and s.app_state not in ("ready", "degraded") for s in r.samples):
        p9.append(Problem("the app said 'protecting' while monitoring was not running"))
    if any(s.device_state == "failed" and s.app_protection == "protecting" and s.t > FREEZE_AT + APP_BOUND_S
           for s in r.samples):
        p9.append(Problem("the app said 'protecting' while the device said FAILED"))
    # The only driver alert in this run is the real closure after recovery:
    # anything alertable before it came from the outage (stale frames).
    types = [e["type"] for e in r.events]
    first_drowsy = types.index("drowsiness") if "drowsiness" in types else len(types)
    stale = [t for t in types[:first_drowsy] if t in ALERTABLE]
    if stale:
        p9.append(Problem(f"driver alert(s) before the real closure (from stale frames?): {stale}"))
    if "monitoring_impaired" not in types:
        p9.append(Problem("no monitoring_impaired event reached the app"))
    if not any(b[0] == "monitoring_impaired" for b in r.buzzer):
        p9.append(Problem("the in-cabin output did not tell the driver monitoring is impaired"))

    ready = r.first(lambda s: s.device_state == "ready", RESTORE_AT)
    app_ready = r.first(lambda s: s.app_state == "ready" and s.app_protection == "protecting", RESTORE_AT)
    if ready is None or ready - RESTORE_AT > RESTORE_BOUND_S:
        p10.append(Problem(f"the device was not READY within {RESTORE_BOUND_S}s of the camera returning ({ready})"))
    if app_ready is None or app_ready - RESTORE_AT > RESTORE_BOUND_S + 2.0:
        p10.append(Problem(f"the app was not shown protecting again in time ({app_ready})"))
    if ready is not None and ready >= CLOSE_EYES_AT:
        p10.append(Problem("monitoring came back only after the eye closure started"))
    after = [s for s in r.samples if (app_ready or DEVICE_RUN_S) <= s.t]
    if any(s.app_protection != "protecting" for s in after):
        p10.append(Problem("protection dropped again after it was restored"))
    drowsy = [e for e in r.events if e["type"] == "drowsiness"]
    if len(drowsy) != 1:
        p10.append(Problem(f"drowsiness fired {len(drowsy)}x after recovery, expected once"))
    else:
        alert = drowsy[0].get("alert") or {}
        if drowsy[0].get("severity") != "loud" or alert.get("decision") != "alerted":
            p10.append(Problem(f"drowsiness was not a loud, alerted event: {drowsy[0].get('severity')}, {alert}"))
        if not drowsy[0].get("simulated"):
            p10.append(Problem("the simulated drowsiness event is not labelled simulated"))
        if not str(drowsy[0].get("signals") or "").strip():
            p10.append(Problem("drowsiness carries no contributing_signals", gap="G2"))
    if not any(b[0] == "drowsiness" for b in r.buzzer):
        p10.append(Problem("the buzzer path did not sound for drowsiness after recovery"))
    if not any(e["type"] == "drowsiness" for e in r.local_events):
        p10.append(Problem("the device's local history (/api/events) has no drowsiness row"))
    return p9, p10


def drowsy_latency(r: DeviceResult) -> Optional[float]:
    """Seconds from the eyes closing to the buzzer sounding for drowsiness (device side)."""
    for s in r.samples:
        if s.t >= CLOSE_EYES_AT and any(b[0] == "drowsiness" for b in r.buzzer[:s.buzzer]):
            return round(s.t - CLOSE_EYES_AT, 2)
    return None


def _transitions(r: DeviceResult, start: float, end: float, origin: float) -> str:
    """The app's monitoring_state/protection as it changed in [start, end), times from `origin`."""
    out: list[str] = []
    last = None
    for s in r.samples:
        if not start <= s.t < end:
            continue
        label = f"{s.app_state}/{s.app_protection}"
        if label != last:
            out.append(label if not out else f"{label} (+{s.t - origin:.1f}s)")
        last = label
    return " -> ".join(out) or "-"


def describe_device(r: DeviceResult) -> list[dict]:
    detected = r.first(lambda s: s.device_state == "failed", FREEZE_AT)
    app_told = r.first(lambda s: s.app_protection == "not_protecting", FREEZE_AT)
    ready = r.first(lambda s: s.device_state == "ready", RESTORE_AT)
    app_ready = r.first(lambda s: s.app_state == "ready" and s.app_protection == "protecting", RESTORE_AT)
    reason = next((s.reasons[0] for s in r.samples if s.t > FREEZE_AT + APP_BOUND_S and s.reasons), "")
    outage_states = "app: " + _transitions(r, FREEZE_AT - 1.0, RESTORE_AT, FREEZE_AT)
    recovery_states = "app: " + _transitions(r, RESTORE_AT - 1.0, DEVICE_RUN_S, RESTORE_AT)
    types = [e["type"] for e in r.events]
    drowsy = next((e for e in r.events if e["type"] == "drowsiness"), {})
    nine = {
        "number": 9, "scenario": "Camera unavailable", "key": "camera_unavailable", "harness": VIRTUAL_DEVICE,
        "expected": "device FAILED + app 'not protecting' naming the camera within the check cadence; "
                    "no driver alert from stale frames; the driver is told (impaired tone)",
        "events": ", ".join(t for t in dict.fromkeys(types) if t in ("monitoring_impaired",)) or "none",
        "phases": outage_states,
        "attribution": "n/a (no frames)",
        "evidence": f"app reason: {reason[:90]}",
        "why": [f"monitoring_impaired: {next((e.get('signals') for e in r.events if e['type'] == 'monitoring_impaired'), '')!r}"],
        "latency_s": None if detected is None else round(detected - FREEZE_AT, 2),
        "latency_detail": {"device_failed_s": None if detected is None else round(detected - FREEZE_AT, 2),
                           "app_not_protecting_s": None if app_told is None else round(app_told - FREEZE_AT, 2)},
        "problems": [{"text": p.text, "gap": p.gap} for p in r.problems9],
        "verdict": verdict(r.problems9),
    }
    ten = {
        "number": 10, "scenario": "Monitoring restored", "key": "monitoring_restored", "harness": VIRTUAL_DEVICE,
        "expected": "READY/protecting again without recalibration; a later eye closure alerts loud through "
                    "the buzzer, in the app and in the device's local history",
        "events": ", ".join(f"{t}" for t in dict.fromkeys(types) if t in ALERTABLE) or "none",
        "phases": recovery_states,
        "attribution": "driver identified (single occupant)",
        "evidence": (f"device READY +{ready - RESTORE_AT:.1f}s, app protecting +{app_ready - RESTORE_AT:.1f}s"
                     if ready is not None and app_ready is not None else "not restored"),
        "why": [f"drowsiness: {drowsy.get('signals')!r} severity {drowsy.get('severity')}"] if drowsy else [],
        "latency_s": drowsy_latency(r),
        "latency_detail": {"device_ready_s": None if ready is None else round(ready - RESTORE_AT, 2),
                           "app_protecting_s": None if app_ready is None else round(app_ready - RESTORE_AT, 2),
                           "drowsiness_buzzer_s": drowsy_latency(r)},
        "problems": [{"text": p.text, "gap": p.gap} for p in r.problems10],
        "verdict": verdict(r.problems10),
    }
    return [nine, ten]


def run_all(fps: float = FPS) -> list[dict]:
    """All ten rows, in order."""
    rows = [describe_case(run_case(case, fps=fps)) for case in CASES]
    rows += describe_device(run_device())
    return rows


def as_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): as_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_json_safe(v) for v in value]
    return value
