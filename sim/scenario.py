"""Scenario timelines: what the virtual device sees, and what is broken, when.

A scenario is a list of segments on the scenario clock (seconds since the
virtual device first booted, scaled by --speed in live mode). Each segment
names a Scene (sim.perception) and the Faults in force. Boot-time conditions
that no timeline can express -- a model that cannot be loaded, a missing
audible output -- are BootFaults.

Every driver scenario opens with >= 12 s of an attentive driver: the real
driver_attention module calibrates its per-driver baselines over the first
10 s (config.yaml calibration_duration_s), exactly as on the Pi.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Optional

from sim.perception import Scene

ATTENTIVE = Scene()
ASLEEP = Scene(eyes_closed=True, label="eyes closed")
LOOKING_DOWN = Scene(head_down_deg=30.0, label="looking down")
PHONE_IN_LAP = Scene(head_down_deg=30.0, phone=True, hand_on_phone=True, label="phone in hand, looking down")
LOOKING_AWAY = Scene(yaw_deg=45.0, label="looking away")
ABSENT = Scene(face=False, person=False, label="driver absent")

#: Seconds of attentive driving every driver scenario starts with (calibration).
CALIBRATION_S = 12.0


@dataclass(frozen=True)
class Faults:
    """What is broken at one moment. Default: nothing."""

    #: ok | frozen (no frames at all) | dropping (every other frame lost)
    #: | stalled (frames arrive in bursts with multi-second gaps)
    camera: str = "ok"
    #: The object detector raises on every call (the module retries with backoff).
    detector_failing: bool = False
    #: The backend is unreachable from the device (uplink requests fail).
    network_down: bool = False
    #: M8.3: the SoC temperature the device reads (None: no sensor, as on a Mac),
    #: and the firmware's live throttle flags (core/device_health.py names).
    soc_temp_c: Optional[float] = None
    throttle: tuple[str, ...] = ()

    def describe(self) -> list[str]:
        out = []
        if self.soc_temp_c is not None:
            out.append(f"SoC {self.soc_temp_c:.1f}°C")
        if self.throttle:
            out.append(f"throttle: {', '.join(self.throttle)}")
        if self.camera != "ok":
            out.append(f"camera {self.camera}")
        if self.detector_failing:
            out.append("detector failing")
        if self.network_down:
            out.append("network down")
        return out


@dataclass(frozen=True)
class BootFaults:
    """Conditions of the boot itself, applied when the device (re)starts."""

    #: The module's model cannot be loaded: main.py boots FAILED (UnavailableModule).
    model_failure: bool = False
    #: No audible alert output (the buzzer could not be set up).
    no_buzzer: bool = False


@dataclass(frozen=True)
class Segment:
    start: float
    scene: Scene = ATTENTIVE
    faults: Faults = Faults()


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    segments: tuple[Segment, ...] = (Segment(0.0),)
    boot: BootFaults = BootFaults()
    #: Scenario times at which the device process restarts (live mode only;
    #: tests restart explicitly).
    reboots: tuple[float, ...] = ()
    #: Driver-attention perception is scripted; anything else is not simulated.
    tags: tuple[str, ...] = field(default_factory=tuple)

    def at(self, t: float) -> Segment:
        """The segment in force at scenario time `t` (the last one holds for ever)."""
        current = self.segments[0]
        for segment in self.segments:
            if segment.start <= t:
                current = segment
            else:
                break
        return current

    @property
    def duration(self) -> float:
        return self.segments[-1].start

    def with_boot(self, **changes) -> "Scenario":
        return replace(self, boot=replace(self.boot, **changes))


def _timeline(*pairs) -> tuple[Segment, ...]:
    """(start, scene[, faults]) tuples -> segments, attentive calibration first."""
    segments = [Segment(0.0)]
    for pair in pairs:
        start, scene, *rest = pair
        segments.append(Segment(float(start), scene, rest[0] if rest else Faults()))
    return tuple(sorted(segments, key=lambda s: s.start))


C = CALIBRATION_S
SCENARIOS: dict[str, Scenario] = {s.name: s for s in (
    Scenario("attentive", "An attentive driver for ever: the device should reach and stay READY."),
    Scenario("drowsy", "Eyes close for 4 s, reopen, then close again for 4 s (drowsiness, twice).",
             _timeline((C, ATTENTIVE), (C + 3, ASLEEP), (C + 7, ATTENTIVE), (C + 22, ASLEEP),
                       (C + 26, ATTENTIVE))),
    Scenario("phone", "Looks down at a phone held in the lap for 6 s (phone_use + eyes_down).",
             _timeline((C + 3, PHONE_IN_LAP), (C + 9, ATTENTIVE))),
    Scenario("distracted", "Head turned 45 degrees away for 5 s (distraction).",
             _timeline((C + 3, LOOKING_AWAY), (C + 8, ATTENTIVE))),
    Scenario("absent", "The driver leaves the seat for 10 s (attention_lost).",
             _timeline((C + 3, ABSENT), (C + 13, ATTENTIVE))),
    Scenario("camera_freeze", "The camera stops delivering frames for 20 s, then recovers "
                              "(FAILED with the reason, then READY).",
             _timeline((C + 5, ATTENTIVE, Faults(camera="frozen")), (C + 25, ATTENTIVE))),
    Scenario("frame_drops", "Every other frame is lost for 30 s (monitoring must hold READY).",
             _timeline((C + 5, ATTENTIVE, Faults(camera="dropping")), (C + 35, ATTENTIVE))),
    Scenario("camera_stall", "Frames arrive in bursts with 4 s gaps for 30 s (stale frames).",
             _timeline((C + 5, ATTENTIVE, Faults(camera="stalled")), (C + 35, ATTENTIVE))),
    Scenario("detector_failure", "The object detector throws for 40 s, then works again "
                                 "(DEGRADED: phone detection lost, face monitoring kept; then READY).",
             _timeline((C + 5, ATTENTIVE, Faults(detector_failing=True)), (C + 45, ATTENTIVE))),
    Scenario("backend_outage", "The backend is unreachable for 60 s: the device keeps monitoring "
                               "offline and the app sees the status go stale, then fresh.",
             _timeline((C + 5, ATTENTIVE, Faults(network_down=True)), (C + 65, ATTENTIVE))),
    Scenario("model_failure", "The perception model cannot be loaded at boot (FAILED, reachable).",
             boot=BootFaults(model_failure=True)),
    Scenario("no_buzzer", "No audible alert output on a driver_safety device (FAILED).",
             boot=BootFaults(no_buzzer=True)),
    Scenario("reboot", "The device process restarts at 40 s: identity, pairing, mode and rules persist.",
             reboots=(40.0,)),
    Scenario("tour", "Everything in one run, for an app demo: drowsy, phone, distracted, detector "
                     "failure, camera freeze, backend outage, each followed by recovery.",
             _timeline(
                 (C + 5, ASLEEP), (C + 9, ATTENTIVE),
                 (C + 20, PHONE_IN_LAP), (C + 26, ATTENTIVE),
                 (C + 36, LOOKING_AWAY), (C + 41, ATTENTIVE),
                 (C + 50, ATTENTIVE, Faults(detector_failing=True)), (C + 80, ATTENTIVE),
                 (C + 95, ATTENTIVE, Faults(camera="frozen")), (C + 110, ATTENTIVE),
                 (C + 125, ATTENTIVE, Faults(network_down=True)), (C + 170, ATTENTIVE),
             )),
)}


def get(name: str) -> Scenario:
    try:
        return SCENARIOS[name]
    except KeyError:
        raise KeyError(f"unknown scenario {name!r}; known: {', '.join(sorted(SCENARIOS))}") from None


class ScenarioClock:
    """Scenario time, from a monotonic clock: (now - origin) * speed."""

    def __init__(self, monotonic: Callable[[], float], speed: float = 1.0,
                 origin: Optional[float] = None) -> None:
        if speed <= 0:
            raise ValueError("speed must be > 0")
        self._monotonic = monotonic
        self.speed = float(speed)
        self.origin = monotonic() if origin is None else origin

    def now(self) -> float:
        return max(0.0, (self._monotonic() - self.origin) * self.speed)


class ScenarioState:
    """The scenario at the scenario clock's current time, plus manual overrides.

    Overrides let a test or an operator force a scene or fault regardless of
    the timeline (e.g. "take the network down now").
    """

    def __init__(self, scenario: Scenario, clock: ScenarioClock) -> None:
        self.scenario = scenario
        self.clock = clock
        self.scene_override: Optional[Scene] = None
        self.fault_overrides: dict = {}

    def segment(self) -> Segment:
        return self.scenario.at(self.clock.now())

    def scene(self) -> Scene:
        return self.scene_override or self.segment().scene

    def current(self) -> Faults:
        faults = self.segment().faults
        return replace(faults, **self.fault_overrides) if self.fault_overrides else faults


# ====================================================================== M8.4 cabin scenarios
# Deterministic behavioural regression scenarios (sim.perception.run_cabin):
# false-positive traps the engine must not alert on, abstention cases it must
# REPORT, and true scenarios that must keep alerting exactly as before. Times
# are seconds after the attentive calibration lead-in. Labels are the script's
# own ground truth -- SYNTHETIC regression evidence, never real-world accuracy.
from sim.perception import CabinLabel, CabinScenario, CabinScene, CabinSegment  # noqa: E402

_ROAD = CabinScene(label="attentive")


def _cabin(name: str, description: str, duration: float, *steps, kind: str = "false_positive",
           expected=(), allowed=(), seed: int = 7) -> CabinScenario:
    segments = [CabinSegment(0.0, _ROAD)] + [CabinSegment(float(start), scene) for start, scene in steps]
    segments.sort(key=lambda s: s.start)
    return CabinScenario(name, description, tuple(segments), float(duration),
                         CabinLabel(tuple(expected), tuple(allowed)), kind, seed)


_MOUNT = CabinScene(phone="mount", label="phone in a dash mount")
_MOUNT_NO_HANDS = CabinScene(phone="mount", wheel_hands=False, label="phone in a mount, hands not seen")
_PASSENGER = CabinScene(passenger="seated", phone="passenger", passenger_hand_on_phone=True,
                        label="passenger holding a phone")
_PASSENGER_LEANING = CabinScene(passenger="leaning", phone="passenger", passenger_hand_on_phone=True,
                                label="passenger (largest box) holding a phone")

CABIN_SCENARIOS: dict[str, CabinScenario] = {s.name: s for s in (
    # ---------------------------------------------------------------- true scenarios
    _cabin("eyes_closed_3s", "Eyes closed for 3 s.", 8.0,
           (2.0, CabinScene(eyes_closed=True)), (5.0, _ROAD),
           kind="true_positive", expected=[("drowsiness", 2.0, 5.0)], allowed=["fatigue_warning"]),
    _cabin("sustained_look_away", "Head turned 45 degrees away for 5 s.", 10.0,
           (2.0, CabinScene(yaw_deg=45.0)), (7.0, _ROAD),
           kind="true_positive", expected=[("distraction", 2.0, 7.0)]),
    _cabin("sustained_look_down", "Looks down 30 degrees for 4 s (no phone).", 8.0,
           (2.0, CabinScene(head_down_deg=30.0)), (6.0, _ROAD),
           kind="true_positive", expected=[("eyes_down", 2.0, 6.0)]),
    _cabin("held_phone_eyes_off_road", "Holds a phone in the lap and looks down at it for 6 s.", 10.0,
           (0.0, CabinScene(phone="lap", hand_on_phone=True)),
           (2.0, CabinScene(phone="lap", hand_on_phone=True, head_down_deg=30.0)),
           (8.0, CabinScene(phone="lap", hand_on_phone=True)),
           kind="true_positive", expected=[("phone_use", 2.0, 8.0), ("eyes_down", 2.0, 8.0)]),
    # ---------------------------------------------------------------- false-positive traps
    _cabin("phone_mounted_unused", "A phone in a dash mount, hands on the wheel, eyes on the road; "
           "one 1.2 s glance at the dashboard.", 25.0,
           (0.0, _MOUNT), (15.0, CabinScene(phone="mount", head_down_deg=25.0)), (16.2, _MOUNT)),
    _cabin("phone_mounted_hands_unseen", "A phone in a dash mount, the hand model sees no hands; eyes on "
           "the road throughout.", 25.0, (0.0, _MOUNT_NO_HANDS)),
    # SUNDAY stream's code-reading case (B1/B3/B4): a long look at a MOUNTED phone,
    # hands visible. After ~2 s of stillness with hands seen elsewhere the shipped
    # PhoneHoldEvidence latches "not held" and phone_use is skipped; a look that
    # starts before that latch is judged on the visible phone alone.
    _cabin("phone_mounted_long_look_latched", "A phone in a dash mount for 10 s, hands on the wheel and "
           "visible; then a 3 s look down at it (eyes_down is true; phone use is not).", 16.0,
           (0.0, _MOUNT), (10.0, CabinScene(phone="mount", head_down_deg=30.0)), (13.0, _MOUNT),
           expected=[("eyes_down", 10.0, 13.0)]),
    _cabin("phone_mounted_look_before_latch", "A phone is put in a dash mount and the driver looks down at "
           "it for 3 s straight away, hands on the wheel and visible (eyes_down is true; phone use is not).",
           8.0, (0.0, CabinScene(phone="mount", head_down_deg=30.0)), (3.0, _MOUNT),
           expected=[("eyes_down", 0.0, 3.0)]),
    _cabin("phone_mounted_look_down", "A phone in a dash mount, hands not seen; the driver looks down 30 deg "
           "for 3.5 s (eyes_down is true; phone use is not).", 15.0,
           (0.0, _MOUNT_NO_HANDS), (5.0, CabinScene(phone="mount", wheel_hands=False, head_down_deg=30.0)),
           (8.5, _MOUNT_NO_HANDS), expected=[("eyes_down", 5.0, 8.5)]),
    _cabin("mirror_checks", "Three wide mirror checks (60 deg, 1.5 s) and one 40 deg glance (1 s).", 20.0,
           (2.0, CabinScene(yaw_deg=60.0)), (3.5, _ROAD), (7.0, CabinScene(yaw_deg=-60.0)), (8.5, _ROAD),
           (12.0, CabinScene(yaw_deg=60.0)), (13.5, _ROAD), (16.0, CabinScene(yaw_deg=40.0)), (17.0, _ROAD)),
    _cabin("dashboard_glances", "Three 1.2 s glances down at the instrument cluster.", 15.0,
           (2.0, CabinScene(head_down_deg=25.0)), (3.2, _ROAD), (6.0, CabinScene(head_down_deg=25.0)),
           (7.2, _ROAD), (10.0, CabinScene(head_down_deg=25.0)), (11.2, _ROAD)),
    _cabin("intersection_look", "Looks left 1.5 s, centre 0.6 s, right 1.5 s at an intersection.", 10.0,
           (2.0, CabinScene(yaw_deg=50.0)), (3.5, _ROAD), (4.1, CabinScene(yaw_deg=-50.0)), (5.6, _ROAD)),
    _cabin("passenger_holding_phone", "The (smaller) passenger holds a phone; the driver looks at the road "
           "with one 1 s glance across.", 20.0,
           (0.0, _PASSENGER), (10.0, CabinScene(passenger="seated", phone="passenger",
                                                passenger_hand_on_phone=True, yaw_deg=40.0)),
           (11.0, _PASSENGER)),
    _cabin("passenger_leaning_with_phone", "A passenger whose box is LARGER than the driver's holds a phone "
           "while the driver looks away for 3.5 s (distraction is true; phone use is not).", 15.0,
           (0.0, _PASSENGER_LEANING),
           (5.0, CabinScene(passenger="leaning", phone="passenger", passenger_hand_on_phone=True, yaw_deg=40.0)),
           (8.5, _PASSENGER_LEANING),
           expected=[("distraction", 5.0, 8.5)]),
    _cabin("camera_blocked", "Something covers the lens for 15 s (cannot monitor; not 'driver absent').", 20.0,
           (2.0, CabinScene(camera_blocked=True)), (17.0, _ROAD),
           kind="abstention", allowed=["attention_lost"]),
    _cabin("camera_shifted", "The camera is knocked 40 deg sideways after calibration; the driver keeps "
           "looking at the road for 60 s.", 60.0,
           (0.0, CabinScene(yaw_deg=40.0, pose_jitter_deg=0.5)), kind="abstention"),
    _cabin("sunglasses", "Dark sunglasses: the eyes cannot be seen; driver attentive for 25 s.", 25.0,
           (0.0, CabinScene(sunglasses=True)), kind="abstention"),
    _cabin("poor_lighting", "A dark cabin: dim, noisy landmark fits and 15% face dropouts for 25 s.", 25.0,
           (0.0, CabinScene(brightness=28, pose_jitter_deg=4.0, face_dropout=0.15)), kind="abstention"),
    _cabin("empty_vehicle", "The driver leaves; nobody in the vehicle for 90 s.", 90.0,
           (2.0, CabinScene(face=False, person=False, wheel_hands=False)),
           kind="abstention", allowed=["attention_lost"]),
    _cabin("partial_face_occlusion", "A hand over the lower face: unstable pose fits, 25% face dropouts, 15 s.",
           15.0, (0.0, CabinScene(pose_jitter_deg=6.0, face_dropout=0.25)), kind="abstention"),
    _cabin("hands_out_of_view_phone", "A phone in the lap, no hand visible to the hand model, looking down "
           "at it for 6 s (holding cannot be established).", 10.0,
           (0.0, CabinScene(phone="lap", wheel_hands=False)),
           (2.0, CabinScene(phone="lap", wheel_hands=False, head_down_deg=30.0)),
           (8.0, CabinScene(phone="lap", wheel_hands=False)),
           kind="abstention", expected=[("eyes_down", 2.0, 8.0)], allowed=["phone_use"]),
    _cabin("stream_stall", "Attentive driver; frames arrive in 0.4 s bursts with 3 s stalls for 30 s.", 30.0,
           (0.0, CabinScene(stall_gap_s=3.0)), kind="abstention"),
    _cabin("fluctuating_confidence", "Attentive driver; 8 deg pose noise, 30% face dropouts, a phone the "
           "detector sees on a third of its ticks, 25 s.", 25.0,
           (0.0, CabinScene(pose_jitter_deg=8.0, face_dropout=0.3, phone="mount", wheel_hands=False,
                            phone_visible_prob=0.33)), kind="abstention"),
    _cabin("detector_failure_look_down", "The object detector fails; the driver looks down for 4 s "
           "(eyes_down true; phone evidence insufficient).", 10.0,
           (0.0, CabinScene(detector_failing=True)), (2.0, CabinScene(detector_failing=True, head_down_deg=30.0)),
           (6.0, CabinScene(detector_failing=True)),
           kind="abstention", expected=[("eyes_down", 2.0, 6.0)]),
)}


# ====================================================================== M8.5 phone handling scenarios
# Phone HANDLING vs prolonged OFF-ROAD ATTENTION (docs/BEHAVIOR_ENGINE.md, "M8.5
# phone handling vs off-road attention"; tests/test_m85_phone_*.py). Kept out of
# CABIN_SCENARIOS so the M8.4 pins stay as they were; scripts/eval_behavior.py
# runs both sets. Labels are the script's own ground truth: SYNTHETIC.
def _mount(**changes) -> CabinScene:
    return CabinScene(**{"phone": "mount", **changes})


def _lap_held(**changes) -> CabinScene:
    return CabinScene(**{"phone": "lap", "hand_on_phone": True, **changes})


M85_PHONE_SCENARIOS: dict[str, CabinScenario] = {s.name: s for s in (
    # ------------------------------------------------ a mounted phone: never phone_use, never silence
    _cabin("m85_mount_unseen_short_glance", "A phone in a dash mount, hands not seen; one 1.5 s look down "
           "at it (under every threshold).", 15.0,
           (0.0, _mount(wheel_hands=False)), (8.0, _mount(wheel_hands=False, head_down_deg=30.0)),
           (9.5, _mount(wheel_hands=False))),
    _cabin("m85_mount_unseen_long_look_side", "A phone in a dash mount, hands not seen; a 4 s look 40 deg "
           "sideways (distraction is true; phone use is not).", 14.0,
           (0.0, _mount(wheel_hands=False)), (5.0, _mount(wheel_hands=False, yaw_deg=40.0)),
           (9.0, _mount(wheel_hands=False)), expected=[("distraction", 5.0, 9.0)]),
    _cabin("m85_mount_flicker_long_look", "A mounted phone the detector sees on 60% of its ticks (the not-held "
           "latch never forms), hands on the wheel; a 4 s look down at it.", 14.0,
           (0.0, _mount(phone_visible_prob=0.6)), (5.0, _mount(phone_visible_prob=0.6, head_down_deg=30.0)),
           (9.0, _mount(phone_visible_prob=0.6)), expected=[("eyes_down", 5.0, 9.0)]),
    _cabin("m85_mount_detector_fails_mid_look", "A mounted phone, hands not seen; the detector fails as the "
           "driver looks down for 4 s (phone evidence insufficient; eyes_down still true).", 14.0,
           (0.0, _mount(wheel_hands=False)),
           (5.0, _mount(wheel_hands=False, head_down_deg=30.0, detector_failing=True)),
           (9.0, _mount(wheel_hands=False)), kind="abstention", expected=[("eyes_down", 5.0, 9.0)]),
    _cabin("m85_sunglasses_mount_long_look", "Sunglasses; a mounted phone, hands not seen; a 3.5 s look down "
           "at it.", 13.0,
           (0.0, _mount(wheel_hands=False, sunglasses=True)),
           (5.0, _mount(wheel_hands=False, sunglasses=True, head_down_deg=30.0)),
           (8.5, _mount(wheel_hands=False, sunglasses=True)), expected=[("eyes_down", 5.0, 8.5)]),
    _cabin("m85_stall_mount_long_look", "Frames in 0.4 s bursts with 1 s stalls; a mounted phone, hands not "
           "seen; a 10 s look down at it.", 20.0,
           (0.0, _mount(wheel_hands=False, stall_gap_s=1.0)),
           (5.0, _mount(wheel_hands=False, stall_gap_s=1.0, head_down_deg=30.0)),
           (15.0, _mount(wheel_hands=False, stall_gap_s=1.0)), expected=[("eyes_down", 5.0, 15.0)]),
    # ------------------------------------------------ real handling: phone_use, unchanged timing
    _cabin("m85_held_phone_eyes_on_road", "A phone held in the lap (hand on it), eyes on the road for 12 s.",
           12.0, (0.0, _lap_held())),
    _cabin("m85_mount_then_picked_up", "A phone in a dash mount (latched not held), then the driver picks it "
           "up and looks down at it in the lap for 5 s.", 14.0,
           (0.0, _mount()), (5.0, _lap_held(head_down_deg=30.0)), (10.0, _lap_held()),
           kind="true_positive", expected=[("phone_use", 5.0, 10.0), ("eyes_down", 5.0, 10.0)]),
    _cabin("m85_mount_then_picked_up_hands_unseen", "As above, but the hand model never sees a hand: the "
           "phone moves from the mount to the lap and is looked at for 5 s.", 14.0,
           (0.0, _mount(wheel_hands=False)), (5.0, CabinScene(phone="lap", wheel_hands=False, head_down_deg=30.0)),
           (10.0, CabinScene(phone="lap", wheel_hands=False)),
           kind="true_positive", expected=[("phone_use", 5.0, 10.0), ("eyes_down", 5.0, 10.0)]),
    _cabin("m85_mount_look_then_pick_up", "A long look down at a mounted phone that, 1.5 s in, the driver "
           "takes out of the mount and keeps looking at in the lap (phone moved mid-episode).", 14.0,
           (0.0, _mount()), (5.0, _mount(head_down_deg=30.0)), (6.5, _lap_held(head_down_deg=30.0)),
           (10.0, _lap_held()),
           kind="true_positive", expected=[("eyes_down", 5.0, 10.0), ("phone_use", 6.5, 10.0)]),
    _cabin("m85_held_then_mounted_mid_look", "Holds the phone in the lap looking down at it 1.5 s, puts it in "
           "the mount and keeps looking at it 4.5 s more.", 12.0,
           (0.0, _lap_held()), (2.0, _lap_held(head_down_deg=30.0)), (3.5, _mount(head_down_deg=30.0)),
           (8.0, _mount()), expected=[("eyes_down", 2.0, 8.0)], allowed=["phone_use"]),
    _cabin("m85_phone_at_ear_look_away", "On a call with the phone at the ear (no hand seen), the driver looks "
           "40 deg sideways for 4.5 s.", 12.0,
           (0.0, CabinScene(phone="ear", wheel_hands=False)),
           (4.0, CabinScene(phone="ear", wheel_hands=False, yaw_deg=40.0)),
           (8.5, CabinScene(phone="ear", wheel_hands=False)),
           kind="true_positive", expected=[("phone_use", 4.0, 8.5), ("distraction", 4.0, 8.5)]),
    _cabin("m85_moving_phone_hands_unseen", "A phone held beside the lap, moving in the hand (no hand seen), "
           "looked down at for 5 s.", 11.0,
           (0.0, CabinScene(phone="side", wheel_hands=False, phone_jitter_px=12.0)),
           (3.0, CabinScene(phone="side", wheel_hands=False, phone_jitter_px=12.0, head_down_deg=30.0)),
           (8.0, CabinScene(phone="side", wheel_hands=False, phone_jitter_px=12.0)),
           kind="true_positive", expected=[("phone_use", 3.0, 8.0), ("eyes_down", 3.0, 8.0)]),
    _cabin("m85_sunglasses_held_phone", "Sunglasses; holds a phone in the lap and looks down at it for 6 s.",
           10.0, (0.0, _lap_held(sunglasses=True)), (2.0, _lap_held(sunglasses=True, head_down_deg=30.0)),
           (8.0, _lap_held(sunglasses=True)),
           kind="true_positive", expected=[("phone_use", 2.0, 8.0), ("eyes_down", 2.0, 8.0)]),
    _cabin("m85_stall_held_phone", "Frames in 0.4 s bursts with 1 s stalls; holds a phone in the lap and looks "
           "down at it for 10 s.", 15.0,
           (0.0, _lap_held(stall_gap_s=1.0)), (2.0, _lap_held(stall_gap_s=1.0, head_down_deg=30.0)),
           (12.0, _lap_held(stall_gap_s=1.0)),
           kind="true_positive", expected=[("phone_use", 2.0, 12.0), ("eyes_down", 2.0, 12.0)]),
)}
