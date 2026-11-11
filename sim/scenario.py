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

    def describe(self) -> list[str]:
        out = []
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
