"""Abstract base class for vision modules, plus the module registry.

To add a new product module you only need to:
  1. Create `modules/your_module.py` subclassing `VisionModule`, decorated
     with `@register_module("your_module")`.
  2. Add a `modules.your_module: {...thresholds...}` section to config.yaml.
  3. Set `module: your_module` in config.yaml.

`main.py` never needs to be edited — modules are auto-discovered by
`modules/__init__.py` scanning this package's directory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from core.detections import DetectionFrame
from core.events import Event

_REGISTRY: dict[str, type["VisionModule"]] = {}


def register_module(name: str):
    """Class decorator that registers a VisionModule subclass under `name`."""

    def decorator(cls: type["VisionModule"]) -> type["VisionModule"]:
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_module_class(name: str) -> type["VisionModule"]:
    """Look up a registered module class by its config.yaml name.

    Raises:
        KeyError: if `name` isn't registered.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Unknown module '{name}'. Available modules: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available_modules() -> list[str]:
    """List the names of all currently registered modules."""
    return sorted(_REGISTRY)


class VisionModule(ABC):
    """A pluggable product: consumes frames, emits Events, optionally draws debug overlays."""

    name: str = "base"

    def __init__(self, config: dict) -> None:
        """Store this module's config section (its `modules.<name>` block from config.yaml)."""
        self.config = config

    @abstractmethod
    def process(self, frame: np.ndarray) -> list[Event]:
        """Process one BGR frame and return any Events triggered by it."""
        raise NotImplementedError

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        """Draw a debug overlay (landmarks, boxes, text) on `frame` for the live display.

        Default implementation is a no-op; override to visualize module state.
        """
        return frame

    def annotate_debug(self, frame: np.ndarray) -> np.ndarray:
        """Draw the full tuning overlay (used by `make dev-debug`).

        Default implementation falls back to `annotate`; override to add
        additional detail (raw signal values, secondary detector boxes, the
        current state-machine label, etc.).
        """
        return self.annotate(frame)

    def is_alert_active(self) -> bool:
        """Whether an alert condition is currently ongoing (drives the red-screen flash).

        Default implementation always returns False; override for modules that
        have an active/inactive alert state.
        """
        return False

    def alert_episode_types(self) -> frozenset[str]:
        """Event types whose condition this module tracks as a live episode.

        For these, the alert policy (core/alerts.py) keeps the episode open while
        `active_alert_conditions()` reports it, repeating and escalating the alert,
        and ends it when the condition clears. Types not listed are one-off
        episodes. Default: none (every event is a one-off).
        """
        return frozenset()

    def active_alert_conditions(self) -> set[str]:
        """The subset of `alert_episode_types()` whose condition is true right now."""
        return set()

    def available_labels(self) -> Optional[frozenset[str]]:
        """Detection labels this module puts in `get_detections()` frames.

        Custom rules evaluated beside this module (additive composition, see
        core/roles.py) can only target these. None means "not declared": such a
        module cannot host additive custom rules.
        """
        return None

    def set_requested_labels(self, labels: set[str]) -> None:  # noqa: B027 - optional hook, no-op by default
        """Widen what this module asks the detector for, for AI-configured rules.

        Modules narrow their detector call to the classes their own logic needs
        (occupancy asks for people and nothing else). That is right for cost, and
        wrong the moment a user writes "alert me if my dog gets on the couch":
        the rule is valid, `dog` and `couch` are real classes, and the detector
        would never have been asked for them — so the rule could be accepted,
        stored, shipped, and silently never fire.

        The runtime calls this whenever the rule set changes. Default is a no-op:
        a module that cannot widen simply keeps its own classes, and the rule is
        reported as unenforceable rather than quietly ignored.
        """

    def set_zones(self, zones: list) -> None:  # noqa: B027 - deliberate no-op default
        """Replace this module's zones with the ones the user drew in the app.

        `zones` is a list of core.zones.Zone, exactly as core.zones.load_zones()
        returns it. An empty list means the user has no zones, so every zone
        rule is removed. Without this hook the app's polygons reached only the
        AI-rule engine: a crib drawn as "Stay inside" never raised
        safe_zone_exit, and baby_monitoring's infant identification never got
        its crib.

        Implementations replace their zone rules in place. They do not rebuild
        the module or touch state unrelated to zones (timers, trackers,
        calibration). Engine state belonging to a removed or redrawn zone is
        dropped so nothing stays latched. A zone whose name, kind and polygon
        are unchanged keeps its in-progress state, so re-sending the same zones
        is a no-op.

        Default is a no-op, for modules that have no zone concept.
        """

    def get_detections(self) -> Optional["DetectionFrame"]:
        """What this module saw in the most recent `process()` frame, undebounced.

        Feeds AI-configured rules (core/rules.py), which need boxes and counts
        that `Event` does not carry — a zone rule ("in the pool area") or a count
        rule ("more than three people") cannot be evaluated from Events.

        Default returns None (opt-in, like `get_telemetry_snapshot`). Modules that
        don't override it simply can't back zone or count rules; RuleEngine reads
        None as "no observation this frame" rather than assuming anything.
        """
        return None

    def get_telemetry_snapshot(self) -> Optional[dict]:
        """Raw per-frame signal snapshot for telemetry logging (future model-training data).

        Default implementation returns None (opt-in); override to expose a flat
        dict of the module's current raw signals. `main.py` logs whatever is
        returned to `data/telemetry.csv` at a throttled rate via TelemetryLogger.
        """
        return None

    def get_debug_state(self) -> Optional[dict]:
        """Human-readable snapshot of why this module is (or isn't) about to alert.

        Developer/debug use only (main.py --debug) -- deliberately more
        detailed and more implementation-specific than get_telemetry_snapshot
        (which IS logged, and is product data). Never sent to the cloud or
        surfaced to an end user. Default returns None (opt-in); override to
        expose current thresholds, durations, and the reasoning behind the
        live alert decision.
        """
        return None

    def close(self) -> None:  # noqa: B027 - optional hook, no-op by default
        """Release module-held resources (models, mediapipe graphs, device handles).

        Called on shutdown and before a preset hot-swap replaces this module,
        so long-running devices don't leak a FaceMesh graph (or an accelerator
        handle) per mode switch. Default is a no-op.
        """


class SustainedConditionTracker:
    """Fires once when a boolean condition has held continuously for `sustain_s` seconds.

    Re-arms automatically once the condition goes False, so it fires at most
    once per continuous "episode" of the condition being True.
    """

    def __init__(self, sustain_s: float) -> None:
        """Create a tracker that requires `sustain_s` seconds of continuous True."""
        self._sustain_s = sustain_s
        self._start: Optional[float] = None
        self._fired = False

    def update(self, condition: bool, now: float) -> Optional[float]:
        """Advance the tracker with the latest condition value and timestamp.

        Args:
            condition: Whether the tracked condition is true this frame.
            now: Current monotonic time in seconds.

        Returns:
            The elapsed sustained duration the moment the threshold is first
            crossed, or None on every other call (including while still
            accumulating, or after already firing for this episode).
        """
        if not condition:
            self._start = None
            self._fired = False
            return None

        if self._start is None:
            self._start = now

        elapsed = now - self._start
        if elapsed >= self._sustain_s and not self._fired:
            self._fired = True
            return elapsed
        return None

    @property
    def is_active(self) -> bool:
        """Whether the condition has been continuously true for at least `sustain_s` already.

        True from the moment the sustained threshold is first crossed until the
        condition clears (goes False). Use this (not the raw per-frame condition)
        to drive a live "alert active" indicator — the raw condition is true on
        every blink/glance, before debouncing; this is only true once debounced.
        """
        return self._fired

    def elapsed(self, now: float) -> float:
        """Seconds the current episode has been running, 0.0 if none is in progress.

        For debug/observability only (e.g. "distraction duration: 6.2s" on a
        developer overlay) — no threshold or event decision reads this.
        """
        return now - self._start if self._start is not None else 0.0


class GracedConditionTracker:
    """A SustainedConditionTracker that survives momentary signal dropouts.

    The plain tracker resets COMPLETELY on a single False frame — start time,
    fired state, everything. That is correct for a clean boolean, and wrong
    for one derived from a noisy measurement: with glasses or glare, MediaPipe
    lands one bad frame every few seconds, and each one silently restarted the
    drowsiness clock. The sleepier the eyes (the more marginal the landmarks),
    the harder the closure became to detect — the failure grew in exactly the
    case the product exists for. The same reset made `is_active` flap, so the
    alert screen flickered with the noise.

    Here, the condition must be False for `release_s` CONTINUOUSLY before the
    tracker resets or deactivates. Real state changes are slower than noise:
    an eye that has genuinely reopened stays open for many frames; a landmark
    glitch lasts one or two. The firing threshold is untouched — a condition
    still fires at exactly `sustain_s` of (grace-bridged) persistence, so
    detection latency does not change; only noise's power to restart the clock
    does. Mirrors what EventEngine's occlusion_grace_s already does for the
    module lifecycle path; this is the same idea for the in-module trackers.

    Two opt-in options, for a condition whose False frames can be REAL rather
    than noise (eye closure: the eye genuinely reopens between blinks):

      * `release_frames`: release after this many consecutive False updates
        (bridging at most `release_frames` frames) instead of after
        `release_s` seconds. A landmark glitch is a frame or two at any frame
        rate; 0.4s is 12 frames at 30fps -- long enough to bridge an eye that
        really did reopen between two blinks.
      * `count_true_time_only`: the sustained duration is the time the
        condition was actually True (sample-and-hold: each True update owns
        the interval until the next update), so bridged False frames pause the
        clock instead of counting toward `sustain_s`. Without it a flurry of
        short blinks with brief openings reads as one continuous closure.

    Both default off, preserving the original semantics for every other user.
    """

    def __init__(
        self,
        sustain_s: float,
        release_s: float = 0.4,
        *,
        release_frames: Optional[int] = None,
        count_true_time_only: bool = False,
    ) -> None:
        self._sustain_s = sustain_s
        self._release_s = release_s
        self._release_frames = None if release_frames is None else max(0, int(release_frames))
        self._count_true_only = bool(count_true_time_only)
        self._start: Optional[float] = None
        self._fired = False
        self._false_since: Optional[float] = None
        self._false_frames = 0
        self._true_time = 0.0
        self._last_update: Optional[tuple[float, bool]] = None

    def _reset(self) -> None:
        self._start = None
        self._fired = False
        self._false_since = None
        self._false_frames = 0
        self._true_time = 0.0

    def update(self, condition: bool, now: float) -> Optional[float]:
        """Advance with this frame's condition. Returns the sustained duration
        the first moment `sustain_s` is crossed, else None (same contract as
        SustainedConditionTracker)."""
        previous = self._last_update
        self._last_update = (now, condition)
        if (
            self._count_true_only
            and self._start is not None
            and previous is not None
            and previous[1]
            and now > previous[0]
        ):
            self._true_time += now - previous[0]

        if condition:
            self._false_since = None
            self._false_frames = 0
            if self._start is None:
                self._start = now
                self._true_time = 0.0
            elapsed = self._true_time if self._count_true_only else now - self._start
            if elapsed >= self._sustain_s and not self._fired:
                self._fired = True
                return elapsed
            return None

        if self._start is None:
            return None
        if self._false_since is None:
            self._false_since = now
        self._false_frames += 1
        if self._release_frames is not None:
            released = self._false_frames > self._release_frames
        else:
            released = now - self._false_since >= self._release_s
        if released:
            self._reset()
        return None

    @property
    def is_active(self) -> bool:
        """True from firing until the condition has been clear for the release grace."""
        return self._fired

    def elapsed(self, now: float) -> float:
        """Seconds the current episode has been running (or, with
        `count_true_time_only`, has been True), 0.0 if none is in progress.

        For debug/observability only -- see SustainedConditionTracker.elapsed.
        """
        if self._start is None:
            return 0.0
        if self._count_true_only:
            pending = 0.0
            if self._last_update is not None and self._last_update[1] and now > self._last_update[0]:
                pending = now - self._last_update[0]
            return self._true_time + pending
        return now - self._start


class CorroboratedGraceTracker:
    """A GracedConditionTracker with two thresholds: a short one that always
    applies, and a longer one usable only while a second, corroborating
    condition has held for the ENTIRE episode so far.

    Built for driver_attention.py's face-lost signal: a driver whose face the
    camera can't find is more patient to alarm on if their body is still
    visible the whole time (they turned to check a mirror) than if nothing
    corroborates their presence at all (they may be genuinely gone) --
    "face unavailable" should not mean "driver disappeared" outright.

    The corroborating evidence can only ever LOWER urgency, never raise it
    past what the short threshold alone would already give: the moment
    corroboration drops out even once mid-episode, the tracker reverts to the
    short threshold for the rest of it, measured from when the primary
    condition first became true -- not from when it dropped out. So a
    situation that gets WORSE partway through (the body disappears too) is
    never slower to fire than the short threshold alone would have been; the
    long threshold is a bonus available only to a fully-corroborated episode,
    never a delay imposed on one that stopped being.
    """

    def __init__(self, short_s: float, long_s: float, release_s: float = 0.4) -> None:
        self._short_s = short_s
        self._long_s = max(short_s, long_s)
        self._release_s = release_s
        self._start: Optional[float] = None
        self._corroborated = True
        self._fired = False
        self._false_since: Optional[float] = None

    def update(self, condition: bool, corroborated: bool, now: float) -> Optional[float]:
        """Advance with this frame's primary condition and corroborating evidence.

        Returns the elapsed duration the moment the applicable threshold is
        first crossed, else None (same contract as GracedConditionTracker).
        """
        if condition:
            self._false_since = None
            if self._start is None:
                self._start = now
                self._corroborated = corroborated
            elif not corroborated:
                self._corroborated = False  # latches for the rest of this episode
            elapsed = now - self._start
            threshold = self._long_s if self._corroborated else self._short_s
            if elapsed >= threshold and not self._fired:
                self._fired = True
                return elapsed
            return None

        if self._start is None:
            return None
        if self._false_since is None:
            self._false_since = now
        if now - self._false_since >= self._release_s:
            self._start = None
            self._fired = False
            self._corroborated = True
            self._false_since = None
        return None

    @property
    def is_active(self) -> bool:
        """True from firing until the condition has been clear for `release_s`."""
        return self._fired

    def elapsed(self, now: float) -> float:
        """Seconds the current episode has been running, 0.0 if none is in progress.

        For debug/observability only -- see SustainedConditionTracker.elapsed.
        """
        return now - self._start if self._start is not None else 0.0

    @property
    def corroborated(self) -> bool:
        """Whether the current (or most recent) episode has had corroborating
        evidence the whole way through -- for callers that want to vary an
        event's reported confidence/severity by which threshold applied."""
        return self._corroborated
