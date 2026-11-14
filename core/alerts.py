"""Pluggable alert actuators and the episode-aware alert policy.

Episodes (M0.1). Every module tracker and EventEngine rule emits ONE event per
episode of a condition. The old policy dropped any alert of the same type within
`cooldown_s` of the previous one, so a new episode that began inside that window
produced no alert at all, and the "continuous" alert was a single 0.6 s pulse
that never repeated however long the driver stayed asleep. The policy now works
in episodes:

  * An alertable event STARTS an episode, keyed by event type (or by the
    caller's `key=`, e.g. a custom rule id), and actuates immediately. A new
    episode is never dropped because an earlier one alerted recently.
  * While an episode is open, a duplicate event for the same key is merged
    into it (SUPPRESSED, no actuation).
  * For conditions a module tracks (`VisionModule.alert_episode_types()`), the
    episode stays ACTIVE while the module reports the condition
    (`active_alert_conditions()`), repeats every `repeat_s` with a bounded
    number of repeats, escalates to a more urgent pattern after
    `escalate_after_s`, and ENDS when the condition clears.
  * Events nobody tracks are single-moment episodes: they end after
    `duplicate_window_s`. Rolling-window NOTICES (the `soft_types`, e.g.
    fatigue_warning) keep the old `cooldown_s` de-duplication, because they
    describe a window, not a new episode of danger.

Actuators. A sink renders a bounded `AlertPattern` (on/off steps) through
`play(pattern, event, category)` when it supports patterns, and otherwise gets
the legacy `emit(event, level)`. Buzzer, speaker, console and test recorders
implement it today; LED, haptic and app actuators can be added without changing
the policy. No test here verifies physical sound: that needs the device.

A sink that raises is logged and skipped; it never stops the other sinks or
propagates into the inference loop.

Delivery evidence (M8.3, docs/EVENT_PIPELINE.md). Detecting an event and
warning someone about it are different facts. For every event it is handed the
policy keeps an `AlertDelivery`: what it decided (a new episode, merged into an
open one, a notice inside its cooldown) and what each in-cabin output (audible
or visual) reported doing about it. An output that sets `reports_completion`
is played with a `report=` callback and says, once, whether its whole pattern
rendered ("played"), raised ("failed") or was never rendered ("skipped"); an
output that cannot report is "unconfirmed", never "played". The uplink sends
the summary with the event, so the app can say "the alert sounded on the
device" only when the device itself said so.

Experimental observations (M8.4, owner decision A). The experimental alert
families (core/behavior/experimental.py: high_risk, elevated_risk,
erratic_hand_movement) are REFUSED by this policy unless it was built with
`experimental_audible=True` (main.build_alert_manager passes it only for an
explicit `alerts.experimental_alerts_audible: true`). A refused event opens no
episode, commands no sink, records no delivery and returns SUPPRESSED -- so no
config list, preset, variant, migration or reset can make one sound.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Optional, Protocol

from core.behavior.experimental import is_experimental
from core.events import Event
from core.logging_setup import get_logger
from core.monitoring_state import Actuator

log = get_logger("alerts")


class AlertLevel(Enum):
    """Severity of a dispatched alert."""

    SUPPRESSED = "suppressed"
    SOFT = "soft"
    CONTINUOUS = "continuous"


#: No single pattern may last longer than this, so a repeat can never overlap
#: itself and an actuator thread is never busy for more than a few seconds.
MAX_PATTERN_S = 3.5
_MAX_PATTERN_STEPS = 8


@dataclass(frozen=True)
class AlertPattern:
    """A bounded on/off rhythm an actuator renders (buzzer pulses, tone bursts, LED blinks)."""

    name: str
    steps: tuple[tuple[float, float], ...]  # (on_s, off_s) pairs

    def __post_init__(self) -> None:
        if not 1 <= len(self.steps) <= _MAX_PATTERN_STEPS:
            raise ValueError(f"pattern {self.name!r}: 1-{_MAX_PATTERN_STEPS} steps required")
        if any(on <= 0 or off < 0 for on, off in self.steps):
            raise ValueError(f"pattern {self.name!r}: on must be > 0 and off >= 0")
        if self.duration_s > MAX_PATTERN_S:
            raise ValueError(f"pattern {self.name!r} lasts {self.duration_s:.2f}s > {MAX_PATTERN_S}s")

    @property
    def duration_s(self) -> float:
        return sum(on + off for on, off in self.steps)


#: Distinguishable rhythms on a single-tone buzzer. Names are stable config values.
PATTERNS: dict[str, AlertPattern] = {
    p.name: p for p in (
        AlertPattern("chirp", ((0.15, 0.0),)),                                  # notice
        AlertPattern("single", ((0.6, 0.0),)),                                  # legacy loud tone
        AlertPattern("double", ((0.3, 0.2), (0.3, 0.0))),                       # warning onset
        AlertPattern("urgent", ((0.4, 0.1), (0.4, 0.1), (0.4, 0.0))),           # warning, escalated
        AlertPattern("critical", ((0.9, 0.15), (0.9, 0.15), (0.9, 0.0))),       # sleep / unresponsive
        AlertPattern("impaired", ((0.1, 0.1), (0.1, 0.1), (0.1, 0.5), (0.7, 0.0))),  # cannot monitor
    )
}


@dataclass(frozen=True)
class AlertCategory:
    """How an episode of one kind of alert sounds, repeats and escalates."""

    name: str
    initial_pattern: str
    escalated_pattern: str
    repeat_s: float = 0.0          # 0 = no repetition
    escalate_after_s: float = 0.0  # seconds from episode start before repeats escalate
    max_repeats: int = 0           # bound on repeats per episode (initial actuation not counted)

    def __post_init__(self) -> None:
        for pattern in (self.initial_pattern, self.escalated_pattern):
            if pattern not in PATTERNS:
                raise ValueError(f"category {self.name!r}: unknown pattern {pattern!r}")
        if self.repeat_s < 0 or self.escalate_after_s < 0 or not 0 <= self.max_repeats <= 120:
            raise ValueError(f"category {self.name!r}: repeat_s/escalate_after_s >= 0, max_repeats 0-120")
        if self.repeat_s > 0:
            longest = max(PATTERNS[self.initial_pattern].duration_s, PATTERNS[self.escalated_pattern].duration_s)
            if self.repeat_s < longest + 0.5:
                raise ValueError(f"category {self.name!r}: repeat_s must leave the pattern 0.5s to finish")

    @property
    def legacy_level(self) -> "AlertLevel":
        """The two-level value given to sinks that only implement emit()."""
        return AlertLevel.SOFT if self.name == "notice" else AlertLevel.CONTINUOUS


#: Defaults. Overridable per config (`alerts.categories`) within these bounds;
#: a deployment role can additionally forbid weakening some of them (core/roles.py).
DEFAULT_CATEGORIES: dict[str, AlertCategory] = {
    c.name: c for c in (
        AlertCategory("critical", "critical", "critical", repeat_s=4.0, escalate_after_s=0.0, max_repeats=30),
        AlertCategory("warning", "double", "urgent", repeat_s=5.0, escalate_after_s=10.0, max_repeats=24),
        AlertCategory("notice", "chirp", "chirp"),
        # Reserved for "monitoring impaired" (M0.2): distinct from every danger pattern.
        AlertCategory("impaired", "impaired", "impaired", repeat_s=15.0, escalate_after_s=0.0, max_repeats=20),
    )
}


class AlertSink(Protocol):
    """A destination that can render an alert (console, sound, GPIO buzzer, ...).

    `play(pattern, event, category)` is optional: sinks that implement it render
    the bounded pattern; sinks that do not receive `emit(event, level)`.
    """

    def emit(self, event: Event, level: AlertLevel) -> None:
        """Render the given event at the given alert level."""
        ...


#: `report(result, error)`: how one output says what became of one actuation (M8.3).
#: result is "played" (every step rendered), "failed" (rendering raised) or
#: "skipped" (never rendered: disabled, unavailable, displaced by a more urgent one).
Report = Callable[[str, Optional[str]], None]


def _safe_report(report: Optional[Report], result: str, error: Optional[str] = None) -> None:
    if report is None:
        return
    try:
        report(result, error)
    except Exception as exc:  # noqa: BLE001 - evidence bookkeeping must never break an output
        log.warning("alert delivery report failed: %s", exc)


class _PatternPlayer:
    """Plays patterns one at a time on a single worker thread.

    One physical buzzer, many overlapping requests: patterns are queued (at most
    `_DEPTH` waiting) instead of interleaving, so a soft chirp can never cut a
    critical pattern short. When the queue is full the least urgent waiting
    pattern is dropped, never the most urgent.

    `render_step(on_s)` blocks for on_s and returns None, or the error that
    stopped it. Each pattern's `report` (if given) is called exactly once:
    played, failed (with the first error), or skipped (dropped for a more
    urgent pattern) -- M8.3 delivery evidence.
    """

    _DEPTH = 2
    _RANK: ClassVar[dict[str, int]] = {"notice": 0, "warning": 1, "impaired": 2, "critical": 3}

    def __init__(self, render_step) -> None:
        self._render_step = render_step  # callable(on_s) -> Optional[str] error; blocks for on_s
        self._lock = threading.Lock()
        self._pending: list[tuple[int, AlertPattern, Optional[Report]]] = []
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def play(self, pattern: AlertPattern, category_name: str, report: Optional[Report] = None) -> None:
        rank = self._RANK.get(category_name, 1)
        dropped: Optional[Report] = None
        with self._lock:
            if len(self._pending) >= self._DEPTH:
                weakest = min(range(len(self._pending)), key=lambda i: self._pending[i][0])
                if self._pending[weakest][0] > rank:
                    dropped = report  # everything waiting is more urgent than this
                    queued = False
                else:
                    dropped = self._pending.pop(weakest)[2]
                    queued = True
            else:
                queued = True
            if queued:
                self._pending.append((rank, pattern, report))
                if self._thread is None or not self._thread.is_alive():
                    self._thread = threading.Thread(target=self._run, daemon=True)
                    self._thread.start()
        _safe_report(dropped, "skipped", "displaced by a more urgent alert pattern on this output")
        self._wake.set()

    def _run(self) -> None:
        while True:
            with self._lock:
                if not self._pending:
                    self._thread = None
                    return
                best = max(range(len(self._pending)), key=lambda i: self._pending[i][0])
                _, pattern, report = self._pending.pop(best)
            error: Optional[str] = None
            for on_s, off_s in pattern.steps:
                step_error = self._render_step(on_s)
                if step_error and error is None:
                    error = str(step_error)
                if off_s:
                    time.sleep(off_s)
            _safe_report(report, "failed" if error else "played", error)


class UnavailableSink:
    """A configured alert output that could not be set up (no GPIO, no Pi, ...).

    It stays in the sink list so health keeps reporting it by name and reason,
    rather than the output silently vanishing while the config still lists it.
    """

    reports_completion = True

    def __init__(self, name: str, kind: str, audible: bool, reason: str) -> None:
        self.name, self.kind, self.audible, self._reason = name, kind, audible, reason

    def readiness(self) -> tuple[bool, str]:
        return False, self._reason

    def play(self, pattern: AlertPattern, event: Event, category: AlertCategory,
             report: Optional[Report] = None) -> None:
        _safe_report(report, "skipped", f"output unavailable: {self._reason}")

    def emit(self, event: Event, level: AlertLevel) -> None:
        pass


@dataclass
class _SinkResults:
    """What happened to the commands sent to one alert output."""

    ok: int = 0
    failed: int = 0
    consecutive_failures: int = 0
    last_error: Optional[str] = None


class ConsoleSink:
    """Prints alerts to stdout."""

    name, kind, audible = "console", "console", False

    def readiness(self) -> tuple[bool, str]:
        return True, "prints to the service log (not an in-cabin signal)"

    def emit(self, event: Event, level: AlertLevel) -> None:
        """Print a one-line alert message."""
        confidence = "n/a" if event.confidence is None else f"{event.confidence:.2f}"
        print(f"[ALERT:{level.value}] {event.module_name}: {event.type} (confidence={confidence})")

    def play(self, pattern: AlertPattern, event: Event, category: AlertCategory) -> None:
        """Print the episode actuation with its category and pattern."""
        confidence = "n/a" if event.confidence is None else f"{event.confidence:.2f}"
        print(f"[ALERT:{category.name}/{pattern.name}] {event.module_name}: {event.type} "
              f"(confidence={confidence})")


class SoundSink:
    """Plays an audible tone: short chime for SOFT, longer tone for CONTINUOUS.

    Prefers simpleaudio (works on Mac and Pi). Falls back to `aplay` on Linux
    if simpleaudio isn't available, and to the terminal bell as a last resort.
    """

    name, kind, audible = "sound", "speaker", True
    reports_completion = True

    def __init__(self) -> None:
        """Detect the best available playback backend."""
        try:
            import simpleaudio  # type: ignore[import-not-found]

            self._simpleaudio = simpleaudio
        except ImportError:
            self._simpleaudio = None
        self._player = _PatternPlayer(self._play_tone_safe)
        self._last_error: Optional[str] = None

    def readiness(self) -> tuple[bool, str]:
        """Software-level only: a backend exists and the last playback did not fail.
        Nothing here can tell whether a speaker is connected or audible."""
        if sys.platform == "darwin":
            return False, "audio output is disabled on macOS (dev machine)"
        if self._simpleaudio is None and not (sys.platform.startswith("linux") and shutil.which("aplay")):
            return False, "no audio backend (simpleaudio or aplay) is available"
        if self._last_error is not None:
            return False, f"last playback failed: {self._last_error}"
        return True, "playback commands succeed (not physically verified)"

    def play(self, pattern: AlertPattern, event: Event, category: AlertCategory,
             report: Optional[Report] = None) -> None:
        """Render the pattern as tone bursts on the sound device (never on macOS; see emit)."""
        if sys.platform == "darwin":
            _safe_report(report, "skipped", "audio output is disabled on macOS (dev machine)")
            return
        self._player.play(pattern, category.name, report)

    def emit(self, event: Event, level: AlertLevel) -> None:
        """Play a tone whose length reflects the alert level.

        Playback happens on a short-lived daemon thread: `wait_done()`/`aplay`
        block for the tone's full duration, and emit() is called from the
        inference loop — a 0.6s continuous-alert tone must not freeze detection
        for 0.6s (≈18 dropped frames at 30fps).
        """
        if level == AlertLevel.SUPPRESSED:
            return
        # simpleaudio can segfault on macOS during native audio cleanup.
        # SensorDyme runs audio alerts through the Pi buzzer in production;
        # on Mac/dev mode, skip audio entirely so the backend stays stable.
        if sys.platform == "darwin":
            return
        duration_s = 0.15 if level == AlertLevel.SOFT else 0.6
        threading.Thread(target=self._play_tone_safe, args=(duration_s,), daemon=True).start()

    def _play_tone_safe(self, duration_s: float) -> Optional[str]:
        """Tone playback that can never crash-spam: a Pi with no PCM audio device
        raises from simpleaudio on every alert — degrade to the terminal bell.

        Returns None when a tone was played, else why not (M8.3): the terminal
        bell is a fallback, never evidence that the speaker sounded."""
        try:
            played = self._play_tone(duration_s)
            self._last_error = None
            return None if played else "no audio backend: terminal bell only"
        except Exception as exc:  # noqa: BLE001 - audio is best-effort by design
            self._last_error = str(exc) or type(exc).__name__
            print("\a", end="", flush=True)
            return self._last_error

    def _play_tone(self, duration_s: float) -> bool:
        """Play (or emulate) a tone of the given length; False when only the bell rang."""
        if self._simpleaudio is not None:
            self._play_via_simpleaudio(duration_s)
            return True
        if sys.platform.startswith("linux") and shutil.which("aplay"):
            self._play_via_aplay(duration_s)
            return True
        print("\a", end="", flush=True)  # terminal bell fallback
        return False

    def _play_via_simpleaudio(self, duration_s: float) -> None:
        """Generate and play a sine tone through simpleaudio."""
        import numpy as np

        fs = 44100
        t = np.linspace(0, duration_s, int(fs * duration_s), False)
        tone = np.sin(880 * t * 2 * np.pi) * 0.5
        audio = (tone * (2**15 - 1)).astype(np.int16)
        play_obj = self._simpleaudio.play_buffer(audio, 1, 2, fs)
        play_obj.wait_done()

    def _play_via_aplay(self, duration_s: float) -> None:
        """Generate a WAV tone and play it via the Linux `aplay` CLI."""
        import os
        import tempfile
        import wave

        import numpy as np

        fs = 44100
        t = np.linspace(0, duration_s, int(fs * duration_s), False)
        tone = (np.sin(880 * t * 2 * np.pi) * 0.5 * (2**15 - 1)).astype(np.int16)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes(tone.tobytes())
        subprocess.run(["aplay", "-q", path], check=False)  # noqa: S603, S607 - fixed argv, our own temp file
        os.unlink(path)


class GPIOBuzzerSink:
    """Drives a GPIO buzzer on Raspberry Pi. Never imported/instantiated on Mac."""

    def __init__(self, pin: int) -> None:
        """Bind to the given BCM pin. `gpiozero` is imported lazily so Mac never touches it."""
        from gpiozero import Buzzer  # lazy import: this line never executes on Mac

        self._buzzer = Buzzer(pin)
        self._pin = pin
        self._last_error: Optional[str] = None
        # One physical buzzer, many overlapping requests. Each buzz used to own
        # an on/sleep/off thread, so a short buzz that overlapped a longer one
        # switched the buzzer OFF in the middle of it: a soft chime 50ms before
        # a loud alarm cut the alarm to ~0.1s. `_off_at` is the latest moment
        # any outstanding request wants the buzzer on; only the request that
        # owns that moment may turn it off.
        self._lock = threading.Lock()
        self._off_at = 0.0

    name, kind, audible = "gpio_buzzer", "buzzer", True
    reports_completion = True

    def readiness(self) -> tuple[bool, str]:
        """Software-level only: the pin was claimed and the last write did not fail.
        A disconnected or dead buzzer is invisible from here (no feedback line)."""
        if self._last_error is not None:
            return False, f"GPIO write failed: {self._last_error}"
        return True, f"GPIO {self._pin} drives succeed (not physically verified)"

    def emit(self, event: Event, level: AlertLevel) -> None:
        """Buzz briefly for SOFT alerts, longer for CONTINUOUS ones.

        The buzz runs on a daemon thread — the inline time.sleep used to stall
        the inference loop for up to 0.6s per alert on the Pi.
        """
        if level == AlertLevel.SUPPRESSED:
            return
        on_s = 0.15 if level == AlertLevel.SOFT else 0.6
        threading.Thread(target=self._buzz, args=(on_s,), daemon=True).start()

    def play(self, pattern: AlertPattern, event: Event, category: AlertCategory,
             report: Optional[Report] = None) -> None:
        """Render the pattern as buzzer pulses on one worker thread (patterns never interleave).

        `report` hears "played" once every pulse of the pattern was driven with
        no GPIO error. That is the pin driven, not a sound heard: a buzzer that
        is unwired or dead is invisible from here (no feedback line).
        """
        if not hasattr(self, "_player"):
            self._player = _PatternPlayer(self._buzz)
        self._player.play(pattern, category.name, report)

    def _buzz(self, on_s: float) -> Optional[str]:
        """Drive the pin on for on_s. Returns None, or the GPIO error."""
        try:
            with self._lock:
                self._off_at = max(self._off_at, time.monotonic() + on_s)
                self._buzzer.on()
            time.sleep(on_s)
            with self._lock:
                # A later or longer request extended _off_at: leave it on.
                if time.monotonic() >= self._off_at - 1e-3:
                    self._buzzer.off()
            self._last_error = None
            return None
        except Exception as exc:  # noqa: BLE001 - a GPIO fault must never kill anything
            self._last_error = str(exc) or type(exc).__name__
            log.warning("buzzer failed: %s", exc)
            return self._last_error


#: Event-level summary of what the device's in-cabin outputs reported, most
#: conclusive first. Only "sounded" means a real output rendered the warning.
DELIVERY_OUTPUTS = ("sounded", "simulated", "pending", "unconfirmed", "failed", "none")
#: What the policy did with the event.
DELIVERY_DECISIONS = ("alerted", "merged", "suppressed")
#: Status changes reported to listeners per event at most (pending -> played, a
#: failed onset then a repeat that played, ...); bounds the updates sent upstream.
_MAX_DELIVERY_REVISIONS = 6


def _in_cabin(sink) -> bool:
    """Whether an output warns someone in the cabin/room (sound or light), as
    opposed to the console log. Only these count as alert delivery."""
    return bool(getattr(sink, "audible", False) or getattr(sink, "visual", False))


@dataclass
class _OutputTally:
    name: str
    kind: str
    audible: bool
    visual: bool
    simulated: bool
    played: int = 0
    failed: int = 0
    skipped: int = 0
    pending: int = 0
    unconfirmed: int = 0
    error: Optional[str] = None

    def describe(self) -> dict:
        out = {"name": self.name, "kind": self.kind, "audible": self.audible, "played": self.played,
               "failed": self.failed, "skipped": self.skipped, "pending": self.pending,
               "unconfirmed": self.unconfirmed}
        if self.visual:
            out["visual"] = True
        if self.simulated:
            out["simulated"] = True
        if self.error:
            out["error"] = self.error[:160]
        return out


class AlertDelivery:
    """What the alert policy did with ONE event, and what the outputs reported (M8.3).

    decision   alerted     the event started an episode; its outputs were commanded
               merged      it joined an episode already open: no new actuation
               suppressed  a rolling-window notice inside its cooldown: no actuation
    output     sounded     a real in-cabin output reported its whole pattern rendered
               simulated   only a simulated output (sim/) recorded the actuation
               pending     commanded; no output has reported back yet
               unconfirmed commanded to outputs that cannot report completion
               failed      commanded; every in-cabin output failed or skipped it
               none        no in-cabin output was commanded for this event

    Repeats of the episode count too: an onset that failed followed by a repeat
    that played is "sounded". `rev` rises with every change of the summary, so
    the backend can keep the newest report whatever order they arrive in.
    """

    def __init__(self, decision: str, episode: Optional[str], category: Optional[str]) -> None:
        self.decision = decision
        self.episode = episode
        self.category = category
        #: For a merged event: the record of the episode it joined (what was
        #: actually actuated). The uplink sends that when the episode's own
        #: event was never sent -- e.g. an episode re-opened after a recovery.
        self.episode_delivery: Optional[AlertDelivery] = None
        self.rev = 0
        self._outputs: dict[int, _OutputTally] = {}
        self._lock = threading.Lock()
        self._listeners: list[Callable[[dict], None]] = []
        self._last_output: Optional[str] = None

    # ------------------------------------------------------------------ recording
    def _tally(self, sink) -> _OutputTally:
        tally = self._outputs.get(id(sink))
        if tally is None:
            tally = self._outputs[id(sink)] = _OutputTally(
                name=str(getattr(sink, "name", type(sink).__name__))[:32],
                kind=str(getattr(sink, "kind", "custom"))[:16],
                audible=bool(getattr(sink, "audible", False)),
                visual=bool(getattr(sink, "visual", False)),
                simulated=bool(getattr(sink, "simulated", False)),
            )
        return tally

    def commanded(self, sink, *, reports: bool) -> Optional[Report]:
        """One actuation sent to `sink`. Returns the callback it reports through."""
        with self._lock:
            tally = self._tally(sink)
            if not reports:
                tally.unconfirmed += 1
            else:
                tally.pending += 1
        self._changed()
        if not reports:
            return None
        done = threading.Event()  # exactly once per actuation

        def report(result: str, error: Optional[str] = None) -> None:
            if done.is_set():
                return
            done.set()
            self._resolve(sink, result, error)

        return report

    def raised(self, sink, error: str, *, reported: Optional[Report]) -> None:
        """The command itself raised: a failure, however the sink reports."""
        if reported is not None:
            reported("failed", error)
            return
        with self._lock:
            tally = self._tally(sink)
            tally.unconfirmed = max(0, tally.unconfirmed - 1)
            tally.failed += 1
            tally.error = error
        self._changed()

    def _resolve(self, sink, result: str, error: Optional[str]) -> None:
        with self._lock:
            tally = self._tally(sink)
            tally.pending = max(0, tally.pending - 1)
            if result == "played":
                tally.played += 1
            elif result == "skipped":
                tally.skipped += 1
                tally.error = error or tally.error
            else:
                tally.failed += 1
                tally.error = error or tally.error
        self._changed()

    # ------------------------------------------------------------------ reading
    def output(self) -> str:
        with self._lock:
            return self._output_locked()

    def _output_locked(self) -> str:
        outs = list(self._outputs.values())
        if not outs:
            return "none"
        if any(o.played and not o.simulated for o in outs):
            return "sounded"
        if any(o.played for o in outs):
            return "simulated"
        if any(o.pending for o in outs):
            return "pending"
        if any(o.unconfirmed for o in outs):
            return "unconfirmed"
        return "failed"

    def summary(self) -> dict:
        """The JSON the uplink sends as the event's `alert` field."""
        with self._lock:
            return self._summary_locked()

    def _summary_locked(self) -> dict:
        return {
            "decision": self.decision, "episode": self.episode, "category": self.category,
            "output": self._output_locked(), "rev": self.rev,
            "outputs": [o.describe() for o in self._outputs.values()],
        }

    @property
    def final(self) -> bool:
        """No output still owes a report (the summary will not change by itself)."""
        return self.output() != "pending"

    def subscribe(self, listener: Callable[[dict], None]) -> None:
        """Call `listener(summary)` whenever the summary's output changes."""
        with self._lock:
            self._listeners.append(listener)

    def _changed(self) -> None:
        with self._lock:
            output = self._output_locked()
            if output == self._last_output:
                return
            self._last_output = output
            if self.rev >= _MAX_DELIVERY_REVISIONS and output not in ("sounded", "failed"):
                return
            self.rev += 1
            summary = self._summary_locked()
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(summary)
            except Exception as exc:  # noqa: BLE001 - a listener must never break an output
                log.warning("alert delivery listener failed: %s", exc)


@dataclass
class _Episode:
    key: str
    event: Event
    source: str
    category: AlertCategory
    episode_id: int
    started_at: float
    last_actuation_at: float
    repeats: int = 0
    escalated: bool = False
    tracked: bool = False
    ended_at: Optional[float] = None
    delivery: Optional[AlertDelivery] = field(default=None, repr=False)

    def describe(self) -> dict:
        return {
            "key": self.key, "type": self.event.type, "source": self.source,
            "category": self.category.name, "episode_id": self.episode_id,
            "state": "active" if self.ended_at is None else "ended",
            "started_at": self.started_at, "ended_at": self.ended_at,
            "actuations": 1 + self.repeats, "escalated": self.escalated,
        }


class AlertManager:
    """Episode-aware alert policy: starts, repeats, escalates and ends alert episodes.

    The name is kept (the runtime and tests construct `AlertManager(sinks=...)`);
    `AlertPolicy` is an alias. See the module docstring for the semantics.
    """

    _RECENT = 100  # bounded history of ended episodes, for health/debug

    def __init__(
        self,
        sinks: list[AlertSink],
        cooldown_s: float = 10.0,
        escalation_window_s: float = 60.0,
        fixed_levels: Optional[dict[str, AlertLevel]] = None,
        *,
        categories: Optional[dict[str, AlertCategory]] = None,
        type_categories: Optional[dict[str, str]] = None,
        duplicate_window_s: float = 2.0,
        experimental_audible: bool = False,
    ) -> None:
        """Create the policy.

        Args:
            sinks: Actuators (console, sound, GPIO buzzer, ...).
            cooldown_s: De-duplication window for rolling-window NOTICES only
                (fixed SOFT types such as fatigue_warning). It never drops a new
                episode of any other type.
            escalation_window_s: For untyped events with no fixed level, a new
                episode within this window of the previous one starts louder.
            fixed_levels: Event type -> legacy AlertLevel (loud_types/soft_types).
            categories: Category name -> AlertCategory (defaults: DEFAULT_CATEGORIES).
            type_categories: Event type -> category name. Types not listed derive
                one from their level: CONTINUOUS -> "warning", SOFT -> "notice".
            duplicate_window_s: An untracked episode (a one-off event) is closed
                this long after it started; a duplicate inside it is merged.
            experimental_audible: The owner's explicit opt-in for the experimental
                alert families (core/behavior/experimental.py). False (default):
                they are refused, whatever fixed_levels/type_categories say.
        """
        self.experimental_audible = experimental_audible is True
        fixed_levels = dict(fixed_levels or {})
        type_categories = dict(type_categories or {})
        if not self.experimental_audible:
            gated = sorted({t for t in (*fixed_levels, *type_categories) if is_experimental(t)})
            for event_type in gated:
                fixed_levels.pop(event_type, None)
                type_categories.pop(event_type, None)
            if gated:
                log.info("alerts: experimental type(s) %s are observations only (not audible); "
                         "alerts.experimental_alerts_audible: true opts in", ", ".join(gated))
        self._sinks = sinks
        self._cooldown_s = cooldown_s
        self._escalation_window_s = escalation_window_s
        self._fixed_levels = fixed_levels
        self._categories = dict(DEFAULT_CATEGORIES if categories is None else categories)
        self._type_categories = type_categories
        self._duplicate_window_s = max(0.0, float(duplicate_window_s))
        self._last_start: dict[str, float] = {}
        self._open: dict[str, _Episode] = {}
        self._recent: deque = deque(maxlen=self._RECENT)
        self._next_id = 0
        self._lock = threading.Lock()
        self._sink_results: dict[int, _SinkResults] = {}  # keyed by id(sink)
        #: Episode ids restart at 1 in every process; this makes "episode 3"
        #: of one boot a different episode from "episode 3" of the next (M8.3).
        self._boot = uuid.uuid4().hex[:12]
        #: event.event_id -> what the policy did with that event (bounded).
        self._deliveries: OrderedDict[str, AlertDelivery] = OrderedDict()

    # ------------------------------------------------------------------ queries
    def category_for(self, event_type: str, level: Optional[AlertLevel] = None) -> AlertCategory:
        """The category an event type alerts with."""
        name = self._type_categories.get(event_type)
        if name is None:
            fixed = level if level is not None else self._fixed_levels.get(event_type)
            name = "notice" if fixed is AlertLevel.SOFT else "warning"
        return self._categories.get(name) or DEFAULT_CATEGORIES[name]

    def permits(self, event_type: str) -> bool:
        """Whether this policy may actuate for `event_type` at all (False only for
        an experimental family without the owner's opt-in)."""
        return self.experimental_audible or not is_experimental(event_type)

    def open_episodes(self) -> list[dict]:
        with self._lock:
            return [e.describe() for e in self._open.values()]

    def recent_episodes(self) -> list[dict]:
        with self._lock:
            return list(self._recent)

    _DELIVERY_MEMORY = 256

    def delivery(self, event: Event) -> Optional[AlertDelivery]:
        """What the policy did with `event` (by its event_id), or None if it was
        never handed to notify(): a log-only event, a companion node's event."""
        event_id = getattr(event, "event_id", None)
        if not event_id:
            return None
        with self._lock:
            return self._deliveries.get(event_id)

    def _record_delivery(self, event: Event, delivery: AlertDelivery) -> AlertDelivery:
        """Remember `delivery` for `event` (caller holds the lock)."""
        event_id = getattr(event, "event_id", None)
        if event_id:
            self._deliveries[event_id] = delivery
            self._deliveries.move_to_end(event_id)
            while len(self._deliveries) > self._DELIVERY_MEMORY:
                self._deliveries.popitem(last=False)
        return delivery

    def _episode_ref(self, episode_id: int) -> str:
        return f"{self._boot}-{episode_id}"

    # ------------------------------------------------------------------ episode start
    def notify(
        self,
        event: Event,
        now: Optional[float] = None,
        *,
        key: Optional[str] = None,
        level: Optional[AlertLevel] = None,
        category: Optional[str] = None,
        source: str = "module",
    ) -> AlertLevel:
        """An alertable event: start a new episode (actuating now), or merge a duplicate.

        Args:
            event: The triggering event.
            now: Monotonic seconds (injected by tests and the runtime).
            key: What the episode is tracked by. Defaults to `event.type`; pass a
                finer key (a custom rule id) when one type covers alerts that must
                not merge with each other.
            level: Force the legacy level (and, unless `category` is given, the
                category derived from it).
            category: Force the category by name.
            source: Which producer owns the episode ("module", "rules"); only that
                producer's `observe()` call advances or ends it.

        Returns:
            The legacy level dispatched, or SUPPRESSED when merged/de-duplicated.
        """
        if not self.permits(event.type):
            # An experimental observation: logged by the caller, never actuated.
            return AlertLevel.SUPPRESSED
        now = time.monotonic() if now is None else now
        episode_key = event.type if key is None else key
        with self._lock:
            current = self._open.get(episode_key)
            if current is not None:
                if current.tracked or now - current.started_at < self._duplicate_window_s:
                    # the same episode: merged
                    merged = AlertDelivery("merged", self._episode_ref(current.episode_id), current.category.name)
                    merged.episode_delivery = current.delivery
                    self._record_delivery(event, merged)
                    return AlertLevel.SUPPRESSED
                self._end(current, current.started_at + self._duplicate_window_s)
            last = self._last_start.get(episode_key)
            if level is not None:
                chosen = level
            elif event.type in self._fixed_levels:
                chosen = self._fixed_levels[event.type]
            elif last is not None and (now - last) < self._escalation_window_s:
                chosen = AlertLevel.CONTINUOUS
            else:
                chosen = AlertLevel.SOFT
            cat = (
                self._categories.get(category) or DEFAULT_CATEGORIES[category]
                if category is not None
                else self.category_for(event.type, level=level if key is not None else chosen)
            )
            # Rolling-window notices keep cooldown de-duplication: a second
            # "PERCLOS over threshold" inside the window is not a new danger.
            if (
                cat.name == "notice" and key is None and event.type in self._fixed_levels
                and last is not None and (now - last) < self._cooldown_s
            ):
                self._record_delivery(event, AlertDelivery("suppressed", None, cat.name))
                return AlertLevel.SUPPRESSED
            self._next_id += 1
            delivery = self._record_delivery(
                event, AlertDelivery("alerted", self._episode_ref(self._next_id), cat.name))
            episode = _Episode(
                key=episode_key, event=event, source=source, category=cat,
                episode_id=self._next_id, started_at=now, last_actuation_at=now, delivery=delivery,
            )
            self._open[episode_key] = episode
            self._last_start[episode_key] = now
        self._actuate(PATTERNS[cat.initial_pattern], event, cat, chosen, delivery)
        return chosen

    # ------------------------------------------------------------------ episode progress
    def observe(
        self,
        now: float,
        *,
        source: str = "module",
        active: Iterable[str] = (),
        tracked: Iterable[str] = (),
    ) -> None:
        """Advance `source`'s open episodes: repeat/escalate the active ones, end the rest.

        Args:
            now: Monotonic seconds.
            source: Only episodes started by this producer are touched.
            active: Keys whose condition is still true this frame.
            tracked: Keys this producer tracks a live condition for. An episode
                whose key is tracked stays open exactly while its key is active;
                any other episode is a one-off that ends after duplicate_window_s.
        """
        active_set, tracked_set = set(active), set(tracked)
        due: list[tuple[AlertPattern, Event, AlertCategory, Optional[AlertDelivery]]] = []
        with self._lock:
            for episode in list(self._open.values()):
                if episode.source != source:
                    continue
                if episode.key in tracked_set:
                    episode.tracked = True
                    if episode.key not in active_set:
                        self._end(episode, now)
                        continue
                    cat = episode.category
                    if (
                        cat.repeat_s > 0
                        and episode.repeats < cat.max_repeats
                        and now - episode.last_actuation_at >= cat.repeat_s
                    ):
                        escalate = now - episode.started_at >= cat.escalate_after_s
                        episode.escalated = episode.escalated or escalate
                        episode.repeats += 1
                        episode.last_actuation_at = now
                        name = cat.escalated_pattern if escalate else cat.initial_pattern
                        due.append((PATTERNS[name], episode.event, cat, episode.delivery))
                elif now - episode.started_at >= self._duplicate_window_s:
                    self._end(episode, now)
        for pattern, event, cat, delivery in due:
            self._actuate(pattern, event, cat, cat.legacy_level, delivery)

    def process_frame(
        self,
        now: float,
        events: Iterable[Event],
        *,
        alertable: Iterable[str],
        active: Iterable[str],
        tracked: Iterable[str],
        source: str = "module",
    ) -> list[AlertLevel]:
        """One frame of a module: start episodes for alertable events, then advance them."""
        alertable_set = set(alertable)
        levels = [self.notify(e, now=now, source=source) for e in events if e.type in alertable_set]
        self.observe(now, source=source, active=active, tracked=tracked)
        return levels

    def end_all(self, now: float) -> None:
        """Close every open episode (monitoring stopped or reconfigured)."""
        with self._lock:
            for episode in list(self._open.values()):
                self._end(episode, now)

    # ------------------------------------------------------------------ internals
    def _end(self, episode: _Episode, now: float) -> None:
        episode.ended_at = now
        self._open.pop(episode.key, None)
        self._recent.append(episode.describe())

    def _actuate(self, pattern: AlertPattern, event: Event, category: AlertCategory, level: AlertLevel,
                 delivery: Optional[AlertDelivery] = None) -> None:
        for sink in self._sinks:
            result = self._sink_results.setdefault(id(sink), _SinkResults())
            play = getattr(sink, "play", None)
            reports = callable(play) and bool(getattr(sink, "reports_completion", False))
            tracked = delivery is not None and _in_cabin(sink)
            report = delivery.commanded(sink, reports=reports) if tracked and delivery is not None else None
            try:
                if callable(play):
                    if reports:
                        play(pattern, event, category, report=report)
                    else:
                        play(pattern, event, category)
                else:
                    sink.emit(event, level)
                result.ok += 1
                result.consecutive_failures = 0
            except Exception as exc:  # noqa: BLE001 - one broken sink must not silence the rest
                result.failed += 1
                result.consecutive_failures += 1
                result.last_error = str(exc) or type(exc).__name__
                log.error("alert sink %s failed: %s", type(sink).__name__, exc)
                if tracked and delivery is not None:
                    delivery.raised(sink, result.last_error, reported=report)

    def actuators(self) -> list[Actuator]:
        """Every configured alert output and its SOFTWARE readiness.

        Ready = the output initialised, reports itself ready, and its last command
        did not raise. "Commanded" is never "heard": physically_verified is False
        for every output today. An output whose command keeps failing is still
        commanded on every alert (bounded by the episode policy), never retried
        in a loop of its own.
        """
        out = []
        for sink in self._sinks:
            readiness: Any = getattr(sink, "readiness", lambda: (True, ""))
            ok, detail = readiness()
            results = self._sink_results.get(id(sink), _SinkResults())
            if results.consecutive_failures:
                ok, detail = False, f"last alert command failed: {results.last_error}"
            out.append(Actuator(
                name=str(getattr(sink, "name", type(sink).__name__)), kind=str(getattr(sink, "kind", "custom")),
                audible=bool(getattr(sink, "audible", False)), ready=ok, detail=detail,
                commands_ok=results.ok, commands_failed=results.failed,
            ))
        return out

    def end_source(self, now: float, source: str) -> None:
        """Close every open episode of one producer (its view of the world went stale)."""
        with self._lock:
            for episode in list(self._open.values()):
                if episode.source == source:
                    self._end(episode, now)

    def open_keys(self, source: str) -> set[str]:
        with self._lock:
            return {e.key for e in self._open.values() if e.source == source}


#: The policy's descriptive name; `AlertManager` is kept for existing callers.
AlertPolicy = AlertManager
