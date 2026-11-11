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
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Optional, Protocol

from core.events import Event
from core.logging_setup import get_logger

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


class _PatternPlayer:
    """Plays patterns one at a time on a single worker thread.

    One physical buzzer, many overlapping requests: patterns are queued (at most
    `_DEPTH` waiting) instead of interleaving, so a soft chirp can never cut a
    critical pattern short. When the queue is full the least urgent waiting
    pattern is dropped, never the most urgent.
    """

    _DEPTH = 2
    _RANK: ClassVar[dict[str, int]] = {"notice": 0, "warning": 1, "impaired": 2, "critical": 3}

    def __init__(self, render_step) -> None:
        self._render_step = render_step  # callable(on_s) that blocks for on_s
        self._lock = threading.Lock()
        self._pending: list[tuple[int, AlertPattern]] = []
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def play(self, pattern: AlertPattern, category_name: str) -> None:
        rank = self._RANK.get(category_name, 1)
        with self._lock:
            if len(self._pending) >= self._DEPTH:
                weakest = min(range(len(self._pending)), key=lambda i: self._pending[i][0])
                if self._pending[weakest][0] > rank:
                    return  # everything waiting is more urgent than this
                self._pending.pop(weakest)
            self._pending.append((rank, pattern))
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()
        self._wake.set()

    def _run(self) -> None:
        while True:
            with self._lock:
                if not self._pending:
                    self._thread = None
                    return
                best = max(range(len(self._pending)), key=lambda i: self._pending[i][0])
                _, pattern = self._pending.pop(best)
            for on_s, off_s in pattern.steps:
                self._render_step(on_s)
                if off_s:
                    time.sleep(off_s)


class ConsoleSink:
    """Prints alerts to stdout."""

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

    def __init__(self) -> None:
        """Detect the best available playback backend."""
        try:
            import simpleaudio  # type: ignore[import-not-found]

            self._simpleaudio = simpleaudio
        except ImportError:
            self._simpleaudio = None
        self._player = _PatternPlayer(self._play_tone_safe)

    def play(self, pattern: AlertPattern, event: Event, category: AlertCategory) -> None:
        """Render the pattern as tone bursts on the sound device (never on macOS; see emit)."""
        if sys.platform == "darwin":
            return
        self._player.play(pattern, category.name)

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

    def _play_tone_safe(self, duration_s: float) -> None:
        """Tone playback that can never crash-spam: a Pi with no PCM audio device
        raises from simpleaudio on every alert — degrade to the terminal bell."""
        try:
            self._play_tone(duration_s)
        except Exception:  # noqa: BLE001 - audio is best-effort by design
            print("\a", end="", flush=True)

    def _play_tone(self, duration_s: float) -> None:
        """Play (or emulate) a tone of the given length using the best available backend."""
        if self._simpleaudio is not None:
            self._play_via_simpleaudio(duration_s)
        elif sys.platform.startswith("linux") and shutil.which("aplay"):
            self._play_via_aplay(duration_s)
        else:
            print("\a", end="", flush=True)  # terminal bell fallback

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
        # One physical buzzer, many overlapping requests. Each buzz used to own
        # an on/sleep/off thread, so a short buzz that overlapped a longer one
        # switched the buzzer OFF in the middle of it: a soft chime 50ms before
        # a loud alarm cut the alarm to ~0.1s. `_off_at` is the latest moment
        # any outstanding request wants the buzzer on; only the request that
        # owns that moment may turn it off.
        self._lock = threading.Lock()
        self._off_at = 0.0

    def emit(self, event: Event, level: AlertLevel) -> None:
        """Buzz briefly for SOFT alerts, longer for CONTINUOUS ones.

        The buzz runs on a daemon thread — the inline time.sleep used to stall
        the inference loop for up to 0.6s per alert on the Pi.
        """
        if level == AlertLevel.SUPPRESSED:
            return
        on_s = 0.15 if level == AlertLevel.SOFT else 0.6
        threading.Thread(target=self._buzz, args=(on_s,), daemon=True).start()

    def play(self, pattern: AlertPattern, event: Event, category: AlertCategory) -> None:
        """Render the pattern as buzzer pulses on one worker thread (patterns never interleave)."""
        if not hasattr(self, "_player"):
            self._player = _PatternPlayer(self._buzz)
        self._player.play(pattern, category.name)

    def _buzz(self, on_s: float) -> None:
        try:
            with self._lock:
                self._off_at = max(self._off_at, time.monotonic() + on_s)
                self._buzzer.on()
            time.sleep(on_s)
            with self._lock:
                # A later or longer request extended _off_at: leave it on.
                if time.monotonic() >= self._off_at - 1e-3:
                    self._buzzer.off()
        except Exception as exc:  # noqa: BLE001 - a GPIO fault must never kill anything
            log.warning("buzzer failed: %s", exc)


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
        """
        self._sinks = sinks
        self._cooldown_s = cooldown_s
        self._escalation_window_s = escalation_window_s
        self._fixed_levels = fixed_levels or {}
        self._categories = dict(DEFAULT_CATEGORIES if categories is None else categories)
        self._type_categories = dict(type_categories or {})
        self._duplicate_window_s = max(0.0, float(duplicate_window_s))
        self._last_start: dict[str, float] = {}
        self._open: dict[str, _Episode] = {}
        self._recent: deque = deque(maxlen=self._RECENT)
        self._next_id = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ queries
    def category_for(self, event_type: str, level: Optional[AlertLevel] = None) -> AlertCategory:
        """The category an event type alerts with."""
        name = self._type_categories.get(event_type)
        if name is None:
            fixed = level if level is not None else self._fixed_levels.get(event_type)
            name = "notice" if fixed is AlertLevel.SOFT else "warning"
        return self._categories.get(name) or DEFAULT_CATEGORIES[name]

    def open_episodes(self) -> list[dict]:
        with self._lock:
            return [e.describe() for e in self._open.values()]

    def recent_episodes(self) -> list[dict]:
        with self._lock:
            return list(self._recent)

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
        now = time.monotonic() if now is None else now
        episode_key = event.type if key is None else key
        with self._lock:
            current = self._open.get(episode_key)
            if current is not None:
                if current.tracked or now - current.started_at < self._duplicate_window_s:
                    return AlertLevel.SUPPRESSED  # the same episode: merged
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
                return AlertLevel.SUPPRESSED
            self._next_id += 1
            episode = _Episode(
                key=episode_key, event=event, source=source, category=cat,
                episode_id=self._next_id, started_at=now, last_actuation_at=now,
            )
            self._open[episode_key] = episode
            self._last_start[episode_key] = now
        self._actuate(PATTERNS[cat.initial_pattern], event, cat, chosen)
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
        due: list[tuple[AlertPattern, Event, AlertCategory]] = []
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
                        due.append((PATTERNS[name], episode.event, cat))
                elif now - episode.started_at >= self._duplicate_window_s:
                    self._end(episode, now)
        for pattern, event, cat in due:
            self._actuate(pattern, event, cat, cat.legacy_level)

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

    def _actuate(self, pattern: AlertPattern, event: Event, category: AlertCategory, level: AlertLevel) -> None:
        for sink in self._sinks:
            try:
                play = getattr(sink, "play", None)
                if callable(play):
                    play(pattern, event, category)
                else:
                    sink.emit(event, level)
            except Exception as exc:  # noqa: BLE001 - one broken sink must not silence the rest
                log.error("alert sink %s failed: %s", type(sink).__name__, exc)


#: The policy's descriptive name; `AlertManager` is kept for existing callers.
AlertPolicy = AlertManager
