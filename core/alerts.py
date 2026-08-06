"""Pluggable alert sinks and escalation logic.

Escalation policy:
  * Event types listed in config as `loud_types` or `soft_types` always dispatch
    at that fixed level (e.g. "drowsiness"/"phone_use"/"eyes_down" are always
    loud; "fatigue_warning"/"frequent_glancing" are always soft) — these are
    already the product of a sustained-condition or rolling-window check, so
    they don't need time-based escalation on top.
  * Any other event type escalates by repetition: a brand-new alert (or one
    whose type hasn't fired in over `escalation_window_s`) plays a soft chime;
    a repeat within `escalation_window_s` escalates to a continuous tone.
  * No alert of the same type fires more than once per `cooldown_s`, regardless
    of level (debounces rapid repeated detections).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time
from enum import Enum
from typing import Optional, Protocol

from core.events import Event


class AlertLevel(Enum):
    """Severity of a dispatched alert."""

    SUPPRESSED = "suppressed"
    SOFT = "soft"
    CONTINUOUS = "continuous"


class AlertSink(Protocol):
    """A destination that can render an alert (console, sound, GPIO buzzer, ...)."""

    def emit(self, event: Event, level: AlertLevel) -> None:
        """Render the given event at the given alert level."""
        ...


class ConsoleSink:
    """Prints alerts to stdout."""

    def emit(self, event: Event, level: AlertLevel) -> None:
        """Print a one-line alert message."""
        print(f"[ALERT:{level.value}] {event.module_name}: {event.type} (confidence={event.confidence:.2f})")


class SoundSink:
    """Plays an audible tone: short chime for SOFT, longer tone for CONTINUOUS.

    Prefers simpleaudio (works on Mac and Pi). Falls back to `aplay` on Linux
    if simpleaudio isn't available, and to the terminal bell as a last resort.
    """

    def __init__(self) -> None:
        """Detect the best available playback backend."""
        try:
            import simpleaudio  # type: ignore

            self._simpleaudio = simpleaudio
        except ImportError:
            self._simpleaudio = None

    def emit(self, event: Event, level: AlertLevel) -> None:
        """Play a tone whose length reflects the alert level.

        Playback happens on a short-lived daemon thread: `wait_done()`/`aplay`
        block for the tone's full duration, and emit() is called from the
        inference loop — a 0.6s continuous-alert tone must not freeze detection
        for 0.6s (≈18 dropped frames at 30fps).
        """
        if level == AlertLevel.SUPPRESSED:
            return
        duration_s = 0.15 if level == AlertLevel.SOFT else 0.6
        threading.Thread(target=self._play_tone, args=(duration_s,), daemon=True).start()

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
        subprocess.run(["aplay", "-q", path], check=False)
        os.unlink(path)


class GPIOBuzzerSink:
    """Drives a GPIO buzzer on Raspberry Pi. Never imported/instantiated on Mac."""

    def __init__(self, pin: int) -> None:
        """Bind to the given BCM pin. `gpiozero` is imported lazily so Mac never touches it."""
        from gpiozero import Buzzer  # lazy import: this line never executes on Mac

        self._buzzer = Buzzer(pin)

    def emit(self, event: Event, level: AlertLevel) -> None:
        """Buzz briefly for SOFT alerts, longer for CONTINUOUS ones.

        The buzz runs on a daemon thread — the inline time.sleep used to stall
        the inference loop for up to 0.6s per alert on the Pi.
        """
        if level == AlertLevel.SUPPRESSED:
            return
        on_s = 0.15 if level == AlertLevel.SOFT else 0.6
        threading.Thread(target=self._buzz, args=(on_s,), daemon=True).start()

    def _buzz(self, on_s: float) -> None:
        self._buzzer.on()
        time.sleep(on_s)
        self._buzzer.off()


class AlertManager:
    """Applies cooldown/escalation policy and fans a notification out to all sinks."""

    def __init__(
        self,
        sinks: list[AlertSink],
        cooldown_s: float = 10.0,
        escalation_window_s: float = 60.0,
        fixed_levels: Optional[dict[str, AlertLevel]] = None,
    ) -> None:
        """Create a manager tracking per-event-type alert history.

        Args:
            sinks: Alert sinks to notify (console, sound, GPIO buzzer, ...).
            cooldown_s: Minimum seconds between two alerts of the same event type.
            escalation_window_s: A repeat within this window escalates to CONTINUOUS
                (only for event types not present in `fixed_levels`).
            fixed_levels: Event type -> AlertLevel overrides that bypass time-based
                escalation entirely and always dispatch at that level.
        """
        self._sinks = sinks
        self._cooldown_s = cooldown_s
        self._escalation_window_s = escalation_window_s
        self._fixed_levels = fixed_levels or {}
        self._last_alert_time: dict[str, float] = {}

    def notify(self, event: Event, now: Optional[float] = None) -> AlertLevel:
        """Apply escalation/cooldown policy for `event` and dispatch to all sinks.

        Args:
            event: The triggering event.
            now: Override for the current time (monotonic seconds); used by tests.

        Returns:
            The AlertLevel that was (or would have been) dispatched.
        """
        now = time.monotonic() if now is None else now
        last = self._last_alert_time.get(event.type)

        if last is not None and (now - last) < self._cooldown_s:
            return AlertLevel.SUPPRESSED

        if event.type in self._fixed_levels:
            level = self._fixed_levels[event.type]
        elif last is not None and (now - last) < self._escalation_window_s:
            level = AlertLevel.CONTINUOUS
        else:
            level = AlertLevel.SOFT

        self._last_alert_time[event.type] = now
        for sink in self._sinks:
            sink.emit(event, level)
        return level
