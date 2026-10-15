"""The monitoring configuration a person asked for, kept on the device.

What someone configured is a decision they made, not a message in flight. It
lived only in memory and only ever arrived from the cloud, which meant three
things were true and none of them were obvious:

  - a reboot silently emptied it, and the device went back to watching whatever
    its config file said
  - offline, it could never be set at all, because the only source was the
    backend — so a device somewhere with no Wi-Fi ignored the whole feature
  - nothing said so. The rule list was simply empty and evaluation was skipped

For a feature whose entire promise is "tell SensorDyme what to watch for", that
is the feature not existing whenever it is most needed.

Stored as one small JSON file next to the other device state. Written
atomically, because the alternative is a half-written file that makes the
device forget its configuration on the next boot — the exact failure this
module exists to prevent.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from core.logging_setup import get_logger

log = get_logger("monitoring")

#: Overridable like the backend's own state paths (SENSORDYME_DB and friends),
#: so a process — the test suite above all — can keep its hands off the real
#: device configuration. Unset on a device: the default is unchanged.
DEFAULT_PATH = Path(os.environ.get("SENSORDYME_MONITORING_PATH") or "data/monitoring.json")


class MonitoringConfig:
    """The active custom configuration: what to watch, and what it was asked as.

    `instruction` is kept alongside the rules deliberately. A rule list is what
    the machine executes; the sentence someone typed is what they will recognise
    when the app asks them to confirm the device is doing the right thing.
    """

    def __init__(self, path: Path | str = DEFAULT_PATH) -> None:
        self._path = Path(path)
        self.instruction: str = ""
        self.rules: list[dict] = []
        self.version: str = ""
        self.updated_at: float = 0.0
        self.overrides_mode: bool = False

    # ------------------------------------------------------------------ state
    @property
    def active(self) -> bool:
        """Whether a custom configuration is in charge of this device."""
        return bool(self.rules) and self.overrides_mode

    def describe(self) -> dict:
        """What the app shows: the sentence asked, and what it became."""
        return {
            "instruction": self.instruction,
            "rule_count": len(self.rules),
            "version": self.version,
            "updated_at": self.updated_at,
            "overrides_mode": self.overrides_mode,
            "active": self.active,
        }

    # ------------------------------------------------------------------ io
    def load(self) -> bool:
        """Read the stored configuration. False if there was none to read."""
        try:
            raw = json.loads(self._path.read_text())
        except FileNotFoundError:
            return False
        except (OSError, ValueError) as exc:
            # A corrupt file must not stop the device monitoring. Better to run
            # the configured base mode than to refuse to start.
            log.warning("monitoring config unreadable (%s) — starting without it", exc)
            return False

        self.instruction = str(raw.get("instruction") or "")
        self.rules = list(raw.get("rules") or [])
        self.version = str(raw.get("version") or "")
        self.updated_at = float(raw.get("updated_at") or 0.0)
        # Defaults FALSE. A file written by an older format, hand-edited, or
        # produced by anything that omits the key used to activate AI-custom
        # monitoring and silence the configured module -- the most disruptive
        # possible reading of an absent field. The safe default is "keep running
        # the module the operator configured"; overriding it is an explicit act.
        self.overrides_mode = bool(raw.get("overrides_mode", False))
        if self.rules:
            log.info(
                "restored custom monitoring: %d rule(s) — %r",
                len(self.rules), self.instruction[:80] or "(no instruction recorded)",
            )
        return True

    def save(self, rules: list[dict], instruction: str = "", version: str = "",
             overrides_mode: bool = True) -> None:
        """Replace the configuration and write it down."""
        self.rules = list(rules or [])
        # Never carry a previous instruction over a new rule set. Keeping it
        # "so something is shown" pairs the sentence someone typed with rules
        # that do something else — the device then reports, confidently, that it
        # is watching for a thing it is not. Silence is better than that.
        self.instruction = instruction
        self.version = version
        self.overrides_mode = overrides_mode
        self.updated_at = time.time()
        self._write()

    def clear(self) -> None:
        """Return the device to its base monitoring mode."""
        self.rules = []
        self.instruction = ""
        self.version = ""
        self.overrides_mode = False
        self.updated_at = time.time()
        self._write()

    def _write(self) -> None:
        payload = {
            "instruction": self.instruction,
            "rules": self.rules,
            "version": self.version,
            "updated_at": self.updated_at,
            "overrides_mode": self.overrides_mode,
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic: a half-written file would make the device forget its
            # configuration on the next boot, which is the failure this whole
            # module exists to prevent.
            fd, tmp = tempfile.mkstemp(dir=str(self._path.parent), suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self._path)
        except OSError as exc:
            # Losing the write is bad; crashing the monitoring loop is worse.
            log.error("could not save monitoring config: %s", exc)


def load_or_empty(path: Path | str = DEFAULT_PATH) -> MonitoringConfig:
    config = MonitoringConfig(path)
    config.load()
    return config
