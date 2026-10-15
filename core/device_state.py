"""What the owner chose on this device, kept across restarts.

Two choices used to live only in memory, so any restart — a power blip, a
watchdog restart, a crash — quietly undid them:

  preset   The mode picked in the app or dashboard. Boot derived the mode from
           config.yaml's `module:` instead, and the backend stops re-sending
           `desired_preset` once the device has reported it, so nothing ever
           re-asserted the choice: a device switched to Elderly Monitoring came
           back as Driver Safety and ran no fall detection.
  zones    The areas drawn in the app. The backend re-sends them once the
           device reconnects, but an offline baby monitor came back without its
           crib zone at all.

The preset is remembered together with the config.yaml `module:` it was chosen
under. Editing `module:` is a newer, deliberate choice (scripts/demo_cycle.sh
cycles modes exactly that way), so a saved preset only applies while config.yaml
still names the module it was saved against.

Writes are atomic (tmp + fsync + replace) for the same reason
core/monitoring_config.py's are: a half-written file on a device that is about
to lose power is worse than none. A file that cannot be read is treated as
empty — every caller then falls back to config.yaml.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional

from core.logging_setup import get_logger

log = get_logger("device_state")

DEFAULT_PATH = "data/device_state.json"

_LOCK = threading.Lock()


def _path() -> Path:
    return Path(os.environ.get("SENSORDYME_DEVICE_STATE_PATH") or DEFAULT_PATH)


def load() -> dict:
    """The saved state, or {} when there is none or it cannot be read."""
    try:
        raw = json.loads(_path().read_text())
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def update(**fields) -> bool:
    """Merge `fields` into the saved state. False if it could not be written."""
    with _LOCK:
        state = load()
        state.update(fields)
        path = _path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            with open(tmp, "w") as handle:
                json.dump(state, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
            return True
        except (OSError, TypeError, ValueError) as exc:
            log.warning("could not save device state to %s: %s", path, exc)
            return False


def save_preset(name: str, config_module: Optional[str]) -> bool:
    """Remember the active preset (optionally "preset:variant")."""
    return update(preset=name, config_module=config_module)


def saved_preset(config_module: Optional[str]) -> Optional[str]:
    """The saved preset, if it was chosen under this same config.yaml module."""
    state = load()
    preset = state.get("preset")
    if isinstance(preset, str) and preset and state.get("config_module") == config_module:
        return preset
    return None


def save_zones(zones: list[dict]) -> bool:
    """Remember the zones the app delivered (the uplink's raw dicts)."""
    return update(zones=[z for z in zones if isinstance(z, dict)])


def saved_zones() -> Optional[list[dict]]:
    """The app's zones from before the restart, or None if it never sent any."""
    zones = load().get("zones")
    if not isinstance(zones, list):
        return None
    return [z for z in zones if isinstance(z, dict)]
