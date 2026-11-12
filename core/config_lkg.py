"""The last configuration this device actually ran with (M8 lifecycle).

A config.yaml that cannot be used -- truncated by a power cut during a write,
hand-edited into invalid YAML, a schema error introduced by a migration --
used to stop the service for good (exit 78, `RestartPreventExitStatus=78`).
That ended the reboot loop it replaced, but it also left the device dark: no
monitoring, no buzzer, no `/api/health`, no BLE, no uplink, until a person with
SSH fixed the file.

Now the device keeps a copy of the last config.yaml that brought it to READY
(`data/config.last_good.yaml`, beside the rest of the device's state, 0600
because it holds the same device key and dashboard token). When config.yaml
cannot be used, boot runs from that copy instead and SAYS SO: the fault is
printed, logged to the event log, and carried in every health report (`config`
block) so the app and the backend see it. config.yaml itself is never touched
-- the broken file stays for a person to inspect -- and the copy is replaced
only after a boot from a good config.yaml reaches READY again.

With no usable copy (a first boot, or both files broken), the old behaviour
stands: exit 78 and stop, never a restart or reboot loop.

Same copy, same identity: this is the device's own earlier config.yaml, so the
device key, cloud address and dashboard token are the ones it already had.
Nothing here generates or rotates a credential.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

#: Overrides where the copy lives (tests, sim/). Default: <config dir>/data/.
ENV_PATH = "SENSORDYME_CONFIG_LKG_PATH"
FILE_NAME = "config.last_good.yaml"


@dataclass(frozen=True)
class ConfigFault:
    """config.yaml could not be used; the device is running on the last good copy."""

    path: str
    reason: str
    fallback_path: str

    def describe(self) -> dict:
        return {"source": "last_known_good", "path": self.path, "fault": self.reason,
                "running_from": self.fallback_path}


def last_good_path(config_path: str | os.PathLike) -> Path:
    override = os.environ.get(ENV_PATH)
    if override:
        return Path(override)
    return Path(config_path).resolve().parent / "data" / FILE_NAME


def read(path: str | os.PathLike) -> tuple[Optional[dict], Optional[str]]:
    """(config, None), or (None, why it cannot be used: unreadable, not YAML, not a mapping)."""
    try:
        with open(path) as handle:
            config = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError, UnicodeDecodeError) as exc:
        return None, f"cannot read {path}: {exc}"
    if not isinstance(config, dict):
        return None, f"{path} must be a YAML mapping, got {type(config).__name__}"
    return config, None


def load_last_good(config_path: str | os.PathLike) -> tuple[Optional[dict], Optional[Path]]:
    """The saved copy if it exists, parses and passes the schema; else (None, None)."""
    from core.config_schema import validate

    path = last_good_path(config_path)
    if not path.exists():
        return None, None
    config, problem = read(path)
    if config is None or problem:
        return None, None
    errors, _warnings = validate(config)
    if errors:
        return None, None
    return config, path


def torn_write(config_path: str | os.PathLike) -> Optional[str]:
    """Why config.yaml looks like an interrupted write of the last good config, or None.

    A truncated YAML file often still parses and can even pass the schema
    (the sections after the cut are simply missing, and their defaults run
    instead). The tell is that the file is a strict prefix of the last good
    copy that stops mid-line: a person deleting trailing sections leaves a
    whole last line; a write cut off by a power loss does not.
    """
    try:
        current = Path(config_path).read_bytes()
        good = last_good_path(config_path).read_bytes()
    except OSError:
        return None
    if len(current) < len(good) and good.startswith(current) and not current.endswith(b"\n"):
        return (f"{config_path} is a truncated copy of the last known good config "
                f"({len(current)} of {len(good)} bytes, cut mid-line): a write was interrupted")
    return None


def save_last_good(config_path: str | os.PathLike) -> bool:
    """Record config.yaml as the last good config. Atomic, 0600, and only when it changed.

    Called once the runtime has reached READY on it. Copies the file's exact
    bytes (comments and all), not a re-serialisation. False when it could not
    be written; the device carries on either way.
    """
    target = last_good_path(config_path)
    try:
        data = Path(config_path).read_bytes()
    except OSError:
        return False
    try:
        if target.exists() and target.read_bytes() == data:
            return True  # unchanged: no SD-card write
    except OSError:
        pass
    return atomic_write_bytes(target, data, mode=0o600)


def atomic_write_bytes(path: str | os.PathLike, data: bytes, *, mode: Optional[int] = None) -> bool:
    """tmp in the same directory + fsync + rename + directory fsync.

    A power cut at any moment leaves either the old file or the new one, never
    a truncated mix. `mode` None keeps the existing file's mode (or the umask
    default for a new file).
    """
    path = Path(path)
    tmp = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if mode is None:
            with contextlib.suppress(OSError):
                mode = path.stat().st_mode & 0o777
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
        tmp = None
        with contextlib.suppress(OSError):  # not every platform can fsync a directory
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        return True
    except OSError:
        return False
    finally:
        if tmp is not None:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
