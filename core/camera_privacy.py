"""Camera privacy while monitoring is stopped (M8.4, owner Decision C).

M8.3 added a camera fallback to the live stream: whenever the runtime was not
drawing -- monitoring stopped included -- the dashboard streamed the raw camera
to anyone holding the device token, on by default. A stop therefore left a live
camera on the LAN that the owner had not asked for.

The decision: **while monitoring is stopped, `/stream` delivers no camera frames
to a remote viewer** unless the device's OWNER has explicitly turned on a
separate *camera-only* mode. This module holds that one setting and the rule.

What a remote viewer may receive (`remote_stream`):

  monitoring          monitoring is not stopped: the analysed view, or the raw
                      camera labelled "camera" while a running module cannot draw
                      (degraded) -- the camera is in use for monitoring anyway
  camera_only         stopped, and the owner enabled camera-only: the raw camera,
                      labelled X-Frame-Source: camera_only ("not monitoring")
  off_while_stopped   stopped, camera-only off (the default): no frames at all;
                      heartbeats say X-Stream-Status: camera_off_while_stopped

What stays allowed locally (docs/PRIVACY.md, docs/VIDEO_TRANSPORT.md §2a):
a viewer on the device itself (a loopback peer: 127.0.0.1 / ::1) still gets the
camera while stopped, labelled X-Frame-Source: camera_local, unless
`dashboard.local_camera_while_stopped: false`. A process on the device can open
the camera directly anyway (scripts/hw_validation/capture_calibration_frames.py
does, with the service stopped), so loopback grants nothing new; the dashboard
token is still required wherever it is configured.

How the setting changes: only by the owner, through the backend
(POST /v1/devices/{id}/camera-mode, owner-scoped) and the signed command path
(`set_camera_mode` in the ingest response, sd-auth-v1). ENABLING it is refused
unless the command arrived with a verified envelope, whatever
`cloud.command_auth` says (core/uplink.py); disabling is the safe direction and
is accepted like any other command. There is deliberately no dashboard endpoint
for it: the dashboard token is a LAN credential that may be shared with people
who are not the owner, and they must not be able to turn the camera back on.

Persistence: data/device_state.json key `camera_privacy`
({"schema": 1, "camera_only": true, ...}), written atomically by
core/device_state.py. Anything else -- no key (every device upgraded from M8.3),
an unreadable file, another schema, a non-boolean -- reads as OFF. The backend
keeps the owner's choice too and re-sends it when the device reports something
else, so a lost file converges back to what the owner chose.
"""

from __future__ import annotations

import ipaddress
import threading
import time
from dataclasses import dataclass
from typing import Optional

from core import device_state
from core.logging_setup import get_logger

log = get_logger("camera_privacy")

SCHEMA = 1
#: Key in data/device_state.json.
STATE_KEY = "camera_privacy"

#: What a remote /stream viewer may receive right now (reported as `remote_stream`).
STREAM_MONITORING = "monitoring"
STREAM_CAMERA_ONLY = "camera_only"
STREAM_OFF = "off_while_stopped"
#: The dashboard is disabled: there is no stream at all.
STREAM_UNAVAILABLE = "unavailable"

#: X-Frame-Source values (core/dashboard.py `_part_header`).
SOURCE_ANNOTATED = "annotated"
SOURCE_CAMERA = "camera"
SOURCE_CAMERA_ONLY = "camera_only"
SOURCE_CAMERA_LOCAL = "camera_local"

#: Heartbeat X-Stream-Status while frames are withheld for privacy.
STATUS_CAMERA_OFF = "camera_off_while_stopped"

#: Longest command id kept (the backend sends 12 hex chars).
_MAX_ID = 64


def peer_is_local(host: Optional[str]) -> bool:
    """Whether a connection comes from this machine (loopback, incl. v4-mapped)."""
    try:
        address = ipaddress.ip_address(str(host or "").split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_loopback


@dataclass(frozen=True)
class CameraPrivacySetting:
    """The owner's choice. Frozen: replaced as a whole, never edited in place."""

    camera_only: bool = False
    since: Optional[float] = None
    command_id: Optional[str] = None
    by: Optional[str] = None

    def to_state(self) -> dict:
        return {"schema": SCHEMA, "camera_only": self.camera_only, "since": self.since,
                "command_id": self.command_id, "by": self.by}

    @classmethod
    def from_state(cls, raw) -> "CameraPrivacySetting":
        """Strict: only an exact `{"schema": 1, "camera_only": true}` is ON."""
        if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
            return cls()
        since = raw.get("since")
        command_id = raw.get("command_id")
        by = raw.get("by")
        return cls(
            camera_only=raw.get("camera_only") is True,
            since=float(since) if isinstance(since, (int, float)) and not isinstance(since, bool) else None,
            command_id=str(command_id)[:_MAX_ID] if isinstance(command_id, str) and command_id else None,
            by=str(by)[:32] if isinstance(by, str) and by else None,
        )


class CameraPrivacy:
    """Thread-safe holder of the setting (the dashboard's HTTP threads read it)."""

    def __init__(self, setting: Optional[CameraPrivacySetting] = None, *, persist: bool = True) -> None:
        self._setting = setting or CameraPrivacySetting()
        self._persist = persist
        self._lock = threading.Lock()
        #: False when the last change could not be written down (reported).
        self._persisted = True

    @classmethod
    def load(cls) -> "CameraPrivacy":
        """The saved setting; OFF when there is none (every M8.3 device) or it is unreadable."""
        return cls(CameraPrivacySetting.from_state(device_state.load().get(STATE_KEY)))

    @property
    def setting(self) -> CameraPrivacySetting:
        with self._lock:
            return self._setting

    @property
    def camera_only(self) -> bool:
        return self.setting.camera_only

    def apply(self, camera_only: bool, *, command_id: Optional[str] = None, by: str = "owner") -> bool:
        """Set the owner's choice and write it down. Returns whether it changed.

        `camera_only` must be a real bool: a command saying "true" or 1 is not
        an owner turning a camera on.
        """
        if not isinstance(camera_only, bool):
            raise TypeError("camera_only must be a bool")
        cid = str(command_id)[:_MAX_ID] if command_id else None
        with self._lock:
            changed = self._setting.camera_only != camera_only
            since = time.time() if changed or self._setting.since is None else self._setting.since
            self._setting = CameraPrivacySetting(camera_only, since, cid or self._setting.command_id, by)
            setting = self._setting
        if self._persist:
            ok = device_state.update(**{STATE_KEY: setting.to_state()})
            with self._lock:
                self._persisted = ok
            if not ok:
                log.error("camera-only mode is %s but could not be saved; after a restart the device "
                          "uses the saved value until the backend re-sends the owner's choice",
                          "ON" if camera_only else "OFF")
        if changed:
            log.warning("[PRIVACY] camera-only while stopped turned %s by %s",
                        "ON" if camera_only else "OFF", by)
        return changed

    def remote_stream(self, stopped: bool) -> str:
        """What a remote /stream viewer may receive now."""
        if not stopped:
            return STREAM_MONITORING
        return STREAM_CAMERA_ONLY if self.camera_only else STREAM_OFF

    def report(self, *, stopped: bool, dashboard: bool = True, local_while_stopped: bool = True) -> dict:
        """The device's own account of camera access (reported upstream as `camera`)."""
        setting = self.setting
        with self._lock:
            persisted = self._persisted
        return {
            "schema": SCHEMA,
            "camera_only": setting.camera_only,
            "remote_stream": self.remote_stream(stopped) if dashboard else STREAM_UNAVAILABLE,
            "local_while_stopped": bool(local_while_stopped),
            "since": setting.since,
            "last_command_id": setting.command_id,
            "persisted": persisted,
        }


def parse_command(raw) -> Optional[dict]:
    """A well-formed `set_camera_mode` command body, or None.

    {"camera_only": <bool>, "id": "<id>"}; anything else is refused whole.
    """
    if not isinstance(raw, dict) or not isinstance(raw.get("camera_only"), bool):
        return None
    command_id = raw.get("id")
    if command_id is not None and (not isinstance(command_id, str) or not 0 < len(command_id) <= _MAX_ID):
        return None
    return {"camera_only": raw["camera_only"], "id": command_id}
