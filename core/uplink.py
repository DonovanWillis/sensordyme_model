"""Cloud uplink: connects the device to the SensorDyme cloud backend (cloud/).

Outbound-only, like any consumer camera device behind a NAT: a daemon thread
periodically (1) pushes queued alert events to the backend and (2) polls for
remote commands (currently: set_preset from the mobile app). Frames NEVER go
through this channel — only events and status, preserving the privacy design.

Failure-tolerant by design: if the backend is unreachable, events queue in
memory (bounded) and the device keeps working standalone; the dashboard and
local alerts are never affected.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional


class CloudUplink:
    """Background sync loop between this device and the cloud backend."""

    def __init__(self, config: dict) -> None:
        self._base_url = str(config.get("base_url", "")).rstrip("/")
        self._device_key = str(config.get("device_key", ""))
        self._poll_interval_s = float(config.get("poll_interval_s", 5.0))
        self._queue: deque[dict] = deque(maxlen=int(config.get("queue_max", 500)))
        self._lock = threading.Lock()
        self._active_preset: Optional[str] = None
        self._pending_preset: Optional[str] = None
        # Rule set currently enforced, reported upstream so the backend only
        # re-sends on a real change instead of on every poll.
        self._rules_version: str = ""
        self._pending_rules: Optional[list[dict]] = None
        self._health: Optional[dict] = None
        self._zones_version: str = ""
        self._pending_zones: Optional[list[dict]] = None
        self._stop = threading.Event()
        self._connected = False
        self._explained_failure = False
        import requests  # already a dependency (VLM judge backends)

        self._requests = requests

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True, name="cloud-uplink").start()

    def close(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------ main-loop side
    def push_event(self, event, severity: str) -> None:
        """Queue one alert event for the next sync cycle."""
        with self._lock:
            self._queue.append(
                {
                    "timestamp": event.timestamp.isoformat(),
                    "type": event.type,
                    "confidence": round(event.confidence, 3),
                    "signals": event.contributing_signals,
                    "severity": severity,
                }
            )

    def set_active_preset(self, name: Optional[str]) -> None:
        with self._lock:
            self._active_preset = name

    def set_health(self, report: Optional[dict]) -> None:
        """Latest device health, sent with the next sync.

        Held as a snapshot rather than collected inside the worker so the health
        read happens on the runtime's own thread, where the frame bus, capture
        service and Wi-Fi manager actually live.
        """
        with self._lock:
            self._health = report

    def pop_pending_preset(self) -> Optional[str]:
        """Remote set_preset command awaiting the main loop, if any."""
        with self._lock:
            pending, self._pending_preset = self._pending_preset, None
            return pending

    def pop_pending_zones(self) -> Optional[list[dict]]:
        """Zones drawn in the app, awaiting the main loop.

        None means nothing new; `[]` means the user deleted their zones and
        every area rule must stop matching rather than keep the old shapes.
        """
        with self._lock:
            pending, self._pending_zones = self._pending_zones, None
            return pending

    def pop_pending_rules(self) -> Optional[list[dict]]:
        """AI-configured rules awaiting the main loop, if a new set arrived.

        None means "nothing new" — distinct from `[]`, which means the user
        deliberately cleared their rules and the device must stop enforcing them.
        """
        with self._lock:
            pending, self._pending_rules = self._pending_rules, None
            return pending

    @property
    def connected(self) -> bool:
        return self._connected

    def _looks_like_bare_ip(self) -> bool:
        """Whether base_url points at a numeric address rather than a name.

        A name resolves wherever the host goes; an IP is only true on the
        network it was written down on.
        """
        import re as _re

        return bool(_re.search(r"//(\d{1,3}\.){3}\d{1,3}(:|/|$)", self._base_url))

    # ------------------------------------------------------------------ worker
    def _run(self) -> None:
        headers = {"X-Device-Key": self._device_key}
        while not self._stop.is_set():
            try:
                with self._lock:
                    batch = list(self._queue)
                    self._queue.clear()
                    active = self._active_preset
                    rules_version = self._rules_version
                    zones_version = self._zones_version
                    health = self._health
                # 1) push events (an empty batch doubles as the online heartbeat)
                resp = self._requests.post(
                    f"{self._base_url}/v1/ingest",
                    json={
                        "events": batch,
                        "active_preset": active,
                        "rules_version": rules_version,
                        "zones_version": zones_version,
                        "health": health,
                    },
                    headers=headers,
                    timeout=10,
                )
                resp.raise_for_status()
                # 2) the ingest response carries any pending command
                command = resp.json().get("command") or {}
                if command.get("set_preset"):
                    with self._lock:
                        self._pending_preset = command["set_preset"]
                if "set_rules" in command:
                    rules = command.get("set_rules")
                    with self._lock:
                        self._pending_rules = rules if isinstance(rules, list) else []
                        # Acknowledge immediately: the main loop applies these on
                        # its next tick, and re-sending every poll until then
                        # would restart in-flight rule tracking each time.
                        self._rules_version = str(command.get("rules_version") or "")
                if "set_zones" in command:
                    zones = command.get("set_zones")
                    with self._lock:
                        self._pending_zones = zones if isinstance(zones, list) else []
                        self._zones_version = str(command.get("zones_version") or "")
                if not self._connected:
                    self._connected = True
                    self._explained_failure = False
                    print(f"cloud_uplink: connected to {self._base_url}")
            except Exception as exc:  # noqa: BLE001 - the device must keep working offline
                if self._connected:
                    print(f"cloud_uplink: connection lost ({exc}); device continues standalone")
                elif not self._explained_failure:
                    # Said once, not every five seconds. A bare IP in base_url is
                    # the commonest cause of a device that looks broken and is
                    # not: the address was correct on the network where it was
                    # configured, and means nothing on any other one.
                    self._explained_failure = True
                    # An HTTP error is not a reachability problem. The backend
                    # answered — it refused. Reporting "cannot reach" for a
                    # rejected device key sends someone to debug a network that
                    # is working perfectly.
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    if status in (401, 403):
                        print(
                            f"cloud_uplink: {self._base_url} rejected this device's key "
                            f"(HTTP {status}). The backend is reachable; the key is wrong or "
                            f"unknown to it — re-issue with `python cloud/manage.py`."
                        )
                    elif status is not None:
                        print(f"cloud_uplink: {self._base_url} returned HTTP {status}")
                    else:
                        print(f"cloud_uplink: cannot reach {self._base_url} ({exc})")
                    if status is None and self._looks_like_bare_ip():
                        print(
                            "cloud_uplink: base_url is a fixed IP address. If this device or "
                            "the backend has moved to a different network, that address now "
                            "points at nothing. Use the backend host's mDNS name instead — "
                            "e.g. http://your-macbook.local:8787 — which follows it between "
                            "networks. See docs/NETWORKING.md."
                        )
                self._connected = False
                with self._lock:
                    # Put the batch back in front of anything recorded since,
                    # but never at the cost of those newer events.
                    #
                    # `appendleft` on a full deque evicts from the RIGHT — the
                    # newest end. So re-queuing a failed batch used to discard
                    # the most recent alerts in favour of the ones that had just
                    # failed to send, which is backwards for a safety product:
                    # during a long outage the device threw away what was
                    # happening now to preserve what happened an hour ago. The
                    # comment here claimed it dropped the oldest.
                    capacity = self._queue.maxlen
                    space = len(batch) if capacity is None else capacity - len(self._queue)
                    if space > 0:
                        # Keep the newest of the failed batch when it cannot all fit.
                        for item in reversed(batch[-space:]):
                            self._queue.appendleft(item)
                    dropped = len(batch) - max(space, 0)
                    if dropped > 0:
                        print(
                            f"cloud_uplink: backlog full — {dropped} older event(s) dropped; "
                            f"newer ones kept"
                        )
            self._stop.wait(self._poll_interval_s)
