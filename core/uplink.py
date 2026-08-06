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
        self._stop = threading.Event()
        self._connected = False
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
                    health = self._health
                # 1) push events (an empty batch doubles as the online heartbeat)
                resp = self._requests.post(
                    f"{self._base_url}/v1/ingest",
                    json={
                        "events": batch,
                        "active_preset": active,
                        "rules_version": rules_version,
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
                if not self._connected:
                    self._connected = True
                    print(f"cloud_uplink: connected to {self._base_url}")
            except Exception as exc:  # noqa: BLE001 - the device must keep working offline
                if self._connected:
                    print(f"cloud_uplink: connection lost ({exc}); device continues standalone")
                self._connected = False
                with self._lock:
                    # put the batch back (bounded deque drops oldest under pressure)
                    for item in reversed(batch):
                        self._queue.appendleft(item)
            self._stop.wait(self._poll_interval_s)
