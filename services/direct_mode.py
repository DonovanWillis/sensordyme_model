"""DirectModeService: runs core/direct_mode.DirectModeManager.tick() on its own thread.

Added to the supervisor only when `network.direct_mode.enabled` is true
(services/runtime.py). The tick does the R3 boot check once and, only when
`fallback_enabled` is also true, the "no network for N minutes" policy. It runs
nmcli, so it never runs on the inference thread.
"""

from __future__ import annotations

import threading

from core.direct_mode import DirectModeManager
from services.supervisor import Service


class DirectModeService(Service):
    name = "direct_mode"

    def __init__(self, manager: DirectModeManager, interval_s: float = 5.0) -> None:
        super().__init__()
        self.manager = manager
        self._interval_s = interval_s

    def run(self, stop: threading.Event) -> None:
        while not stop.is_set():
            self.manager.tick()  # never raises (it logs instead)
            stop.wait(self._interval_s)

    def status(self) -> tuple[bool, str]:
        """(healthy, detail) for the health monitor (non-critical)."""
        state = self.manager.status()
        mode = state["mode"]
        if mode == "direct":
            return True, f"direct mode: the device's own access point '{state['ssid']}' is up"
        if mode == "unavailable":
            return True, f"direct mode unavailable: {state['last_error']}"
        if state["last_error"]:
            return True, f"{mode}; last direct-mode error: {state['last_error']}"
        return True, mode
