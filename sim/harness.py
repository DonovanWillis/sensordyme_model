"""SimStack: the real backend in-process + a virtual device, on one SimClock.

For deterministic end-to-end tests (tests/test_sim_e2e.py) and for poking at
the system from a Python prompt. Everything is asserted through the HTTP API
the app uses; the device is reached only through its uplink, like a real one.

    with SimStack(tmp_path) as stack:
        device = stack.boot(scenario.get("drowsy"))
        stack.claim()
        device.advance(20.0)
        stack.status()["monitoring_state"]   # "ready"
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from sim.backend import InProcessBackend
from sim.clock import SimClock
from sim.device import VirtualDevice
from sim.scenario import Scenario
from sim.uplink import TestClientTransport


class SimStack:
    def __init__(self, root: Path | str, *, clock: Optional[SimClock] = None) -> None:
        self.root = Path(root)
        self.clock = clock or SimClock()
        self._clock_ctx = self.clock.installed()
        self._clock_ctx.__enter__()
        try:
            self.backend = InProcessBackend(self.root / "backend", self.clock)
            self.factory = self.backend.provision()
            self.headers = self.backend.signup()
        except Exception:
            self._clock_ctx.__exit__(None, None, None)
            raise
        self.device: Optional[VirtualDevice] = None
        self.device_id: Optional[int] = None
        self.transport = TestClientTransport(self.backend.client)

    @property
    def api(self):
        return self.backend.client

    @property
    def state_dir(self) -> Path:
        return self.root / "device"

    # ------------------------------------------------------------------ device lifecycle
    def boot(self, scenario: Scenario, *, transport=None, **kwargs) -> VirtualDevice:
        """Power the virtual device on (first boot provisions its identity)."""
        first_boot = not (self.state_dir / "identity.json").exists()
        self.device = VirtualDevice(
            state_dir=self.state_dir, backend_url=self.backend.base_url, scenario=scenario,
            device_key=self.factory["device_key"] if first_boot else None,
            claim_code=self.factory["claim_code"] if first_boot else None,
            clock=self.clock, transport=transport or self.transport, log_level="warning", **kwargs,
        )
        self.device.start()
        return self.device

    def power_off(self) -> None:
        if self.device is not None:
            self.device.stop()
            self.device = None

    def restart(self, scenario: Scenario, **kwargs) -> VirtualDevice:
        """A process restart: power off, then boot again from the same state dir."""
        self.power_off()
        return self.boot(scenario, **kwargs)

    def power_cut(self) -> None:
        """Power lost (M8): no shutdown code runs -- nothing is closed, flushed or
        saved on the way down. What survives is exactly what was already on disk."""
        self.device = None

    def cold_restart(self, scenario: Scenario, **kwargs) -> VirtualDevice:
        """Power cut, then power on again from the same state dir."""
        self.power_cut()
        return self.boot(scenario, **kwargs)

    def restart_backend(self) -> None:
        """The backend process restarts (M8.4): same database, a fresh import of
        cloud/app.py, no in-memory state carried over. The device keeps its link."""
        self.backend.close()
        self.backend = InProcessBackend(self.root / "backend", self.clock)
        self.transport.retarget(self.backend.client)

    def close(self) -> None:
        try:
            self.power_off()
            self.backend.close()
        finally:
            self._clock_ctx.__exit__(None, None, None)

    def __enter__(self) -> "SimStack":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ app-side API
    def claim(self) -> int:
        response = self.api.post("/v1/devices/claim", json={"claim_code": self.factory["claim_code"]},
                                 headers=self.headers)
        response.raise_for_status()
        self.device_id = int(response.json()["device_id"])
        return self.device_id

    def devices(self) -> list[dict]:
        response = self.api.get("/v1/devices", headers=self.headers)
        response.raise_for_status()
        return response.json()["devices"]

    def row(self) -> dict:
        """This device as GET /v1/devices shows it."""
        return next(d for d in self.devices() if d["id"] == self.device_id)

    def status(self) -> dict:
        return self.row()["status"]

    def events(self, limit: int = 200) -> list[dict]:
        response = self.api.get(f"/v1/devices/{self.device_id}/events", params={"limit": limit},
                                headers=self.headers)
        response.raise_for_status()
        return list(reversed(response.json()["events"]))  # oldest first

    def event_types(self) -> list[str]:
        return [e["type"] for e in self.events()]

    def set_preset(self, preset: str, **body):
        return self.api.post(f"/v1/devices/{self.device_id}/preset", json={"preset": preset, **body},
                             headers=self.headers)

    def protection(self, action: str, confirm: bool = False):
        return self.api.post(f"/v1/devices/{self.device_id}/protection",
                             json={"action": action, "confirm": confirm}, headers=self.headers)

    def ai_configure(self, prompt: str):
        return self.api.post("/v1/ai/configure", json={"device_id": self.device_id, "prompt": prompt},
                             headers=self.headers)
