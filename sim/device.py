"""VirtualDevice: one software Raspberry Pi, built the way main.py builds the real one.

    device = VirtualDevice(state_dir=..., backend_url=..., scenario=scenario.get("drowsy"),
                           device_key=...)
    device.run()                      # live: real threads, real time (the CLI)

    device = VirtualDevice(..., clock=SimClock())   # deterministic (tests)
    device.start(); device.advance(30.0); device.stop()

Both paths construct the runtime through `main.build_runtime` from a config
file derived from the repo's config.yaml -- so config validation, the role
floor, preset restore after a restart and the boot-FAILED path for a module
that cannot be built are main.py's own -- and differ only in who turns the
crank: DeviceRuntime.run()'s threads, or `advance()`, which runs the
production inference pass (`_inference_pass`), the monitoring check and the
uplink sync cycle (`sync_once`) on the injected clock.

One virtual device per process: core.device_state reads its path from the
environment, which this sets to the device's own state directory.
"""

from __future__ import annotations

import copy
import functools
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Optional

import yaml

from sim import identity as sim_identity
from sim.camera import SimCamera
from sim.clock import SimClock
from sim.runtime import SimBuzzerSink, SimContext, SimRuntime
from sim.scenario import Scenario, ScenarioClock, ScenarioState
from sim.uplink import NetworkGate

REPO = Path(__file__).resolve().parent.parent
#: Health-monitor and uplink cadence in the deterministic path (production: 2 s each).
MONITORING_CHECK_S = 2.0


class _SteppedCapture:
    """Stands in for CaptureService when advance() is the camera pump.

    The real pump thread stays running through camera faults (it catches the
    read error and reopens), so `is_running` is True until stopped; whether
    frames ARRIVE is what the camera fault decides, and the bus shows it.
    """

    name = "camera"

    def __init__(self) -> None:
        self.is_running = True

    def start(self, health=None) -> None:
        self.is_running = True

    def stop(self, timeout: float = 5.0) -> None:
        self.is_running = False

    def restart(self, stop_timeout: float = 5.0) -> None:
        self.is_running = True


class _NoWait:
    """`stop` for a stepped inference pass: never set, never sleeps."""

    def is_set(self) -> bool:
        return False

    def wait(self, _timeout: Optional[float] = None) -> bool:
        return False


class VirtualDevice:
    def __init__(
        self,
        *,
        state_dir: Path | str,
        backend_url: str,
        scenario: Scenario,
        device_key: Optional[str] = None,
        claim_code: Optional[str] = None,
        clock: Optional[SimClock] = None,
        transport=None,
        speed: float = 1.0,
        video: Optional[str] = None,
        dashboard_port: Optional[int] = None,
        dashboard_host: str = "127.0.0.1",
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        poll_interval_s: float = 2.0,
        echo_alerts: bool = False,
        log_level: str = "info",
    ) -> None:
        self.state_dir = Path(state_dir).resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.identity = sim_identity.load_or_create(self.state_dir, device_key=device_key, claim_code=claim_code)
        self.backend_url = backend_url.rstrip("/")
        self.scenario = scenario
        self.clock = clock
        self.manual = clock is not None
        self.video = video
        self.dashboard_port = dashboard_port
        self.dashboard_host = dashboard_host
        self.width, self.height, self.fps = int(width), int(height), float(fps)
        self.poll_interval_s = float(poll_interval_s)
        self.log_level = log_level
        self.buzzer = SimBuzzerSink(echo=echo_alerts)
        monotonic = clock.monotonic if clock is not None else time.monotonic
        self.scenario_clock = ScenarioClock(monotonic, speed)
        self.state = ScenarioState(scenario, self.scenario_clock)
        self.camera = SimCamera(self.state, width=self.width, height=self.height, fps=self.fps, video_path=video)
        if transport is None:
            import requests

            transport = requests
        self.network = NetworkGate(transport, lambda: self.state.current().network_down)
        self.sim: Optional[SimContext] = None
        self.runtime: Optional[SimRuntime] = None
        self._loop = None
        self._next_check = 0.0
        self._next_sync = 0.0
        self._reboot_requested = threading.Event()

    # ------------------------------------------------------------------ config
    @property
    def dashboard_token(self) -> str:
        """The device's local API / live-feed token (what the app calls the device token)."""
        path = self.state_dir / "dashboard_token"
        try:
            token = path.read_text().strip()
        except OSError:
            token = ""
        if not token:
            token = secrets.token_urlsafe(18)
            path.write_text(token)
        return token

    def config(self) -> dict:
        """The repo's config.yaml, with the hardware and paths a Mac can honestly have."""
        config = copy.deepcopy(yaml.safe_load((REPO / "config.yaml").read_text()))
        data = self.state_dir
        config["display"] = False
        config["camera"] = {**config.get("camera", {}), "width": self.width, "height": self.height,
                            "fps": self.fps}
        config["dashboard"] = {
            **config.get("dashboard", {}),
            "enabled": self.dashboard_port is not None,
            "host": self.dashboard_host,
            "port": int(self.dashboard_port or 0),
            "token": self.dashboard_token if self.dashboard_port is not None else None,
        }
        # No NPU here, and nothing pretends there is one.
        config["accelerator"] = {"enabled": False, "yolo_hef": None, "required": False}
        config["network"] = {"provisioning": {"method": "off"}}
        config["health"] = {**config.get("health", {}), "allow_reboot": False}
        config["cloud"] = {
            **config.get("cloud", {}), "enabled": True, "base_url": self.backend_url,
            "device_key": self.identity.device_key, "poll_interval_s": self.poll_interval_s,
        }
        config["logging"] = {
            **config.get("logging", {}), "level": self.log_level,
            "csv_path": str(data / "events.csv"), "telemetry_csv_path": str(data / "telemetry.csv"),
        }
        alerts = config.setdefault("alerts", {})
        # gpio_buzzer stays configured (the driver_safety floor requires an
        # audible output); the virtual device supplies a simulated one for it.
        alerts["sinks"] = ["console", "gpio_buzzer"] if not self.manual else ["gpio_buzzer"]
        alerts["save_evidence"] = False
        alerts["evidence_dir"] = str(data / "evidence")
        config.setdefault("training", {}).setdefault("capture", {})["enabled"] = False
        modules = config.get("modules", {})
        if "driver_attention" in modules:
            modules["driver_attention"]["calibration_path"] = str(data / "calibration.json")
        if "driver_attention_dashsentinel" in modules:
            modules["driver_attention_dashsentinel"]["profile_path"] = str(data / "dashsentinel_profile.json")
        return config

    def write_config(self) -> Path:
        path = self.state_dir / "config.yaml"
        path.write_text(yaml.safe_dump(self.config(), sort_keys=False))
        return path

    # ------------------------------------------------------------------ build
    def build(self) -> SimRuntime:
        """main.build_runtime with the simulated edges: the device, wired, not started."""
        import main

        os.environ["SENSORDYME_DEVICE_STATE_PATH"] = str(self.state_dir / "device_state.json")
        self.sim = SimContext(
            state=self.state, camera=self.camera, identity=self.identity, state_dir=self.state_dir,
            transport=self.network, manual=self.manual, scripted=self.video is None,
            model_failure=self.scenario.boot.model_failure, scenario_name=self.scenario.name,
        )
        overrides = None if self.scenario.boot.no_buzzer else {"gpio_buzzer": lambda: self.buzzer}
        self.runtime = main.build_runtime(
            ["--config", str(self.write_config()), "--no-display"],
            runtime_cls=functools.partial(SimRuntime, sim=self.sim),
            module_factory=self.sim.build_module,
            sink_overrides=overrides,
        )
        return self.runtime

    # ------------------------------------------------------------------ deterministic
    def start(self) -> SimRuntime:
        """Build and bring the device up without starting any thread (manual mode)."""
        if not self.manual:
            raise RuntimeError("start()/advance() need a SimClock; use run() for a live device")
        if time.monotonic() != self.clock.monotonic():
            raise RuntimeError("install the SimClock first (`with clock.installed():`)")
        runtime = self.build()
        runtime._build()
        runtime.capture = _SteppedCapture()
        runtime._heartbeat_reset_at = time.monotonic()  # what run() does before starting threads
        self._loop = runtime._new_inference_loop_state()
        self._loop.frame_timeout_s = 0.0
        self._next_check = self._next_sync = time.monotonic()
        return runtime

    def advance(self, seconds: float, step: float = 0.05) -> None:
        """Run the device for `seconds` of sim time: a camera frame and an
        inference pass every `step`, the monitoring check and an uplink sync
        cycle on their production cadence."""
        assert self.runtime is not None and self._loop is not None and self.clock is not None
        for _ in range(max(1, round(seconds / step))):
            self.clock.advance(step)
            frame = self.camera.grab()
            if frame is not None:
                self.runtime.bus.publish(frame)
            self.runtime._inference_pass(self._loop, _NoWait())
            now = self.clock.monotonic()
            if now >= self._next_check:
                self.runtime._monitoring_check()
                self._next_check = now + MONITORING_CHECK_S
            if self.runtime.uplink is not None and now >= self._next_sync:
                self.sync()
                self._next_sync = now + self.poll_interval_s

    def sync(self) -> None:
        """One uplink cycle, with a fresh health sample (as _HealthReportService does)."""
        uplink = self.runtime.uplink if self.runtime is not None else None
        if uplink is None:
            return
        uplink.set_health(self.runtime._collect_health("uplink"))
        for _ in range(20):  # a backlog drains in full batches, then stops
            if not uplink.sync_once():
                break

    def stop(self) -> None:
        """Shut the device down (manual mode): everything _build made is released."""
        if self.runtime is not None:
            self.runtime.shutdown.set()
            self.runtime._shutdown_partial()

    # ------------------------------------------------------------------ live
    def request_reboot(self) -> None:
        """Restart the device process (live mode): same state dir, same identity."""
        self._reboot_requested.set()
        if self.runtime is not None:
            self.runtime.shutdown.set()

    def run(self) -> None:
        """Run live until interrupted, restarting at the scenario's reboot times."""
        if self.manual:
            raise RuntimeError("run() is the live path; a SimClock device is driven with advance()")
        pending = sorted(self.scenario.reboots)
        while True:
            self._reboot_requested.clear()
            runtime = self.build()
            timer = None
            if pending:
                delay = max(0.0, (pending[0] - self.scenario_clock.now()) / self.scenario_clock.speed)
                timer = threading.Timer(delay, self.request_reboot)
                timer.daemon = True
                timer.start()
            try:
                runtime.run()
            finally:
                if timer is not None:
                    timer.cancel()
            if not self._reboot_requested.is_set():
                return
            if pending:
                pending.pop(0)
            print("SensorDyme SIM: SIMULATED reboot — restarting the device process with the same state")
