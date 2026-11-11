"""SimRuntime: services.runtime.DeviceRuntime with the hardware edges replaced.

Everything the device decides is the real code: the inference loop, alert
episodes and role floors, the monitoring state machine, protection stop/resume,
preset swaps and role refusals, custom rules, uplink and health reporting.
What this subclass replaces, and only this:

  _open_camera                SimCamera (the scenario) instead of Picamera2/cv2
  _build_module               the real module class, then scripted perception
                              installed on it (driver_attention only)
  _make_uplink                SimUplink (real CloudUplink) with the sim transport
  _verify_detection_backend   for a scripted module: the scripted detector IS
                              the backend (identity says "simulated"); any other
                              module is verified for real -- and on a Mac without
                              ultralytics or a Hailo that honestly means DEGRADED
  _collect_health             the real report, marked simulated, with the sim's
                              hardware id and a network block that says what it is
  _issue_reboot               never reboots the host: logs and refuses
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from core.alerts import AlertCategory, AlertLevel, AlertPattern
from core.events import Event
from services.runtime import DeviceRuntime
from sim import perception

log = logging.getLogger("sensordyme.sim")


class SimBuzzerSink:
    """The driver's in-cabin buzzer, simulated: records every actuation.

    Audible in the policy sense (it stands in for gpio_buzzer, which a
    driver_safety role requires), and says plainly that nothing makes a sound.
    """

    name, kind, audible = "sim_buzzer", "buzzer", True

    def __init__(self, echo: bool = False) -> None:
        self.played: list[tuple[str, str, str]] = []
        self._echo = echo

    def readiness(self) -> tuple[bool, str]:
        return True, "SIMULATED buzzer: actuations are recorded, nothing makes a sound"

    def play(self, pattern: AlertPattern, event: Event, category: AlertCategory) -> None:
        self.played.append((event.type, category.name, pattern.name))
        if self._echo:
            print(f"[SIM BUZZER] {category.name}/{pattern.name}: {event.type}")

    def emit(self, event: Event, level: AlertLevel) -> None:
        self.played.append((event.type, level.value, level.value))


class SimContext:
    """What a SimRuntime needs from the virtual device that owns it."""

    def __init__(self, *, state, camera, identity, state_dir: Path, transport=None,
                 manual: bool = False, scripted: bool = True, model_failure: bool = False,
                 scenario_name: str = "") -> None:
        self.state = state
        self.camera = camera
        self.identity = identity
        self.state_dir = Path(state_dir)
        self.transport = transport
        self.manual = manual
        self.scripted = scripted
        self.model_failure = model_failure
        self.scenario_name = scenario_name
        self.detectors: list = []
        #: The device's one detector, shared by every module build and every
        #: monitoring node, as on hardware (core/detectors.py loads it once).
        #: The runtime's perception engine wraps it; `detector.calls` counts
        #: every real inference, whoever asked.
        self.detector = perception.ScriptedDetector(state)
        self.uplink: Optional[object] = None
        #: Every monitoring state this boot has passed through, in order.
        self.transitions: list[str] = []

    def build_module(self, module_cls, module_config: dict):
        """Construct a module the way main.py/_switch_preset do, then script it."""
        if self.model_failure:
            raise RuntimeError("SIMULATED model failure: the perception model could not be loaded")
        module = module_cls(module_config)
        if self.scripted and perception.supports(module):
            self.detectors.append(perception.install(module, self.state, detector=self.detector))
        return module


class SimRuntime(DeviceRuntime):
    def __init__(self, *, sim: SimContext, **kwargs) -> None:
        self.sim = sim
        self.monitoring_path = str(sim.state_dir / "monitoring.json")
        super().__init__(**kwargs)
        # Every Observation from this device says so (core/perception.py).
        self._perception_source = "sim"
        self._perception_simulated = True
        # Instance attribute, read as self._REBOOT_STATE_PATH: never data/ in the cwd.
        self._REBOOT_STATE_PATH = sim.state_dir / "reboot_state.json"

    # ------------------------------------------------------------------ hardware edges
    def _open_camera(self):
        return self.sim.camera

    def _build_module(self, module_name: str, module_config: dict):
        from modules import get_module_class

        return self.sim.build_module(get_module_class(module_name), module_config)

    def _make_uplink(self, cloud_cfg: dict):
        from sim.uplink import SimUplink

        uplink = SimUplink(cloud_cfg, transport=self.sim.transport, manual=self.sim.manual)
        self.sim.uplink = uplink
        return uplink

    def _scripted(self) -> bool:
        return self.sim.scripted and perception.supports(self.module)

    def _load_shared_detector(self):
        """A node beside a detector-less module gets the device's scripted detector."""
        if self.sim.scripted:
            return self.sim.detector
        return super()._load_shared_detector()

    def _verify_detection_backend(self) -> None:
        from modules.base import UnavailableModule

        if self.sim.scripted and isinstance(self.module, UnavailableModule):
            # The module itself could not be built: there is no detector to
            # verify, and the module's own FAILED report says why.
            self._detection_degraded = None
            self._model_identity = {"module": self._module_identity_name(), "detector": None}
            return
        if not self._scripted():
            super()._verify_detection_backend()
            return
        self._detection_degraded = None
        self._model_identity = {
            "module": self._module_identity_name(),
            "detector": {"name": perception.ScriptedDetector.model_name, "sha256": None, "backend": "simulated"},
        }

    def _collect_health(self, caller: str = "loop") -> dict:
        report = super()._collect_health(caller)
        faults = self.sim.state.current()
        report["device_id"] = self.sim.identity.hardware_id
        report["simulated"] = True
        report["simulation"] = {
            "scenario": self.sim.scenario_name,
            "scenario_time_s": round(self.sim.state.clock.now(), 1),
            "scene": self.sim.state.scene().label,
            "faults": faults.describe(),
            "perception": "scripted (no camera, no models)" if self._scripted()
                          else "not simulated for this module (real code, no models on this host)",
            "not_simulated": ["camera sensor", "Hailo NPU", "BLE provisioning", "GPIO buzzer",
                              "Wi-Fi", "thermals"],
        }
        report["network"] = (
            {"state": "failed", "detail": "SIMULATED network outage", "metrics": {}}
            if faults.network_down else
            {"state": "ok", "detail": "SIMULATED: host network, no Wi-Fi radio", "metrics": {}}
        )
        return report

    def _monitoring_check(self) -> tuple[bool, str]:
        result = super()._monitoring_check()
        if not self.sim.transitions or self.sim.transitions[-1] != self._monitoring_state:
            self.sim.transitions.append(str(self._monitoring_state))
        return result

    def _issue_reboot(self) -> bool:
        log.critical("SIMULATED: the runtime asked to reboot the device; a virtual device never "
                     "reboots its host (restart the virtual device instead)")
        return False
