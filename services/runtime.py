"""DeviceRuntime: builds, wires, supervises, and cleanly shuts down every service.

This is what `python main.py` actually runs after resolving the profile/module.
Thread architecture (each box restartable on its own):

    CaptureService ──publish──> FrameBus ──wait_for_fresh──> inference loop
        (camera thread)                        (main thread with --display,
                                                InferenceService thread headless)
    DashboardServer  -- its own HTTP threads; /stream pulls JPEGs at stream_fps
    CloudUplink      -- its own sync thread (events up / commands down)
    ProvisioningService -- Wi-Fi watcher + BLE window (Pi only)
    HealthMonitor    -- checks everything above; restarts, escalates, feeds the
                        systemd watchdog, and serves /api/health via the dashboard

Startup order: accelerator probe -> camera (with retries) -> capture -> dashboard
-> uplink -> provisioning -> health -> sd_notify READY. Shutdown is the reverse,
runs on SIGTERM (systemd stop) and SIGINT alike, and always releases the camera.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional, cast

import cv2

from core import accel, device_health, device_state, health_stamp, monitoring_mode, roles, sdnotify
from core import monitoring_state as ms
from core import nodes as monitoring_nodes
from core import perception as pc
from core.alerts import AlertLevel, AlertManager
from core.camera import Camera, CameraError
from core.dataset import CaptureStore
from core.events import Event
from core.frame_bus import FrameBus
from core.health import HealthMonitor
from core.logger import EventLogger
from core.logging_setup import get_logger, setup_logging
from core.monitoring_config import load_or_empty
from core.network import WifiManager, nmcli_available
from core.presets import DEVICE_POLICY_FEATURES
from core.privilege import privileged
from core.rules import RuleEngine, validate_rule_set
from core.telemetry import TelemetryLogger
from core.zones import load_zones
from modules import get_module_class
from services.capture import CaptureService
from services.provisioning import ProvisioningService
from services.supervisor import Service, Supervisor

log = get_logger("runtime")

# Matches the uplink's default poll interval: sampling faster just discards
# readings, and /proc + vcgencmd reads are not free.
_HEALTH_SAMPLE_S = 5.0
_CAMERA_OPEN_RETRIES = 5
_CAMERA_OPEN_BACKOFF_S = 2.0
_CAMERA_VERIFY_TIMEOUT_S = 10.0
_VERSION = "0.2"
#: The custom rules' consumer id on the shared perception engine (the rules node's id), and
#: how often their detector demand is due: 0.2 s, which the driver module's every-3rd-frame
#: pass usually serves from the same run, so it costs no extra inference there.
RULES_CONSUMER = "rules"
RULES_DEMAND_INTERVAL_S = 0.2
# A module.process() call that keeps failing this many frames IN A ROW is not
# a transient hiccup -- propagate so the existing recovery ladder (in-thread
# restart headless, process exit under --display) can recover a genuinely
# wedged module, rather than looping forever silently dropping every frame.
_MAX_CONSECUTIVE_PROCESS_FAILURES = 10
#: How often to re-log a persistently failing annotate(). It runs at stream
#: cadence while a viewer is connected, so logging every failure would bury the
#: rest of the log within seconds of opening the feed.
_ANNOTATE_FAILURE_LOG_EVERY = 300
# How long the video overlay (_apply_alert_overlay, below) stays on after a new
# event reaches Recent Activity -- edge-triggered on push_event, not tied to
# module.is_alert_active()'s continuous "still ongoing" state, so a persistent
# condition (an object still visible, a driver still distracted) does not keep
# the screen flashing. Tracked independently of core.dashboard.FLASH_DURATION_S
# (that one drives the web page's own DOM flash) -- same value, kept in sync
# deliberately rather than imported, since this module only imports
# DashboardServer lazily and this constant governs a different renderer (the
# baked-in frame overlay used by both the cv2 --display window and /stream).
_FLASH_DURATION_S = 1.5
#: Longest a module swap may run before the inference check treats it as hung.
#: Model loading on a Pi CPU fallback has been measured in tens of seconds.
_MAX_SWAP_S = 120.0
#: Quiet time after which a run of last-resort reboots counts as over.
_REBOOT_CAMPAIGN_RESET_S = 24 * 3600.0
#: A process restart for a component counts as "already tried" for this long.
#: Long enough to cover the watchdog firing, the restart, boot and the
#: component's own restart budget being spent again in the new process.
_PROCESS_RESTART_WINDOW_S = 900.0
#: How long the inference check accepts "starting" (heartbeat reset, not yet
#: stamped). The loop stamps on every pass — even a no-frame timeout every
#: 0.5 s — so a loop that has not stamped for this long is not starting, it is
#: wedged, and must not read as healthy (or the watchdog is fed again).
_INFERENCE_STARTING_GRACE_S = 30.0
#: While DEGRADED (no detector), how often to try loading it again.
_DETECTOR_RETRY_S = 60.0
#: A module that could not be built at boot (main.py's UnavailableModule) is
#: rebuilt after this long, then at doubling intervals up to the cap (M8).
_MODULE_REBUILD_FIRST_S = 30.0
_MODULE_REBUILD_MAX_S = 600.0


@dataclass
class _InferenceLoopState:
    """What one generation of the inference loop carries from pass to pass."""

    last_seq: int
    last_detector_retry: float
    stream_interval: float
    #: How long a pass waits for a fresh frame. 0.5 s in production; a caller
    #: stepping the loop on its own clock (sim/) uses 0 so a pass never sleeps.
    frame_timeout_s: float = 0.5


class _InferenceService(Service):
    """Runs the inference loop in a supervised thread (headless mode)."""

    name = "inference"

    def __init__(self, runtime: "DeviceRuntime") -> None:
        super().__init__()
        self._runtime = runtime

    def run(self, stop: threading.Event) -> None:
        self._runtime._inference_loop(stop)


class _HealthReportService(Service):
    """Samples device health for the uplink on its own thread.

    It used to be sampled inside the inference loop, after a successfully
    processed frame — so in exactly the failures it exists to report (camera
    dead, detector degraded, a module failing every frame, a wedged loop) it
    stopped being refreshed, and the backend kept serving the last healthy
    snapshot ("capturing", monitoring_active true) for as long as the outage
    lasted. Sampling also runs nmcli/vcgencmd, which must not stall inference.
    """

    name = "health_report"

    def __init__(self, runtime: "DeviceRuntime") -> None:
        super().__init__()
        self._runtime = runtime

    def run(self, stop: threading.Event) -> None:
        while True:
            uplink = self._runtime.uplink
            if uplink is not None:
                # Guarded: a few failed samples (nmcli restarting, say) must not
                # spend the thread's crash budget and stop reporting for good.
                try:
                    uplink.set_health(self._runtime._collect_health("uplink"))
                except Exception:
                    log.exception("health sample failed; retrying next interval")
            if stop.wait(_HEALTH_SAMPLE_S):
                return


class _NullEventLog:
    """Stands in for EventLogger/TelemetryLogger when data/ cannot be written.

    A read-only or full SD card is a documented field failure. Failing to open
    a log file must cost the log, not the detection and alerts it records.
    """

    def log(self, *_args, **_kwargs) -> None:
        return None

    def log_if_due(self, *_args, **_kwargs) -> bool:
        return False

    def close(self) -> None:
        return None


#: Custom-rule severity -> alert category (core/alerts.py).
_RULE_CATEGORY = {"critical": "critical", "warning": "warning", "info": "notice"}


def _episode_state(module) -> tuple[frozenset, set]:
    """(tracked, active) alert conditions of a module; empty for modules that
    predate the hooks (test doubles not derived from VisionModule)."""
    tracked: Any = getattr(module, "alert_episode_types", frozenset)
    active: Any = getattr(module, "active_alert_conditions", set)
    return frozenset(tracked()), set(active())


def _zones_from_app(raw) -> list:
    """core.zones.Zone objects from the uplink's zone dicts {name, polygon, kind}."""
    return load_zones({
        z["name"]: {"polygon": z.get("polygon"), "kind": z.get("kind", "restricted")}
        for z in (raw or []) if isinstance(z, dict) and z.get("name")
    })


def _systemd_watchdog_armed() -> bool:
    """Whether systemd will restart this process if health stops pinging."""
    return sdnotify.enabled() and bool(os.environ.get("WATCHDOG_USEC"))


class DeviceRuntime:
    """Owns every long-running piece of the device for one process lifetime."""

    # Safe defaults for the M0.1 policy state, so a runtime assembled without
    # __init__ (several tests do) is a general-role device that is not stopped.
    role = roles.ROLES[roles.DEFAULT_ROLE]
    _stopped: Optional[dict] = None
    _last_rejection: Optional[dict] = None
    _custom_rules_rejected = False
    # M0.2 monitoring-state defaults (see _evaluate_monitoring).
    _heartbeat = 0.0
    _heartbeat_reset_at = 0.0
    # M7 shared-perception defaults: such a runtime has no nodes and a fresh engine.
    _perception_source = "camera"
    _perception_simulated = False
    _last_observation = None
    _detector_unavailable = None
    _node_zone_signature = None
    _preset_nodes_suspended = False
    # Read-only on purpose: a write raises TypeError instead of mutating one dict
    # shared by every such runtime. __init__ binds real dicts, the declared type;
    # the casts say "read-only on the class, a dict on the instance", which an
    # annotation cannot express.
    _companions = cast("dict[str, monitoring_nodes.MonitoringNode]", MappingProxyType({}))
    _preset_nodes = cast("dict[str, monitoring_nodes.MonitoringNode]", MappingProxyType({}))
    _node_errors = cast("dict[str, str]", MappingProxyType({}))
    _consecutive_process_failures = 0
    _last_process_error: Optional[str] = None
    _detection_degraded: Optional[str] = None
    _camera_error: Optional[str] = None
    _camera_stale_after_s = 2.0
    _camera_grace_until = 0.0
    _ready_grace_from = 0.0
    _monitoring_state: Optional[str] = None
    _last_protection_command: Optional[str] = None
    _model_identity: Optional[dict] = None
    #: Where saved custom rules live; None = core.monitoring_config's default
    #: (SENSORDYME_MONITORING_PATH or data/monitoring.json). A seam for sim/,
    #: which keeps each virtual device's state in its own directory.
    monitoring_path: Optional[str] = None
    #: The config file this runtime was built from (main.build_runtime sets it)
    #: and, when that file could not be used, the fault: the runtime is then
    #: running on core.config_lkg's last-known-good copy (M8).
    config_path: Optional[str] = None
    config_fault = None
    #: When to try building an UnavailableModule again (inf: nothing to rebuild).
    _module_rebuild_at = float("inf")
    _module_rebuild_attempts = 0

    def __init__(
        self,
        *,
        config: dict,
        profile: str,
        module,
        module_name: str,
        alert_manager: AlertManager,
        alert_cfg: dict,
        alert_event_types: set,
        alert_types_for,
        presets: dict,
        active_preset: Optional[str],
        display: bool,
        debug: bool,
        window_name: str = "SensorDyme",
        module_config: Optional[dict] = None,
    ) -> None:
        self.config = config
        # The config the boot module was ACTUALLY built from. For the pi profile
        # that is config["modules"][module_name]; for mac_max it is
        # config["mac_max"][...] with the device pin, while module_name still
        # says "driver_attention" — so re-reading config["modules"] verified and
        # labelled the pi detector while the module ran a different one on MPS.
        self._boot_module = module
        self._boot_module_config = module_config
        # The config a module built by a PRESET SWITCH was built from: the
        # module's config.yaml block plus the variant overrides and owner
        # settings (M8). Without it, _active_module_config() re-read the bare
        # config.yaml block, so reports ignored the owner's settings.
        self._switched_module: Any = None
        self._switched_module_config: Optional[dict] = None
        # Which exact model events from this module come from; see
        # _verify_detection_backend and core.detectors.model_identity.
        self._model_identity: Optional[dict] = None
        self.profile = profile
        self.module = module
        self.module_name = module_name
        self.alert_manager = alert_manager
        self.alert_event_types = alert_event_types
        self.alert_types_for = alert_types_for
        self.loud_types = set(alert_cfg.get("loud_types", []))
        self.soft_types = set(alert_cfg.get("soft_types", []))
        self.presets = presets
        self.active_preset = active_preset
        self.display = display
        self.debug = debug
        self.window_name = window_name

        alerts_cfg = config.get("alerts", {}) if isinstance(config.get("alerts"), dict) else {}
        self._evidence_dir: Optional[Path] = (
            Path(str(alerts_cfg.get("evidence_dir", "data/evidence")))
            if bool(alerts_cfg.get("save_evidence", False)) else None
        )
        self._evidence_max = int(alerts_cfg.get("evidence_max_files", 50))

        # Zones drawn in the app; None until the app has sent any. They are the
        # device's zones whatever mode runs — the app's editor is where a person
        # draws a crib or a keep-out area — so they are handed to every module
        # and to the rule engine, and remembered across restarts so an offline
        # device does not come back without them. See _effective_zones.
        self._app_zones: Optional[list] = None
        # Saved state that exists but cannot be read is restored as empty (every
        # loader falls back safely) -- and reported, not silently forgotten (M8).
        self._state_faults = self._check_saved_state()
        saved_zones = device_state.saved_zones()
        if saved_zones is not None:
            self._app_zones = _zones_from_app(saved_zones)

        # The shared perception engine (core/perception.py): every model runs at
        # most once per analysed frame, whoever asks, and each frame ends in one
        # typed Observation that the module, custom rules and monitoring nodes
        # all read. The boot module was built before the engine existed, so its
        # full-frame models are adopted now; later builds are adopted as made.
        perception_cfg = config.get("perception") if isinstance(config.get("perception"), dict) else {}
        self._perception_engine = pc.PerceptionEngine(
            max_age_s=float(perception_cfg.get("max_age_s", pc.DEFAULT_MAX_AGE_S)))
        self.perception.set_detector_loader(self._load_shared_detector)
        self.perception.adopt(module)
        #: Where analysed frames come from (the simulator says "sim").
        self._perception_source = "camera"
        self._perception_simulated = False
        self._last_observation: Optional[pc.Observation] = None
        self._detector_unavailable: Optional[str] = None

        # AI-configured rules. A rule naming "pool" resolves against the same
        # polygon the module watches.
        self.rule_engine = RuleEngine(zones=self._effective_zones(module_name))
        self._push_zones_to(module)

        # What someone configured survives the device being switched off.
        #
        # Rules used to arrive only from the cloud and live only in memory, so a
        # reboot silently emptied them and a device with no Wi-Fi could never be
        # configured at all. Restoring here means the device wakes up watching
        # what it was asked to watch, with or without a network.
        # The deployment role (core/roles.py): which safeguards must keep running
        # while this device reports that it is protecting. Configs without one
        # are "general" and carry no floor.
        self.role = roles.role_for_config(config)
        #: Set only by the explicit, confirmed stop command; never persisted, so a
        #: reboot always comes back protecting. See _apply_protection.
        self._stopped: Optional[dict] = None
        #: The last policy change this device refused, and why (reported upstream).
        self._last_rejection: Optional[dict] = None
        self.monitoring = load_or_empty(self.monitoring_path) if self.monitoring_path else load_or_empty()
        # Restored rules are validated exactly like delivered ones: a rule set
        # saved by an older version that this device cannot run is not loaded.
        self._custom_rules_rejected = False
        if self.monitoring.rules:
            problems = self._rule_problems(self.monitoring.rules, module, module_name)
            if problems:
                self._custom_rules_rejected = True
                self._record_rejection("rules", self.monitoring.version, problems)
                log.error("[POLICY] saved custom rules NOT loaded: %s", "; ".join(problems))
            else:
                self.rule_engine.replace_rules(self.monitoring.rules)
        # What the selected preset wants to run, remembered separately from what
        # is running now — AI_CUSTOM swaps the module out and must be able to
        # hand it back when the custom configuration is cleared.
        self._preset_module = module_name
        self.mode = self._resolve_mode(module_name)

        # Opt-in training capture. Disabled unless config says otherwise, and
        # local-only either way — nothing here can upload.
        training_cfg = config.get("training", {}) if isinstance(config.get("training"), dict) else {}
        self.capture_store = CaptureStore(training_cfg.get("capture", {}))
        if self.capture_store.enabled:
            log.info(
                "training capture ON: writing to %s (max %d samples, never uploaded)",
                self.capture_store.directory, self.capture_store.max_samples,
            )

        self.bus = FrameBus()
        self.shutdown = threading.Event()
        self._module_lock = threading.Lock()  # guards module swaps vs. the inference loop
        self._heartbeat = 0.0  # monotonic time of the last completed inference iteration
        #: Stamps every health snapshot where it is generated (M8.1, core/health_stamp.py):
        #: a new boot id per process start, and a generation per sample.
        self._health_stamper = health_stamp.HealthStamper()
        # When the heartbeat was last reset to "starting"; bounds that grace.
        self._heartbeat_reset_at = time.monotonic()
        self._inference_service: Optional[_InferenceService] = None
        self._last_stream_push = 0.0
        self._flash_until = 0.0  # monotonic time; see _FLASH_DURATION_S
        self._consecutive_process_failures = 0  # see _MAX_CONSECUTIVE_PROCESS_FAILURES
        self._consecutive_annotate_failures = 0  # see _ANNOTATE_FAILURE_LOG_EVERY
        #: Why object detection is unavailable, or None when it is available.
        #: Set at boot by _verify_detection_backend. When set, the device runs in
        #: DEGRADED mode: it boots, serves health, the dashboard and provisioning,
        #: and says plainly that monitoring is NOT active -- rather than either
        #: crash-looping (telling the owner nothing) or running the module anyway
        #: (claiming to watch something it cannot see).
        self._detection_degraded: Optional[str] = None
        self._last_debug_print = 0.0
        # Readiness grace: capabilities still coming up read STARTING until this
        # long after boot or a mode switch, then a capability that never came
        # up is IMPAIRED. Camera grace mirrors camera_check's.
        self._ready_grace_from = time.monotonic()
        self._camera_grace_until = time.monotonic() + 10.0
        self._module_rebuild_attempts = 0
        self._module_rebuild_at = (time.monotonic() + _MODULE_REBUILD_FIRST_S
                                   if self._module_unavailable() else float("inf"))

        self.camera: Optional[Camera] = None
        self.capture: Optional[CaptureService] = None
        self.wifi: Optional[WifiManager] = None
        self.dashboard = None
        self.uplink = None
        self.provisioning: Optional[ProvisioningService] = None
        self.health: Optional[HealthMonitor] = None
        self.supervisor: Optional[Supervisor] = None
        self.logger: Optional[EventLogger] = None
        self.telemetry: Optional[TelemetryLogger] = None
        self.hardware = None

        # Monitoring nodes (core/nodes.py) beside the module. Companions the
        # owner enabled survive a restart (device_state); a preset may also
        # own nodes (presets.yaml `nodes:`), rebuilt with every preset switch.
        self._companions: dict[str, monitoring_nodes.MonitoringNode] = {}
        self._preset_nodes: dict[str, monitoring_nodes.MonitoringNode] = {}
        self._node_errors: dict[str, str] = {}
        self._node_zone_signature: Optional[tuple] = None
        self._set_preset_nodes(active_preset)
        self._restore_nodes()

    # ------------------------------------------------------------------ build
    def _build_module(self, module_name: str, module_config: dict):
        """Construct a module for a mode switch. A seam: sim/ substitutes its
        scripted perception here; production builds the registered class."""
        return get_module_class(module_name)(module_config)

    def _make_uplink(self, cloud_cfg: dict):
        """Construct the cloud uplink. A seam: sim/ steps it on its own clock."""
        from core.uplink import CloudUplink

        return CloudUplink(cloud_cfg)

    def _open_camera(self) -> Camera:
        """Open the camera with retries — transient 'device busy' at boot must not
        burn a whole systemd restart cycle (and model reload) per attempt."""
        last_exc: Optional[Exception] = None
        for attempt in range(1, _CAMERA_OPEN_RETRIES + 1):
            camera: Optional[Camera] = None
            try:
                camera = Camera(self.config.get("camera", {}))
                # Prove it produces frames before declaring victory (the "verify
                # the camera" boot step): one real read, not just a successful open.
                # Bounded: a read that hangs used to hold the whole boot (no
                # health, dashboard or BLE) until systemd's TimeoutStartSec.
                camera.read(timeout_s=_CAMERA_VERIFY_TIMEOUT_S)
                if attempt > 1:
                    log.info("camera opened on attempt %d", attempt)
                return camera
            except (CameraError, RuntimeError) as exc:
                last_exc = exc
                # Release before retrying. `Camera()` can succeed (the sensor is
                # started) and the verifying `read()` then fail, which left a live
                # handle on /dev/video* behind on every attempt — so attempts 2-5
                # were contending with the fds attempt 1 still owned, making each
                # one likelier to fail than the last. Five failures inside
                # StartLimitIntervalSec then reach StartLimitAction=reboot, so a
                # transient glitch could become a reboot cycle.
                if camera is not None:
                    with suppress(Exception):  # nothing to do if even this fails
                        camera.release()
                log.warning(
                    "camera open attempt %d/%d failed: %s", attempt, _CAMERA_OPEN_RETRIES, exc
                )
                if attempt < _CAMERA_OPEN_RETRIES:
                    time.sleep(_CAMERA_OPEN_BACKOFF_S)
        raise CameraError(f"camera unavailable after {_CAMERA_OPEN_RETRIES} attempts: {last_exc}")

    def _build(self) -> None:
        setup_logging(str(self.config.get("logging", {}).get("level", "INFO")))

        # Accelerator: probe once, install the config, verify before inference starts.
        accel_cfg = self.config.get("accelerator", {})
        info = accel.configure(accel_cfg)
        if accel_cfg.get("required", False) and not info.usable:
            # DEGRADED, for the same reason as a missing detector: raising here
            # exited before health, the dashboard or BLE existed, and under
            # Restart=always that was a crash loop into StartLimitAction=reboot.
            self._detection_degraded = f"accelerator.required is set but: {info.detail}"
            log.error("DEGRADED: %s — monitoring is NOT active", self._detection_degraded)
        else:
            self._verify_detection_backend()

        logging_config = self.config.get("logging", {})
        try:
            self.logger = EventLogger(logging_config.get("csv_path", "data/events.csv"))
        except OSError as exc:
            log.error("event log unavailable (%s); detection and alerts continue without it", exc)
            self.logger = _NullEventLog()
        try:
            self.telemetry = TelemetryLogger(
                logging_config.get("telemetry_csv_path", "data/telemetry.csv"),
                interval_s=float(logging_config.get("telemetry_interval_s", 0.5)),
                # Bounded: an unrotated telemetry file grew 15-26 MB/day until
                # the SD card filled, which is what stops the event log too.
                max_bytes=int(float(logging_config.get("telemetry_max_mb", 25)) * 1024 * 1024),
            )
        except OSError as exc:
            log.error("telemetry log unavailable (%s); continuing without it", exc)
            self.telemetry = _NullEventLog()

        try:
            self.camera = self._open_camera()
        except CameraError as exc:
            # Boot DEGRADED rather than exit. Exiting here took health, the
            # dashboard, the uplink and BLE provisioning down with it, and under
            # Restart=always + StartLimitAction=reboot a missing ribbon became an
            # unbounded reboot loop that also bypassed _last_resort's total
            # bound. An unopened camera lets the capture service keep retrying
            # the open, and the health ladder report and escalate it.
            log.error(
                "camera unavailable (%s); starting without it — capture keeps retrying "
                "and /api/health reports it", exc,
            )
            self._camera_error = str(exc)
            self.camera = Camera(self.config.get("camera", {}), open_now=False)
        self.capture = CaptureService(self.camera, self.bus)

        # Warm-up inference on a dummy frame BEFORE services start: model loads
        # (mediapipe graph, YOLO weights — a download on first ever run) happen
        # here, not inside the live loop where they'd starve capture and stall
        # the first seconds of detection. READY then genuinely means "one full
        # inference has completed". Warm-up events are discarded, never logged.
        import numpy as np

        cam_cfg = self.config.get("camera", {})
        dummy = np.zeros((int(cam_cfg.get("height", 480)), int(cam_cfg.get("width", 640)), 3), dtype=np.uint8)
        try:
            self.module.process(dummy)
        except Exception as exc:  # noqa: BLE001 - a no-face frame must never kill boot
            log.warning("model warm-up raised (continuing): %s", exc)

        # Health monitor + last-resort policy (created before dashboard so the
        # dashboard can serve /api/health from day one).
        health_cfg = self.config.get("health", {})
        self.health = HealthMonitor(
            check_interval_s=float(health_cfg.get("check_interval_s", 2.0)),
            on_last_resort=self._last_resort,
            watchdog_ping=sdnotify.notify_watchdog if sdnotify.enabled() else None,
        )

        dash_cfg = self.config.get("dashboard", {})
        if bool(dash_cfg.get("enabled", False)):
            from core.dashboard import DashboardServer

            self.dashboard = DashboardServer(
                host=str(dash_cfg.get("host", "127.0.0.1")),
                port=int(dash_cfg.get("port", 8899)),
                module_name=self.module_name,
                presets=self.presets,
                active_preset=self.active_preset,
                token=dash_cfg.get("token") or None,
                events_csv_path=logging_config.get("csv_path", "data/events.csv"),
                jpeg_quality=int(dash_cfg.get("jpeg_quality", 60)),
                stream_fps=float(dash_cfg.get("stream_fps", 12)),
                # Both halves of the picture. `health.snapshot()` answers "is each
                # supervised component up"; `device_health.collect()` answers "what
                # is the hardware actually doing" — frame-interval distribution,
                # accelerator inference evidence, thermals. The second was already
                # computed every tick for the app's status card and reached the
                # backend over the uplink, but the DEVICE'S OWN endpoint could not
                # see it. That mattered for the frame-interval percentiles: a mean
                # fps cannot show a stall, and the stall is what loses evidence.
                # caller="dashboard": fps sampling keeps a per-caller baseline,
                # because this runs on the HTTP thread on every request while the
                # inference loop samples on its own cadence. Sharing one baseline
                # made each consume the other's window and report ~0 fps.
                health_provider=lambda: {**self.health.snapshot(),
                                         "device": self._collect_health("dashboard")},
            )
            # Refuse a role-violating preset at the request, with the reason, rather
            # than accept it and silently drop it in the inference loop.
            from main import resolve_preset

            self.dashboard.preset_guard = lambda name: roles.preset_violation(
                self.role, resolve_preset(name, self.presets), self.presets)
            self.dashboard.set_policy_state(self._policy_state())
            try:
                dashboard_url = self.dashboard.start()
            except OSError as exc:
                # The dashboard is the live feed and local API, not the detector.
                # A port still held by a stale process, or a host this IPv4
                # server cannot bind, used to abort the whole device -- and five
                # aborts in two minutes make systemd reboot it. Detection and
                # alerts carry on without it.
                log.error(
                    "dashboard could not start on %s:%s (%s); continuing without "
                    "the live feed and local API",
                    dash_cfg.get("host", "127.0.0.1"), dash_cfg.get("port", 8899), exc,
                )
                self.dashboard = None
            else:
                print(
                    f"SensorDyme: dashboard live at {dashboard_url} — annotated frames stream "
                    "in-memory to connected browsers only (localhost by default); never written to disk"
                )

        cloud_cfg = self.config.get("cloud", {})
        if bool(cloud_cfg.get("enabled", False)) and cloud_cfg.get("device_key_file"):
            # The key lives in its own 0600 file (core/secret_store.py). A file
            # that cannot be used leaves the device running standalone, said
            # once -- never a crash loop, never a fallback to no key.
            from core.secret_store import CredentialError, resolve_cloud_credentials

            try:
                cloud_cfg, credential_warnings = resolve_cloud_credentials(cloud_cfg)
                for warning in credential_warnings:
                    print(f"SensorDyme: credentials: {warning}")
            except CredentialError as exc:
                print(f"SensorDyme: cloud uplink OFF -- the device key file cannot be used: {exc}")
                cloud_cfg = {**cloud_cfg, "device_key": ""}
        if bool(cloud_cfg.get("enabled", False)) and cloud_cfg.get("device_key"):
            # M8: the undelivered-event backlog lives on disk beside the event log,
            # so alerts raised during an outage survive a restart or a power cut.
            if not cloud_cfg.get("spool_path"):
                events_csv = str(logging_config.get("csv_path", "data/events.csv"))
                cloud_cfg = {**cloud_cfg,
                             "spool_path": str(Path(events_csv).parent / "uplink_spool.jsonl")}
            self.uplink = self._make_uplink(cloud_cfg)
            self.uplink.set_active_preset(self.active_preset)
            # Report the restored configuration at startup too, not only when a
            # new one arrives — otherwise a device that rebooted into a custom
            # configuration shows the base mode until the next change.
            self._publish_monitoring()
            self.uplink.start()
            print(
                f"SensorDyme: cloud uplink -> {cloud_cfg.get('base_url')} "
                "(events + mode commands; never frames)"
            )

        # Wire a RESTORED rule set into the running system. Both calls used to
        # happen only in the inference loop, when a NEW rule set arrived over the
        # uplink -- and `_apply_monitoring_mode()` additionally sat inside the
        # `cloud.enabled` block above, so on a cloud-disabled device it never ran
        # at all. A device that rebooted with rules already saved therefore came
        # up in AI_CUSTOM mode, discarded every event its module produced (see
        # the `self.monitoring.active` branch in the inference loop), never
        # switched to the module the rules needed, never told any module which
        # labels to request -- and reported itself healthy the whole time.
        # Verified in Phase 6: 7 frames processed, 1 rule active, 0 events.
        #
        # Unconditional and before the loop starts, because "the device was
        # restarted" must not be a different code path from "a rule arrived".
        if self.monitoring.active:
            self._apply_monitoring_mode()
            self._apply_requested_labels()

        wifi = WifiManager()
        self.wifi = wifi  # also read by the health report the app displays
        network_cfg = self.config.get("network", {})
        self.provisioning = ProvisioningService(
            wifi, network_cfg.get("provisioning", {}), version=_VERSION,
            deployment=str(self.config.get("deployment", "development")),
        )

        # Hardware HAL: null implementations today; health-visible either way.
        from hardware.hal import build_hardware

        self.hardware = build_hardware(self.config.get("hardware", {}))

        # ---- health registrations -------------------------------------------------
        camera_fps = max(self.camera.fps, 1.0)
        stale_after = max(2.0, 5.0 / camera_fps)  # 5 frame intervals, 2s floor
        boot_time = time.monotonic()
        self._camera_stale_after_s = stale_after
        self._camera_grace_until = boot_time + 10.0

        def camera_check() -> tuple[bool, str]:
            age = self.bus.age_s()
            if age == float("inf"):  # no frame yet: grace during service startup
                starting = time.monotonic() - boot_time < 10.0
                return starting, "waiting for first frame"
            return age < stale_after, f"last frame {age:.1f}s ago"

        self.health.register(
            "camera",
            critical=True,
            check=camera_check,
            restart=lambda: self.capture.restart(),  # noqa: PLW0108 - late-binds self.capture
            check_interval_s=2.0,
        )
        self.health.register(
            "inference",
            critical=True,
            check=self._inference_check,
            # With --display the loop runs on the main thread and cannot be
            # bounced. A no-op restart "succeeded" three times in ~6s, spent the
            # budget, and reached the last-resort reboot on a hand-run Pi.
            restart=None if self.display else self._restart_inference,
            check_interval_s=2.0,
        )
        # live_status(), not status(): the endpoint must report what is actually
        # computing, not what the config asked for. See core/accel.py:live_status.
        # 10s, not 60s: inference_stats() calls a result "live" only within 10s, so
        # a 60s interval would spend most of its time reporting a stale verdict.
        # Gated on whether the ACTIVE module uses the accelerator, for the same
        # reason device_health's monitoring_active is. `accel.live_status()` reads
        # a process-global inference counter that does NOT reset on a module swap,
        # so after switching into Elderly Monitoring (fall_detection is
        # MediaPipe-Pose-only and never touches the NPU) the previous module's
        # inferences age out and it reports "Hailo NPU idle — pipeline stalled or
        # fell back to CPU". Three checks later the component is DEGRADED and the
        # top-level `state` in /api/health follows -- for the whole Elderly
        # segment of the demo, on a perfectly working fall detector.
        self.health.register(
            "accelerator",
            check=self._accelerator_health,
            check_interval_s=10.0,
        )
        # The product-level state (READY/DEGRADED/FAILED/...). Evaluated on the
        # monitor thread, not the inference thread, so a wedged inference loop
        # is still detected, reported and signalled (impaired alert). Not
        # critical and no restart: the components it reads have their own.
        self.health.register(
            "monitoring",
            check=self._monitoring_check,
            check_interval_s=2.0,
        )
        if self.dashboard is not None:
            self.health.register(
                "dashboard",
                check=lambda: (self.dashboard.is_alive(), self.dashboard.url),
                restart=lambda: self.dashboard.restart(),  # noqa: PLW0108 - late-binds self.dashboard
                check_interval_s=10.0,
            )
        if self.uplink is not None:
            self.health.register(
                "cloud_uplink",
                check=lambda: (
                    self.uplink.connected,
                    "connected" if self.uplink.connected else "offline — events queued on device",
                ),
                check_interval_s=15.0,
            )
        # Non-critical: a hot device is still a working device, and the Pi's
        # firmware protects the hardware on its own. This is here so a person
        # can see the temperature and whether the fan is actually turning,
        # rather than having to SSH in to find out.
        from core import thermal
        self.health.register("thermal", check=thermal.status, check_interval_s=20.0)
        self.health.register("wifi", check=wifi.status, check_interval_s=15.0)
        self.health.register(
            "provisioning",
            check=lambda: self.provisioning.status(),  # noqa: PLW0108 - late-binds self.provisioning
            check_interval_s=15.0,
        )
        for name, component in self.hardware.health_checks().items():
            self.health.register(name, check=component, check_interval_s=60.0)

        if self.config_fault is not None:
            # Reported, not hidden: in the event log here, in every health report
            # (_config_report), and on the journal by main.load_device_config.
            log.error("[CONFIG] %s", self.config_fault.reason)
            self._log_system_event("config_fault", "running on the last known good config "
                                   f"({self.config_fault.fallback_path}): {self.config_fault.reason}")
        for fault in getattr(self, "_state_faults", []) or []:
            self._log_system_event("saved_state_fault", fault)

        # ---- supervisor -----------------------------------------------------------
        self.supervisor = Supervisor(self.health)
        self.supervisor.add(self.capture)
        if not self.display:
            self._inference_service = _InferenceService(self)
            self.supervisor.add(self._inference_service)
        self.supervisor.add(self.provisioning)
        if self.uplink is not None:
            self.supervisor.add(_HealthReportService(self))

    def _apply_requested_labels(self) -> None:
        """Tell the running module which objects the active rules need.

        Without this the chain breaks at its last link: a rule naming `dog` and
        `couch` is valid, stored and delivered, but the module asks the detector
        only for people, so the rule can never fire. The user would see a rule
        listed as active that does nothing.
        """
        labels = {rule.detection for rule in self.rule_engine.rules if rule.detection}
        labels |= {rule.secondary for rule in self.rule_engine.rules if rule.secondary}
        with self._module_lock:
            module = self.module
        module.set_requested_labels(labels)
        # M7: the rules are a perception consumer in their own right. Their demand
        # makes the shared detector look for exactly the labels they name, on the
        # same per-frame run the module triggers when one happens (a module such as
        # driver_attention asks only for its own classes), and lists "rules" among
        # the detector's consumers in the device's perception report.
        self.perception.withdraw(RULES_CONSUMER)
        if labels:
            self.perception.demand(pc.Demand(consumer=RULES_CONSUMER, component=pc.OBJECT_DETECTOR,
                                             labels=frozenset(labels), min_interval_s=RULES_DEMAND_INTERVAL_S))
        if labels and module.get_detections() is None:
            # The module cannot report per-frame detections at all, so these
            # rules will never be evaluated. Say so rather than fail silently.
            log.warning(
                "module '%s' does not expose per-frame detections; %d AI-configured rule(s) "
                "cannot be enforced in this mode",
                self.module_name, len(self.rule_engine.rules),
            )
        elif labels:
            # A module can expose detections and still not produce THIS label —
            # occupancy reports objects, never a posture, so a "sleeping"
            # (lying_down) rule left running under it would validate, look
            # active in the app, and simply never see the label it needs.
            # `_apply_monitoring_mode` already switches to the module that
            # covers a pose-only rule set; this covers the rule sets it
            # deliberately leaves on occupancy — mixed with an object rule, or
            # running under a STANDARD preset that never runs pose either.
            from core.capabilities import registry as capability_registry

            reg = capability_registry()
            covered = set(monitoring_mode.capabilities_covered_by(self.module_name))
            if self.perception.component(pc.OBJECT_DETECTOR).state != pc.UNAVAILABLE:
                # The rules' own demand (above) has the shared detector serve COCO
                # labels whatever module runs; only labels no component can give
                # (a posture under a module that runs no pose) stay uncovered.
                covered.add("objects.coco")
            uncovered = {
                label for label in labels
                if not ({c.id for c in reg.providers_of_label(label) if c.available} & covered)
                and reg.providers_of_label(label)  # only warn about a label SOME module can serve
            }
            if uncovered:
                log.warning(
                    "module '%s' does not produce %s; the matching AI-configured rule(s) "
                    "are active but cannot fire until monitoring switches to a module that does",
                    self.module_name, ", ".join(sorted(uncovered)),
                )

    #: Config keys through which a module names its detector weights. A module
    #: declaring none (fall_detection is MediaPipe-Pose-only; dashsentinel loads
    #: no detector) runs no object detection at all and must NOT be made to load
    #: a model just to prove a backend it never uses exists.
    _WEIGHTS_KEYS = ("yolo_weights", "model_weights", "phone_weights", "detector_weights")

    def _verify_detection_backend(self) -> None:
        """Resolve the object-detection backend at boot, or fail with one clear line.

        WHY THIS EXISTS. Since ultralytics left the device image (AGPL-3.0, see
        docs/LICENSING.md), the Hailo path is the ONLY detector present -- "no
        backend" is now a total loss of object detection rather than a quiet
        fallback to a slower one. occupancy.py already surfaces that at startup
        because it loads its model in __init__, but driver_attention.py loads
        lazily on its first stride tick, INSIDE the inference loop, where the
        failure reads as a warm-up warning followed by a restart loop instead of
        an explanation. This collapses both cases into the same message, at the
        same moment, before the camera is even opened.

        It LOADS the model rather than probing for one, deliberately: this runs in
        the process that will use it, so the model lands in the shared cache and
        the lazy caller gets it for free. scripts/preflight.py does the opposite
        and only probes -- that is a separate process which exits moments before
        this one starts, so loading there would race the service for the NPU.
        """
        # Re-decides from scratch, so a stale reason must not survive. Without
        # this, a device that booted degraded and then switched to a module
        # needing no detector (fall_detection is MediaPipe-Pose-only) would stay
        # marked degraded while working perfectly -- the mirror of the bug this
        # method exists to prevent. Assigned once the answer is known, not
        # cleared up front: health is sampled on other threads, and a cleared
        # reason during a seconds-long load read as "monitoring active".
        #
        # Unknown while verifying, then assigned ONCE, complete. Filling it in
        # piecemeal left a window (a CPU load can take seconds) in which events
        # went out naming the module with detector=None — or, before the swap
        # sites cleared it, naming the PREVIOUS module's model.
        self._model_identity = None
        identity: dict = {"module": self._module_identity_name(), "detector": None}

        module_cfg = self._active_module_config()
        weights = next((module_cfg[k] for k in self._WEIGHTS_KEYS if module_cfg.get(k)), None)
        if not weights:
            log.info("object detection: %s uses no object detector", self.module_name)
            self._detection_degraded = None
            self._model_identity = identity
            return
        from core import detectors

        try:
            # `phone_device` is the comparison modules' name for the same pin.
            device = module_cfg.get("device") or module_cfg.get("phone_device")
            backend = detectors.require_backend(str(weights), device)
        except Exception as exc:  # noqa: BLE001 - the reason is reported, not swallowed
            # DEGRADED, not dead, and not pretending. Crash-looping here would
            # take down health, the dashboard and BLE provisioning too, leaving
            # an owner with a device that says nothing at all. Continuing as
            # normal would be worse: the module would run against a detector
            # that does not exist and the app would show "monitoring active".
            # So: boot, refuse to analyse, and be loud about which one it is.
            self._detection_degraded = str(exc)
            log.error(
                "DEGRADED: object detection unavailable for %s — monitoring is NOT "
                "active. %s", self.module_name, exc,
            )
            self._model_identity = identity
            return
        identity["detector"] = detectors.model_identity(str(weights), device)
        self._detection_degraded = None
        self._model_identity = identity
        log.info(
            "object detection backend for %s: %s (%s) — model %s", self.module_name, backend, weights,
            identity["detector"],
        )

    def _retry_detection_backend(self) -> None:
        """While DEGRADED, try the detector again.

        Boot verified it once, and only a preset switch ever re-checked: a
        transient Hailo init failure (or a model provisioned after boot) left
        the device refusing to analyse until someone restarted it. Wrapped as
        a module swap so a slow load reads as busy, not hung.
        """
        accel_cfg = self.config.get("accelerator", {}) or {}
        if accel_cfg.get("required", False):
            # A required HAT is re-probed rather than bypassed: a CPU detector
            # must not quietly stand in for an accelerator the config demands.
            info = accel.configure(accel_cfg)
            if not info.usable:
                self._detection_degraded = f"accelerator.required is set but: {info.detail}"
                return
        with self._module_swap():
            self._verify_detection_backend()
        if self._detection_degraded is None:
            log.info("object detection recovered for %s; monitoring is active again",
                     self.module_name)

    def _module_unavailable(self) -> bool:
        from modules.base import UnavailableModule

        return isinstance(getattr(self, "module", None), UnavailableModule)

    def _retry_unavailable_module(self) -> bool:
        """Rebuild a module that could not be built at boot (M8). True once it runs.

        main.py boots FAILED-but-reachable with an UnavailableModule when the
        module raised (a model file not mounted yet, a transient load error).
        Nothing ever tried again, so a fault that cleared a second later kept
        the device FAILED -- impaired buzzer and all -- until a person restarted
        it. Retried here, on the inference thread like every module swap, after
        _MODULE_REBUILD_FIRST_S and then at doubling intervals up to
        _MODULE_REBUILD_MAX_S. Each failure keeps the reason current.
        """
        current = self.module
        if not self._module_unavailable() or self.profile == "mac_max":
            self._module_rebuild_at = float("inf")
            return not self._module_unavailable()
        self._module_rebuild_attempts += 1
        delay = min(_MODULE_REBUILD_MAX_S, _MODULE_REBUILD_FIRST_S * 2 ** (self._module_rebuild_attempts - 1))
        if current is self._boot_module and self._boot_module_config is not None:
            module_cfg = self._boot_module_config
        else:
            module_cfg = (self.config.get("modules", {}) or {}).get(self.module_name, {}) or {}
        try:
            with self._module_swap():
                new_module = self.perception.adopt(self._build_module(self.module_name, dict(module_cfg)))
        except Exception as exc:  # noqa: BLE001 - still FAILED, with the current reason
            self._module_rebuild_at = time.monotonic() + delay
            current.reason = str(exc)
            log.warning("module '%s' still cannot be built (attempt %d, next in %.0f s): %s",
                        self.module_name, self._module_rebuild_attempts, delay, exc)
            return False
        self._push_zones_to(new_module)
        with self._module_lock:
            self.module = new_module
            if current is self._boot_module:
                self._boot_module = new_module
            self._model_identity = None
            self._consecutive_process_failures = 0
            self._ready_grace_from = time.monotonic()
        self._module_rebuild_at = float("inf")
        self._verify_detection_backend()
        if self._custom_active():
            self._apply_requested_labels()
        with suppress(Exception):
            current.close()
        log.info("module '%s' built on attempt %d; monitoring resumes", self.module_name,
                 self._module_rebuild_attempts)
        self._log_system_event("module_recovered", f"module '{self.module_name}' built after "
                               f"{self._module_rebuild_attempts} retr{'y' if self._module_rebuild_attempts == 1 else 'ies'}")
        self._publish_monitoring()
        return True

    def _active_module_config(self) -> dict:
        """The config the RUNNING module was built from (see __init__)."""
        if self._boot_module_config is not None and self.module is self._boot_module:
            return self._boot_module_config
        switched_config = getattr(self, "_switched_module_config", None)
        if switched_config is not None and self.module is getattr(self, "_switched_module", None):
            return switched_config
        return (self.config.get("modules", {}) or {}).get(self.module_name, {}) or {}

    def _module_identity_name(self) -> str:
        """The module's own registered name: "driver_attention_max", not the
        "driver_attention" the mac_max profile passes as module_name."""
        return str(getattr(self.module, "name", None) or self.module_name)

    @property
    def model_identity(self) -> Optional[dict]:
        """Which exact model this runtime's events come from, for the uplink."""
        return self._model_identity

    def _model_version(self) -> str:
        """Which model produced a sample — useless data without it, since you
        cannot tell an old model's mistakes from a new one's."""
        detector = (self._model_identity or {}).get("detector")
        if detector:
            sha = detector.get("sha256")
            return f"{detector['name']}@{sha[:12]}" if sha else str(detector["name"])
        # Before (or without) a verified load: the configured name, from every
        # weights key a module can declare.
        module_cfg = self._active_module_config()
        # Every key a module might use, not just model_weights. driver_attention
        # -- the flagship -- declares `yolo_weights`, so reading only
        # `model_weights` stamped every one of its captured samples with the
        # hardcoded "yolo11n.pt" default. A training sample labelled with a model
        # that did not produce it is worse than an unlabelled one: you cannot tell
        # an old model's mistakes from a new one's, which is this method's whole
        # purpose. _WEIGHTS_KEYS is the same tuple _verify_detection_backend uses.
        weights = next((module_cfg[k] for k in self._WEIGHTS_KEYS if module_cfg.get(k)), None)
        # On-device the .pt name is only a cache key -- the real weights are the
        # .hef -- so record the accelerator's model when one is in use.
        hef = (self.config.get("accelerator", {}) or {}).get("yolo_hef")
        if hef and accel.inference_stats()["count"] > 0:
            import os

            return f"{weights or 'unknown'}+{os.path.basename(hef)}"
        return str(weights or "unknown")

    def _capture_training_sample(self, frame, alert, detection_frame) -> None:
        """Keep the frame that triggered a rule, if capture is switched on."""
        if not self.capture_store.should_capture():
            return
        self.capture_store.capture(
            frame,
            event_type=alert.detection,
            rule_id=alert.rule_id,
            confidence=alert.confidence,
            module=self.module_name,
            model_version=self._model_version(),
            # The model's guess, so a labeller starts from it rather than a blank frame.
            detections=[
                {"label": d.label, "confidence": d.confidence, "box": list(d.box) if d.box else None}
                for d in detection_frame.detections
            ],
        )

    # ------------------------------------------------------------------ health reporting
    def _collect_health(self, caller: str = "loop") -> dict:
        """This device's real state, for the app's status indicators."""
        report = self._collect_component_health(caller)
        # The product-level answer next to the component-level one: the
        # monitoring state decides monitoring_active (READY or DEGRADED), not
        # the components alone -- a stopped, failed or still-starting device is
        # not monitoring however healthy its parts look.
        monitoring = self.monitoring_report()
        report["monitoring"] = monitoring.to_dict()
        report["monitoring_active"] = monitoring.monitoring
        report["protection"] = self.protection_state()
        report["config"] = self._config_report()
        report["saved_state"] = {"faults": list(getattr(self, "_state_faults", []) or [])}
        backlog = getattr(self.uplink, "backlog", None)
        if callable(backlog):
            report["uplink_backlog"] = backlog()
        # Stamped HERE, where the sample is taken, not where it is sent (M8.1): a
        # snapshot re-sent unchanged keeps its generation, so the backend can tell
        # a frozen report from a new one whatever the device's wall clock says.
        return self._stamper().stamp(report)

    def _stamper(self) -> health_stamp.HealthStamper:
        stamper = getattr(self, "_health_stamper", None)
        if stamper is None:  # a runtime built without __init__ (unit-test doubles)
            stamper = self._health_stamper = health_stamp.HealthStamper()
        return stamper

    def _check_saved_state(self) -> list[str]:
        """Faults in the files that carry the owner's choices across a restart.

        device_state.json (mode, zones, companion nodes) and monitoring.json
        (custom rules) are read as empty when damaged -- the device boots on its
        configured mode, which is the safe direction -- but the owner's choices
        are gone, and the next save overwrites the evidence. So each damaged file
        is copied aside (<name>.corrupt) and named in every health report.
        """
        from core.monitoring_config import DEFAULT_PATH as MONITORING_DEFAULT

        paths = {
            "device_state": Path(os.environ.get("SENSORDYME_DEVICE_STATE_PATH") or device_state.DEFAULT_PATH),
            "custom_rules": Path(self.monitoring_path or MONITORING_DEFAULT),
        }
        faults: list[str] = []
        for what, path in paths.items():
            try:
                raw = path.read_bytes()
            except FileNotFoundError:
                continue
            except OSError as exc:
                faults.append(f"{what} ({path.name}) unreadable: {exc}")
                continue
            try:
                ok = isinstance(json.loads(raw), dict)
            except (ValueError, UnicodeDecodeError):
                ok = False
            if ok:
                continue
            faults.append(f"{what} ({path.name}) was damaged and could not be restored; "
                          f"kept as {path.name}.corrupt")
            with suppress(OSError):
                path.with_name(path.name + ".corrupt").write_bytes(raw)
        for fault in faults:
            log.error("[STATE] %s", fault)
        return faults

    def _config_report(self) -> dict:
        """Which configuration is in force, and the fault when it is not config.yaml (M8)."""
        if self.config_fault is not None:
            return self.config_fault.describe()
        return {"source": "config", "path": self.config_path, "fault": None}

    def _record_known_good_config(self) -> None:
        """Once READY: this config.yaml is the one to fall back to (core/config_lkg.py).

        Never while running FROM the fallback -- the broken config.yaml must not
        become the "good" copy -- and never for a runtime built without a file.
        """
        if not self.config_path or self.config_fault is not None:
            return
        from core import config_lkg

        if not config_lkg.save_last_good(self.config_path):
            log.warning("could not record %s as the last known good config (%s)",
                        self.config_path, config_lkg.last_good_path(self.config_path))

    def _collect_component_health(self, caller: str) -> dict:
        heartbeat_age = None if self._heartbeat == 0.0 else time.monotonic() - self._heartbeat
        return device_health.collect(
            bus=self.bus,
            capture=self.capture,
            wifi=self.wifi,
            heartbeat_age_s=heartbeat_age,
            module_name=self.module_name,
            uplink_connected=self.uplink.connected if self.uplink is not None else None,
            # A module that keeps running without its detector (face-based
            # driver monitoring) is analysing; the monitoring block says what
            # it cannot do. Reported as "not active" only when nothing runs.
            detection_degraded=(self._detection_degraded
                                if not self._module_runs_without_detector() else None),
            caller=caller,
            # Whether the ACTIVE module uses the accelerator at all. Recomputed
            # per call because the active module changes at runtime when the
            # operator switches preset mid-demo.
            detector_required=self._module_uses_a_detector(),
            # Whatever the accelerator probe says, no detector is no detector:
            # never "OK, running on CPU" on a device with no CPU backend (M8).
            detector_unavailable=self._detection_degraded,
        )

    def _accelerator_health(self) -> tuple[bool, str]:
        """Accelerator health, scoped to whether this mode actually uses it."""
        if not self._module_uses_a_detector():
            return True, f"not used by this mode ({self.module_name}) — {accel.status()[1]}"
        if self._detection_degraded is not None:
            return False, f"object detection unavailable — {self._detection_degraded}"
        return accel.live_status()

    def _module_uses_a_detector(self) -> bool:
        """Does the currently running module consume the object detector?

        Same _WEIGHTS_KEYS signal `_verify_detection_backend` uses, so health and
        the boot check can never disagree about which modules need a detector.
        """
        module_cfg = self._active_module_config()
        return any(module_cfg.get(key) for key in self._WEIGHTS_KEYS)

    # ------------------------------------------------------------------ AI-configured rules
    def _zones_for_module(self, module_name: str) -> list:
        """Zones configured for `module_name`, or none if it has no zone support."""
        modules_cfg = self.config.get("modules", {})
        module_cfg = modules_cfg.get(module_name, {}) if isinstance(modules_cfg, dict) else {}
        return load_zones(module_cfg.get("zones", {}) if isinstance(module_cfg, dict) else {})

    def _effective_zones(self, module_name: str) -> list:
        """The zones in force for `module_name`: the app's once it has sent any
        (an empty list included — the owner deleted them), else config.yaml's."""
        if self._app_zones is not None:
            return list(self._app_zones)
        return self._zones_for_module(module_name)

    def _push_zones_to(self, module) -> None:
        """Hand the app's zones to a module, so its own zone alerts use them.

        Without this the app's zones only ever reached AI-configured rules: a
        crib drawn as "Stay inside" never produced safe_zone_exit, because
        modules read config.yaml's zones once, at construction. A no-op until
        the app has sent zones, so config.yaml zones keep working.
        """
        if self._app_zones is None:
            return
        set_zones = getattr(module, "set_zones", None)
        if not callable(set_zones):
            return
        try:
            set_zones(list(self._app_zones))
        except Exception:  # bad zones must not stop detection
            log.exception(
                "module '%s' could not apply the app's zones; it keeps its previous zones",
                getattr(module, "name", "?"),
            )

    def _apply_monitoring_mode(self) -> None:
        with self._module_swap():
            self._apply_monitoring_mode_now()

    def _apply_monitoring_mode_now(self) -> None:
        """Run the detector the active mode calls for, and nothing else.

        This is what makes "AI custom monitoring" mean something. Suppressing a
        module's alerts leaves it running: MediaPipe FaceMesh, Hands and Pose
        cost 85% of one core on this device, computing drowsiness and gaze for
        a person who asked about phones. Switching the module stops the work.
        """
        mode = self._resolve_mode(self._preset_module or self.module_name)
        requested_labels = None
        if mode is monitoring_mode.MonitoringMode.AI_CUSTOM:
            requested_labels = {rule.detection for rule in self.rule_engine.rules if rule.detection}
            requested_labels |= {rule.secondary for rule in self.rule_engine.rules if rule.secondary}
        wanted = monitoring_mode.module_for(
            mode, self._preset_module or self.module_name, requested_labels=requested_labels
        )
        if wanted is None or wanted == self.module_name:
            self.mode = mode
            return

        try:
            module_cfg = self.config.get("modules", {}).get(wanted, {}) or {}
            new_module = self.perception.adopt(self._build_module(wanted, {**module_cfg, "recalibrate": False}))
        except Exception as exc:  # noqa: BLE001 - a bad switch must not kill the device
            log.error("could not switch to '%s' for %s: %s", wanted, mode.value, exc)
            return
        self._push_zones_to(new_module)

        with self._module_lock:
            old_module, self.module, self.module_name = self.module, new_module, wanted
            self._model_identity = None  # unknown, not the old module's, until verified below
            self._consecutive_process_failures = 0  # the new module starts with a clean slate
        # The new module may need a detector the old one did not, or none at all.
        # Verified once at boot was not enough: a device could boot on Elderly
        # Monitoring (Pose-only, no detector required, so nothing was degraded),
        # then be switched in the app to Security or Baby, whose modules load YOLO
        # lazily inside process() -- raising every frame while monitoring_active
        # stayed true and the banner said "active".
        self._verify_detection_backend()
        # Release the old graphs explicitly. MediaPipe holds native resources
        # that a dropped reference does not free promptly, and a preset switch
        # used to leak one FaceMesh graph per swap.
        close = getattr(old_module, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:  # noqa: BLE001 - teardown must never raise
                log.debug("old module close() failed during swap: %r", exc)

        # Zones survive the swap when the swap is INTO AI-custom monitoring.
        #
        # The app sends zones and rules in a single poll: zones are applied
        # first, then the rules, and the rules are what trigger this swap. Taking
        # the new module's config.yaml zones here therefore replaced the polygon
        # the user had just drawn -- with `{}` in the shipped config -- so the
        # rule referencing it could never fire. The uplink had already acked
        # zones_version, so the app never re-sent, and nothing recovered.
        # Verified: 400 frames of a person inside the drawn polygon, zero alerts,
        # with the video correct, people being detected, monitoring_active True
        # and the app still listing the rule as active.
        #
        # Keyed on the MODE rather than on tracking the delivery, because in
        # AI-custom the zones always come from the app by definition, however
        # they arrived. Otherwise the effective zones apply: the app's when it
        # has sent any, else the new module's config.yaml zones.
        if mode is not monitoring_mode.MonitoringMode.AI_CUSTOM:
            self.rule_engine.set_zones(self._effective_zones(wanted))
        self.mode = mode
        log.info(
            "[MONITORING] mode=%s module=%s base_monitoring=%s%s",
            mode.value, wanted,
            "DISABLED" if mode is monitoring_mode.MonitoringMode.AI_CUSTOM else "enabled",
            f" instruction={self.monitoring.instruction!r}" if self.monitoring.instruction else "",
        )

    def _emit_rule_alert(self, alert, now: Optional[float] = None) -> None:
        """Route a confirmed AI-rule alert through the same path as built-in events.

        Custom rules are not a second-class channel: they log, notify locally, and
        reach the phone exactly the way a preset's own detections do.
        """
        event = Event.now(
            type=alert.detection,
            duration=alert.duration_s,
            confidence=alert.confidence,
            module_name=self.module_name,
            # message(), not description: for a counting rule the NUMBER is the
            # answer someone asked for. "7 bottles visible" is what they wanted;
            # "Count the bottles" is only what they typed.
            contributing_signals=alert.message(),
        )
        # Alert before recording, for the same reason as the module-event path.
        # Keyed per RULE, at the rule's own severity: keyed on the event type
        # (the target label), an info rule about "person at the door" put a
        # critical "child in the pool" rule into cooldown, and a critical rule
        # sounded as a one-chirp SOFT alert the first time.
        self.alert_manager.notify(
            event,
            key=alert.rule_id,
            level=AlertLevel.CONTINUOUS if alert.severity == "critical" else AlertLevel.SOFT,
            category=_RULE_CATEGORY.get(alert.severity, "warning"),
            source="rules",
            now=now,
        )
        self.logger.log(event)
        severity = "loud" if alert.severity == "critical" else "soft"
        if self.dashboard is not None:
            self.dashboard.push_event(event, severity)
        self._flash_until = time.monotonic() + _FLASH_DURATION_S
        if self.uplink is not None and alert.notify != "silent":
            # Marked as configured. `notify: "push"` on a rule IS someone asking
            # to be told; the severity field is about how prominent it looks,
            # not about whether it is worth interrupting for.
            # The rule's own delivery choice (push / sms) travels with the event:
            # the backend routes on it instead of texting for every alert.
            self.uplink.push_event(
                event, severity, source="custom_rule", model=self._model_identity,
                notify=alert.notify,
            )

    # ------------------------------------------------------------------ health hooks
    @contextmanager
    def _module_swap(self):
        """Mark the inference thread as busy swapping modules, not hung.

        A swap builds the new module, loads its models and closes the old one
        on the inference thread; on a Pi that can take longer than the 10s the
        heartbeat check allows. The check then "restarted" a loop that was
        working, the restart was refused while the swap finished, and the
        device lost ~40s of inference and two of its three restarts -- a second
        slow swap within five minutes reached the last-resort reboot.
        """
        self._swap_started = time.monotonic()
        before = getattr(self, "module", None)
        try:
            yield
        finally:
            self._swap_started = None
            self._heartbeat = time.monotonic()
            if getattr(self, "module", None) is not before:
                self._ready_grace_from = time.monotonic()  # a new module: STARTING until it analyses

    def _inference_check(self) -> tuple[bool, str]:
        swap_started = getattr(self, "_swap_started", None)
        if swap_started is not None:
            busy = time.monotonic() - swap_started
            # Bounded: a swap that never finishes IS a hang.
            return busy < _MAX_SWAP_S, f"switching module ({busy:.0f}s)"
        if self._heartbeat == 0.0:
            # Bounded. An unbounded "starting" read a loop that never stamped —
            # wedged on its first frame, or a restart that never really
            # happened — as healthy for ever, and resumed feeding the watchdog.
            starting_for = time.monotonic() - self._heartbeat_reset_at
            return starting_for < _INFERENCE_STARTING_GRACE_S, f"starting ({starting_for:.0f}s)"
        age = time.monotonic() - self._heartbeat
        return age < 10.0, f"last iteration {age:.1f}s ago"

    def _restart_inference(self) -> None:
        if self.display:
            return  # display mode runs inference on the main thread; nothing to bounce
        # Back to "starting": the restarted loop gets the same grace a fresh
        # boot does, instead of being declared dead again on the very next
        # check because the stamp still describes the loop being replaced.
        # Only for a restart that HAPPENED: when the wedged thread outlives
        # stop(), restart() raises, and the old stamp must keep reporting it.
        previous = self._heartbeat
        self._heartbeat = 0.0
        self._heartbeat_reset_at = time.monotonic()
        try:
            self.supervisor.restart("inference")
        except Exception:
            self._heartbeat = previous
            raise

    def _last_resort(self, component: str) -> None:
        """A critical component is down and restarts are spent: reboot the Pi —
        config-gated, rate-limited, TOTAL-bounded, and never on a dev machine.

        Two independent limits, because they fail differently:

          * `reboot_min_interval_s` spaces reboots out. On its own it does not
            bound anything: a permanently dead camera satisfies it forever and
            reboots the device every 30 minutes until someone notices.
          * `reboot_max_total` bounds the campaign. Past it the device stays up
            in its degraded state and keeps reporting through /api/health, which
            is strictly more useful than a box that power-cycles all night: a
            fault that survived N reboots is not going to be fixed by N+1.

        The rate-limit state is CONSULTED AND PERSISTED BEFORE the reboot, and a
        failure to persist it BLOCKS the reboot rather than being swallowed. That
        ordering is the whole point. `data/` going read-only or full is a real
        field failure on a worn SD card (see core/logger.py's own notes on it) —
        and it is precisely the condition under which an unrecorded reboot would
        loop unbounded, because every boot would then re-read "no reboots yet".
        Refusing to reboot when we cannot remember having done so is the only
        safe direction.
        """
        health_cfg = self.config.get("health", {})
        reboot_allowed = bool(health_cfg.get("allow_reboot", False)) and nmcli_available()
        min_interval = float(health_cfg.get("reboot_min_interval_s", 1800))
        max_total = int(health_cfg.get("reboot_max_total", 5))
        state = self._reboot_state()
        now = time.time()
        # A campaign ends once the device has gone a day without needing one.
        # Without this the total was a LIFETIME count: after `reboot_max_total`
        # last resorts spread over months, auto-recovery was off for good.
        if state["last_at"] and now - state["last_at"] > _REBOOT_CAMPAIGN_RESET_S:
            state = {**state, "count": 0}

        # Rung 4 before rung 5: a process restart before a reboot. The restart
        # budget runs out in well under WatchdogSec, so this used to reboot
        # before the watchdog ever restarted the process. While a critical
        # component is DOWN the health monitor withholds the watchdog ping, so
        # declining here IS the process restart — but only when systemd's
        # watchdog is armed; without one nothing would restart us.
        if _systemd_watchdog_armed():
            tried_at = state["process_restarts"].get(component, 0.0)
            if not 0.0 <= now - tried_at <= _PROCESS_RESTART_WINDOW_S:
                state["process_restarts"] = {**state["process_restarts"], component: now}
                if not self._write_reboot_state(state):
                    log.critical(
                        "'%s' is down and unrecoverable, and the recovery record could not "
                        "be written to %s; staying up degraded.",
                        component, self._REBOOT_STATE_PATH,
                    )
                    self._stay_up_degraded(component)
                    return
                log.critical(
                    "'%s' is down and its restarts are spent; the systemd watchdog will now "
                    "restart the process. A reboot follows only if it goes down again within "
                    "%.0f min.", component, _PROCESS_RESTART_WINDOW_S / 60,
                )
                return

        if not reboot_allowed:
            # This used to return with the ping still withheld whenever reboot
            # was disabled: systemd then restarted the process every
            # WatchdogSec, for ever (StartLimitBurst never trips at that rate).
            log.critical(
                "%s is down and unrecoverable; reboot disabled (health.allow_reboot) — "
                "staying up degraded and reporting it", component,
            )
            self._stay_up_degraded(component)
            return

        if state["count"] >= max_total > 0:
            log.critical(
                "reboot suppressed: %d reboot(s) already requested (health.reboot_max_total=%d) "
                "and '%s' is still down — a fault that survived those will not be fixed by "
                "another. Staying up degraded; see /api/health.",
                state["count"], max_total, component,
            )
            self._stay_up_degraded(component)
            return
        if state["last_at"] and now - state["last_at"] < min_interval:
            log.critical(
                "reboot suppressed: last one was %.0fs ago (<%.0fs)",
                now - state["last_at"], min_interval,
            )
            self._stay_up_degraded(component)
            return

        if not self._record_reboot_request(state, now):
            log.critical(
                "reboot ABORTED: could not persist the reboot record to %s. Without it "
                "the next boot cannot know this reboot happened, so the rate limit and "
                "the total bound would both reset and the device could power-cycle "
                "indefinitely. Staying up degraded instead ('%s' remains down).",
                self._REBOOT_STATE_PATH, component,
            )
            self._stay_up_degraded(component)
            return

        log.critical(
            "LAST RESORT: rebooting because '%s' is unrecoverable (reboot %d of %d)",
            component, state["count"] + 1, max_total,
        )
        if not self._issue_reboot():
            # Nothing rebooted (a missing sudoers rule, say), so the attempt must
            # not spend the bound: put the record back as it was.
            self._write_reboot_state(state)
            self._stay_up_degraded(component)

    def _stay_up_degraded(self, component: str) -> None:
        """Every "staying up degraded" decision above: stop withholding the watchdog
        ping for this outage, or systemd restarts the process regardless (see
        HealthMonitor.release_watchdog_hold). The monitoring state reports why."""
        health = getattr(self, "health", None)
        if health is not None:
            health.release_watchdog_hold(component)

    def _issue_reboot(self) -> bool:
        """Run the reboot command. False when it could not be issued."""
        try:
            result = subprocess.run(  # noqa: S603 - fixed argv; sudoers- or polkit-authorised (core/privilege.py)
                privileged(["systemctl", "reboot"]),
                check=False, timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            log.critical("reboot command could not run: %s — staying up degraded", exc)
            return False
        if result.returncode != 0:
            log.critical(
                "reboot command failed (exit %s) — check the sudoers rule from "
                "setup_pi.sh (or, under the hardened unit, the polkit rule); staying up degraded",
                result.returncode,
            )
            return False
        return True

    #: Persisted across reboots, which is what makes the total bound meaningful --
    #: systemd's own StartLimitBurst counter does NOT survive a reboot, so it can
    #: never bound a reboot campaign.
    _REBOOT_STATE_PATH = Path("data/.reboot_state.json")

    def _reboot_state(self) -> dict:
        """{"count": int, "last_at": float, "process_restarts": {component: time}}
        — empty values when absent or unreadable.

        An unreadable file reads as "no reboots yet", which is the permissive
        direction; that is safe ONLY because _record_reboot_request must succeed
        before any reboot proceeds, so an unwritable state file stops the reboot
        rather than uncapping it.
        """
        try:
            raw = json.loads(self._REBOOT_STATE_PATH.read_text())
            restarts = raw.get("process_restarts") or {}
            return {
                "count": int(raw.get("count", 0)),
                "last_at": float(raw.get("last_at", 0.0)),
                "process_restarts": {
                    str(k): float(v) for k, v in restarts.items()
                } if isinstance(restarts, dict) else {},
            }
        except (OSError, ValueError, TypeError, AttributeError):
            return {"count": 0, "last_at": 0.0, "process_restarts": {}}

    def _record_reboot_request(self, state: dict, now: float) -> bool:
        """Persist the incremented reboot record. False if it could not be written."""
        return self._write_reboot_state({**state, "count": state["count"] + 1, "last_at": now})

    def _write_reboot_state(self, payload: dict) -> bool:
        """Persist the recovery record. False if it could not be written.

        Atomic (tmp + fsync + replace) for the same reason core/monitoring_config.py
        is: a half-written record on a device that is about to power-cycle is worse
        than no record, and this file's whole job is to survive that power-cycle.
        """
        try:
            self._REBOOT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._REBOOT_STATE_PATH.with_suffix(".json.tmp")
            with open(tmp, "w") as handle:
                json.dump(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self._REBOOT_STATE_PATH)
            return True
        except OSError as exc:
            log.critical("could not persist reboot state: %s", exc)
            return False

    # ------------------------------------------------------------------ roles and protection
    def _custom_active(self) -> bool:
        return self.monitoring.active and not self._custom_rules_rejected

    def _resolve_mode(self, preset_module: Optional[str]) -> "monitoring_mode.MonitoringMode":
        """The mode for `preset_module`: custom rules are ADDITIVE when the role
        requires that module, and replace it (AI_CUSTOM) otherwise."""
        return monitoring_mode.resolve(
            monitoring_active=self._custom_active(),
            preset_module=preset_module,
            additive=self.role.additive_custom_rules and preset_module in self.role.mandatory_modules,
        )

    def _rule_problems(self, raw_rules, module, preset_module: Optional[str]) -> list[str]:
        """Why this device cannot run `raw_rules` (empty = it can), all or nothing.

        On an additive role the rules run beside the mandatory module, so they may
        only target what that module detects. Otherwise the capability registry
        answers, as the backend's own validation does.
        """
        if self.role.additive_custom_rules and preset_module in self.role.mandatory_modules:
            declared = module.available_labels() if hasattr(module, "available_labels") else None
            labels = frozenset(declared or ())

            def label_ok(label: str) -> bool:
                return label in labels
        else:
            from core.capabilities import registry

            reg = registry()

            def label_ok(label: str) -> bool:
                return any(c.available for c in reg.providers_of_label(label))
        # Areas are not checked here: a rule may name an area the owner has not
        # drawn yet (the backend's own proposals do). It loads inert, the
        # monitoring state reports it as waiting for that area, and it runs once
        # the area arrives. Rejecting it lost it for good (M0.1 review): the
        # rules_version was already acknowledged, so it was never re-sent.
        return validate_rule_set(raw_rules, label_ok=label_ok)

    def _record_rejection(self, kind: str, subject: Optional[str], reasons: list[str],
                          code: Optional[str] = None) -> None:
        self._last_rejection = {
            "kind": kind, "subject": subject or "", "reasons": list(reasons), "at": time.time(),
        }
        if code:
            self._last_rejection["code"] = code
        if getattr(self, "dashboard", None) is not None:
            setter = getattr(self.dashboard, "set_policy_state", None)
            if callable(setter):
                setter(self._policy_state())
        if getattr(self, "uplink", None) is not None:
            self._publish_monitoring()

    # ------------------------------------------------------------------ shared perception (M7)
    @property
    def perception(self) -> pc.PerceptionEngine:
        """The shared perception engine (one per runtime; created on first use for a
        runtime assembled without __init__, as several tests build them)."""
        engine = self.__dict__.get("_perception_engine")
        if engine is None:
            engine = self.__dict__["_perception_engine"] = pc.PerceptionEngine()
        return engine

    def _shared_detector_weights(self) -> Optional[str]:
        """The detector a model-less consumer runs: perception.detector_weights, else
        the running module's, else any configured module's (they share one model)."""
        cfg = self.config.get("perception") if isinstance(self.config.get("perception"), dict) else {}
        if cfg.get("detector_weights"):
            return str(cfg["detector_weights"])
        module_cfg = self._active_module_config()
        weights = next((module_cfg[k] for k in self._WEIGHTS_KEYS if module_cfg.get(k)), None)
        if weights:
            return str(weights)
        modules = self.config.get("modules", {}) if isinstance(self.config.get("modules"), dict) else {}
        for name in ("shared_perception", "driver_attention", "occupancy"):
            section = modules.get(name) if isinstance(modules.get(name), dict) else {}
            if section.get("yolo_weights"):
                return str(section["yolo_weights"])
        return None

    def _load_shared_detector(self):
        """PerceptionEngine's detector loader (a node beside a detector-less module)."""
        from core import detectors

        weights = self._shared_detector_weights()
        if not weights:
            raise RuntimeError("no object-detector weights are configured (perception.detector_weights)")
        return detectors.get_yolo_model(weights)

    def _sync_perception_facts(self) -> None:
        """What the runtime knows about the detector that the engine cannot see:
        a backend that is missing altogether, and which exact model loaded."""
        reason = (f"object detection unavailable: {self._detection_degraded}"
                  if self._detection_degraded is not None else None)
        if reason != self._detector_unavailable:
            self.perception.set_unavailable(pc.OBJECT_DETECTOR, reason)
            self._detector_unavailable = reason
        detector = (self._model_identity or {}).get("detector")
        if detector:
            self.perception.set_model(pc.OBJECT_DETECTOR, detector)

    def _perceive(self, module, frame, frame_id: int, captured_at: Optional[float]):
        """One analysed frame through the shared perception engine: (events, Observation).

        Model-less consumers first: whatever a node asked for and is due runs,
        and the module's own model calls on this frame are then answered from
        the same results (core/perception.py). The frame always ends, also when
        the module raises, so no half-open frame is left behind.
        """
        engine = self.perception
        self._sync_perception_facts()
        engine.begin_frame(frame, frame_id=frame_id, captured_at=captured_at,
                           source=self._perception_source, simulated=self._perception_simulated)
        try:
            with engine.activated():
                engine.run_due(frame)
                with engine.consuming(self.module_name):
                    events = module.process(frame)
        except Exception:
            self._last_observation = engine.end_frame()
            raise
        observation = engine.end_frame(self._module_signals(module))
        self._last_observation = observation
        return events, observation

    @staticmethod
    def _module_signals(module) -> Optional[dict]:
        hook = getattr(module, "perception_signals", None)
        if not callable(hook):
            return None
        try:
            signals = hook()
        except Exception:  # a broken hook loses the signals, never the frame
            log.debug("perception_signals() failed", exc_info=True)
            return None
        return signals if isinstance(signals, dict) else None

    def _rules_detection_frame(self, module, observation: Optional[pc.Observation]):
        """The module's per-frame view for custom rules, only when it is current.

        A module that runs the shared detector on a stride used to hand rules
        the SAME boxes on every frame between detector runs, and its last boxes
        after a failure. Now rules observe a frame only when the detector ran
        on it for this module, and nothing while detection is failed, stale or
        unavailable (the rule engine holds its state; health says why).
        """
        if observation is None:
            return module.get_detections()
        if pc.OBJECT_DETECTOR not in self.perception.consumed_by(self.module_name):
            return module.get_detections()  # no shared detector behind its view (e.g. pose labels)
        status = observation.components.get(pc.OBJECT_DETECTOR)
        if status is not None and status.state in (pc.FAILED, pc.UNAVAILABLE, pc.STALE):
            return None
        if self.module_name not in observation.detection_consumers:
            return None  # nothing new for this module since its last detector run
        # The module's boxes ARE this run's boxes: give them the shared tracker's
        # ids, so per-subject rules (dwell) follow the device's one tracker (M8).
        return observation.attach_tracks(module.get_detections())

    # ------------------------------------------------------------------ monitoring nodes (M7)
    def _node_zones(self) -> list:
        return list(self._effective_zones(self.module_name))

    def _active_companions(self) -> list:
        """Companion nodes running now: the preset's own, then the owner's (deduplicated).

        When custom rules REPLACE the preset (AI_CUSTOM, M8), the preset's own
        nodes are part of what was replaced and stop with it; the owner's
        companions are additive and keep running."""
        nodes = {} if self._preset_replaced() else dict(self._preset_nodes)
        for node_id, node in self._companions.items():
            nodes.setdefault(node_id, node)
        return list(nodes.values())

    def _preset_replaced(self) -> bool:
        return getattr(self, "mode", None) is monitoring_mode.MonitoringMode.AI_CUSTOM

    def _sync_preset_node_demands(self) -> None:
        """Withdraw the preset's own nodes' detector demands while custom rules
        replace the preset, and restore them when it is back."""
        replaced = self._preset_replaced()
        if replaced == self._preset_nodes_suspended:
            return
        self._preset_nodes_suspended = replaced
        for node in self._preset_nodes.values():
            if node.id in self._companions:
                continue  # the owner's own copy keeps the (shared-key) demand
            if replaced:
                self.perception.withdraw(node.id)
            else:
                for demand in node.demands():
                    self.perception.demand(demand)

    def _register_node(self, node) -> None:
        node.set_zones(self._node_zones())
        for demand in node.demands():
            self.perception.demand(demand)

    def _retire_node(self, node) -> None:
        self.perception.withdraw(node.id)
        try:
            node.close()
        except Exception:
            log.debug("closing node %s failed", node.id, exc_info=True)
        # A node id the other set still runs keeps its demands.
        survivor = self._preset_nodes.get(node.id) or self._companions.get(node.id)
        if survivor is not None and survivor is not node:
            for demand in survivor.demands():
                self.perception.demand(demand)

    def _set_preset_nodes(self, preset_name: Optional[str]) -> None:
        """Build the companion nodes the active preset owns (presets.yaml `nodes:`)."""
        base = str(preset_name or "").partition("@")[0].partition(":")[0]
        wanted = {}
        for raw in ((self.presets.get(base) or {}).get("nodes") or []) if base else []:
            if isinstance(raw, dict) and raw.get("node"):
                wanted[str(raw["node"])] = raw.get("settings") or {}
        for node_id in [n for n in self._preset_nodes if n not in wanted]:
            node = self._preset_nodes.pop(node_id)
            self._retire_node(node)
        for node_id, settings in wanted.items():
            if node_id in self._preset_nodes:
                continue
            try:
                node = monitoring_nodes.build_companion(node_id, settings)
            except (KeyError, ValueError) as exc:
                log.error("[NODES] preset '%s' names node '%s', which cannot run: %s", base, node_id, exc)
                self._record_rejection("nodes", node_id, [f"preset {base}: {exc}"], code="node_unavailable")
                continue
            self._preset_nodes[node_id] = node
            self._register_node(node)
        if self._preset_nodes_suspended:  # kept nodes had their demands withdrawn: restore them
            for node in self._preset_nodes.values():
                for demand in node.demands():
                    self.perception.demand(demand)
        self._preset_nodes_suspended = False  # every demand registered: the next frame re-checks the mode

    @staticmethod
    def _node_items(raw) -> Optional[list[tuple[Any, Any]]]:
        """(node id, {enabled, settings}) pairs from either command shape:
        {node_id: {enabled, settings}} (the backend's) or [{node, enabled, settings}]."""
        if isinstance(raw, dict):
            return list(raw.items())
        if isinstance(raw, list):
            return [(item.get("node") if isinstance(item, dict) else None, item) for item in raw]
        return None

    def _node_problems(self, raw) -> tuple[list[tuple[str, str, str]], dict, set]:
        """Why a set_nodes command cannot be applied (all or nothing): a list of
        (node id, code, reason); the enabled nodes with validated settings; and
        the nodes it explicitly disables."""
        items = self._node_items(raw)
        if items is None:
            return [("", "malformed", "nodes must map node id -> {enabled, settings}")], {}, set()
        problems: list[tuple[str, str, str]] = []
        wanted: dict = {}
        disabled: set[str] = set()
        seen: set[str] = set()
        for node_id, item in items:
            if not isinstance(node_id, str) or not node_id:
                problems.append(("", "malformed", "a node entry needs a node id"))
                continue
            if node_id in seen:
                problems.append((node_id, "malformed", f"'{node_id}' is listed twice"))
                continue
            seen.add(node_id)
            if not isinstance(item, dict):
                problems.append((node_id, "malformed", f"'{node_id}' needs {{enabled, settings}}"))
                continue
            spec = monitoring_nodes.companion_spec(node_id)
            if spec is None:
                known = ", ".join(sorted(monitoring_nodes.COMPANIONS))
                problems.append((node_id, "unknown_node",
                                 f"unknown monitoring node '{node_id}' (this device runs: {known})"))
                continue
            enabled = item.get("enabled", True)
            if not isinstance(enabled, bool):
                problems.append((node_id, "malformed", f"'{node_id}': enabled must be true or false"))
                continue
            if not enabled:
                disabled.add(node_id)
                continue
            try:
                settings = monitoring_nodes.validate_node_settings(spec, item.get("settings"))
            except ValueError as exc:
                problems.append((node_id, "invalid_setting", str(getattr(exc, "reason", exc))))
                continue
            if "objects" in spec.required_capabilities:
                missing = self._object_detection_missing()
                if missing:
                    problems.append((node_id, "capability_unavailable",
                                     f"'{node_id}' needs object detection, which {missing}"))
                    continue
            wanted[node_id] = settings
        running = (set(self._companions) - disabled) | set(wanted)
        if len(running) > monitoring_nodes.MAX_COMPANIONS:
            problems.append((",".join(sorted(running)), "too_many_nodes",
                             f"at most {monitoring_nodes.MAX_COMPANIONS} companion nodes can run at once"))
        return problems, wanted, disabled

    def _object_detection_missing(self) -> Optional[str]:
        """Why object detection cannot run on this device at all (None = it can)."""
        if self._detection_degraded is not None:
            return f"is unavailable on this device ({self._detection_degraded})"
        status = self.perception.component(pc.OBJECT_DETECTOR)
        if status.state == pc.UNAVAILABLE:
            return f"is unavailable on this device ({status.reason})"
        if not self._shared_detector_weights() and self.perception.detector() is None:
            return "has no detector configured on this device"
        return None

    def _apply_nodes(self, raw, source: str) -> None:
        """A set_nodes command: the wanted state of each named companion node.

        Idempotent: a node already running with the same settings is left alone
        (its counts and timers carry on), so a re-delivered command has no
        second effect; nodes the command does not name are left as they are.
        Refused all-or-nothing, reported as last_rejection {kind: "nodes",
        subject: <node id>, reasons, code}.
        """
        problems, wanted, disabled = self._node_problems(raw)
        if problems:
            subject, code, _ = problems[0]
            self._record_rejection("nodes", subject, [reason for _, _, reason in problems], code=code)
            log.warning("[NODES] %s refused: %s", source, "; ".join(r for _, _, r in problems))
            return
        changed = False
        for node_id in [n for n in self._companions if n in disabled]:
            self._retire_node(self._companions.pop(node_id))
            changed = True
        for node_id, settings in wanted.items():
            current = self._companions.get(node_id)
            if current is not None and current.settings == settings:
                continue
            if current is not None:
                self._retire_node(self._companions.pop(node_id))
            node = monitoring_nodes.build_companion(node_id, settings)
            self._companions[node_id] = node
            self._register_node(node)
            changed = True
        if changed:
            device_state.update(nodes={n.id: {"enabled": True, "settings": dict(n.settings)}
                                       for n in self._companions.values()})
            log.info("[NODES] %s: running %s", source, ", ".join(self._companions) or "no companion nodes")
            if getattr(self, "uplink", None) is not None:
                self._publish_monitoring()

    def _restore_nodes(self) -> None:
        saved = device_state.load().get("nodes")
        if saved:
            self._apply_nodes(saved, source="restore")

    def _run_nodes(self, observation: pc.Observation, now: float) -> None:
        self._sync_preset_node_demands()
        nodes = self._active_companions()
        if not nodes:
            return
        zones = self._node_zones()
        signature = tuple((z.name, tuple(z.polygon), z.kind) for z in zones)
        if signature != self._node_zone_signature:
            for node in nodes:
                node.set_zones(zones)
            self._node_zone_signature = signature
        for node in nodes:
            try:
                produced = node.consume(observation, now)
            except Exception as exc:  # one node must never stop the others or the module
                if self._node_errors.get(node.id) is None:
                    log.exception("[NODES] %s failed on a frame; it keeps running", node.id)
                self._node_errors[node.id] = str(exc) or type(exc).__name__
                continue
            self._node_errors.pop(node.id, None)
            for item in produced:
                self._emit_node_event(node, item)

    def _emit_node_event(self, node, item) -> None:
        """A node's event: logged, shown, and sent to the app like any other.

        Not routed to the local alert outputs: a companion's count or area
        event must not sound the driver's buzzer, which is reserved for the
        safeguards the preset itself raises.
        """
        event = item.event
        if self.logger is not None:
            self.logger.log(event)
        if self.dashboard is not None:
            self.dashboard.push_event(event, item.severity)
        if item.severity != "log":
            self._flash_until = time.monotonic() + _FLASH_DURATION_S
        if self.uplink is not None and item.notify != "silent":
            self.uplink.push_event(event, item.severity, source=f"node:{node.id}",
                                   model=self._model_identity, notify=item.notify)

    def _node_entries(self, capabilities) -> list[dict]:
        """`perception.nodes` (sd-perception/1): every node running now, with its state."""
        stopped = self._stopped is not None
        entries: list[dict] = []
        base = str(getattr(self, "active_preset", None) or "").partition("@")[0].partition(":")[0]
        preset = self.presets.get(base) if base else None
        if preset is not None:
            spec = monitoring_nodes.preset_spec(base, preset)
            state, missing, reason = monitoring_nodes.node_state(spec, capabilities, stopped=stopped)
            if self.mode is monitoring_mode.MonitoringMode.AI_CUSTOM and not stopped:
                state, missing, reason = monitoring_nodes.INACTIVE, [], "custom rules replace this mode"
            features = monitoring_nodes.preset_experimental_features(base, preset, self._active_module_config())
            entries.append(monitoring_nodes.wire_entry(
                spec, state=state, missing=missing, reason=reason,
                validation=monitoring_nodes.validation_tier(base), experimental_features=features))
        if self._custom_active():
            spec = monitoring_nodes.RULES_SPEC
            state, missing, reason = monitoring_nodes.node_state(spec, capabilities, stopped=stopped)
            entries.append(monitoring_nodes.wire_entry(
                spec, state=state, missing=missing, reason=reason,
                validation=monitoring_nodes.validation_tier("rules"),
                metrics={"rules": len(self.rule_engine.rules)}))
        for node in self._active_companions():
            state, missing, reason = monitoring_nodes.node_state(node.spec, capabilities, stopped=stopped)
            error = self._node_errors.get(node.id) or getattr(node, "degraded_reason", None)
            if error and state == monitoring_nodes.ACTIVE:
                state, reason = monitoring_nodes.DEGRADED, error
            entry = monitoring_nodes.wire_entry(
                node.spec, state=state, missing=missing, reason=reason,
                validation=monitoring_nodes.validation_tier(node.id), metrics=node.metrics(),
                settings=node.settings)
            entry["enabled_by"] = "preset" if node.id in self._preset_nodes else "owner"
            entries.append(entry)
        if self._preset_replaced():
            running = {e["id"] for e in entries}
            for node in self._preset_nodes.values():
                if node.id in running:
                    continue
                entry = monitoring_nodes.wire_entry(
                    node.spec, state=monitoring_nodes.INACTIVE, missing=[],
                    reason="custom rules replace this mode", validation=monitoring_nodes.validation_tier(node.id),
                    settings=node.settings)
                entry["enabled_by"] = "preset"
                entries.append(entry)
        return entries

    def perception_report(self, capabilities=None) -> dict:
        """`monitoring.perception` exactly as the wire contract describes it."""
        if capabilities is None:
            capabilities = self._evaluate_monitoring().capabilities
        wire = self.perception.to_wire(simulated=self._perception_simulated,
                                       nodes=self._node_entries(capabilities))
        if getattr(self, "_stopped", None) is not None:
            # Stopped on purpose (M8.2): a component that is only stale or idle
            # because nothing runs is reported "paused" with the reason; failed
            # and unavailable components keep their own state (real faults).
            for status in (wire.get("components") or {}).values():
                if isinstance(status, dict) and status.get("state") in (pc.OK, pc.STALE, pc.IDLE):
                    status["state"], status["reason"] = "paused", "monitoring is stopped"
        return wire

    # ------------------------------------------------------------------ monitoring state (M0.2)
    def _module_runs_without_detector(self, module=None) -> bool:
        """Can the module still raise some event with object detection down?"""
        module = getattr(self, "module", None) if module is None else module
        hook: Any = getattr(module, "event_requirements", dict)
        reqs: dict[str, frozenset[str]] = hook()
        return any("objects" not in needs for needs in reqs.values())

    def _camera_capability(self, now: float) -> ms.Capability:
        bus: Optional[FrameBus] = getattr(self, "bus", None)
        if bus is None:
            return ms.Capability(ms.IMPAIRED, "camera pipeline not started")
        age = bus.age_s()
        if age == float("inf"):
            if now < self._camera_grace_until:
                return ms.Capability(ms.STARTING, "waiting for the first frame")
            why = f": {self._camera_error}" if self._camera_error else ""
            return ms.Capability(ms.IMPAIRED, f"no frame has been captured{why}")
        capture = getattr(self, "capture", None)
        if capture is not None and not capture.is_running:
            return ms.Capability(ms.IMPAIRED, "capture service stopped")
        if age > self._camera_stale_after_s:
            return ms.Capability(ms.IMPAIRED, f"no new frame for {age:.1f}s")
        return ms.Capability()

    def _inference_capability(self, now: float, camera_ok: bool) -> ms.Capability:
        swap_started = getattr(self, "_swap_started", None)
        if swap_started is not None:
            busy = now - swap_started
            if busy < _MAX_SWAP_S:
                return ms.Capability(ms.STARTING, f"switching mode ({busy:.0f}s)")
            return ms.Capability(ms.IMPAIRED, f"mode switch stuck for {busy:.0f}s")
        if self._heartbeat == 0.0:
            if now - self._heartbeat_reset_at < _INFERENCE_STARTING_GRACE_S:
                return ms.Capability(ms.STARTING, "inference loop starting")
            return ms.Capability(ms.IMPAIRED, "inference loop has not started")
        loop_age = now - self._heartbeat
        if loop_age >= 10.0:
            return ms.Capability(ms.IMPAIRED, f"inference loop unresponsive for {loop_age:.0f}s")
        if self._consecutive_process_failures:
            return ms.Capability(
                ms.IMPAIRED, f"analysis failing ({self._consecutive_process_failures} frame(s) in a row): "
                             f"{self._last_process_error}")
        if getattr(self, "_stopped", None) is not None:
            # The loop is alive (checked above) but analyses nothing by design.
            return ms.Capability(ms.PAUSED, "paused while monitoring is stopped")
        analysing = self._detection_degraded is None or self._module_runs_without_detector()
        bus: Optional[FrameBus] = getattr(self, "bus", None)
        if camera_ok and analysing and bus is not None:
            processed = bus.processed_age_s()
            if processed is not None and now - processed < self._ready_grace_from:
                processed = None  # nothing analysed since this (re)start or resume
            if processed is None:
                if now - self._ready_grace_from < _INFERENCE_STARTING_GRACE_S:
                    return ms.Capability(ms.STARTING, "no frame analysed yet")
                return ms.Capability(ms.IMPAIRED, "no frame has been analysed")
            if processed > max(5.0, self._camera_stale_after_s):
                return ms.Capability(ms.IMPAIRED, f"no frame analysed for {processed:.0f}s")
        return ms.Capability()

    _PERCEPTION_KINDS = ("objects", "face", "hands", "pose")

    def _module_capabilities(self, module) -> dict[str, ms.Capability]:
        """The perception capabilities the running module and nodes use, as the
        shared engine observed them (core/perception.py), not as declared.

        A module's own `capability_status()` says WHICH capabilities it needs
        and may know more (a detector in backoff); whether each model actually
        produced results recently is the engine's to say. A model that has not
        run yet is STARTING (bounded by the readiness grace), never "available".
        """
        hook: Any = getattr(module, "capability_status", dict)
        try:
            declared = dict(hook())
        except Exception as exc:  # noqa: BLE001 - a broken report is itself a fact to report
            declared = {"module": ms.Capability(ms.IMPAIRED, f"capability report failed: {exc}")}
        caps = {k: v for k, v in declared.items() if k not in self._PERCEPTION_KINDS}
        expected = {k for k in declared if k in self._PERCEPTION_KINDS}
        expected |= {pc.CAPABILITY_OF[c] for c in self.perception.consumed_by(self.module_name)}
        for node in self._active_companions():
            expected |= {k for k in (*node.spec.required_capabilities, *node.spec.optional_capabilities)
                         if k in self._PERCEPTION_KINDS}
        if not expected and not declared:
            # A module this software does not describe, that has not touched a
            # shared model: judged only by whether it processes frames (camera,
            # inference), which is all that can honestly be said about it.
            caps = {kind: ms.Capability() for kind in self._PERCEPTION_KINDS}
        status = self.perception.status()
        now = time.monotonic()
        for kind in sorted(expected):
            caps[kind] = self._perception_capability(status[pc.COMPONENT_OF[kind]], declared.get(kind), now)
        if self._detection_degraded is not None:
            caps["objects"] = ms.Capability(ms.IMPAIRED, f"object detection unavailable: {self._detection_degraded}")
        return caps

    def _perception_capability(self, status: pc.ComponentStatus, declared: Optional[ms.Capability],
                               now: float) -> ms.Capability:
        label = status.name.replace("_", " ")
        if declared is not None and declared.state == ms.IMPAIRED:
            return declared  # the module knows more (its own backoff, a refused load)
        if status.state in (pc.FAILED, pc.UNAVAILABLE):
            return ms.Capability(ms.IMPAIRED, f"{label} {status.state}: {status.reason}")
        if getattr(self, "_stopped", None) is not None:
            # Stopped on purpose: no result is expected, so a stale or missing
            # one is not a fault (M8.2). Real faults are reported above.
            return ms.Capability(ms.PAUSED, f"{label}: paused while monitoring is stopped")
        restarted_at = self._ready_grace_from
        if (status.last_run_age_s is not None and now - status.last_run_age_s < restarted_at
                and now - restarted_at <= self.perception.max_age_s):
            # Monitoring just (re)started (boot, resume, mode switch): a result
            # from before that does not count, and the model has one max_age_s
            # to produce a fresh one before it is judged stale. Meanwhile the
            # inference capability says whether anything has been analysed.
            return declared if declared is not None else ms.Capability(
                ms.STARTING, f"{label}: waiting for its first result since monitoring started")
        if status.state == pc.OK:
            return ms.Capability()
        if status.state == pc.STALE and status.last_run_age_s is not None:
            return ms.Capability(ms.IMPAIRED, f"{label} stale: {status.reason}")
        if status.last_run_age_s is not None:
            # It has produced results before but not lately (e.g. right after a
            # stop and resume): an old result, not a cold start.
            return ms.Capability(ms.IMPAIRED, f"{label} stale: no result for {status.last_run_age_s:.1f}s")
        # Never produced a result yet (idle, or demanded but not run).
        if declared is not None and declared.state == ms.STARTING and declared.reason:
            return declared
        return ms.Capability(ms.STARTING, f"{label}: waiting for its first result")

    def _capabilities(self, now: float, module) -> dict[str, ms.Capability]:
        caps = {"camera": self._camera_capability(now)}
        caps["inference"] = self._inference_capability(now, caps["camera"].available)
        caps.update(self._module_capabilities(module))
        if getattr(self, "_stopped", None) is not None:
            # Nothing starts while stopped (e.g. the driver calibration waits for
            # frames): paused, not "not ready after 30 s" (M8.2).
            return {name: ms.Capability(ms.PAUSED, f"{name}: paused while monitoring is stopped")
                    if cap.state == ms.STARTING else cap for name, cap in caps.items()}
        if now - self._ready_grace_from >= _INFERENCE_STARTING_GRACE_S:
            # Bounded: a capability "starting" past the grace never came up.
            caps = {
                name: ms.Capability(ms.IMPAIRED, f"not ready after {_INFERENCE_STARTING_GRACE_S:.0f}s"
                                                 f" ({cap.reason or 'starting'})")
                if cap.state == ms.STARTING and name != "inference" else cap
                for name, cap in caps.items()
            }
        return caps

    _LABEL_KINDS = frozenset({"objects", "pose", "face", "hands"})

    def _label_needs(self, label: str) -> frozenset[str]:
        """The perception capability a custom rule's label comes from."""
        if self.role.additive_custom_rules and self.module_name in self.role.mandatory_modules:
            return frozenset({"objects"})  # additive rules read the module's own detections
        from core.capabilities import registry

        kinds = {c.kind for c in registry().providers_of_label(label) if c.available} & self._LABEL_KINDS
        return frozenset(sorted(kinds)[:1]) if kinds else frozenset({"objects"})

    def _active_policies(self, module) -> list[ms.Policy]:
        policies: list[ms.Policy] = []
        if self.mode is not monitoring_mode.MonitoringMode.AI_CUSTOM:
            hook: Any = getattr(module, "event_requirements", dict)
            reqs: dict[str, frozenset[str]] = dict(hook())
            if not reqs:
                reqs = {self.module_name: frozenset({"objects"}) if self._module_uses_a_detector() else frozenset()}
            name = self.active_preset or self.module_name
            policies.append(ms.Policy(
                id=f"preset:{name}", kind="preset", name=name,
                events={event: ms.BASE_REQUIREMENTS | needs for event, needs in reqs.items()},
            ))
        if self._custom_active():
            from core.rules import _zone_key

            waiting = {_zone_key(z) for z in self.rule_engine.pending_zones()}
            for rule in self.rule_engine.rules:
                needs = self._label_needs(rule.detection)
                if rule.secondary:
                    needs |= self._label_needs(rule.secondary)
                blocked = ({rule.rule_id: f"area '{rule.zone}' is not drawn yet"}
                           if rule.zone and _zone_key(rule.zone) in waiting else {})
                policies.append(ms.Policy(
                    id=f"rule:{rule.rule_id}", kind="rule", name=rule.description or rule.rule_id,
                    events={rule.rule_id: ms.BASE_REQUIREMENTS | needs}, blocked=blocked,
                ))
        for node in self._active_companions():
            policies.append(ms.Policy(
                id=f"node:{node.id}", kind="node", name=node.spec.title,
                events={event: ms.BASE_REQUIREMENTS | needs for event, needs in node.spec.events.items()},
            ))
        return policies

    def _evaluate_monitoring(self) -> ms.Evaluation:
        """The device's monitoring state from what is observed right now."""
        now = time.monotonic()
        module = getattr(self, "module", None)
        stopped = None if self._stopped is None else f"monitoring stopped by {self._stopped['by']}"
        return ms.evaluate(
            stopped=stopped,
            capabilities=self._capabilities(now, module),
            policies=self._active_policies(module),
            role_required=self.role.required_capabilities,
            actuators=self.alert_manager.actuators() if hasattr(self.alert_manager, "actuators") else (),
        )

    def monitoring_report(self) -> ms.MonitoringReport:
        """The public, typed monitoring state (dashboard /api/health, uplink, app)."""
        evaluation = self._evaluate_monitoring()
        bus: Optional[FrameBus] = getattr(self, "bus", None)
        frame_age = None if bus is None or bus.age_s() == float("inf") else round(bus.age_s(), 2)
        processed = None if bus is None else bus.processed_age_s()
        identity = self._model_identity or {}
        return ms.MonitoringReport(
            state=evaluation.state,
            reasons=evaluation.reasons,
            role=self.role.name,
            active_preset=getattr(self, "active_preset", None),
            module=getattr(self, "module_name", ""),
            mode=self.mode.value,
            custom_rules=tuple(r.rule_id for r in self.rule_engine.rules) if self._custom_active() else (),
            capabilities=evaluation.capabilities,
            policies=evaluation.policies,
            actuators=tuple(self.alert_manager.actuators()) if hasattr(self.alert_manager, "actuators") else (),
            last_frame_age_s=frame_age,
            last_inference_age_s=None if processed is None else round(processed, 2),
            model={"detector": identity.get("detector"), "status": accel.live_status()[1]},
            versions={"software": _VERSION, "model": self._model_version()},
            perception=self.perception_report(evaluation.capabilities),
        )

    def _monitoring_check(self) -> tuple[bool, str]:
        """Health-monitor tick: evaluate, act on transitions, drive the impaired alert."""
        report = self.monitoring_report()
        state = report.state
        now = time.monotonic()
        previous, self._monitoring_state = self._monitoring_state, state.value
        if previous != state.value:
            detail = "; ".join(report.reasons) or "all monitoring policies operational"
            (log.warning if state in (ms.MonitoringState.FAILED, ms.MonitoringState.DEGRADED) else log.info)(
                "[MONITORING] %s -> %s: %s", previous or "boot", state.value, detail)
            self._log_system_event(f"monitoring_{state.value}", detail)
            if state in (ms.MonitoringState.FAILED, ms.MonitoringState.STOPPED):
                # What the module and rules last believed is stale now: end
                # their episodes. A condition still live after recovery re-opens
                # (_reopen_live_episodes), so nothing is lost and nothing doubled.
                self.alert_manager.end_source(now, "module")
                self.alert_manager.end_source(now, "rules")
            self._publish_monitoring()
        self._signal_impaired(state is ms.MonitoringState.FAILED, report, now)
        return state in (ms.MonitoringState.READY, ms.MonitoringState.STARTING, ms.MonitoringState.STOPPED), (
            f"{state.value}: {'; '.join(report.reasons[:3])}" if report.reasons else state.value)

    _IMPAIRED_KEY = "monitoring_impaired"

    def _signal_impaired(self, impaired: bool, report: ms.MonitoringReport, now: float) -> None:
        """While FAILED (not stopped, not starting), signal "cannot monitor" on any
        output that still works, with the bounded impaired pattern (every 15 s, at
        most 20 repeats, then silent until recovery). DEGRADED is reported, not
        sounded: an unexplained chirp for a lost phone detector would teach a
        driver to ignore the buzzer."""
        if impaired and self._IMPAIRED_KEY not in self.alert_manager.open_keys("health"):
            event = Event.now(self._IMPAIRED_KEY, 0.0, None, "system", "; ".join(report.reasons))
            self.alert_manager.notify(event, now=now, key=self._IMPAIRED_KEY, category="impaired", source="health")
            if self.uplink is not None:
                self.uplink.push_event(event, "loud", model=self._model_identity)
        self.alert_manager.observe(now, source="health", active={self._IMPAIRED_KEY} if impaired else (),
                                   tracked={self._IMPAIRED_KEY})

    def protection_state(self) -> dict:
        """The truthful answer to "is this device protecting right now?".

        stopped        an explicit, confirmed stop is in force
        protecting     every safeguard the role requires is running
        not_protecting something required is not running; `reasons` says what
        """
        if self._stopped is not None:
            return {"state": "stopped", "role": self.role.name, "since": self._stopped["since"],
                    "by": self._stopped["by"], "reasons": ["monitoring was stopped explicitly"],
                    "monitoring_state": ms.MonitoringState.STOPPED.value,
                    "last_command_id": self._last_protection_command}
        evaluation = self._evaluate_monitoring()
        reasons: list[str] = []
        if evaluation.state in (ms.MonitoringState.FAILED, ms.MonitoringState.STARTING):
            reasons += list(evaluation.reasons)
        if self.role.has_floor and self.module_name not in self.role.mandatory_modules:
            reasons.append(
                f"role '{self.role.name}' requires {', '.join(sorted(self.role.mandatory_modules))}; "
                f"running {self.module_name}"
            )
        degraded = list(evaluation.reasons) if evaluation.state is ms.MonitoringState.DEGRADED else []
        return {"state": "not_protecting" if reasons else "protecting", "role": self.role.name,
                "reasons": reasons, "degraded": degraded, "monitoring_state": evaluation.state.value,
                "last_command_id": self._last_protection_command}

    def _policy_state(self) -> dict:
        return {
            "role": self.role.describe(),
            "protection": self.protection_state(),
            "mode": self.mode.value,
            "last_rejection": self._last_rejection,
            # What this software understands beyond older firmware (preset
            # settings, dwell rules): the backend sends those only to a device
            # that reports them (core/presets.py DEVICE_POLICY_FEATURES).
            "features": list(DEVICE_POLICY_FEATURES),
        }

    def _publish_monitoring(self) -> None:
        """Tell the backend what runs, the role, the protection state and the last refusal."""
        if self.uplink is not None:
            self.uplink.set_monitoring({**self.monitoring.describe(), **self._policy_state(),
                                        "base_monitoring_disabled":
                                            self.mode is monitoring_mode.MonitoringMode.AI_CUSTOM,
                                        "state": self._monitoring_state,
                                        "perception": self.perception_report()})
            refresh = getattr(self.uplink, "refresh_health_monitoring", None)
            if callable(refresh):
                # Same moment, same facts: without this the 5 s health sample still
                # carried the previous state and the app saw a stop or a failure late.
                report = self.monitoring_report()
                refresh(report.to_dict(), report.monitoring, self.protection_state())
        if self.dashboard is not None:
            setter = getattr(self.dashboard, "set_policy_state", None)
            if callable(setter):
                setter(self._policy_state())

    def _apply_protection(self, request: dict, source: str) -> None:
        """Explicit stop / resume. A stop must be confirmed; it is never implied.

        A request carrying an id (the backend's) is echoed as last_command_id once
        handled -- applied or refused -- so the backend converges on THIS request.
        """
        action = str((request or {}).get("action", "")).lower()
        command_id = (request or {}).get("id")
        if command_id:
            self._last_protection_command = str(command_id)
        if action == "stop":
            if (request or {}).get("confirm") is not True:
                self._record_rejection("protection", "stop", ["stopping monitoring requires confirm: true"])
                return
            if self._stopped is None:
                self._stopped = {"since": time.time(), "by": source}
                self.alert_manager.end_all(time.monotonic())
                log.warning("[PROTECTION] STOPPED by %s: nothing is monitored until resumed", source)
                self._log_system_event("protection_stopped", f"monitoring stopped by {source}")
        elif action == "resume":
            if self._stopped is not None:
                self._stopped = None
                self._ready_grace_from = time.monotonic()  # STARTING until frames are analysed again
                # Re-arm custom rules: a rule still true at resume re-confirms and
                # alerts, instead of EventEngine treating it as already reported.
                # Module conditions re-open through _reopen_live_episodes.
                if self._custom_active():
                    self.rule_engine.replace_rules(self.monitoring.rules)
                log.info("[PROTECTION] resumed by %s", source)
                self._log_system_event("protection_resumed", f"monitoring resumed by {source}")
        else:
            self._record_rejection("protection", action, [f"unknown protection action {action!r}"])
            return
        self._publish_monitoring()

    def _log_system_event(self, kind: str, detail: str) -> None:
        event = Event.now(kind, 0.0, None, "system", detail)
        if self.logger is not None:
            self.logger.log(event)
        if self.dashboard is not None:
            self.dashboard.push_event(event, "log")

    # ------------------------------------------------------------------ preset switching
    def _exit_custom_monitoring(self) -> None:
        """Someone explicitly picked a mode: that always wins over custom rules.

        Zones are reset to the effective zones for the running module: the
        app's, when it has sent any (they are the device's zones, not the custom
        configuration's — a crib drawn in the app must keep protecting the baby
        preset), otherwise the module's own config.yaml zones.

        Before this, `_switch_preset` hot-swapped the running module but left
        `self.monitoring` (the saved rules + `overrides_mode`) untouched. That
        config drives `_apply_monitoring_mode()`, which runs on every rules/zones
        poll — every few seconds, via the uplink. So a preset switch away from AI
        Custom Monitoring would visibly take, then get silently reverted back to
        AI_CUSTOM/occupancy on the next poll, because `self.monitoring.active`
        had never stopped being true. Picking "Driver Safety" is not a request
        to keep evaluating custom rules on top of it.

        A no-op when custom monitoring was not active, so a normal
        preset-to-preset switch does not write the monitoring file or push an
        extra state update for nothing.
        """
        if not self.monitoring.active:
            return
        self.monitoring.clear()
        self._custom_rules_rejected = False
        self.rule_engine.replace_rules([])
        self.rule_engine.set_zones(self._effective_zones(self.module_name))
        if self.uplink is not None:
            self._publish_monitoring()
        log.info("[MONITORING] custom monitoring exited: explicit mode switch")

    def _switch_preset(self, name: str) -> None:
        """Hot-swap the running module to a preset (optionally "preset:variant")."""
        with self._module_swap():
            self._switch_preset_now(name)

    def _switch_preset_now(self, name: str) -> None:
        # Translate superseded names first. A `set_preset` command can have been
        # sitting in the backend since before a mode was renamed, and the old
        # code path simply returned on an unknown preset — so the device would
        # ignore the instruction silently, for ever, with the app showing a mode
        # the device was never in.
        from main import resolve_preset

        name = resolve_preset(name, self.presets)
        # "" is not an unknown preset name — it is cloud/app.py's own sentinel
        # for "apply AI Custom" (POST .../preset with preset=""), which names no
        # module to hot-swap to at all: entering custom monitoring is driven
        # entirely by self.monitoring.active, set independently by the rules
        # (set_rules) delivery path and applied by _apply_monitoring_mode(),
        # not by this method. There is nothing for _switch_preset to DO with
        # it beyond what that path already does — so this is a real, named
        # no-op, not "ignoring unknown preset ''" (which reads as a dropped or
        # misdelivered command rather than the expected shape of this one).
        if name == "":
            return
        # Post-resolution no-op guard, independent of the caller's gate: a
        # rebuild of the module a device is already running costs a FaceMesh
        # teardown and every accumulated timer/window, which is never worth it
        # for a command that changes nothing.
        if name == self.active_preset and not self.monitoring.active:
            return
        violation = roles.preset_violation(self.role, name, self.presets)
        if violation:
            self._record_rejection("preset", name, [violation], code="role_forbids_preset")
            log.warning("[POLICY] preset '%s' refused: %s", name, violation)
            return
        base_name, _, variant_name = name.partition("@")[0].partition(":")
        preset = self.presets.get(base_name)
        if preset is None or self.profile == "mac_max":
            if preset is None:
                print(f"SensorDyme: ignoring unknown preset '{name}'")
                self._record_rejection("preset", name, [f"unknown preset '{name}' on this device"],
                                       code="unknown_preset")
            else:
                self._record_rejection("preset", name, ["the mac_max profile cannot switch presets"],
                                       code="profile_fixed")
            return
        # Variant overrides plus any owner settings ("preset:variant@key=value",
        # core/presets.py), validated against the preset's schema. A setting this
        # device cannot apply is refused with the reason, never half-applied.
        from core.presets import PresetError, preset_overrides

        try:
            name, _preset, overrides = preset_overrides(name, self.presets)
        except PresetError as exc:
            self._record_rejection("preset", name, [exc.reason], code="invalid_setting")
            print(f"SensorDyme: preset '{name}' refused: {exc.reason}")
            return
        new_module_name = preset["module"]
        new_module_config = {**self.config["modules"][new_module_name], **overrides, "recalibrate": False}
        try:
            new_module = self.perception.adopt(self._build_module(new_module_name, dict(new_module_config)))
        except Exception as exc:  # noqa: BLE001 - a bad preset must not kill the device
            print(f"SensorDyme: preset switch to '{name}' failed: {exc}")
            # Refused with the reason, so the backend stops re-sending a switch
            # this device cannot make and the app can say why (M7).
            self._record_rejection("preset", name, [f"the mode could not start on this device: {exc}"],
                                   code="module_failed")
            return
        self._push_zones_to(new_module)
        # Exit custom monitoring BEFORE swapping the module, not after: an
        # explicit preset choice always overrides whatever custom rules were
        # running, and the module below must land as STANDARD, never as a
        # STANDARD module with AI_CUSTOM still recorded as the active mode.
        try:
            self._exit_custom_monitoring()
        except Exception:
            # The new module never went live; release its graphs (a dropped
            # MediaPipe graph keeps its threads until close()) and give up.
            with suppress(Exception):
                new_module.close()
            raise
        with self._module_lock:
            old_module = self.module
            self.module, self.module_name, self.active_preset = new_module, new_module_name, name
            self._switched_module, self._switched_module_config = new_module, new_module_config
            self._model_identity = None  # unknown, not the old module's, until verified below
            self.alert_event_types = self.alert_types_for(new_module_name, preset)
            self._consecutive_process_failures = 0  # the new module starts with a clean slate
        # Same reason as the AI-custom swap: a preset switch changes which module
        # runs, and different modules need different backends. Verified once at
        # boot leaves a device that switches into a detector-needing preset
        # failing every frame while health still reports monitoring active.
        self._verify_detection_backend()
        # What the preset wants, remembered separately from what is running:
        # AI_CUSTOM swaps the module out and must be able to hand it back.
        self._preset_module = new_module_name
        # self.mode was previously left stale here — only ever set in __init__
        # and _apply_monitoring_mode(), never on a preset switch. It happened to
        # be harmless when nothing but the preset changed (mode was STANDARD
        # before and after), but AI_CUSTOM exiting through this path makes it
        # actively wrong if left alone: resolve it the same way the mode
        # decision is made everywhere else, so self.mode can never disagree
        # with self.module_name.
        self.mode = self._resolve_mode(new_module_name)
        # Re-point custom rules at the zones in force for the new module: the
        # app's when it has sent any, otherwise the new module's config.yaml.
        self.rule_engine.set_zones(self._effective_zones(new_module_name))
        self._set_preset_nodes(name)
        # Release the old FaceMesh/graph (it used to leak). Guarded: the switch
        # has already happened, and a close() that raised used to skip the
        # dashboard/uplink updates below, leaving the app showing the old mode.
        try:
            old_module.close()
        except Exception:  # teardown must never raise
            log.exception("closing the previous module failed; continuing")
        if overrides:
            print(f"SensorDyme: variant '{variant_name}' overrides applied: {overrides}")
        if self.dashboard is not None:
            self.dashboard.set_active_preset(name, new_module_name)
        if self.uplink is not None:
            self.uplink.set_active_preset(name)
        # The owner's choice survives a restart (see core/device_state.py).
        device_state.save_preset(name, self.config.get("module"))
        # A mode change is not a detection: no confidence (core/confidence.py).
        switch_event = Event.now("preset_changed", 0.0, None, "system", f"mode -> {name}")
        self.logger.log(switch_event)
        self._flash_until = time.monotonic() + _FLASH_DURATION_S
        if self.dashboard is not None:
            self.dashboard.push_event(switch_event, "log")
        print(f"SensorDyme: switched to preset '{name}' (module '{new_module_name}')")

    # ------------------------------------------------------------------ the inference loop
    def _apply_pending_commands(self) -> None:
        """Apply any mode, zone or rule change the app or dashboard has sent."""
        for producer, name in ((self.dashboard, "dashboard"), (self.uplink, "uplink")):
            pop = getattr(producer, "pop_pending_protection", None) if producer is not None else None
            request = pop() if callable(pop) else None
            if isinstance(request, dict):
                self._apply_protection(request, source=name)
        # `is not None`, not `or`: dashboard's own "" is cloud/app.py's sentinel
        # for "apply AI Custom" (see POST .../preset), a real, meaningful value
        # distinct from "nothing pending" (None) — an `or` chain treats an
        # empty-string result from the dashboard as falsy and silently falls
        # through to whatever the uplink has instead, discarding the actual
        # command.
        dashboard_pending = self.dashboard.pop_pending_preset() if self.dashboard else None
        pending = dashboard_pending if dashboard_pending is not None else (
            self.uplink.pop_pending_preset() if self.uplink else None
        )
        # Resolve superseded names BEFORE the equality gate, not only inside
        # _switch_preset. A stale "driver_distraction" command from an old
        # backend or app never string-equals the device's "driver_safety",
        # so the unresolved comparison passed every time and each delivery
        # rebuilt the running module — closing FaceMesh and zeroing every
        # timer, PERCLOS window and risk EMA mid-drive. With a backend old
        # enough to re-send the alias on every poll, that was a full module
        # teardown every few seconds: nothing needing sustained evidence
        # could ever fire.
        #
        # `is not None`, not truthy: the same "" sentinel above must reach
        # resolve_preset/_switch_preset rather than being silently dropped
        # here — see _switch_preset's own handling of an empty name.
        if pending is not None:
            from main import resolve_preset

            pending = resolve_preset(pending, self.presets)
        # `pending == self.active_preset` is not "nothing to do" while custom
        # monitoring is active: active_preset still names whichever preset was
        # running before custom rules took over (entering custom monitoring
        # never changes it), so a request to return to that exact preset has
        # real work left — exiting custom monitoring — that this equality
        # check would otherwise skip silently, with no error and no log line.
        if pending is not None and (pending != self.active_preset or self.monitoring.active):
            self._switch_preset(pending)

        # A new AI-configured rule set from the app replaces the live one.
        # `None` means no news; `[]` means the user deleted their rules and
        # enforcement must actually stop.
        if self.uplink is not None:
            # Zones first: a rule that names an area is inert until the area
            # exists, so applying rules before zones would leave a freshly
            # activated rule silently doing nothing for one cycle.
            new_zones = self.uplink.pop_pending_zones()
            if new_zones is not None:
                zones = _zones_from_app(new_zones)
                self._app_zones = zones
                device_state.save_zones(new_zones)
                self.rule_engine.set_zones(zones)
                # The running module's own zone alerts (crib exit, keep-out,
                # line) use them too — not only AI-configured rules.
                with self._module_lock:
                    module = self.module
                self._push_zones_to(module)
                print(f"zones: {len(zones)} area(s) active ({', '.join(z.name for z in zones) or 'none'})")

            pop_nodes = getattr(self.uplink, "pop_pending_nodes", None)
            new_nodes = pop_nodes() if callable(pop_nodes) else None
            if new_nodes is not None:
                self._apply_nodes(new_nodes, source="uplink")

            new_rules = self.uplink.pop_pending_rules()
            if new_rules is not None:
                problems = self._rule_problems(new_rules, self.module, self._preset_module or self.module_name)
                if problems:
                    # All or nothing: keep the rules already running and say why.
                    self._record_rejection("rules", self.uplink.rules_version(), problems)
                    log.warning("[POLICY] custom rules refused: %s", "; ".join(problems))
                    return
                self._custom_rules_rejected = False
                self.rule_engine.replace_rules(new_rules)
                # Written down as it arrives, so the next boot still has it.
                self.monitoring.save(
                    new_rules,
                    instruction=self.uplink.pending_instruction(),
                    version=self.uplink.rules_version(),
                )
                # A new configuration can change which detector is needed.
                self._apply_monitoring_mode()
                self._apply_requested_labels()
                # Reported after the mode is applied, so the report names the
                # mode actually running (ADDITIVE vs AI_CUSTOM), not the old one.
                self._publish_monitoring()
                pending_zones = self.rule_engine.pending_zones()
                print(
                    f"rules: enforcing {len(self.rule_engine.rules)} AI-configured rule(s)"
                    + (f"; awaiting zones: {', '.join(pending_zones)}" if pending_zones else "")
                )

    # ------------------------------------------------------------------ per-frame alert path
    def _dispatch_module_events(self, module, events, now: float) -> bool:
        """Alert, record and publish one frame's module events; advance its episodes.

        Returns whether a loud event was seen (evidence capture is due). Split out
        of the inference loop so the exact production path can be driven with an
        injected clock (tests/test_m01_role_floors.py).
        """
        # Belt and braces. The module running under AI_CUSTOM is a general
        # detector with no opinions to suppress, so this should already be
        # empty — but a mode switch that has not landed yet, or a detector
        # that grows its own events later, must not leak alerts nobody asked
        # for. The real saving is that the expensive module is not running
        # at all; see _apply_monitoring_mode. ADDITIVE never discards: the
        # module there is a safeguard the role requires.
        if self.mode is monitoring_mode.MonitoringMode.AI_CUSTOM:
            events = []
        evidence_due = False
        for event in events:
            # Alert first, then record. The alert is the product; a log
            # write that fails (full or read-only SD card) must never be
            # what stands between a confirmed event and the buzzer.
            # Each event starts an alert EPISODE; it is never dropped
            # because an earlier episode alerted recently (M0.1).
            if event.type in self.alert_event_types:
                self.alert_manager.notify(event, now=now)
            self.logger.log(event)
            severity = (
                "loud" if event.type in self.loud_types
                else "soft" if event.type in self.soft_types or event.type in self.alert_event_types
                else "log"
            )
            if severity == "loud":
                evidence_due = True
            if self.dashboard is not None:
                self.dashboard.push_event(event, severity)
            # Edge-triggered: every genuinely new event reaching Recent
            # Activity arms the flash window, regardless of severity --
            # see _apply_alert_overlay's caller below and _FLASH_DURATION_S.
            self._flash_until = time.monotonic() + _FLASH_DURATION_S
            if self.uplink is not None and event.type in self.alert_event_types:
                self.uplink.push_event(event, severity, model=self._model_identity)

        # Advance this module's open episodes: repeat/escalate those whose
        # condition is still true, end the ones that cleared.
        tracked, active = _episode_state(module)
        if self.mode is not monitoring_mode.MonitoringMode.AI_CUSTOM:
            self._reopen_live_episodes(module, tracked, active, now)
        self.alert_manager.observe(now, source="module", active=active, tracked=tracked)
        return evidence_due

    def _reopen_live_episodes(self, module, tracked, active, now: float) -> None:
        """Open an episode for a tracked condition that is live but has none.

        Trackers fire once per episode, so after episodes were closed from outside
        -- an explicit stop and resume, monitoring FAILED and recovered, a mode
        change -- a driver still asleep produced no new event and no alert (M0.1
        review). A live alertable condition without an open episode gets one now.
        In normal operation the module's own event opened it first, so this adds
        nothing.
        """
        missing = (set(active) & set(tracked) & set(self.alert_event_types)) - self.alert_manager.open_keys("module")
        for event_type in sorted(missing):
            event = Event.now(event_type, 0.0, None, getattr(module, "name", self.module_name),
                              "condition still active")
            self.alert_manager.notify(event, now=now)

    def _evaluate_custom_rules(self, module, frame, now: float,
                               observation: Optional[pc.Observation] = None) -> None:
        """Evaluate custom rules on the module's raw per-frame view; advance rule episodes.

        AI-configured rules evaluate the module's raw per-frame view, not its
        Events: zone and count rules need boxes and counts that an Event does
        not carry, and Events are already debounced.
        """
        detection_frame = self._rules_detection_frame(module, observation)
        if detection_frame is not None and self.rule_engine.rules:
            for alert in self.rule_engine.observe_frame(detection_frame, now=now):
                self._emit_rule_alert(alert, now=now)
                self._capture_training_sample(frame, alert, detection_frame)
        # Custom-rule episodes: a critical rule repeats while it stays
        # confirmed; every other rule alert is a one-off episode.
        self.alert_manager.observe(
            now, source="rules",
            active=self.rule_engine.active_alerts(),
            tracked={r.rule_id for r in self.rule_engine.rules
                     if r.severity == "critical" and r.notify != "silent"},
        )

    # ------------------------------------------------------------------ the inference loop
    def _inference_loop(self, stop: threading.Event) -> None:
        # Each generation of the loop gets its own failure budget. Carried over,
        # a loop restarted after escalating raised on its very first bad frame.
        self._consecutive_process_failures = 0
        loop = self._new_inference_loop_state()
        while not (stop.is_set() or self.shutdown.is_set()):
            if not self._inference_pass(loop, stop):
                break

    def _new_inference_loop_state(self) -> _InferenceLoopState:
        """The state one generation of the inference loop carries between passes."""
        return _InferenceLoopState(
            last_seq=0,
            last_detector_retry=time.monotonic(),  # see _retry_detection_backend
            stream_interval=self.dashboard.stream_interval if self.dashboard is not None else 0.0,
        )

    def _inference_pass(self, loop: _InferenceLoopState, stop: threading.Event) -> bool:
        """One pass of the inference loop; False when the loop must end (quit key).

        The loop body, unchanged, as a method so a caller with its own clock can
        drive the exact production path one pass at a time (sim/, the virtual
        device's deterministic end-to-end tests). `stop.wait` is the only pause
        besides the frame wait, which is bounded by `loop.frame_timeout_s`.
        """
        # Commands first, and on EVERY pass -- not only after a fresh frame.
        # They used to be read further down, past the degraded short-circuit
        # and the no-frame `continue`, so a device whose detector was missing
        # (or whose camera was down) could not be switched to a mode that
        # would work: the app's command sat pending for ever.
        try:
            self._apply_pending_commands()
        except Exception:  # one bad command must not stop detection
            # A switch that raised part-way (a module that would not build,
            # a close() that threw) is logged and dropped; the loop -- and
            # the module still installed -- carries on.
            log.exception("applying a mode/rule/zone command failed; continuing")
        if self._stopped is not None:
            # Explicitly stopped: nothing is analysed and nothing alerts, and
            # every report says "stopped". The loop stays alive for commands.
            self._heartbeat = time.monotonic()
            stop.wait(0.5)
            return True
        if time.monotonic() >= self._module_rebuild_at and self._module_unavailable():
            try:
                self._retry_unavailable_module()
            except Exception:  # a failed retry must not crash the loop
                log.exception("module rebuild failed; still FAILED")
        if self._detection_degraded is not None:
            self._heartbeat = time.monotonic()
            if time.monotonic() - loop.last_detector_retry >= _DETECTOR_RETRY_S:
                loop.last_detector_retry = time.monotonic()
                try:
                    self._retry_detection_backend()
                except Exception:  # a failed retry must not crash the loop
                    log.exception("detector retry failed; still DEGRADED")
            if not self._module_runs_without_detector():
                # No detector and nothing this module can do without one:
                # there is nothing honest to compute. The heartbeat still
                # ticks -- the loop IS alive, and health must be able to
                # tell "degraded" from "hung".
                stop.wait(0.5)
                return True
            # Otherwise keep running what still works (audit F05: a missing
            # detector used to switch off face-only drowsiness too). The
            # monitoring state reports object detection as unavailable.
        frame, seq, captured_at = self.bus.wait_for_fresh_stamped(loop.last_seq, timeout=loop.frame_timeout_s)
        if seq == loop.last_seq or frame is None:
            # The heartbeat measures whether THIS LOOP is responsive; frame
            # availability is the camera component's own check. Without this
            # stamp, a camera outage >10s read as "inference dead", burned
            # the inference restart budget on a healthy thread, and could
            # escalate all the way to a mid-drive Pi reboot for a camera
            # problem the camera check was already recovering.
            self._heartbeat = time.monotonic()
            return True  # timeout: no fresh frame — health watches bus.age_s()
        loop.last_seq = seq

        with self._module_lock:
            module = self.module
        # A single bad frame must never crash the process -- found in a repo-
        # wide audit: previously this call was unguarded, so in --display mode
        # (a documented, first-class runtime mode, not dev-only) an exception
        # here took down the entire process instead of the graceful in-place
        # recovery headless mode already got from Service._guarded_run. Even
        # in headless mode, letting ONE bad frame propagate meant restarting
        # the whole inference loop (reopening everything, a visible pause)
        # rather than just skipping that frame — the finer-grained recovery
        # here is strictly better for both. A module that keeps failing is
        # NOT silently tolerated forever, though: past
        # _MAX_CONSECUTIVE_PROCESS_FAILURES in a row this re-raises, handing
        # off to the same recovery ladder a truly wedged module already had.
        try:
            setter = getattr(module, "set_frame_time", None)
            if callable(setter):
                setter(captured_at)  # decisions rest on when the frame was SEEN
            events, observation = self._perceive(module, frame, seq, captured_at)
            self._consecutive_process_failures = 0
            self._last_process_error = None
            # Processed-frame cadence, not capture cadence: a stall here is
            # what makes EventEngine drop in-progress evidence.
            self.bus.record_processed()
        except Exception as exc:
            self._consecutive_process_failures += 1
            self._last_process_error = str(exc) or type(exc).__name__
            log.error(
                "module.process() failed (%d/%d consecutive): %s",
                self._consecutive_process_failures, _MAX_CONSECUTIVE_PROCESS_FAILURES, exc,
                exc_info=True,
            )
            if self._consecutive_process_failures >= _MAX_CONSECUTIVE_PROCESS_FAILURES:
                log.critical(
                    "module.process() failed %d times in a row; propagating "
                    "for a full restart instead of dropping frames forever",
                    self._consecutive_process_failures,
                )
                raise
            # The inference LOOP is alive and responsive; only the module call
            # failed, so the heartbeat should say so rather than let a string
            # of caught failures look like a wedged/dead loop to health checks.
            self._heartbeat = time.monotonic()
            return True

        alert_now = time.monotonic()
        evidence_due = self._dispatch_module_events(module, events, alert_now)

        # Evidence snapshots: OPT-IN (alerts.save_evidence), local-disk only,
        # bounded count — the single, deliberate exception to "frames are never
        # written to disk", captured only at the moment of a loud alert.
        if evidence_due and self._evidence_dir is not None:
            self._save_evidence(module, frame, events)

        self._evaluate_custom_rules(module, frame, alert_now, observation)
        self._run_nodes(observation, alert_now)

        # A periodic frame with nothing happening. Negatives are what stop a
        # detector learning to answer yes to everything, and they are gated
        # behind their own opt-in because they are footage of ordinary life.
        if self.capture_store.should_capture(negative=True):
            self.capture_store.capture(
                frame, module=self.module_name,
                model_version=self._model_version(),
            )

        # Health for the uplink is sampled by _HealthReportService on its own
        # thread, so it keeps reporting while this loop is not producing.

        snapshot = module.get_telemetry_snapshot()
        if snapshot is not None:
            self.telemetry.log_if_due(
                time.monotonic(), datetime.now(UTC).isoformat(), self.module_name, snapshot
            )

        # Developer-only: "why did/didn't that alert" as a readable block,
        # not a re-read of this module's source. --debug only, throttled
        # (every frame would be unreadable), and never sent anywhere off
        # this machine -- get_debug_state() is opt-in per module (most
        # return None) and deliberately more detailed than the telemetry
        # snapshot above, which IS logged and IS product data.
        if self.debug:
            now_dbg = time.monotonic()
            if now_dbg - self._last_debug_print >= 1.0:
                state = module.get_debug_state()
                if state is not None:
                    self._last_debug_print = now_dbg
                    print(f"{self.module_name.upper()} STATE")
                    for key, value in state.items():
                        print(f"  {key}: {value}")
        if self.dashboard is not None:
            # Either source can raise the alert state. A configured rule is
            # someone explicitly asking to be alerted, so it must drive the
            # red screen exactly as a preset's own detection does — the
            # module's own opinion is not the only one that counts.
            alert_active = module.is_alert_active() or self.rule_engine.is_alert_active()
            self.dashboard.push_status(alert_active, snapshot)

        # Annotate + encode only when someone is watching, and never faster than
        # stream_fps — inference cadence and stream cadence are decoupled, so a
        # phone at 12fps costs nothing at 30fps detection.
        now = time.monotonic()
        want_stream = (
            self.dashboard is not None
            and self.dashboard.has_viewers()
            and now - self._last_stream_push >= loop.stream_interval
        )
        view = None
        if self.display or want_stream:
            # Guarded exactly like module.process() above, and for a sharper
            # reason: this call happens ONLY while someone is watching the
            # live feed (see `want_stream`). Unguarded, a module that drew
            # badly took the whole inference loop down at the precise moment
            # the app opened the camera -- and at no other time. The device
            # looked healthy right up until a user looked at it, and the
            # supervisor's restart then rebuilt the module, zeroing every
            # timer, PERCLOS window and risk EMA, so nothing needing
            # sustained evidence could fire while the feed was open.
            #
            # Unlike process(), a persistent failure here is never escalated
            # into a restart: annotation is presentation, detection is the
            # product. Losing the picture must not cost the monitoring.
            try:
                view = module.annotate_debug(frame.copy()) if self.debug else module.annotate(frame.copy())
                self._consecutive_annotate_failures = 0
            except Exception as exc:
                self._consecutive_annotate_failures += 1
                # First failure loudly, then throttled: this runs at stream
                # cadence, so an unthrottled log would bury everything else.
                if self._consecutive_annotate_failures == 1 or (
                    self._consecutive_annotate_failures % _ANNOTATE_FAILURE_LOG_EVERY == 0
                ):
                    log.error(
                        "module.annotate() failed (%d consecutive); the live feed will "
                        "hold its last frame, detection is unaffected: %s",
                        self._consecutive_annotate_failures, exc, exc_info=True,
                    )
                view = None
            if view is not None:
                # Event-driven, not level-driven: flashes once per new event
                # (see push_event/_emit_rule_alert above), not continuously for
                # as long as module.is_alert_active() stays true -- an object
                # that remains visible must not keep the screen flashing.
                if now < self._flash_until:
                    view = _apply_alert_overlay(view)
                if want_stream:
                    self.dashboard.push_frame(view)
                    self._last_stream_push = now
        if self.display and view is not None:
            cv2.imshow(self.window_name, view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.shutdown.set()
                return False
            if key != 255 and hasattr(module, "handle_key"):
                module.handle_key(key)  # module-specific controls (e.g. openpilot wheel touch)

        self._heartbeat = time.monotonic()
        return True

    # ------------------------------------------------------------------ evidence
    def _save_evidence(self, module, frame, events) -> None:
        """Write one annotated snapshot for a loud alert (opt-in, bounded, local-only)."""
        try:
            self._evidence_dir.mkdir(parents=True, exist_ok=True)
            view = module.annotate(frame.copy())
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
            types = "+".join(sorted({e.type for e in events}))[:60] or "alert"
            cv2.imwrite(str(self._evidence_dir / f"{stamp}_{types}.jpg"), view)
            # Bounded retention: newest N only.
            files = sorted(self._evidence_dir.glob("*.jpg"))
            for old in files[:-self._evidence_max]:
                old.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001 - evidence must never break detection
            log.warning("evidence snapshot failed: %s", exc)

    def _shutdown_partial(self) -> None:
        """Release whatever `_build()` managed to construct before it failed.

        Mirrors run()'s finally block, but every step is independent and
        best-effort: this runs on a half-built object where any attribute may be
        None, and a second exception here would mask the real startup failure.

        The step that actually matters is the camera. A held camera makes the
        NEXT start fail too, which is how a single unplugged ribbon became a
        reboot loop rather than one clear error repeated.
        """
        for label, attribute, method in (
            ("health", "health", "stop"),
            ("services", "supervisor", "stop_all"),
            ("uplink", "uplink", "close"),
            ("dashboard", "dashboard", "close"),
            ("event log", "logger", "close"),
            ("telemetry", "telemetry", "close"),
            ("module", "module", "close"),
            ("camera", "camera", "release"),
        ):
            target = getattr(self, attribute, None)
            if target is None:
                continue  # never built, nothing to release
            try:
                getattr(target, method)()
            except Exception as exc:  # noqa: BLE001 - never mask the startup error
                log.debug("partial shutdown: %s not released (%s)", label, exc)

    # ------------------------------------------------------------------ run
    def run(self) -> None:
        try:
            self._build()
        except Exception:
            # _build() used to sit OUTSIDE this try, so anything it raised —
            # a camera that will not open, a dashboard port still held by a
            # stale manual run — escaped before the try/finally existed. The
            # camera was left open, nothing was released, and systemd's
            # Restart=always retried every 3s until StartLimitAction=reboot
            # power-cycled the Pi. Verified in Phase 6: an unopenable camera
            # raised after 10s, with `dashboard built? None  health built? None`.
            #
            # This does NOT convert a build failure into a running device — the
            # exception still propagates and the service still exits non-zero.
            # It guarantees the shutdown path runs, so the camera and sockets are
            # released before the retry, which is what turns a reboot loop into
            # an ordinary restart loop that systemd can report.
            log.exception("startup failed; releasing whatever was already built")
            self._shutdown_partial()
            raise

        # systemd sends SIGTERM on stop; a clean release of the camera + flushed
        # logs must happen on that path too, not just Ctrl+C. (Handlers can only
        # be installed from the main thread — tests drive run() from a worker.)
        def _request_shutdown(signum, _frame):
            log.info("received %s — shutting down", signal.Signals(signum).name)
            self.shutdown.set()

        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, _request_shutdown)
            signal.signal(signal.SIGINT, _request_shutdown)

        # The "starting" grace counts from when the loop is actually started,
        # not from __init__ — _build (model warm-up, camera retries) can take
        # longer than the grace itself.
        self._heartbeat_reset_at = time.monotonic()
        self.supervisor.start_all()
        self.health.start()
        sdnotify.notify_ready()
        self._record_known_good_config()
        log.info(
            "runtime up: profile=%s module=%s display=%s accelerator=%s%s",
            self.profile, self.module_name, self.display, accel.status()[1],
            "" if self._detection_degraded is None
            else "  *** DEGRADED: monitoring NOT active ***",
        )
        # One block naming everything that decides what this device is about to
        # do. Before this, confirming a demo was set up correctly meant grepping
        # several log lines written by different subsystems at different times,
        # and the most important facts -- which model, which thresholds, whether
        # anything is overriding the configured mode -- were the ones spread
        # furthest apart. Phase 5 lost a driver-attention demo to a stale rule
        # that nothing printed.
        accel_cfg = self.config.get("accelerator", {}) or {}
        module_cfg = (self.config.get("modules", {}) or {}).get(self.module_name, {}) or {}
        hef = accel_cfg.get("yolo_hef")
        pinned = str(accel_cfg.get("yolo_hef_sha256") or "")
        lines = [
            "----- demo configuration -----",
            f"  mode              {self.module_name}"
            + ("" if self.config.get("module") == self.module_name
               else f"   (config.yaml says '{self.config.get('module')}')"),
            f"  detector          {hef or 'CPU (no .hef configured)'}",
            f"  model sha256      {pinned[:16] + '…' if pinned else '(not pinned)'}",
            f"  phone conf        {module_cfg.get('yolo_confidence_threshold', '(n/a)')}",
            f"  person conf       {module_cfg.get('person_confidence_threshold', '(n/a)')}",
            f"  accelerator       {accel.status()[1]}",
            f"  monitoring        {'NOT ACTIVE — DEGRADED' if self._detection_degraded else 'active'}",
        ]
        if self.monitoring.active:
            lines.append("  override          AI-configured rule set IS overriding the mode")
        log.info("\n".join(lines))

        # Say WHY this module is running when it is not the configured one. An
        # AI-configured rule set puts the device into AI-Custom mode, which is a
        # feature -- but a rule saved weeks ago silently outranking config.yaml's
        # `module:` is indistinguishable, from the logs, from the config simply
        # being wrong. On this device a stale "watch for a bottle" rule from a
        # fortnight earlier was quietly in charge of a driver-attention demo.
        if self.monitoring.active:
            configured = self.config.get("module")
            age_h = (
                (time.time() - self.monitoring.updated_at) / 3600.0
                if self.monitoring.updated_at else None
            )
            log.warning(
                "module '%s' is running because an AI-configured rule set is ACTIVE "
                "and overrides config.yaml's module: '%s'. Instruction: %r. "
                "Saved %s. Clear it from the app to return to the configured module.",
                self.module_name, configured,
                self.monitoring.instruction or "(none recorded)",
                f"{age_h:.1f}h ago" if age_h is not None else "at an unknown time",
            )

        try:
            if self.display:
                # cv2.imshow must run on the main thread on macOS.
                self._inference_loop(self.shutdown)
            else:
                while not self.shutdown.wait(0.5):
                    pass
        except KeyboardInterrupt:
            print("\nSensorDyme: shutting down (Ctrl+C)")
        finally:
            sdnotify.notify_stopping()
            self.shutdown.set()
            self.health.stop()
            self.supervisor.stop_all()
            if self.uplink is not None:
                self.uplink.close()
            if self.dashboard is not None:
                self.dashboard.close()
            self.logger.close()
            self.telemetry.close()
            # Native teardown only for what nothing is still using. A thread
            # that outlived its join (a wedged read, a hung NPU call) may still
            # be inside the module, the detector or the camera, and freeing
            # them under it turns a hang into a native crash during stop.
            inference_alive = (
                self._inference_service is not None and self._inference_service.is_running
            )
            capture_alive = self.capture is not None and self.capture.is_running
            if inference_alive:
                log.error("inference thread did not stop; leaving the module and detector open")
            else:
                with self._module_lock:
                    self.module.close()
                from core import detectors

                detectors.release_all()
            if capture_alive:
                log.error("capture thread did not stop; leaving the camera open")
            elif self.camera is not None:
                self.camera.release()
            if self.display:
                cv2.destroyAllWindows()
            log.info("shutdown complete")


def _apply_alert_overlay(view):
    """Flash the frame red and stamp an ALERT banner on it (moved from main.py)."""
    overlay = view.copy()
    overlay[:] = (0, 0, 255)
    view = cv2.addWeighted(overlay, 0.45, view, 0.55, 0)
    cv2.putText(
        view,
        "ALERT",
        (max(0, view.shape[1] // 2 - 90), 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.6,
        (255, 255, 255),
        3,
    )
    return view
