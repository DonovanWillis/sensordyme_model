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
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2

from core import accel, sdnotify
from core.alerts import AlertManager
from core.camera import Camera, CameraError
from core.events import Event
from core.frame_bus import FrameBus
from core.health import HealthMonitor
from core.logger import EventLogger
from core.logging_setup import get_logger, setup_logging
from core import device_health
from core.network import WifiManager, nmcli_available
from core.dataset import CaptureStore
from core import monitoring_mode
from core.monitoring_config import load_or_empty
from core.rules import RuleEngine
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
_VERSION = "0.2"
# A module.process() call that keeps failing this many frames IN A ROW is not
# a transient hiccup -- propagate so the existing recovery ladder (in-thread
# restart headless, process exit under --display) can recover a genuinely
# wedged module, rather than looping forever silently dropping every frame.
_MAX_CONSECUTIVE_PROCESS_FAILURES = 10
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


class _InferenceService(Service):
    """Runs the inference loop in a supervised thread (headless mode)."""

    name = "inference"

    def __init__(self, runtime: "DeviceRuntime") -> None:
        super().__init__()
        self._runtime = runtime

    def run(self, stop: threading.Event) -> None:
        self._runtime._inference_loop(stop)


class DeviceRuntime:
    """Owns every long-running piece of the device for one process lifetime."""

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
    ) -> None:
        self.config = config
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

        # AI-configured rules. Zones come from the active module's config, so a
        # rule naming "pool" resolves against the same polygon the module draws.
        self.rule_engine = RuleEngine(zones=self._zones_for_module(module_name))

        # What someone configured survives the device being switched off.
        #
        # Rules used to arrive only from the cloud and live only in memory, so a
        # reboot silently emptied them and a device with no Wi-Fi could never be
        # configured at all. Restoring here means the device wakes up watching
        # what it was asked to watch, with or without a network.
        self.monitoring = load_or_empty()
        if self.monitoring.rules:
            self.rule_engine.replace_rules(self.monitoring.rules)
        # What the selected preset wants to run, remembered separately from what
        # is running now — AI_CUSTOM swaps the module out and must be able to
        # hand it back when the custom configuration is cleared.
        self._preset_module = module_name
        self.mode = monitoring_mode.resolve(
            monitoring_active=self.monitoring.active, preset_module=module_name
        )

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
        self._last_stream_push = 0.0
        self._flash_until = 0.0  # monotonic time; see _FLASH_DURATION_S
        self._consecutive_process_failures = 0  # see _MAX_CONSECUTIVE_PROCESS_FAILURES
        self._last_debug_print = 0.0

        self.camera: Optional[Camera] = None
        self.capture: Optional[CaptureService] = None
        self.wifi: Optional[WifiManager] = None
        self._last_health_sample = 0.0
        self.dashboard = None
        self.uplink = None
        self.provisioning: Optional[ProvisioningService] = None
        self.health: Optional[HealthMonitor] = None
        self.supervisor: Optional[Supervisor] = None
        self.logger: Optional[EventLogger] = None
        self.telemetry: Optional[TelemetryLogger] = None
        self.hardware = None

    # ------------------------------------------------------------------ build
    def _open_camera(self) -> Camera:
        """Open the camera with retries — transient 'device busy' at boot must not
        burn a whole systemd restart cycle (and model reload) per attempt."""
        last_exc: Optional[Exception] = None
        for attempt in range(1, _CAMERA_OPEN_RETRIES + 1):
            try:
                camera = Camera(self.config.get("camera", {}))
                # Prove it produces frames before declaring victory (the "verify
                # the camera" boot step): one real read, not just a successful open.
                camera.read()
                if attempt > 1:
                    log.info("camera opened on attempt %d", attempt)
                return camera
            except (CameraError, RuntimeError) as exc:
                last_exc = exc
                log.warning(
                    "camera open attempt %d/%d failed: %s", attempt, _CAMERA_OPEN_RETRIES, exc
                )
                time.sleep(_CAMERA_OPEN_BACKOFF_S)
        raise CameraError(f"camera unavailable after {_CAMERA_OPEN_RETRIES} attempts: {last_exc}")

    def _build(self) -> None:
        setup_logging(str(self.config.get("logging", {}).get("level", "INFO")))

        # Accelerator: probe once, install the config, verify before inference starts.
        accel_cfg = self.config.get("accelerator", {})
        info = accel.configure(accel_cfg)
        if accel_cfg.get("required", False) and not info.usable:
            raise RuntimeError(f"accelerator.required is set but: {info.detail}")

        logging_config = self.config.get("logging", {})
        self.logger = EventLogger(logging_config.get("csv_path", "data/events.csv"))
        self.telemetry = TelemetryLogger(
            logging_config.get("telemetry_csv_path", "data/telemetry.csv"),
            interval_s=float(logging_config.get("telemetry_interval_s", 0.5)),
        )

        self.camera = self._open_camera()
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
                health_provider=lambda: self.health.snapshot(),
            )
            print(
                f"SensorDyme: dashboard live at {self.dashboard.start()} — annotated frames stream "
                "in-memory to connected browsers only (localhost by default); never written to disk"
            )

        cloud_cfg = self.config.get("cloud", {})
        if bool(cloud_cfg.get("enabled", False)) and cloud_cfg.get("device_key"):
            from core.uplink import CloudUplink

            self.uplink = CloudUplink(cloud_cfg)
            self.uplink.set_active_preset(self.active_preset)
            # Report the restored configuration at startup too, not only when a
            # new one arrives — otherwise a device that rebooted into a custom
            # configuration shows the base mode until the next change.
            self.uplink.set_monitoring(self.monitoring.describe())
            self._apply_monitoring_mode()
            self.uplink.start()
            print(
                f"SensorDyme: cloud uplink -> {cloud_cfg.get('base_url')} "
                "(events + mode commands; never frames)"
            )

        wifi = WifiManager()
        self.wifi = wifi  # also read by the health report the app displays
        network_cfg = self.config.get("network", {})
        self.provisioning = ProvisioningService(
            wifi, network_cfg.get("provisioning", {}), version=_VERSION
        )

        # Hardware HAL: null implementations today; health-visible either way.
        from hardware.hal import build_hardware

        self.hardware = build_hardware(self.config.get("hardware", {}))

        # ---- health registrations -------------------------------------------------
        camera_fps = max(self.camera.fps, 1.0)
        stale_after = max(2.0, 5.0 / camera_fps)  # 5 frame intervals, 2s floor
        boot_time = time.monotonic()

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
            restart=lambda: self.capture.restart(),
            check_interval_s=2.0,
        )
        self.health.register(
            "inference",
            critical=True,
            check=self._inference_check,
            restart=self._restart_inference,
            check_interval_s=2.0,
        )
        self.health.register("accelerator", check=accel.status, check_interval_s=60.0)
        if self.dashboard is not None:
            self.health.register(
                "dashboard",
                check=lambda: (self.dashboard.is_alive(), self.dashboard.url),
                restart=lambda: self.dashboard.restart(),
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
        self.health.register("provisioning", check=lambda: self.provisioning.status(), check_interval_s=15.0)
        for name, component in self.hardware.health_checks().items():
            self.health.register(name, check=component, check_interval_s=60.0)

        # ---- supervisor -----------------------------------------------------------
        self.supervisor = Supervisor(self.health)
        self.supervisor.add(self.capture)
        if not self.display:
            self.supervisor.add(_InferenceService(self))
        self.supervisor.add(self.provisioning)

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
            covered = monitoring_mode.capabilities_covered_by(self.module_name)
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

    def _model_version(self) -> str:
        """Which model produced a sample — useless data without it, since you
        cannot tell an old model's mistakes from a new one's."""
        weights = (self.config.get("modules", {}).get(self.module_name, {}) or {}).get(
            "model_weights", "yolo11n.pt"
        )
        return str(weights)

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
    def _collect_health(self) -> dict:
        """This device's real state, for the app's status indicators."""
        heartbeat_age = None if self._heartbeat == 0.0 else time.monotonic() - self._heartbeat
        return device_health.collect(
            bus=self.bus,
            capture=self.capture,
            wifi=self.wifi,
            heartbeat_age_s=heartbeat_age,
            module_name=self.module_name,
            uplink_connected=self.uplink.connected if self.uplink is not None else None,
        )

    # ------------------------------------------------------------------ AI-configured rules
    def _zones_for_module(self, module_name: str) -> list:
        """Zones configured for `module_name`, or none if it has no zone support."""
        modules_cfg = self.config.get("modules", {})
        module_cfg = modules_cfg.get(module_name, {}) if isinstance(modules_cfg, dict) else {}
        return load_zones(module_cfg.get("zones", {}) if isinstance(module_cfg, dict) else {})

    def _apply_monitoring_mode(self) -> None:
        """Run the detector the active mode calls for, and nothing else.

        This is what makes "AI custom monitoring" mean something. Suppressing a
        module's alerts leaves it running: MediaPipe FaceMesh, Hands and Pose
        cost 85% of one core on this device, computing drowsiness and gaze for
        a person who asked about phones. Switching the module stops the work.
        """
        mode = monitoring_mode.resolve(
            monitoring_active=self.monitoring.active,
            preset_module=self._preset_module or self.module_name,
        )
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
            new_module = get_module_class(wanted)({**module_cfg, "recalibrate": False})
        except Exception as exc:  # noqa: BLE001 - a bad switch must not kill the device
            log.error("could not switch to '%s' for %s: %s", wanted, mode.value, exc)
            return

        with self._module_lock:
            old_module, self.module, self.module_name = self.module, new_module, wanted
        # Release the old graphs explicitly. MediaPipe holds native resources
        # that a dropped reference does not free promptly, and a preset switch
        # used to leak one FaceMesh graph per swap.
        close = getattr(old_module, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 - teardown must never raise
                pass

        self.rule_engine.set_zones(self._zones_for_module(wanted))
        self.mode = mode
        log.info(
            "[MONITORING] mode=%s module=%s base_monitoring=%s%s",
            mode.value, wanted,
            "DISABLED" if mode is monitoring_mode.MonitoringMode.AI_CUSTOM else "enabled",
            f" instruction={self.monitoring.instruction!r}" if self.monitoring.instruction else "",
        )

    def _emit_rule_alert(self, alert) -> None:
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
        self.logger.log(event)
        self.alert_manager.notify(event)
        severity = "loud" if alert.severity == "critical" else "soft"
        if self.dashboard is not None:
            self.dashboard.push_event(event, severity)
        self._flash_until = time.monotonic() + _FLASH_DURATION_S
        if self.uplink is not None and alert.notify != "silent":
            # Marked as configured. `notify: "push"` on a rule IS someone asking
            # to be told; the severity field is about how prominent it looks,
            # not about whether it is worth interrupting for.
            self.uplink.push_event(event, severity, source="custom_rule")

    # ------------------------------------------------------------------ health hooks
    def _inference_check(self) -> tuple[bool, str]:
        if self._heartbeat == 0.0:
            return True, "starting"
        age = time.monotonic() - self._heartbeat
        return age < 10.0, f"last iteration {age:.1f}s ago"

    def _restart_inference(self) -> None:
        if self.display:
            return  # display mode runs inference on the main thread; nothing to bounce
        # Back to "starting": the restarted loop gets the same grace a fresh
        # boot does, instead of being declared dead again on the very next
        # check because the stamp still describes the loop being replaced.
        self._heartbeat = 0.0
        self.supervisor.restart("inference")

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
        if not bool(health_cfg.get("allow_reboot", False)) or not nmcli_available():
            log.critical(
                "%s is down and unrecoverable; reboot disabled (health.allow_reboot) — "
                "leaving recovery to systemd's Restart/StartLimitAction",
                component,
            )
            return

        min_interval = float(health_cfg.get("reboot_min_interval_s", 1800))
        max_total = int(health_cfg.get("reboot_max_total", 5))
        state = self._reboot_state()
        now = time.time()

        if state["count"] >= max_total > 0:
            log.critical(
                "reboot suppressed: %d reboot(s) already requested (health.reboot_max_total=%d) "
                "and '%s' is still down — a fault that survived those will not be fixed by "
                "another. Staying up degraded; see /api/health.",
                state["count"], max_total, component,
            )
            return
        if state["last_at"] and now - state["last_at"] < min_interval:
            log.critical(
                "reboot suppressed: last one was %.0fs ago (<%.0fs)",
                now - state["last_at"], min_interval,
            )
            return

        if not self._record_reboot_request(state, now):
            log.critical(
                "reboot ABORTED: could not persist the reboot record to %s. Without it "
                "the next boot cannot know this reboot happened, so the rate limit and "
                "the total bound would both reset and the device could power-cycle "
                "indefinitely. Staying up degraded instead ('%s' remains down).",
                self._REBOOT_STATE_PATH, component,
            )
            return

        log.critical(
            "LAST RESORT: rebooting because '%s' is unrecoverable (reboot %d of %d)",
            component, state["count"] + 1, max_total,
        )
        subprocess.run(["sudo", "-n", "systemctl", "reboot"], check=False)

    #: Persisted across reboots, which is what makes the total bound meaningful --
    #: systemd's own StartLimitBurst counter does NOT survive a reboot, so it can
    #: never bound a reboot campaign.
    _REBOOT_STATE_PATH = Path("data/.reboot_state.json")

    def _reboot_state(self) -> dict:
        """{"count": int, "last_at": float} — zeros when absent or unreadable.

        An unreadable file reads as "no reboots yet", which is the permissive
        direction; that is safe ONLY because _record_reboot_request must succeed
        before any reboot proceeds, so an unwritable state file stops the reboot
        rather than uncapping it.
        """
        try:
            raw = json.loads(self._REBOOT_STATE_PATH.read_text())
            return {"count": int(raw.get("count", 0)), "last_at": float(raw.get("last_at", 0.0))}
        except (OSError, ValueError, TypeError):
            return {"count": 0, "last_at": 0.0}

    def _record_reboot_request(self, state: dict, now: float) -> bool:
        """Persist the incremented reboot record. False if it could not be written.

        Atomic (tmp + fsync + replace) for the same reason core/monitoring_config.py
        is: a half-written record on a device that is about to power-cycle is worse
        than no record, and this file's whole job is to survive that power-cycle.
        """
        payload = {"count": state["count"] + 1, "last_at": now}
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

    # ------------------------------------------------------------------ preset switching
    def _exit_custom_monitoring(self) -> None:
        """Someone explicitly picked a mode: that always wins over custom rules.

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
        self.rule_engine.replace_rules([])
        if self.uplink is not None:
            self.uplink.set_monitoring(self.monitoring.describe())
        log.info("[MONITORING] custom monitoring exited: explicit mode switch")

    def _switch_preset(self, name: str) -> None:
        """Hot-swap the running module to a preset (optionally "preset:variant")."""
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
        base_name, _, variant_name = name.partition(":")
        preset = self.presets.get(base_name)
        if preset is None or self.profile == "mac_max":
            if preset is None:
                print(f"SensorDyme: ignoring unknown preset '{name}'")
            return
        overrides = {}
        if variant_name:
            variant = (preset.get("variants") or {}).get(variant_name)
            if variant is None:
                print(f"SensorDyme: preset '{base_name}' has no variant '{variant_name}'")
                return
            overrides = variant.get("overrides") or {}
        new_module_name = preset["module"]
        try:
            new_module = get_module_class(new_module_name)(
                {**self.config["modules"][new_module_name], **overrides, "recalibrate": False}
            )
        except Exception as exc:  # noqa: BLE001 - a bad preset must not kill the device
            print(f"SensorDyme: preset switch to '{name}' failed: {exc}")
            return
        # Exit custom monitoring BEFORE swapping the module, not after: an
        # explicit preset choice always overrides whatever custom rules were
        # running, and the module below must land as STANDARD, never as a
        # STANDARD module with AI_CUSTOM still recorded as the active mode.
        self._exit_custom_monitoring()
        with self._module_lock:
            old_module = self.module
            self.module, self.module_name, self.active_preset = new_module, new_module_name, name
            self.alert_event_types = self.alert_types_for(new_module_name, preset)
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
        self.mode = monitoring_mode.resolve(
            monitoring_active=self.monitoring.active, preset_module=new_module_name
        )
        # Zones belong to the module's config, so a mode switch re-points custom
        # rules at the new module's polygons instead of the previous mode's.
        self.rule_engine.set_zones(self._zones_for_module(new_module_name))
        old_module.close()  # release the old FaceMesh/graph — was leaked before
        if overrides:
            print(f"SensorDyme: variant '{variant_name}' overrides applied: {overrides}")
        if self.dashboard is not None:
            self.dashboard.set_active_preset(name, new_module_name)
        if self.uplink is not None:
            self.uplink.set_active_preset(name)
        switch_event = Event.now("preset_changed", 0.0, 1.0, "system", f"mode -> {name}")
        self.logger.log(switch_event)
        self._flash_until = time.monotonic() + _FLASH_DURATION_S
        if self.dashboard is not None:
            self.dashboard.push_event(switch_event, "log")
        print(f"SensorDyme: switched to preset '{name}' (module '{new_module_name}')")

    # ------------------------------------------------------------------ the inference loop
    def _inference_loop(self, stop: threading.Event) -> None:
        last_seq = 0
        stream_interval = (
            self.dashboard.stream_interval if self.dashboard is not None else 0.0
        )
        while not (stop.is_set() or self.shutdown.is_set()):
            frame, seq = self.bus.wait_for_fresh(last_seq, timeout=0.5)
            if seq == last_seq or frame is None:
                # The heartbeat measures whether THIS LOOP is responsive; frame
                # availability is the camera component's own check. Without this
                # stamp, a camera outage >10s read as "inference dead", burned
                # the inference restart budget on a healthy thread, and could
                # escalate all the way to a mid-drive Pi reboot for a camera
                # problem the camera check was already recovering.
                self._heartbeat = time.monotonic()
                continue  # timeout: no fresh frame — health watches bus.age_s()
            last_seq = seq

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
                    zones = load_zones({
                        z["name"]: {"polygon": z.get("polygon"), "kind": z.get("kind", "restricted")}
                        for z in new_zones if isinstance(z, dict) and z.get("name")
                    })
                    self.rule_engine.set_zones(zones)
                    print(f"zones: {len(zones)} area(s) active ({', '.join(z.name for z in zones) or 'none'})")

                new_rules = self.uplink.pop_pending_rules()
                if new_rules is not None:
                    self.rule_engine.replace_rules(new_rules)
                    # Written down as it arrives, so the next boot still has it.
                    self.monitoring.save(
                        new_rules,
                        instruction=self.uplink.pending_instruction(),
                        version=self.uplink.rules_version(),
                    )
                    self.uplink.set_monitoring(self.monitoring.describe())
                    # A new configuration can change which detector is needed.
                    self._apply_monitoring_mode()
                    self._apply_requested_labels()
                    pending_zones = self.rule_engine.pending_zones()
                    print(
                        f"rules: enforcing {len(self.rule_engine.rules)} AI-configured rule(s)"
                        + (f"; awaiting zones: {', '.join(pending_zones)}" if pending_zones else "")
                    )

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
                events = module.process(frame)
                self._consecutive_process_failures = 0
            except Exception as exc:  # noqa: BLE001 - see comment above
                self._consecutive_process_failures += 1
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
                continue

            # Belt and braces. The module running under AI_CUSTOM is a general
            # detector with no opinions to suppress, so this should already be
            # empty — but a mode switch that has not landed yet, or a detector
            # that grows its own events later, must not leak alerts nobody asked
            # for. The real saving is that the expensive module is not running
            # at all; see _apply_monitoring_mode.
            if self.monitoring.active:
                events = []
            evidence_due = False
            for event in events:
                self.logger.log(event)
                if event.type in self.alert_event_types:
                    self.alert_manager.notify(event)
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
                    self.uplink.push_event(event, severity)

            # Evidence snapshots: OPT-IN (alerts.save_evidence), local-disk only,
            # bounded count — the single, deliberate exception to "frames are never
            # written to disk", captured only at the moment of a loud alert.
            if evidence_due and self._evidence_dir is not None:
                self._save_evidence(module, frame, events)

            # AI-configured rules evaluate the module's raw per-frame view, not its
            # Events: zone and count rules need boxes and counts that an Event does
            # not carry, and Events are already debounced.
            detection_frame = module.get_detections()
            if detection_frame is not None and self.rule_engine.rules:
                for alert in self.rule_engine.observe_frame(detection_frame):
                    self._emit_rule_alert(alert)
                    self._capture_training_sample(frame, alert, detection_frame)

            # A periodic frame with nothing happening. Negatives are what stop a
            # detector learning to answer yes to everything, and they are gated
            # behind their own opt-in because they are footage of ordinary life.
            if self.capture_store.should_capture(negative=True):
                self.capture_store.capture(
                    frame, module=self.module_name,
                    model_version=self._model_version(),
                )

            # Health is sampled here, on the thread that owns the frame bus and
            # capture service, and handed to the uplink as a snapshot. Sampled at
            # the uplink's own cadence rather than every frame: reading /proc and
            # vcgencmd 30 times a second would cost more than it reports.
            sampled_at = time.monotonic()
            if self.uplink is not None and sampled_at - self._last_health_sample >= _HEALTH_SAMPLE_S:
                self._last_health_sample = sampled_at
                self.uplink.set_health(self._collect_health())

            snapshot = module.get_telemetry_snapshot()
            if snapshot is not None:
                self.telemetry.log_if_due(
                    time.monotonic(), datetime.now(timezone.utc).isoformat(), self.module_name, snapshot
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
                and now - self._last_stream_push >= stream_interval
            )
            view = None
            if self.display or want_stream:
                view = module.annotate_debug(frame.copy()) if self.debug else module.annotate(frame.copy())
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
                    break
                if key != 255 and hasattr(module, "handle_key"):
                    module.handle_key(key)  # module-specific controls (e.g. openpilot wheel touch)

            self._heartbeat = time.monotonic()

    # ------------------------------------------------------------------ evidence
    def _save_evidence(self, module, frame, events) -> None:
        """Write one annotated snapshot for a loud alert (opt-in, bounded, local-only)."""
        try:
            self._evidence_dir.mkdir(parents=True, exist_ok=True)
            view = module.annotate(frame.copy())
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            types = "+".join(sorted({e.type for e in events}))[:60] or "alert"
            cv2.imwrite(str(self._evidence_dir / f"{stamp}_{types}.jpg"), view)
            # Bounded retention: newest N only.
            files = sorted(self._evidence_dir.glob("*.jpg"))
            for old in files[:-self._evidence_max]:
                old.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001 - evidence must never break detection
            log.warning("evidence snapshot failed: %s", exc)

    # ------------------------------------------------------------------ run
    def run(self) -> None:
        self._build()

        # systemd sends SIGTERM on stop; a clean release of the camera + flushed
        # logs must happen on that path too, not just Ctrl+C. (Handlers can only
        # be installed from the main thread — tests drive run() from a worker.)
        def _request_shutdown(signum, _frame):
            log.info("received %s — shutting down", signal.Signals(signum).name)
            self.shutdown.set()

        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, _request_shutdown)
            signal.signal(signal.SIGINT, _request_shutdown)

        self.supervisor.start_all()
        self.health.start()
        sdnotify.notify_ready()
        log.info(
            "runtime up: profile=%s module=%s display=%s accelerator=%s",
            self.profile, self.module_name, self.display, accel.status()[1],
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
            with self._module_lock:
                self.module.close()
            if self.camera is not None:
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
