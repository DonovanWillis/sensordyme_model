"""The digital twin's fidelity map (M8.4), machine-readable.

For every product component: does the virtual device run the REAL production
code, or a stand-in -- and what exactly the stand-in fakes. docs/DIGITAL_TWIN.md
is the prose version of this table (tests/test_m84_twin_fidelity.py keeps the
two in step). The device's health report carries the stand-ins
(`simulation.stand_ins`) and the acceptance summary carries the whole map, so no
twin result can be read without seeing what was simulated.

`code`:  real       production code runs unchanged
         real+seam  production code, with one documented seam replaced
         stand-in   simulated; `fakes` says what is fabricated
         absent     not exercised by the twin at all
"""

from __future__ import annotations

from typing import TypedDict


class Fidelity(TypedDict):
    code: str
    runs: str
    fakes: str
    proven_only_on_hardware: str


FIDELITY: dict[str, Fidelity] = {
    "camera": {
        "code": "stand-in", "runs": "sim/camera.py SimCamera (rendered scene, SIMULATED banner)",
        "fakes": "the image: a drawn face/phone, the scene encoded in row 0; faults frozen/dropping/stalled",
        "proven_only_on_hardware": "Camera Module 3 exposure, IR, motion blur, real faces, Picamera2 reopen"},
    "capture": {
        "code": "real+seam", "runs": "live: services/capture.CaptureService pumping SimCamera; "
                                     "stepped (SimClock): sim/device._SteppedCapture, advance() is the pump",
        "fakes": "stepped mode only: no pump thread (one frame per 50 ms step)",
        "proven_only_on_hardware": "frame timing under CPU load"},
    "frame_bus": {
        "code": "real", "runs": "core/frame_bus.FrameBus", "fakes": "nothing",
        "proven_only_on_hardware": "-"},
    "perception_engine": {
        "code": "real", "runs": "core/perception.py (one detector run per frame shared by every node)",
        "fakes": "nothing; it wraps the scripted detector", "proven_only_on_hardware": "-"},
    "face_landmarks": {
        "code": "stand-in", "runs": "sim/perception.ScriptedFaceMesh (--video: real MediaPipe FaceMesh)",
        "fakes": "478 geometrically real landmarks for the scene's pose and eye state",
        "proven_only_on_hardware": "FaceMesh accuracy on real drivers, glasses, IR"},
    "hand_landmarks": {
        "code": "stand-in", "runs": "sim/perception.ScriptedHands (--video: real MediaPipe Hands)",
        "fakes": "one hand box wrapped round a held phone", "proven_only_on_hardware": "real hands"},
    "object_detector": {
        "code": "stand-in", "runs": "sim/perception.ScriptedDetector (person/phone boxes; raises on detector_failing)",
        "fakes": "boxes from the scene; the model identity says backend 'simulated'",
        "proven_only_on_hardware": "Hailo-8L INT8 accuracy, latency, NPU faults"},
    "pose": {
        "code": "stand-in", "runs": "sim/perception.ScriptedPose", "fakes": "never finds a pose (pose signals unclaimed)",
        "proven_only_on_hardware": "MediaPipe Pose"},
    "hailo_decoder": {
        "code": "real+seam", "runs": "core/accel.HailoYOLO (parse, letterbox, decode, NMS, validation) on "
                                     "scripts/hw_validation/fake_hailo.py; M8.4 runs it inside the runtime",
        "fakes": "the chip: hailo_platform returns synthetic head tensors (no quantisation, timing, PCIe)",
        "proven_only_on_hardware": "the real .hef outputs, quantisation, firmware"},
    "driver_attention_logic": {
        "code": "real", "runs": "modules/driver_attention (calibration, EAR/iris, pose, gaze, trackers, risk)",
        "fakes": "nothing", "proven_only_on_hardware": "-"},
    "monitoring_state": {
        "code": "real", "runs": "core/monitoring_state + services/runtime (READY/DEGRADED/FAILED/STOPPED, role floors)",
        "fakes": "nothing", "proven_only_on_hardware": "-"},
    "alert_policy": {
        "code": "real", "runs": "core/alerts.AlertManager (episodes, cooldowns, delivery evidence)",
        "fakes": "nothing", "proven_only_on_hardware": "-"},
    "buzzer": {
        "code": "stand-in", "runs": "sim/runtime.SimBuzzerSink in place of gpio_buzzer",
        "fakes": "actuation: recorded and reported 'simulated', never 'sounded'; can be told to fail or report late",
        "proven_only_on_hardware": "the GPIO buzzer actually sounding"},
    "thermal": {
        "code": "real+seam", "runs": "core/device_health.thermal_condition + monitoring degradation/hysteresis",
        "fakes": "the reading: Faults.soc_temp_c / throttle replace sysfs and vcgencmd",
        "proven_only_on_hardware": "real temperatures, throttling, fan, frame-rate loss"},
    "storage": {
        "code": "real", "runs": "the real filesystem (state dir): events.csv, spool, saved state, LKG config; "
                               "M8.4 fills a real small volume (hdiutil) for genuine ENOSPC",
        "fakes": "nothing on macOS/Linux filesystems", "proven_only_on_hardware": "SD-card wear, power-cut "
                                                                                  "durability of a real card"},
    "uplink": {
        "code": "real+seam", "runs": "core/uplink.CloudUplink (queue, batching, retry, event keys, spool, commands)",
        "fakes": "the transport: TestClientTransport / NetworkGate / sim.faults.LinkFaults instead of requests",
        "proven_only_on_hardware": "real Wi-Fi loss patterns"},
    "network": {
        "code": "stand-in", "runs": "sim/uplink.NetworkGate (outage = refused requests)",
        "fakes": "the radio: no NetworkManager, no Wi-Fi", "proven_only_on_hardware": "Wi-Fi drop, handover"},
    "backend": {
        "code": "real", "runs": "cloud/app.py (TestClient in-process on the SimClock, or uvicorn live)",
        "fakes": "AI provider 'unavailable' by default: the deterministic fallback, no model calls",
        "proven_only_on_hardware": "-"},
    "dashboard_stream": {
        "code": "real", "runs": "core/dashboard.py local API + MJPEG /stream (live mode only)",
        "fakes": "frames are SimCamera's", "proven_only_on_hardware": "MJPEG encode cost on the Pi CPU"},
    "ble_provisioning": {
        "code": "real+seam", "runs": "services/provisioning + services/ble_pop in sim/ble_peripheral.py; the app's "
                                     "Swift onboarding code via sim/ble_harness.py",
        "fakes": "the radio: an in-memory GATT table over localhost instead of BlueZ/CoreBluetooth",
        "proven_only_on_hardware": "BlueZ peripheral, iPhone CoreBluetooth, Wi-Fi join"},
    "identity": {
        "code": "stand-in", "runs": "sim/identity.py (SD-SIM-XXXXXXXX, device key, claim code in the state dir)",
        "fakes": "the hardware serial", "proven_only_on_hardware": "the Pi serial / factory provisioning"},
    "config_boot": {
        "code": "real", "runs": "main.build_runtime, config validation, role floors, LKG fallback (core/config_lkg)",
        "fakes": "nothing", "proven_only_on_hardware": "-"},
    "health_ladder": {
        "code": "real+seam", "runs": "live: core/health.HealthMonitor + services/supervisor; stepped: monitoring "
                                     "check on the production cadence, optional supervise_inference",
        "fakes": "reboot: refused and logged (_issue_reboot); stepped mode has no restart backoff/budget",
        "proven_only_on_hardware": "systemd watchdog, Restart=, start-limit, reboot ladder"},
    "clock": {
        "code": "stand-in", "runs": "sim/clock.SimClock (time.monotonic; backend time.time via wall_clock_module)",
        "fakes": "time, in the deterministic suites only; live runs use real time",
        "proven_only_on_hardware": "RTC-less boot, NTP"},
}

#: What the device's own health report says it is simulating (sim/runtime.py).
STAND_INS = {name: f["fakes"] for name, f in FIDELITY.items() if f["code"] in ("stand-in", "real+seam")}


def summary() -> dict[str, list[str]]:
    """Component names by code kind, for evidence summaries."""
    out: dict[str, list[str]] = {}
    for name, f in FIDELITY.items():
        out.setdefault(f["code"], []).append(name)
    return out
