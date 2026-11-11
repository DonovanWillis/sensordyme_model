# Hardware dependency matrix (M6)

What each feature has been verified against **in software or simulation** on
this branch, versus what can only be established on the physical Raspberry Pi 5,
Hailo-8L AI HAT+ and Camera Module 3 Wide NoIR. Nothing in the right-hand column
has been executed. `docs/MONDAY_RUNBOOK.md` gives the exact procedure, and
`docs/HAILO_HARDWARE_VALIDATION.md` the M5 measurement package it reuses.

Key: **SW** = unit/integration test on a Mac or worker, **SIM** = replay or
simulated input, **—** = none.

| Feature | Verified in software / simulation (where) | Needs the physical device (runbook §) | Hardware involved |
|---|---|---|---|
| Service start ordering, no network dependency | SW: unit-file assertions (`tests/test_m6_autonomy_boot.py`) | `systemd-analyze verify`; real boot with no Wi-Fi (§K) | Pi, systemd on Bookworm |
| Exit-78 config refusal stops the unit (no restart, no reboot) | SW: `tests/test_m02_recovery.py`, `test_runtime_review_fixes.py::test_f7_*`; unit has `RestartPreventExitStatus=78` | Observe `systemctl status` after a deliberately bad config (§K4) | Pi |
| Bounded start-limit reboot (≤3 per 6 h, never for exit 78) | SW: guard decision logic, state persistence, dry-run, unrecordable reboot refused, unit wiring (`test_m6_autonomy_boot.py`, 18 tests) | That `OnFailure=` fires on `start-limit-hit` under Bookworm's systemd 252; the reboot happens; the budget survives reboots (§K5) | Pi, systemd |
| In-process last resort (process restart before a rate-limited reboot) | SW: `test_m02_recovery.py`, `test_health.py` | Real watchdog expiry; `sudo -n systemctl reboot` sudoers rule (§K) | Pi |
| Watchdog (`WatchdogSec=45`, READY after warm-up) | SW: health monitor pings and holds; `TimeoutStartSec` covers worst-case camera bring-up (5×12 s) + 60 s | Actual READY time with the Hailo load and the camera (§H1), wedged-loop restart | Pi, Hailo, camera |
| Camera open, verify read, retry | SW: `test_camera_open_retries.py`, `test_capture_service.py` (fake camera) | `rpicam-hello --list-cameras`, sensor mode, ScalerCrop, NoIR exposure in the cabin (§D, §I) | Camera Module 3 Wide NoIR, CSI ribbon |
| Camera calibration (intrinsics) | SW: `calibrate_intrinsics.py` tests in `test_m5_hw_validation.py` | Checkerboard capture (§I; M5 step 9) | Camera |
| Model provenance (hash pin) | SW: setup_pi.sh / config / registry pin agreement (`test_phase5_demo_readiness.py`, manifest `model_pins.consistent`); preflight refuses a mismatch (`test_m02_deploy_preflight.py`) | sha256 of the `.hef` on the device (§C3, §F2) | SD card |
| Hailo model load and inference | SW: `test_hailo_lifecycle.py`, `test_accel*.py` with fakes; no real HailoRT on a Mac | `hailortcli scan`, `fw-control identify`, `parse-hef` (six outputs), service start with `accelerator.live` (§D1, §H) | Hailo-8L, PCIe, HailoRT/driver versions |
| Legacy vs reference decoder; INT8 parity | SIM: float ONNX surrogate on the worker (`docs/DECODER_CANDIDATE.md`); decode unit tests | Raw INT8 output capture plus `compare_float_vs_int8.py` (§H2; M5 steps 6 and 12). **The reference decoder stays opt-in until this passes** | Hailo-8L |
| Processed FPS and latency per stage | SW: frame-bus statistics unit tests | `stage_timing.py health` / `stages`; camera-to-alert latency with `ms_clock.html` (§H3; M5 steps 3, 7, 11) | Pi CPU, Hailo, camera |
| Thermal behaviour, throttling, fan | SW: `core/thermal.py` parsing tests | `thermal_soak.sh` for 30 min, `vcgencmd get_throttled` (§J; M5 step 4) | Pi 5 SoC, active cooler, HAT airflow |
| Power supply adequacy (undervoltage) | — | `vcgencmd get_throttled` bits 0/16, `pmic_read_adc`, PSU rating (§B2) | 27 W USB-C PSU / vehicle supply |
| GPIO buzzer alert output | SW: pattern player and `GPIOBuzzerSink` with a fake pin; the alert policy (M0.1) | `buzzer_check.py run` plus a person recording audio; scope on BCM 17 (§G; M5 step 8). Software cannot tell audible from silent | Buzzer, wiring, BCM 17 |
| Monitoring state machine (READY/DEGRADED/FAILED, impaired alert) | SW: `test_m02_monitoring_state.py` | Real camera unplug and Hailo absence during operation (§K) | Camera, Hailo |
| Driver-attention / presets / custom rules logic | SW + SIM: module unit tests, M4 replay harness | Real in-cabin behaviour, which also needs the consent protocol (§L) | Camera, person |
| BLE Wi-Fi provisioning | SW: GATT payload sizing, long-write reassembly, credential validation, window policy (`test_ble_*.py`, `test_provisioning.py`, `test_m6_security_device.py`) with no BlueZ | Real BlueZ GATT registration, advertising, iPhone discovery, write, join, status notify (§K2) | Bluetooth radio, BlueZ, iPhone |
| Wi-Fi join, rollback, reconnect | SW: nmcli argv and failure mapping with fake `_run` (`test_wifi_*.py`) | NetworkManager on the device; AP loss and recovery (§K3) | Wi-Fi radio, AP |
| Uplink reconnection (bounded backoff and jitter) | SW: `test_m6_autonomy_uplink.py` (scripted outages) | Real AP loss and return; events delivered after the outage (§K3) | Wi-Fi, backend |
| Command authorization (id dedupe, malformed refusal) | SW: `test_m6_security_uplink.py` | — (logic only) | — |
| Dashboard auth, `/stream`, LAN exposure | SW: real HTTP server on an ephemeral port (`test_m6_security_device.py`, `test_dashboard.py`) | The phone's live feed over the real LAN with a token (§L1) | Wi-Fi, iPhone |
| mDNS advertisement (`_sensordyme._tcp`) | SW: `publish_mdns.py` record generation | avahi on the device; discovery by the app (§L1) | Wi-Fi, avahi |
| Persistent identity (`SD-XXXXXXXX`) | SW: identity derivation unit tests | `/proc/cpuinfo` Serial present on the Pi 5; the id is stable across reboot and reinstall (§C2, §K) | Pi 5 SoC |
| iPhone pairing (backend claim + device token) | SW: backend claim tests (`test_cloud_*`); app logic tests (Swift) | End-to-end with the real app on the real LAN (§L1) | iPhone, Wi-Fi, backend |
| Storage health, free space | SW: preflight disk check | `df`, SD card wear, `dmesg` I/O errors (§B3) | SD card |
| Release integrity (installed = manifest) | SW: `test_m6_security_release_manifest.py` | Device snapshot plus `release-verify` (§F3) | — |
| Developer-only Tailscale | — (not product software) | Optional, developer access only (§M). It does **not** provision Wi-Fi | — |
