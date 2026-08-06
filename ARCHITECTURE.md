# SensorDyme runtime architecture

How the device boots, runs, recovers, and talks to the app. (Product/algorithm
docs live in README.md; this file covers the service architecture added for
production readiness.)

## Process layout

One systemd service (`sensordyme.service`), one Python process, several
supervised threads — chosen over multiple processes because every stage's heavy
work (capture I/O, mediapipe/onnx inference, JPEG encode) releases the GIL, and
one process means zero-copy frame sharing.

```
                 ┌────────────────────────────────────────────────────────┐
 Camera Module 3 │ CaptureService (thread)                                │
 (Picamera2 /    │   owns the camera; opened ONCE; read → FrameBus        │
  cv2 on Mac)    │   read fails → retry → reopen in place                 │
                 └───────────────┬────────────────────────────────────────┘
                                 ▼  FrameBus (latest-frame slot, zero-copy)
                 ┌────────────────────────────────────────────────────────┐
                 │ Inference loop (main thread w/ --display, else thread) │
                 │   always processes the NEWEST frame (never a backlog)  │
                 │   module.process → events → alerts/logs/uplink         │
                 │   annotate+encode ONLY if viewers, at stream_fps       │
                 └───────────────┬────────────────────────────────────────┘
                                 ▼
   DashboardServer (HTTP threads): /stream MJPEG, /api/state, /api/health,
                                   /api/presets, /setup
   CloudUplink (thread):           events up / commands down; offline-tolerant
   ProvisioningService (thread):   Wi-Fi watcher + BLE provisioning window
   HealthMonitor (thread):         checks everything; restarts; watchdog
```

## Boot sequence (Pi)

1. systemd starts `sensordyme.service` (`Type=notify`).
2. `ExecStartPre` runs `scripts/preflight.py`: config validation (fatal on
   error), camera produces a real frame, AI HAT+ detection, disk space.
   Results → journal + `data/preflight.json`. Only a bad config stops boot.
3. `services/runtime.py` builds everything: accelerator configured
   (`accelerator.required: true` makes an unusable HAT fatal), camera opened
   with retries **and verified with a real read**, services started in
   dependency order, health checks registered.
4. `READY=1` is sent to systemd only once everything is actually up.
5. NetworkManager auto-joins saved Wi-Fi; no network after
   `network.provisioning.grace_s` → BLE provisioning window opens.

## Recovery ladder

1. **In place** — capture retries reads, then reopens the camera device.
2. **Component restart** — health checks failing 3× trigger the component's
   registered restart (bounded budget per `restart_window_s`).
3. **Service-level** — a crashed service thread restarts with backoff
   (supervisor), independent of every other service.
4. **Process** — the health monitor stops feeding the systemd watchdog while a
   critical component is DOWN → systemd restarts the process (`WatchdogSec`).
   A wedged-but-alive process gets caught here.
5. **Reboot (last resort)** — critical component down + restart budget spent →
   rate-limited `systemctl reboot` (config `health.allow_reboot`, sudoers rule
   from setup_pi.sh). Behind even that: `StartLimitAction=reboot`.

Every step logs with timestamps (`journalctl -u sensordyme`), and the full
picture is served at `/api/health` → the app's Settings → Device health card.

## Networking workflow

- **Every boot**: try saved Wi-Fi (NetworkManager profiles — credentials the
  app provisioned are saved automatically by `nmcli ... connect`).
- **No Wi-Fi** (new device, or moved): after `grace_s`, advertise BLE as
  `SensorDyme-XXXX` (services/ble_gatt.py). The app (BLEProvisioning.swift)
  connects, writes `{ssid, psk}`, watches status; on success the profile is
  saved, BLE closes, mDNS + dashboard serve over Wi-Fi.
- **BLE is setup/config/commands only** — frames never move over Bluetooth.
  The live feed is always Wi-Fi (`/stream` MJPEG, direct device→phone).
- **Discovery**: avahi advertises `_sensordyme._tcp`; the app's
  "Find my device" (Discovery.swift) resolves it — no IP typing.
- **Offline**: detection, local alerts, dashboard all keep working; cloud
  events queue in memory (bounded) until connectivity returns.

## AI acceleration (AI HAT+ / Hailo)

`core/accel.py` detects the HAT (`/dev/hailo0` + HailoRT runtime) at boot and
verifies it before inference. With `accelerator.yolo_hef` pointing at a
compiled model, `core.detectors.get_yolo_model()` transparently returns a
Hailo-backed detector for every module. No/broken HAT → automatic CPU fallback,
with the reason in `/api/health`. See docs/ACCELERATOR.md for compiling models
— stock `.pt`/`.onnx` files cannot run on the HAT, and MediaPipe Face Mesh has
no Hailo build (stays on CPU by design).

## Adding hardware

`hardware/hal.py` defines interfaces (IMU, GPS, IO expander, audio amp, LEDs,
buttons, ESP32 bridge) with null implementations. A new driver plugs into
`build_hardware()` and gets health monitoring for free; nothing above the HAL
changes.
