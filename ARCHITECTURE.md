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
- **Which camera is which**: each device reports its `SD-` id in its health
  report; the backend keeps it as `devices.hardware_id`. The app shows a device
  its live feed only when that id matches the camera the phone linked
  (`ios/Sources/FeedLink.swift`); with an id unknown, only a one-device account
  is unambiguous enough to stream.
- **Cleartext**: plain HTTP is for the LAN only. ATS keeps
  `NSAllowsArbitraryLoads` because `NSAllowsLocalNetworking` blocks IP literals in
  public-range space (measured on a ResNet-style network), and the app enforces
  the real rule itself: `BackendResolution.cleartextPermitted` refuses to send a
  password or token over HTTP to any internet DNS name.

### Known limitation: BLE provisioning is unauthenticated

The Wi-Fi credential write is BLE "Just Works": anyone in radio range while the
window is open (offline, or 5 minutes after boot) can point the device at their
own network. Planned fix, which needs firmware + app + backend together and a
test against real hardware: the backend gives a signed-in owner a
provisioning proof derived from the device's `device_key` (which the device and
backend already share); the app includes it, HMAC'd over a nonce read from the
device-info characteristic, in the credential write; the device rejects writes
without a valid proof. Unclaimed devices would need a printed setup code.

## AI acceleration (AI HAT+ / Hailo)

`core/accel.py` detects the HAT (`/dev/hailo0` + HailoRT runtime) at boot and
verifies it before inference. With `accelerator.yolo_hef` pointing at a
compiled model, `core.detectors.get_yolo_model()` transparently returns a
Hailo-backed detector for every module. No/broken HAT → automatic CPU fallback,
with the reason in `/api/health`. See docs/ACCELERATOR.md for compiling models
— stock `.pt`/`.onnx` files cannot run on the HAT, and MediaPipe Face Mesh has
no Hailo build (stays on CPU by design).

The chip is one exclusive device and config names one `.hef`, so `core/accel.py`
keeps **one** `HailoYOLO` per `.hef` (every weights name a module asks for is a
cache key for it). Its vstreams are opened and the network group activated
once, then held for every frame; a device fault tears them down and the next
frame rebuilds them. Inference is serialised by a per-instance lock.
`core.detectors.release_all()` closes it at shutdown.

**Integrity.** The `.hef`'s sha256 is checked against
`accelerator.yolo_hef_sha256` on every load (boot, a preset switch that needs a
detector, a reload) — never per frame. A mismatch, truncation or replacement is
refused: the model never reaches the chip, `/api/health` names the expected and
actual hashes, and on a device (no CPU detector) the runtime goes DEGRADED
rather than detecting with an unexpected file. Recovery: re-provision with
`setup_pi.sh`, or update the pin if the change is intended.

## Events: confidence and model identity

`Event.confidence` is the certainty, in [0, 1], that the condition the event
names really occurred — or `None` when there is no graded evidence (a count, a
mode change). It is never a raw measurement or a threshold. Three sources,
defined in `core/confidence.py`: a model's own score; for threshold rules,
`Phi(margin / sigma)` with sigma measured live from the signal (0.5 = exactly on
the threshold); or none. Severity (log/soft/loud) is separate and lives in the
alert tiers. Nothing gates on confidence after an event exists.

Every uplinked event carries `model`: the module plus the detector file that
ran, its sha256 and backend (`DeviceRuntime.model_identity`, taken from what was
actually loaded). The backend stores it per event (`events.model`, NULL for
history) and returns it from `GET /v1/devices/{id}/events`; the app decodes it
as `EventModel` and the explanation lists it as "Model".

## Adding hardware

`hardware/hal.py` defines interfaces (IMU, GPS, IO expander, audio amp, LEDs,
buttons, ESP32 bridge) with null implementations. A new driver plugs into
`build_hardware()` and gets health monitoring for free; nothing above the HAL
changes.
