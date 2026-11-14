# Diagnostics (M10 OBSERVE): per-stage latency and developer diagnostics

Two things, with different audiences:

| | What | Who sees it | Default |
|---|---|---|---|
| `performance` in `GET /api/health` (`sd-stage-timing/1`) | per-stage latency, processed fps, memory, SoC temperature, throttle state, tagged with the host | anyone holding the device token (as `/api/health` always was) | **always on** |
| `GET /api/diagnostics` (`sd-diagnostics/1`) | detections, evidence, open conditions, capabilities, degraded reasons, model versions, the latencies | developers holding the device token | **off** |

Neither is sent upstream: the uplink's health sample and the event payload are
unchanged. Neither ever carries a frame or a secret. The consumer app shows
neither, except that the event detail now shows the event's duration and
model where the record carries them (§5).

## 1. Host tag: a Mac number is never a Pi number

Every timing block carries `platform` (`core/stage_timing.py platform_tag()`):

| `class` | Means | `note` |
|---|---|---|
| `pi` | `/proc/device-tree/model` says Raspberry Pi | measured on a Raspberry Pi |
| `desktop` | anything else (a Mac, a Linux workstation) | measured on a desktop host: NOT a Raspberry Pi measurement |
| `simulated` | the frames or the accelerator are stand-ins (`sim/`, `fake_hailo`) | SIMULATED inputs or accelerator: not a hardware measurement |

`hailo_runtime` is `loaded`, `not loaded` or `simulated` (fake_hailo). The
recording tools (§4) repeat the class in every output, print a NOTE when it is
not `pi`, and `summarize_record` sets `is_pi_measurement` only for an all-`pi`
recording. The app's diagnostics screen shows the host banner first.

## 2. `performance` (sd-stage-timing/1)

Always on, in `GET /api/health` next to `device`:

```json
"performance": {
  "schema": "sd-stage-timing/1",
  "platform": {"class": "pi", "board": "Raspberry Pi 5 Model B Rev 1.0", "machine": "aarch64", ...},
  "window_samples": 256,
  "stages": {
    "hailo_inference": {"n": 256, "p50_ms": 9.8, "p95_ms": 12.0, "max_ms": 31.0, "mean_ms": 10.1,
                        "last_ms": 9.7, "count": 48211},
    "face_landmarks": null,
    ...
  },
  "processed_fps": 29.6,
  "rss_mb": 412.3, "rss_source": "current (/proc/self/statm)",
  "soc_temperature_c": 61.2, "throttle_now": [], "throttled_raw": "0x0",
  "uptime_s": 5120.4
},
"diagnostics": {"enabled": false, "schema": "sd-diagnostics/1"}
```

(The values above show the shape only. Nothing in this document is a Pi
measurement: the Pi has not run this code yet.)

A stage with no sample yet is `null`, never `0`. `n` is the samples in the
window (the last 256), `count` the samples since the process started.
`max_ms` is the window's maximum.

| Stage | Measured | Where |
|---|---|---|
| `camera_read` | one `Camera.read()` in the capture thread (sensor + ISP + copy) | `services/capture.py` |
| `frame_age` | frame published → the inference loop picks it up | `services/runtime.py _inference_pass` |
| `detector_total` | one shared detector run (`predict`), any backend; a cached answer is not a run | `core/perception.py SharedDetector` |
| `hailo_preprocess` | letterbox + colour conversion before the chip | `core/accel.py HailoYOLO.predict` |
| `hailo_inference` | `pipeline.infer()`: host → chip → host, incl. PCIe | same |
| `decoder_cpu` | off-chip DFL decode + NMS (or the on-chip NMS output parse) | same |
| `face_landmarks` | one MediaPipe Face Mesh `process()` | `core/perception.py SharedGraph` |
| `hands` | one MediaPipe Hands `process()` | same |
| `tracking` | the shared tracker's association for one detector run | `core/perception.py detector_ran` |
| `behaviour_reasoning` | `module.process()` minus every model call and the tracker inside it (colour conversion, landmark maths, trackers, the behaviour engine) | `services/runtime.py _perceive` |
| `alert_dispatch` | the alert policy for one frame's events, up to the commands being issued | `services/runtime.py` |
| `alert_output` | alert command → GPIO pin driven, first pulse of each pattern (queue wait + GPIO write; **pin driven, not sound heard**) | `core/alerts.py GPIOBuzzerSink` |
| `frame_to_decision` | frame published → that frame's alert decision is made | `services/runtime.py` |
| `frame_interval` | between two analysed frames; `processed_fps = 1000 / mean` | `services/runtime.py` |

`soc_temperature_c` and the throttle fields are the same health sample's
`device.system.metrics` (read once, not twice). `rss_mb` is the current RSS
on Linux; on macOS it is the PEAK RSS (`rss_source` says so).

### Bounded and cheap

- One preallocated `array('d')` of 256 doubles per stage, fixed stage set: 14
  stages × 2 KiB. An unknown stage name or a NaN / negative / infinite value is
  refused, never added. Nothing grows with uptime.
- `add()` is a lock, two stores and three integer updates. Percentiles are
  computed only when someone reads `/api/health`, from a sorted copy of the
  window, on the HTTP thread.
- Unlike `device.inference.metrics` (the frame-bus interval stats), these
  windows are not drained per caller: any number of clients can poll.

Overhead, measured by `tests/test_m10_observe.py::test_the_instrumentation_is_cheap`
on the development Mac (Apple Silicon, Python 3.12; **a desktop figure, not a Pi
figure**): `add()` 0.42 µs, `snapshot()` 0.015 ms. A frame records about a
dozen samples (~5 µs), about 0.015 % of a 33 ms frame. The Pi's Cortex-A76 is
slower per call; re-measure it there with the same test before quoting a Pi
number.

## 3. `GET /api/diagnostics` (sd-diagnostics/1)

### Enabling

```yaml
dashboard:
  token: <per-device token>        # required; without it diagnostics are refused
  developer_diagnostics: true      # default false; config.yaml ships false
```

`dashboard.developer_diagnostics` must be a YAML bool (`core/config_schema.py`
refuses `"yes"`, `1`, …) and warns when it is true with no token. Restart the
service after changing it.

### Access rules (in order)

| Request | Answer |
|---|---|
| token missing or wrong (with a token configured) | **401** `invalid or missing device token`, like every `/api` path |
| switch off | **404** `{"error": "developer diagnostics are off on this device: …", "enabled": false}` |
| switch on, no `dashboard.token` configured (even on loopback, where `/api/health` would answer) | **403** `… never served without the device token …` |
| switch on, token matches | **200** the body below |
| the body could not be built | **500** a fixed sentence (no traceback, no state); details in the device log |

The token is accepted only as the `X-Device-Token` header (the `?token=`
fallback exists for `GET /stream` only).

### Body

```json
{
  "schema": "sd-diagnostics/1", "enabled": true,
  "note": "developer diagnostics: local, token-gated, never uploaded; no frames or secrets",
  "platform": {"class": "desktop", "note": "measured on a desktop host: NOT a Raspberry Pi measurement", ...},
  "snapshot_age_s": 0.2,
  "warming_up": false,
  "frame": {
    "module": "driver_attention", "frame_id": 1834, "frame_age_ms": 4.1, "simulated": false, "source": "camera",
    "detections": {"state": "ok", "fresh": true, "age_s": 0.03, "total": 3,
                   "items": [{"label": "cell phone", "confidence": 0.61, "box": [..], "class_id": 67, "track_id": 4}]},
    "face": {"face": "found", "landmarks": 478,
             "eye_state": {"ear": 0.27, "iris_openness": 0.41, "closed": false},
             "head_pose": {"yaw_deg": -4.2, "pitch_deg": 1.0, "calibrated": true},
             "gaze": {"yaw_offset_deg": 2.1, "pitch_offset_deg": -0.4}},
    "hands": {"count": 1, "ran": true},
    "evidence": {"driver_association": {...}, "hand_phone": {"phone_detected": true, "phone_confidence": 0.61,
                 "hand_overlaps_phone": false, "phone_hold": "...", "phone_handling": {...},
                 "engine_phone_evidence": {...}},
                 "state": "OK", "gaze_zone": "Forward", "abstaining": {...}, "events_fired_this_frame": []},
    "open_conditions": [
      {"source": "module_tracker", "condition": "distraction", "state": "developing", "running_s": 1.4},
      {"source": "alert_episode:module", "condition": "phone_use", "state": "active", "category": "warning",
       "running_s": 6.2, "actuations": 2}]
  },
  "capabilities": {"state": "ready", "monitoring": true, "available": [...], "impaired": [...], "paused": [],
                   "detail": {"objects": {"state": "available", "reason": null}, ...}},
  "unknown_or_degraded": ["alert_output: no alert output is configured", ...],
  "versions": {"software": "0.2", "model_version": "...", "detector": {"name": "...hef", "sha256": "...", "backend": "hailo"},
               "detector_pin_sha256": "...", "decoder": "legacy",
               "mediapipe": {"mediapipe_face_mesh": "mediapipe-0.10.14", "mediapipe_hands": "...",
                             "mediapipe_pose": "...", "installed": "mediapipe-0.10.14"}},
  "latency": { ...the `performance` block of §2... }
}
```

- `frame` is `null` and `warming_up` is `true` on the first request: the
  snapshot is only built while requests keep arriving (below).
- `detections.items` is `null` when the detector could not look (failed, stale,
  unavailable, idle) with `state`/`reason`; `[]` means it looked and saw nothing.
  At most 20 items, highest confidence first; `total` says how many there were.
- `face.eye_state` / `head_pose` / `gaze` are what the module derived this
  frame; "not measured" when it derived nothing.
- `evidence` comes from the module's `get_debug_state()` (driver_attention
  reports the driver association, the hand-phone association and the M8.5
  handling cue, which is an unvalidated heuristic and is labelled so there).
  A module without a debug state reports "not reported by this module".
- `open_conditions`: the alert policy's open episodes (`running_s` since the
  episode started) and the module's trackers in `DEVELOPING` / `CONFIRMED`
  (`running_s` = the tracker's own elapsed time).
- `unknown_or_degraded`: the monitoring state's reasons, then each capability
  that is not available with its reason, de-duplicated.

### What it costs the device

- Switch off: one attribute check per analysed frame.
- Switch on, nobody asking: the same. The snapshot is built only while a
  request has arrived within the last 10 s (`DEMAND_TTL_S`).
- Someone asking: one snapshot at most every 0.5 s (`SNAPSHOT_INTERVAL_S`), on
  the inference thread (where the module's state is consistent): one
  `get_debug_state()` call plus a bounded copy. The request itself takes a
  health sample on the HTTP thread, as `/api/health` does.
- A failure building the snapshot is logged at debug level and never costs a
  frame.

## 4. Security

- Off by default, on every shipped device (`config.yaml`
  `developer_diagnostics: false`, pinned by `tests/test_m10_observe.py`).
- Token-gated twice: the dashboard's `/api` gate (401), then a refusal when no
  token is configured at all (403), so it is never served unauthenticated, also
  not on loopback.
- No pixels: `core/diagnostics.py sanitize()` drops numpy arrays, `bytes`,
  and any key named `frame`, `image`, `jpeg`, `png`, `pixels`, `thumbnail` or
  `crop`, at any depth. The detection boxes are four numbers each.
- No secrets: keys matching `token|secret|password|passwd|psk|ssid|device_key|
  api_key|apikey|credential|cookie|auth` are dropped at any depth. The payload
  is assembled from the observation, the module's debug state, the alert
  episodes, the monitoring report and the accelerator config's
  `yolo_hef_sha256` / `yolo_decoder` only: never from `dashboard`, `cloud` or
  `network` config.
- Bounded: depth 6, 32 items per list or mapping, 240 characters per string.
- Never sent upstream: the uplink and the backend are untouched.
- Tests: `tests/test_m10_observe.py` (auth 401/403/404/200, the off switch,
  no secrets or pixels in a real runtime's body, schema, the config switch,
  a failing provider leaks nothing).

## 5. iPhone

- **Event detail** ("What the record shows"): "How long it lasted" from the
  `duration` the backend already stores and returns (`cloud/app.py _event_row`;
  the app did not read it before M10), and "Detected by" from the event's
  model identity ("damoyolo_tinynasL20_T.hef · e69136d3ebf3 on the Hailo
  accelerator"). Both rows appear only when the record carries the value. The
  contributing-signals line is shown in plain words (`SignalWords`): token
  lines such as `phone + hand_overlap` are translated, sentences are left
  alone, unknown tokens are only de-underscored. Confidence was already shown.
  No device, uplink or backend change was needed: every field already
  travelled end to end.
- **Developer diagnostics** (Settings → Live feed → Developer → Developer
  diagnostics): the section exists only when the device's `/api/health`
  reports `diagnostics.enabled: true` AND the phone holds the device token
  (`DiagnosticsAccess.available`). Shipped devices report false, so the
  consumer never sees it. The screen shows the host banner first, then
  detections with confidence and track, face and evidence, open conditions
  with running time, monitoring state and degraded reasons, per-stage latency
  in pipeline order, and versions. It refreshes every 2 s while open and stops
  on a refusal (401/403/404, each explained). Logic and wording:
  `ios/Sources/DeviceDiagnostics.swift`, tested by
  `ios/Tests/DeviceDiagnosticsTests.swift` against a real captured body.

## 6. Hardware measurement commands (prepared; run on the Pi)

See docs/HAILO_HARDWARE_VALIDATION.md "M10 additions". In short:

```bash
# service RUNNING, read-only: the runtime's own stage timings for 10 min -> JSON + CSV + provenance
SENSORDYME_TOKEN=$TOKEN $PY $V/hwv_stage.py observe --minutes 10 --interval 10 --install-dir $D --out $EV/stages
# service STOPPED: Arch A (production .hef) vs Arch B (+ a second .hef), ABBA, equal conditions
$PY $V/hwv_stage.py arch --second-hef /path/to/arch_b.hef --second-hef-sha256 <sha> --install-dir $D --out $EV/stages
# Mac dry runs (SIMULATED, exit 3; never evidence)
python scripts/hw_validation/hwv_stage.py observe --fake
python scripts/hw_validation/hwv_stage.py arch --fake --second-hef <an operator-provided placeholder file>
```
