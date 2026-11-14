# The virtual SensorDyme device

A software Raspberry Pi. It runs the **real device runtime** (`services/runtime.py`,
built by `main.build_runtime` from the repo's `config.yaml`) and talks to the
**real backend** (`cloud/app.py`) over the **real uplink** (`core/uplink.py`,
`X-Device-Key`, `/v1/ingest`), so the iPhone app in the iOS Simulator and the
end-to-end tests can exercise the product with no hardware.

What it replaces is the hardware edge and nothing else: the camera is a
renderer, and the three model calls of the driver-attention module (MediaPipe
FaceMesh, MediaPipe Hands, the object detector) are scripted. Everything that
*decides* is production code: calibration, EAR and iris closure, head pose and
gaze, every tracker, risk fusion, alert episodes and role floors, the M0.2
monitoring state machine, explicit stop/resume, preset swaps and role
refusals, custom rules, health and status reporting.

It is marked **simulated end to end** and never claims what it does not have:

| Where | Marker |
|---|---|
| hardware id | `SD-SIM-XXXXXXXX` (health `device_id`, mDNS `id`) |
| health report (`GET /v1/devices` → `health`, device `/api/health`) | `simulated: true`, plus a `simulation` block: scenario, scenario time, scene, active faults, what is *not* simulated |
| monitoring report (`GET /v1/devices` → `monitoring`) | `simulated: true` |
| live feed (`/stream`) and every frame | a red **SIMULATED** bar and corner tag in the pixels |
| model identity on every event | detector `sim-scripted-detector`, backend `simulated` |
| alert output | `sim_buzzer`: "SIMULATED buzzer: actuations are recorded, nothing makes a sound" |
| network block | "SIMULATED: host network, no Wi-Fi radio" (or "SIMULATED network outage") |
| backend device name (from `sim.backend`) | "Virtual SensorDyme N (SIMULATED)" |

Capabilities are reported by the real module and runtime from what actually
happens to the scripted frames: a frozen camera is FAILED, a failing detector
is DEGRADED, a missing buzzer is FAILED. There is no always-green path.

---

## Run it

All commands from the repo root. The interpreter needs the device + backend
requirements (`~/.venvs/sensordyme-m01` has them); no model is downloaded.

**One command** (backend + device, throwaway data):

```bash
make sim SIM_PY=~/.venvs/sensordyme-m01/bin/python SCENARIO=tour
# = python -m sim.virtual_device --with-backend --backend http://127.0.0.1:8787 --scenario tour
```

**Separately** (persistent, restartable):

```bash
# 1. the real backend on :8787, its data in that dir, demo account, factory.json
python -m sim.backend --port 8787 --data-dir ~/.sensordyme-sim/backend
# 2. a virtual device provisioned from that factory file
python -m sim.virtual_device --factory ~/.sensordyme-sim/backend/factory.json \
    --state-dir ~/.sensordyme-sim/device --scenario drowsy --mdns
```

Stopping the device (Ctrl+C) and starting it again with the same
`--state-dir` is a **reboot**: same hardware id, same device key (so still
paired), same mode (`device_state.json`), same custom rules
(`monitoring.json`), same driver calibration.

Against any other backend (e.g. one you deployed), create the device row the
factory way and pass its key:

```bash
python cloud/manage.py create-device "Virtual car unit"     # prints device_key + claim_code
python -m sim.virtual_device --backend https://... --device-key dk_... --claim-code ABCD-...
```

Useful flags: `--list-scenarios`, `--auto-claim` (pair to the factory's demo
account, i.e. do the app's claim step for you), `--speed 2` (play the scenario
timeline faster — the device's own thresholds stay real-time, so a 4 s eye
closure played at 2x lasts 2 s and may correctly *not* alert), `--video PATH`
(see below), `--dashboard-port` / `--dashboard-host`, `--no-dashboard`,
`--fps` (default 15), `--log-level`.

### iOS Simulator

1. Start the stack (`make sim ...` or the two commands above). Note the
   printed **claim code**, **demo account** and **device token**.
2. Backend URL: a Simulator build with no `SENSORDYME_API_BASE_URL` already
   uses `http://localhost:8787` (`AppState.productionBaseURL`). If the build
   has one baked in, or you used another port, set it in the app's
   **Settings → Server → Manual server address** to `http://localhost:<port>`
   — `localhost` is the Mac from inside the Simulator.
3. Sign in as `demo@sensordyme.local` / `sensordyme-demo` (or sign up), then
   add the device with the claim code. Until it is claimed the device reports
   to the backend but appears on no account — exactly like a new Pi.
4. Watch the status card follow the scenario (`ready`, `degraded`, `failed`
   with reasons, `stopped`, `unknown`/stale), Recent Activity fill with
   events, and drive it from the app: mode switch, stop/resume, Configure with
   AI → Activate.
5. Live feed: start the device with `--mdns`, then **Settings → Live feed →
   Find my device**, and paste the printed device token into **Settings → Device token**.

**LAN discovery limits in the Simulator.** The Simulator browses Bonjour
through the Mac's mDNSResponder, so `--mdns` (`_sensordyme._tcp`, verified
visible to `dns-sd -B`) and `python -m sim.backend --mdns`
(`_sensordyme-api._tcp`, `cloud/mdns.py`) are discoverable; without the flags
nothing is advertised and you use the manual addresses above. CoreBluetooth
does not work in the Simulator, so the BLE Wi-Fi setup flow cannot be
exercised (it is also not simulated on the device side). The backend stores
only ids shaped exactly `SD-XXXXXXXX`, so for a virtual device `GET
/v1/devices` reports `hardware_id: null` (the `SD-SIM-` id is in its health
report); the app's feed linking therefore cannot match the camera by id and
falls back to what it does for an unidentified one. The iOS UI itself is
exercised by the UI tests below.

### iOS UI tests (XCUITest)

```bash
scripts/sim/ios_uitest.sh OUT_DIR [SIMULATOR_UDID] [-only-testing:SensorDymeUITests/OrynJourneysUITests/test05_Stop]
```

One command: `scripts/sim/uitest_control.py` starts the backend (port 8797),
an unclaimed virtual device and a proxy on `localhost:8787` (the Simulator
build's default address), then the `SensorDymeUITests` target
(`ios/UITests`) runs 20 journeys and an accessibility audit. Launch, sign-in,
discovery (claim through Add device), ready, stop, resume, preset browse,
activate and reject, the Jupiter prompt, policy preview, zone requirement,
rule disable/enable, monitoring status, offline, stale, recovery, alert
history, invalid responses and error messages. The tests fault the stack
through the controller's control port (127.0.0.1:8790):
- `mode=down` stops the app's server address;
- `mode=garbage` answers invalid JSON;
- the device can be killed, or restarted with any scenario.

The device always reports to the real backend, so a proxy fault is seen only
by the app. Outputs:
- `OUT_DIR/summary.md`: pass/fail per journey;
- `OUT_DIR/shots/`: a PNG per step, plus `audit_*.txt`;
- `OUT_DIR/xcodebuild.log`;
- the result bundle.

Needs about 1.5 GB free; the build products are deleted afterwards. Sign-in is typed for real, and
every other launch uses a token from a real `/v1/login` (`-orynSession`). The DEBUG-only
`-orynSignOut` starts each launch signed out.

From a **real phone** on the same Wi-Fi: run the backend with `--host 0.0.0.0`,
the device with `--dashboard-host 0.0.0.0 --mdns`, and point the app at
`http://<mac-lan-ip>:8787`.

---

## Scenarios

`python -m sim.virtual_device --list-scenarios`. Every driver scenario opens
with 12 s of an attentive driver: the real module calibrates over its first
10 s (`calibration_duration_s`), as on the Pi. The last segment holds for ever.

| Scenario | What happens | What the app should show |
|---|---|---|
| `attentive` | an attentive driver | starting → ready, protecting |
| `drowsy` | eyes closed 4 s, open, closed 4 s again | two `drowsiness` events (loud), `fatigue_warning`/`elevated_risk` |
| `phone` | looks down at a phone in hand for 6 s | `phone_use`, `eyes_down` |
| `distracted` | head turned 45° for 5 s | `distraction` |
| `absent` | driver leaves for 10 s | `attention_lost` |
| `camera_freeze` | no frames for 20 s | failed (camera), `monitoring_impaired` event, then ready |
| `frame_drops` | every other frame lost for 30 s | stays ready |
| `camera_stall` | frames in bursts with 4 s gaps for 30 s | flips to failed in the gaps, then ready |
| `detector_failure` | detector throws for 40 s | degraded (objects impaired, face monitoring kept), then ready after the module's own retry backoff |
| `backend_outage` | backend unreachable for 60 s | stale → `unknown` after 30 s; device keeps alerting locally; fresh again, queued events delivered once |
| `model_failure` | the module cannot be built at boot | failed, `module` capability impaired with the cause; device stays reachable |
| `no_buzzer` | no audible output on a driver_safety device | failed: `audible_alert` |
| `reboot` | process restart at 40 s (live mode) | brief silence, same device, same mode |
| `tour` | drowsy, phone, distracted, detector failure, camera freeze, outage — each with recovery | everything above in ~3 min |

Scenes and faults are plain data (`sim/scenario.py`); a test or a REPL can
build its own timeline, or force one with `ScenarioState.scene_override` /
`fault_overrides`.

### `--video PATH`

Frames come from a video file (looped, SIMULATED banner added) and the module
runs its **real** models instead of scripted perception. On a Mac the
detector backend is absent by design (ultralytics is AGPL and not installed;
no Hailo), so the device honestly reports DEGRADED for object detection while
face monitoring runs on real MediaPipe.

---

## Deterministic end-to-end tests

```bash
make sim-e2e SIM_PY=~/.venvs/sensordyme-m01/bin/python
# = pytest tests/test_sim_e2e.py tests/test_sim_units.py tests/test_sim_seams.py   (~8 s, 50 tests)
```

`sim.harness.SimStack` starts `cloud/app.py` in-process (FastAPI TestClient,
own database), a virtual device whose uplink posts to it, and puts both on one
`SimClock`: the device's `time.monotonic()` and the backend's `time.time()`.
`device.advance(seconds)` runs, every 50 ms of sim time, one camera frame and
one production inference pass (`DeviceRuntime._inference_pass`), the
monitoring check every 2 s and an uplink cycle (`CloudUplink.sync_once`) every
poll interval. A 30 s outage takes milliseconds and every run is identical.
Assertions use only the app's API: `GET /v1/devices` (`status`, `health`,
`monitoring`), `GET /v1/devices/{id}/events`, and refusal codes.

```python
from sim import scenario
from sim.harness import SimStack

with SimStack(tmp_path) as stack:
    device = stack.boot(scenario.get("drowsy"))
    stack.claim()
    device.advance(20.0)
    assert stack.status()["monitoring_state"] == "ready"
```

Covered: first boot unclaimed → claim → starting → ready; one drowsiness
episode visible once (including a lost-response retry); phone / distraction /
absence through the real module; preset switch applied, role-forbidden preset
`409 role_forbids_preset`, a forbidden preset queued before the device
reported its role dropped with `status.preset_rejection`; confirmed stop →
stopped (command id echoed), nothing alerts, resume → protecting; camera
freeze → failed with reasons + impaired signal → ready; drops vs stalls;
detector failure → degraded → ready; model failure and missing buzzer →
failed; backend outage → stale/unknown, local alerting, reconnect, queued
alert delivered once; restart keeps identity, pairing, mode, rules and
calibration; custom rule draft → activate (stale draft `409
draft_superseded`) → running additively beside driver safety → fires;
unsupported prompt refused; a rule the device cannot run reported in
`status.rules.rejected` with the previous rule kept.

### Reliability invariants and stress (M6.5)

`sim/invariants.py` `InvariantMonitor(stack)` hooks every device the stack boots and
checks, after every monitoring check, command application and uplink cycle (device
side) and every uplink cycle (app side, `GET /v1/devices`):

| Invariant | Holds when |
|---|---|
| `failed_not_protecting` | FAILED never reports `protecting` or `monitoring_active: true` |
| `stopped_not_ready` | an explicit stop always reports `stopped`/`stopped`, never `ready` |
| `stale_not_healthy` | a stale device is `unknown`/`unknown`/`null`, offline, health withheld |
| `rejected_rule_inactive` | the running rule set is always one the device accepts |
| `no_duplicate_effect` | one protection command id = one state change; one alert = one stored row |
| `driver_floor` | a protecting driver_safety device runs driver_attention; never `ai_custom` |
| `queue_bounded` | the uplink backlog never exceeds its capacity |

`tests/test_sim_reliability.py` (multi-stage scenarios: offline boot, replayed /
overwritten / lost / malformed commands, an old backend's role-forbidden preset,
invalid policies, two rules, network flapping, close episodes, detector → camera →
actuator faults, damaged state files, malformed config, 20 min with faults, the
production backoff) and `tests/test_sim_stress.py` (backlog, retry storm,
producer/consumer race, the three production threads together, restart leaks, a
misbehaving link) assert it, each capped in iterations and wall time.
`VirtualDevice(honour_backoff=True)` syncs on the production reconnection backoff
(`CloudUplink.next_delay`) instead of the fixed poll interval: with the fixed
cadence a flap period equal to the poll interval phase-locks every sync into a
down window, which production's jitter does not.

---

## Simulated vs real

| Part | Virtual device |
|---|---|
| Runtime, inference loop, monitoring state, episodes, role floors, stop/resume, presets, custom rules, health/status reporting | **real** code |
| driver_attention decision logic (calibration, EAR/iris, pose, gaze, trackers, risk) | **real** code |
| Backend, auth, claim, ingest, status contract, events, AI-configure fallback | **real** `cloud/app.py` |
| Uplink (batching, retry, idempotent event keys, commands, device key) | **real** `CloudUplink` (transport swapped in tests / outages) |
| Device local API + MJPEG `/stream` + token auth | **real** `core/dashboard.py` |
| Config validation, role floors, preset restore, boot-FAILED | **real** `main.py` (`build_runtime`) |
| Camera | simulated (rendered scene, scripted faults) |
| FaceMesh / Hands / object detector | scripted (`--video` uses the real models) |
| MediaPipe Pose (baby monitoring) | scripted to find **no** pose: pose-based signals never fire in the simulator |
| Room scenes | `Scene.people`: standing people at given positions (sim/perception.py), for counting, tracking, areas and dwell |
| Buzzer | simulated (recorded, silent) |
| Network outage | simulated at the uplink transport |
| Reboot | process restart of the virtual device; the host is never rebooted |
| `shared_perception`, `occupancy` (Property), `baby_monitoring` (Child) on a `general` role | real code on scripted perception (M8) |
| Other modules (fall detection, …) on a `general` role | real code, **no** scripted perception: on a Mac they honestly report DEGRADED/FAILED |

### Still needs the physical Pi

Camera Module 3 (exposure, IR, motion blur, real faces and glasses), the
Hailo-8L INT8 detector and its decoder (`docs/HAILO_HARDWARE_VALIDATION.md`),
real FaceMesh accuracy on real drivers, BLE provisioning and the Wi-Fi
handover, the GPIO buzzer actually sounding, systemd watchdog / restart /
reboot ladder, thermals and throttling under load, SD-card durability, and
avahi advertising on a real LAN.

---

## Production-code seams

Added for the virtual device; every default is the previous code, pinned by
`tests/test_sim_seams.py` and the existing suites:

| Seam | Default (production) |
|---|---|
| `core/uplink.py` `CloudUplink.sync_once()` | the old loop body; `_run()` loops it |
| `services/runtime.py` `_inference_pass(loop, stop)` + `_InferenceLoopState` | the old loop body; `_inference_loop` loops it; frame wait 0.5 s |
| `services/runtime.py` `_build_module(name, cfg)` | `get_module_class(name)(cfg)` |
| `services/runtime.py` `_make_uplink(cfg)` | `CloudUplink(cfg)` |
| `services/runtime.py` `DeviceRuntime.monitoring_path` | `None` → `load_or_empty()` |
| `main.py` `build_runtime(argv, runtime_cls=, module_factory=, sink_overrides=)` | `run()` = `build_runtime(argv).run()`, all keywords `None` |
| `main.py` `build_alert_manager(..., sink_overrides=None)` | unchanged sink construction |

## Known limitations

- One virtual device per process (`core/device_state` takes its path from the
  environment). Run several processes with different `--state-dir`s.
- Live mode samples health every 5 s (`_HEALTH_SAMPLE_S`), so the status card
  can lag a scene change by a few seconds — as on the Pi.
- The deterministic path runs the monitoring check and uplink on the
  production cadence but not the `HealthMonitor` component ladder (camera
  restart, watchdog); the live path runs all of it.
- On a first boot (no stored calibration) `driver_attention` reports
  `starting` (`attention: starting`) during its 10 s gaze calibration, because
  distraction, eyes down/up, glancing and phone use cannot fire until it
  settles (M8). A restart loads the stored calibration and skips it.
- For a module that cannot be built, the top-level `status.reasons` name the
  lost capability (`face: not provided by this device`); the cause is on
  `status.capabilities.impaired[name=module]`.

## The real iPhone app against the virtual device (verified 2026-11-11)

`scripts/sim/ios_e2e.sh [OUT_DIR] [SIMULATOR_UDID]` (with `SIM_PY` pointing at a Python
that has the repo's requirements) reproduces the end-to-end run: real backend
(`sim.backend`), virtual device, a real `/v1/login` + claim, a Debug build of the app,
and six screenshots driven by DEBUG-only launch arguments
(`ios/Sources/DebugLaunch.swift`: `-orynSession <token> <email>` signs in with that
session, `-orynNoPrompt` skips the notification-permission prompt; Release builds
compile both out). Observed in the Simulator:

| Step | What the app showed |
|---|---|
| ready | ONLINE · ALL CLEAR · Driver Safety active · "NOT YET VALIDATED" · forbidden presets disabled with the backend's role reason |
| stop via `/protection` | ONLINE · STOPPED (amber) |
| "stays inside the marked area for longer than 10 seconds" | typed dwell policy (person, area, 10 s), "Draw the area on the device to activate this" |
| "a stranger I don't recognize" | refused: identity recognition not possible (face recognition deliberately not shipped) |
| device process killed, 30 s+ | "isn't reporting … status is unknown" · OFFLINE · STATUS UNKNOWN, nothing green |
| device restarted (same state dir) | same hardware id, ONLINE · ALL CLEAR |

Build note: an unsigned Simulator build (`CODE_SIGNING_ALLOWED=NO`) has no Keychain, so
the session token is silently dropped ("missing bearer token"); build ad-hoc signed
(`CODE_SIGN_IDENTITY=-`). Not covered by this run: tapping UI controls (there is no
XCUITest target), BLE, the live MJPEG feed (needs `--mdns` + device token), push
notifications.
