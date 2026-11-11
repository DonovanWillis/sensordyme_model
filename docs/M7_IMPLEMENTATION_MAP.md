# Mission 7 implementation map: shared perception engine and monitoring nodes

Branch `feat/m7-perception-engine` from `cfe2a32` (M6.5; last fully verified `1a83e40`).
This is a targeted seam assessment (2026-11-11), not a repository audit.

## What exists (reuse, do not rebuild)

| Seam | Today | File |
|---|---|---|
| Frame path | Camera → `CaptureService` → `FrameBus` → `DeviceRuntime._inference_pass`: commands, `module.process`, module events, custom rules, telemetry | `services/runtime.py:2061-2240` |
| Modules | One `VisionModule` at a time (`self.module`, swapped under `_module_lock`), each owning its own models | `modules/base.py`, `modules/*.py` |
| Detector | `get_yolo_model` caches **weights** per `(weights, device)`; every `.predict()` is a fresh inference; no per-frame result sharing | `core/detectors.py:22-85` |
| MediaPipe | Each module builds its own FaceMesh, Hands or Pose; nothing shared | `modules/driver_attention.py:1318,1352-1369` |
| Custom rules | `RuleEngine.observe_frame(module.get_detections())`; typed predicates, zones, dwell; all-or-nothing validation; additive on the `driver_safety` role, replacing the preset elsewhere | `core/rules.py`, `runtime.py:2022-2041,1373-1398` |
| Presets | Declarative: module, required/optional capabilities, hardware, temporal policies, settings, limitations, evidence; role floors refuse violations | `presets.yaml`, `core/presets.py`, `core/roles.py` |
| Health | Per-capability READY / STARTING / DEGRADED / FAILED / STOPPED against the role's required capabilities, reported upstream | `core/monitoring_state.py`, `runtime.py:1419-1508` |
| Commands | App → `POST /v1/devices/{id}/preset` → `desired_preset` → `/v1/ingest` returns `set_preset` → uplink → `_switch_preset_now`; the device-reported `active_preset` is the acknowledgment; sd-auth-v1 signing | `cloud/app.py:2048,3404-3779`, `core/uplink.py` |
| Prompts | Jupiter: refusal, grammar, LLM to JSON (never code), validation, draft, confirm; the device re-validates | `cloud/app.py:3041-3152`, `jupiter/` |
| Simulator | Scene in pixels; scripted FaceMesh/detector (call counters) and Hands; faults; `SimStack`, `VirtualDevice` | `sim/` |

## Answers to the nine seam questions

1. **Shared perception components:** weight caching only. No per-frame observation, no shared results.
2. **Modes on shared signals:** none. Custom rules read the active module's cached boxes, which is the only reuse.
3. **Duplicated inference:**
   - Today one module runs per frame, so cross-mode duplication is latent: any second consumer would re-run the detector and its own MediaPipe graphs.
   - `baby_monitoring` already runs extra crop passes per frame.
   - Rules re-read the same stride boxes on three consecutive frames.
4. **Preset activation:** yes, it reaches `DeviceRuntime._switch_preset_now` and rebuilds the module. Unknown presets and module-build failures are not recorded as rejections.
5. **Distracted Driving card:** this is the existing **Driver Safety** preset (`driver_safety`).
   - The row tap posts the preset optimistically and discards the response.
   - The info sheet activates with settings and polls.
   - The checkmark is `active_preset` alone, so it shows active even when monitoring is FAILED.
   - Device refusals are not shown, and the backend re-sends a refused preset forever.
6. **Incompatible presets:** refused, both by the backend (409 `role_forbids_preset`) and by the device (`preset_violation`).
7. **Custom rules:** additive only on the `driver_safety` role; elsewhere they replace the preset (AI_CUSTOM swaps in `occupancy`).
8. **Capability truthfulness:**
   - Modules without `capability_status` report objects, pose, face and hands all available.
   - `driver_attention` reports `face` unconditionally and copies `objects` into `hands`.
   - Rule validation on general roles uses a static label registry.
   - Jupiter validates against the backend's environment, not the device.
   - The tiers endpoint exists, but the app never reads it.
9. **Justified modularity:**
   - a per-frame perception engine with shared component results;
   - companion monitoring nodes that consume observations without owning a model;
   - truthful per-component health;
   - one activation path in the app;
   - device-truth capability chips.

## Design (incremental)

### Perception engine (`core/perception.py`)
- **Per-frame cache.** The engine wraps the existing loaded models; it does not replace them. `SharedDetector` wraps the model from `get_yolo_model`, and `SharedFaceMesh` / `SharedHands` wrap the MediaPipe graphs. Each keeps a per-frame result cache keyed on the engine's current frame id, so a model runs at most once per frame. It runs with the union of requested classes and the lowest requested confidence, and filters per caller. Modules call the same `.predict` / `.process` API as today.
- **Demand-driven runs.** Nodes declare demand (component, classes, minimum interval); the module's own stride calls also register. The engine runs a component proactively only when an active consumer is due, so idle components never run.
- **Typed observation per frame.** Each frame produces an `Observation` with:
  - frame id, monotonic capture time, source, and a simulated flag;
  - detections (label, confidence, box, optional track id), or **None when unavailable**, never an empty substitute;
  - face, hands and pose landmarks when run;
  - module-provided attention signals (eye state, head pose, gaze) only when actually computed;
  - per-component status and staleness.
- **Failure isolation.**
  - A component error is recorded and its output is `None`.
  - Other components continue.
  - Results older than `max_age_s` become `stale`.
  - Health and the app see the state.

### Monitoring nodes (`core/nodes.py`)
- **Contract:** id, version, preset, required and optional capabilities, hardware, settings schema, `consume(observation, now) -> events`, `health()`, bounded state, and validation state (from the registry). Experimental nodes are flagged.
- **Node kinds:**
  - **preset node:** the existing module, adapted.
  - **rules node:** the existing `RuleEngine`, fed from the shared observation.
  - **companion nodes:** consume shared observations only, with no model of their own. First: `occupancy` (count, entry/exit, zone occupancy, dwell); property and workplace presets are typed rule bundles.
- **Role safety.** On the `driver_safety` role, companions and rules are additive; nothing can remove `driver_attention`.

### Wire contract (device → backend → app)

`monitoring.perception` (device), forwarded unchanged by the backend in `GET /v1/devices` as `status.perception`:

```json
{
  "schema": "sd-perception/1",
  "simulated": false,
  "components": {
    "object_detector": {"state": "ok|stale|failed|unavailable|idle", "model": {"name": "...", "sha256": "..."},
                         "last_run_age_s": 0.12, "runs": 1234, "errors": 0, "reason": null,
                         "consumers": ["driver_safety", "rules"]},
    "face_landmarks": {"state": "ok"},
    "hand_landmarks": {"state": "idle"},
    "pose": {"state": "unavailable", "reason": "..."}
  },
  "nodes": [
    {"id": "driver_safety", "kind": "preset", "version": "...", "state": "active|degraded|inactive|failed",
     "required_capabilities": ["face", "objects"], "missing_capabilities": [],
     "validation": "synthetic_tests", "experimental": false, "reason": null}
  ]
}
```

- **Commands.** New `set_nodes` in the ingest response, from `POST /v1/devices/{id}/nodes {node, enabled, settings?}`. The device acknowledges by reporting the node in `perception.nodes` and refuses with `last_rejection {kind: "nodes"}`.
- **Device refusals.** A refused preset (role, unknown, or build failure) is reported as `last_rejection {kind: "preset", ...}`. The backend clears `desired_preset` and sets `preset_rejection`.
- **Status forwarding.** `_device_status` also forwards `last_command_id`, `policies`, `actuators` and `last_rejection`.

## Work split

| Stream | Owns |
|---|---|
| **Device** | `core/perception.py`, `core/nodes.py`, `services/runtime.py`, `modules/*`, `sim/*`, `core/monitoring_state.py`, `core/device_health.py`, `core/uplink.py`, `presets.yaml`, `core/presets.py`, `core/model_registry*`, device tests |
| **Backend** | `cloud/*`, `jupiter/*`, backend tests |
| **iOS** | `ios/*` (Driver Safety card flow, device-truth status, rejections, tiers, accessibility, the BLE proof-of-possession setup screen) |
| **Research** (running) | `research/pilot/*`, `docs/research/M7_*` |

Integration, the end-to-end simulator, the combined suite, the release manifest and the runbook stay with the integrator.

## Hardware gates (not attempted)

- Hailo throughput with shared scheduling.
- Real camera FPS and backpressure.
- Real face/hand model latency on the Pi 5.
- BLE proof of possession on hardware.
- `required` signed commands on the device.
