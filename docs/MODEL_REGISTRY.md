# Model registry

What SensorDyme runs, where each model's bytes come from, who may use them, and
what has been measured about them. The registry is the evidence layer under the
presets and the app.

- Data: `core/model_registry.yaml` (tracked; `models/` is gitignored and holds only the binaries).
- Code: `core/model_registry.py`. It is pure: yaml, the standard library, and the
  pure `core.roles` and `core.presets`. The backend can import it the same way
  the device does.
- Tests: `tests/test_model_registry.py`.

## Why it exists

Three layers already answer three different questions. None of them answers the
fourth.

| Layer | Question it answers |
|---|---|
| `core/capabilities.py`, `core/taxonomy.py` | Can SensorDyme watch for this at all (SUPPORTED / PARTIAL / PLANNED / UNSUPPORTED)? |
| `core/monitoring_state.py` | Is each capability producing results right now (available / impaired / starting)? |
| `core/roles.py` | Which capabilities must this deployment never lose? |
| **`core/model_registry.py`** | **Has the model behind each capability ever been shown to be right, and on what evidence?** |

`monitoring_state.py` says: "Available means the component is producing
results now, not that the results are right." Without a registry, the app had
nothing else to show, so a model that loaded looked as good as a validated one.
The registry keeps those two facts separate.

## The model card

One card per model, as frozen dataclasses (`ModelCard` and its parts):

| Field | Content |
|---|---|
| `id`, `version`, `family`, `task` | Identity. `version` names the exact build (for example, `hailo-model-zoo-v2.15.0-hailo8l`). |
| `artifacts[]` | `{path, sha256, url, size, format, notes}`. A shipping model must pin a sha256 on every artifact. |
| `vocabulary`, `classes[]` | Each output class: `{source, index, label, capability}`. `source` is the model's own name, `label` is the SensorDyme label rules use, and `capability` is the capability kind it feeds. |
| `provides_capabilities` | Perception kinds only: `face`, `objects`, `hands`, `pose`. |
| `compute` | `targets` (for example `hailo8l`, `pi5-cpu`, `dev-mps`), `published_fps[]` with `{value, hardware, conditions, source}`, and `measured[]` with `{device, metric, value, evidence}`. Published and measured numbers stay separate: a vendor's x86-host figure is never a Pi figure. |
| `licence` | `{code, weights, training_data, commercial_status, notes}`. `commercial_status` is one of `cleared`, `cleared_with_caveats`, `restricted`, `non_commercial`, or `unreviewed`. |
| `evaluation` | `{status, evidence[], shipping_exception?}`. See below. |
| `known_limitations[]` | Plain sentences that the app and docs can show. |
| `deployment` | `shipping`, `dev_only`, or `candidate`. |

Two top-level maps connect cards to the product:

- `providers` gives the shipping model behind each perception kind on a device.
  There is one provider per kind, because every module shares the single
  detector from `core/detectors.py` and the same MediaPipe graphs.
- `modules` gives the kinds each module uses in its default configuration, plus
  `optional` kinds that only a config switch enables.

## The evaluation ladder

```
unevaluated  <  benchmarked_offline  <  evaluated_in_domain  <  validated_on_device
```

`EvaluationStatus` members compare by rank, not alphabetically, and
`weakest()` returns the lowest rung.

| Rung | Means | Evidence kind that reaches it | Extra requirements |
|---|---|---|---|
| `unevaluated` | Nothing measured. A model that loads and runs is here. | `load_check` (can be recorded, but supports nothing) | none |
| `benchmarked_offline` | Measured on public data. A surrogate backend is allowed, for example the float ONNX of a `.hef`. | `offline_benchmark` | `manifest_id`, `split` (not `train`), and metrics |
| `evaluated_in_domain` | Measured with the harness on SensorDyme's own consented footage. | `in_domain_harness` | `consented: true` |
| `validated_on_device` | Measured on the Pi + Hailo hardware package. | `on_device_hardware` | `device`, `granularity: event`, and an `artifact_sha256` equal to one of the card's pinned artifacts |

Rules:

1. **A status claim needs evidence.** A status above `unevaluated` needs at least
   one evidence item whose kind reaches that rung and that was measured on the
   shipped configuration (`shipped_config` defaults to true). An item for an
   undeployed configuration, such as the DAMO-YOLO decoder fix, is recorded so
   the gap stays visible, but it supports no claim.
2. **"Validated" means one thing.** `ModelCard.validated` is true only at
   `validated_on_device` with event-level on-device evidence. That evidence must
   come from the exact pinned binary, so a new artifact sha256 makes the claim
   fail validation until the model is re-validated.
3. **Loading never changes status.** No function takes runtime facts as
   evidence. A `load_check` item reaches only `unevaluated`. `display_state()`
   lets liveness lower what the app shows (impaired, starting) but never raise
   it.
4. **A binary the registry doesn't know gets no status.** If the device reports
   a detector sha256 that matches no artifact on the provider's card, the link
   becomes `provided_by: "unregistered"` at `unevaluated`.

### Recording evidence

Add an item under `evaluation.evidence` with:

- `kind`
- `report_id`: where the report lives, for example `detector_bench/bench_val:damo_t.production`
- `manifest_id` and `split`: which frames were measured
- `protocol`: how they were scored
- `backend`: for example `onnx-fp32 surrogate` or `hailo`
- `granularity`: `image`, `frame`, or `event`
- `metrics`: a mapping of numbers
- `date`: an ISO date, in quotes
- `device` and `artifact_sha256`, when the measurement ran on hardware

Then raise `status` only as far as that evidence reaches. The test suite loads
the file and runs full validation, so an overclaim fails CI.

### Shipping exceptions

The validator refuses a shipping model below `benchmarked_offline`. Three
MediaPipe models already ship with no evaluation: FaceMesh, Hands, and Pose.
They need a decision, not a silent pass. Each carries a dated
`shipping_exception` with `{reason, recorded, review_by}`:

- The exception is valid only on a shipping model below the floor.
- It is reported downstream as `shipping_exception: true`.
- `registry_violations(..., today=...)` refuses it after `review_by`, and the
  file test passes today's date. When the review date passes, CI fails until
  someone evaluates the model or writes a new reason. Today's exceptions expire
  on 2027-02-28.

### What validation refuses

`registry_violations()` returns every problem. `load_registry()` raises
`RegistryError` with all of them. It refuses:

- A missing licence field (`code`, `weights`, `training_data`, or `commercial_status`), or an unknown commercial status.
- A sha256 that is not 64 lowercase hex characters, on an artifact or on evidence.
- A shipping artifact without a pin, or a shipping card without artifacts.
- A shipping model below `benchmarked_offline` without a valid, unexpired exception.
- A status claim without supporting evidence, including evidence that is only a `load_check`, only for an unshipped configuration, or measured on the `train` split.
- Incomplete evidence: no metrics, no manifest or split, a malformed date, or an unknown kind.
- In-domain evidence without consent. On-device evidence that is not event-level, names no device, or names a binary not on the card.
- An unknown capability kind in `provides_capabilities`, in `classes`, in `modules`, or in `providers`. Also a runtime kind such as `camera` claimed by a model.
- On a shipping card: GPL or AGPL code or weights, non-commercial terms (`NC`, "non-commercial"), or a `commercial_status` other than `cleared` or `cleared_with_caveats`. LGPL is not treated as GPL.
- A provider that is unregistered, not `shipping`, or does not provide that kind.
- Duplicate ids.

## What is registered today (registry_version 2026-11-11.1)

| Model | Deployment | Provides | Status | Note |
|---|---|---|---|---|
| `damo_yolo_tinynasl20_t_coco80` (Hailo zoo v2.15.0 `.hef`, sha256 `e69136d3…`) | shipping | objects | **benchmarked_offline** | Open Images V7 val (manifest `b878e8429d221f3f`), float ONNX surrogate, as the device decodes it: phone AP50 0.728, person AP50 0.656. Decoder fix (offset 0 + multilabel), not shipped: 0.795 / 0.713. State Farm on the Pi + Hailo: phone recall 0.883 / precision 0.771 at 0.20 (image-level, public data). The 124 fps figure is Hailo's, from an x86 PCIe Gen3x4 host; the measured Pi latency is 22.07 ms. |
| `mediapipe_face_mesh` (0.10.14) | shipping | face | unevaluated + exception | Model card: Apache-2.0, "not intended for human life-critical decisions". |
| `mediapipe_hands` (0.10.14) | shipping | hands | unevaluated + exception | |
| `mediapipe_pose` (0.10.14) | shipping | pose | unevaluated + exception | Used by presets that a driver_safety-role device cannot select. |
| `ultralytics_yolo11n_coco80` | dev_only | objects | unevaluated | AGPL-3.0. Validation refuses it as shipping. |
| `damo_yolo_t_exp08_person_phone_hand` | candidate | objects, hands | unevaluated | EXP-08 slot. No artifact, licence review pending. |

The MediaPipe hashes come from the copies of the `.tflite` files in the macOS
arm64 `mediapipe==0.10.14` wheel. The Pi's aarch64 wheel has not been hashed
yet. Do that before any runtime check enforces these hashes.

## Presets, roles, and custom policies

`preset_view(registry, "driver_safety:strict")` follows preset → module
(`presets.yaml`) → kinds (`modules`) → models (`providers`) → status. A variant
only retunes thresholds, so it shares its preset's chain. An unknown preset, or
a module the registry doesn't describe, raises `KeyError`, because an
undescribed module must not read as "needs nothing".

`ChainView.weakest` is the lowest rung over the required links that a model
backs. Optional links are listed but do not lower it. `ChainView.validated` is
true only when every required model-backed link is validated, and at least one
exists.

| Subject | Models | Weakest |
|---|---|---|
| preset `driver_safety` | DAMO-YOLO, FaceMesh, Hands | unevaluated |
| preset `elderly_monitoring` | Pose | unevaluated |
| preset `baby_monitoring` | DAMO-YOLO, Pose | unevaluated |
| preset `security_monitoring` | DAMO-YOLO (Pose optional) | benchmarked_offline |
| role `driver_safety` (requires face, audible_alert) | FaceMesh | unevaluated |
| roles `occupancy`, `property` (require objects) | DAMO-YOLO | benchmarked_offline |
| roles `general`, `child_monitoring`, `workplace` | none (camera + inference only) | none |

No preset is validated today, and none can be until the hardware package
produces event-level evidence.

**Custom monitoring policies.** The runtime already maps each AI-configured
rule's labels to capability kinds (`services/runtime.py` `_label_needs`, through
`core/capabilities.py`). `policy_view(registry, rule_id, kinds)` gives that
rule the same chain, so an accepted custom rule ("alert me if the dog is on the
couch") reports `benchmarked_offline` on the detector. It does not inherit the
status of any preset.

The registry informs and never gates. It refuses no rule and no preset. Gating
remains a product decision; one example would be a role that declares a
minimum status. If that decision is made, it belongs in `core/roles.py`, beside
`required_capabilities`.

## Re-evaluation when a card changes

`reevaluation_impact(old, new)` compares two registries and lists, for each
added, removed, or changed card and for each provider swap, the presets and
roles whose chain runs through that model. Any field change counts, because the
registry cannot tell a cosmetic edit from a new binary. With today's file:

- A new detector binary affects `baby_monitoring`, `driver_safety`, and `security_monitoring`, and the roles `occupancy` and `property`.
- A FaceMesh change affects `driver_safety` (preset and role).
- A Pose change affects `baby_monitoring`, `elderly_monitoring`, and `security_monitoring` (optional use).
- A dev-only model change affects nothing on the device.

Run it in review against `git show <base>:core/model_registry.yaml` to get the
re-evaluation list for a pull request.

## Integration plan (not wired yet)

The plan is deliberately left unwired in this branch. `MonitoringReport` is a
wire format that the app status contract (branch `feat/app-status-contract`)
already consumes. Adding a field there, or to `/api/health`, would change that
contract from a branch that does not own it. The registry is pure, so the
smallest integration is on the backend and needs no device change.

### Backend: one new block on each device in `GET /v1/devices`

This is a sibling of the existing `status.capabilities`. That block keeps its
meaning (liveness only: `available`, `impaired`, `starting`) and is not changed.

```jsonc
"status": {
  // ... existing fields from docs/APP_STATUS_CONTRACT.md ...
  "capability_evidence": {
    "registry_version": "2026-11-11.1",
    "preset_known": true,
    "weakest_evaluation_status": "unevaluated",
    "validated": false,
    "capabilities": [
      { "capability": "face", "model_backed": true,
        "provided_by": "mediapipe_face_mesh", "model_version": "mediapipe-0.10.14",
        "evaluation_status": "unevaluated", "validated": false,
        "shipping_exception": true, "optional": false, "note": "",
        "live_state": "available", "display": "available_not_validated" },
      { "capability": "objects", "provided_by": "damo_yolo_tinynasl20_t_coco80",
        "evaluation_status": "benchmarked_offline", "display": "available_not_validated" },
      { "capability": "camera", "model_backed": false, "display": "available" }
    ]
  }
}
```

`core.model_registry.capability_evidence()` already builds this block.
`cloud/app.py` `_device_status` would call it with values the device already
reports:

| Input | Source |
|---|---|
| `preset` | the monitoring report's `active_preset` |
| `role` | `core.roles.ROLES[...]`, from the reported role |
| `live` | the state of each capability, from the report's `capabilities`. Leave it empty when stale. |
| `running_sha256` | `{"objects": report["model"]["detector"]["sha256"]}`, when present |

How each field behaves:

- **Liveness:** when `stale` is true, `live_state` is null and `display` is
  `not_reported`. Following the contract's rule, an old `available` is never
  repeated. `evaluation_status` is kept, because it describes the
  configuration, like `allowed_presets`.
- **`display` values for the app:**
  - `validated`
  - `available_not_validated` ("Running — not yet validated")
  - `starting`
  - `impaired`
  - `available` (for kinds no model backs)
  - `not_reported`

  The app never derives "validated" from anything else.
- **Custom rules:** the backend can attach `policy_view(...).to_dict()` to each
  rule in its activation response, so the user sees the status of the chain
  their rule depends on.
- **Older apps** ignore the new key. Apps built for it treat a missing key (an
  older backend) as "no evidence information", never as validated.

### Later, optional

1. **Device:** add `model_registry.as_dict(load_registry(validate=False))` to
   `/api/health` as a read-only `model_registry` block. Only do it once the
   contract owner agrees. Use `validate=False` so a bad registry file cannot
   stop a device. CI already validates the file.
2. **Device:** report the MediaPipe `.tflite` hashes, and the hashes from the Pi
   wheel. The `running_sha256` check then covers face, hands, and pose, not
   only objects.
3. **`setup_pi.sh` / preflight:** read the `.hef` pin from the registry instead
   of repeating it, keeping one source for the pin. Today the registry test
   checks that `config.yaml`'s `yolo_hef_sha256` matches the card.
4. **Evidence:**
   - Run the in-cabin pilot (consented) and record `in_domain_harness` items.
   - Run the hardware package and record event-level `on_device_hardware`
     items against `e69136d3…`.
   - Then, and only then, move DAMO-YOLO, and later FaceMesh, up the ladder.
