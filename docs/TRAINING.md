# Feeding SensorDyme data: collection, labeling, training, evaluation

How to make the detection models better with your own data — what to collect,
how to organize and label it, how to train without fooling yourself, and how to
prove a new model is better before it ships to a device.

## What the codebase already gives you

- **`data/telemetry.csv`** — continuous raw signals (EAR, yaw, pitch, PERCLOS,
  risk, motion energy…) logged at 2Hz while the device runs. This is the
  labeled-data raw material: line it up with ground truth and you have training
  rows without touching video.
- **`data/events.csv`** — every emitted event with timestamp/confidence/signals.
  Each row is a *model prediction* you can mark true/false.
- **Evidence snapshots** (`alerts.save_evidence: true`) — one annotated JPEG per
  loud alert, local-only. Your fastest source of hard positives *and* false
  positives: each file is either "yes that was real" or "no it wasn't".
- **`scripts/prepare_state_farm.py` + `import_state_farm_labelstudio.py`** — a
  worked example of the whole pipeline against the State Farm distracted-driver
  dataset (already in `data/state_farm/`), including Label Studio import.
- **Privacy defaults**: frames are never uploaded and never stored except the
  opt-in evidence path. Any collection beyond that must be an explicit opt-in
  you build UI for — never silent.

## Dataset layout

One directory per detection task, split first, class second:

```
data/datasets/<task>/                  # e.g. phone_use, fall, zone_person
  train/
    positive/   img_0001.jpg ...
    negative/   img_0002.jpg ...
  val/
    positive/ ...
    negative/ ...
  test/                                # touch ONLY for final numbers
    positive/ ...
    negative/ ...
  clips/                               # temporal tasks: short 3-10s mp4 per event
    fall_0001.mp4
    fall_0001.json                     # {"label": "fall", "onset_s": 2.4}
  manifest.jsonl                       # one line per sample: provenance + metadata
```

Every sample gets a manifest line — this is what makes datasets versionable
and debuggable:

```json
{"file": "train/positive/img_0001.jpg", "label": "phone_use", "source": "device_a3",
 "captured_at": "2026-08-05T22:14:03Z", "lighting": "night_ir", "camera_angle": "dash_left",
 "model_version": "yolo11n+hef-v2.14", "confirmed_by": "user", "notes": "sunglasses"}
```

## Labeling: formats and tools

| Task | Annotation | Format | Tool |
|---|---|---|---|
| Object detection (phone, package…) | Bounding boxes | **YOLO txt** (`class cx cy w h`, normalized) — ultralytics-native | Label Studio, CVAT, Roboflow |
| Zones/segmentation | Polygons | COCO JSON | CVAT |
| Pose (falls, posture) | 17/33 keypoints | COCO keypoints | CVAT |
| Eye/face state | Frame-level class | folder-per-class | Label Studio |
| Temporal events (fall, stillness) | Onset/offset times per clip | sidecar JSON (above) | Label Studio video |
| Event verdicts (cheapest, start here) | true/false per emitted event | extra `verdict` column on events.csv | any spreadsheet |

Start with **event verdicts**: run the device for a week, then mark every event
right/wrong. That alone measures your false-alert rate and builds the
hard-negative list — no images needed.

## Collection rules that decide whether training works

1. **Hard negatives are worth more than easy positives.** For every false alert
   the device fires, save that moment as a negative. "Sitting down" clips are
   what teach a fall detector not to cry wolf.
2. **Split by session/person, never by frame.** Frames from one video are
   near-duplicates; putting some in train and some in test is leakage and will
   overstate accuracy. Keep each recording session (and ideally each person)
   entirely inside one split.
3. **Cover the deployment matrix**: day/night-IR, each mounting angle, multiple
   people, partial occlusion, TV-in-frame, empty scenes. A model trained on one
   kitchen learns that kitchen.
4. **Freeze the test set.** Build it once, version it, and never train on it or
   tune against it. Report all decisions from val; touch test only for the final
   number of a release.

## Training and deploying (per model type)

- **YOLO (objects)**: `yolo train model=yolo11n.pt data=<task>.yaml epochs=100` —
  ultralytics is already a dependency. For the AI HAT, compile the trained
  weights to `.hef` with Hailo's Dataflow Compiler (x86 machine + calibration
  images), then point `accelerator.yolo_hef` at it. See docs/ACCELERATOR.md.
- **Thresholds (EAR, PERCLOS, velocities)**: these are config, not weights.
  Sweep them offline against telemetry.csv + verdicts (see `benchmarks/`) and
  ship winners as preset variants.
- **Temporal events**: before reaching for an action-recognition network, tune
  the EventEngine rule (persistence/confidence/occlusion) against labeled clips
  — most false-alert problems are rule-shaped, not model-shaped.
- **Never auto-train on raw user footage.** The pipeline is: opt-in capture →
  human labeling → versioned dataset → offline training → evaluation gate →
  manual deploy.

## Evaluation gate (before any model reaches a device)

For each detection, on the frozen test set:

- **Precision / recall / F1** per class, plus the confusion matrix.
- **False alerts per 24h** — the number users actually feel. Compute it by
  replaying a full day of negative-only footage/telemetry.
- **Time-to-detect** for temporal events (fall onset → alert).
- **On-device reality check**: FPS, latency, temperature on the Pi with the
  candidate model (`benchmarks/run_benchmarks.py`), because a model that wins
  F1 but halves the frame rate can lose real-world recall.

Ship only if the candidate beats the incumbent on F1 *and* doesn't regress
false-alerts/24h or FPS. Record the comparison in the dataset changelog.

## Versioning

- Name datasets `<task>-vN` and never mutate a released version — additions
  create `vN+1` (keep `manifest.jsonl` diffs).
- Tag trained weights with the dataset they came from: `phone_use-v3-yolo11n.pt`.
- The device reports its model in `/api/state` and events carry the module
  name, so field reports are traceable to (model, dataset) pairs.
- Store datasets outside git (`data/` is gitignored); track only manifests and
  eval reports.
