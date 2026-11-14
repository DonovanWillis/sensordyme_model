# Real-video evaluation (Mission 4, Checkpoint B)

`evaluation/` turns "is it accurate?" into numbers that can be reproduced and
compared. Code: `evaluation/{manifest,annotations,replay,metrics,run}.py`.
Tests: `tests/test_eval_harness.py`.

**Status:** the harness exists and is tested on synthetic clips. No real-video
number exists yet. That needs an approved dataset (see licensing below), or our
own consented recordings.

## Pipeline

1. **Manifest** (`python -m evaluation.manifest build|verify`). Every file of a
   dataset version, with:
   - its sha256, subject, session, split, conditions (lighting, IR, camera,
     mount, eyewear) and annotation file;
   - the dataset's licence class, plus consent and retention for our own
     recordings.

   The manifest id is a hash of all of it. Splits are assigned per **subject**
   and never moved once set. A subject or session in two splits, a duplicated
   file, a near-duplicate across splits (dHash), or a passed retention date is
   refused.
2. **Annotations** (JSON per clip). Event intervals for event-level scoring,
   binary state spans for frame-level scoring, and ignore spans (calibration,
   occluded camera) that count as neither error nor evaluated time.
3. **Replay.** The clip runs through the real module:
   - each frame is timestamped from the clip (frame index / fps), and that time
     is handed to `set_frame_time` (the M0.2 seam);
   - `time.monotonic` is replaced by the clip clock for the replay;
   - `--target-fps` drops frames to the device's processing rate;
   - frames are resized to the device's 640×480.

   The same clip and config give the same events on any host.
4. **Metrics.**
   - Per event type: recall, precision, F1, false alarms per hour, alerts per
     hour, detection latency (p50/p90), and repeats (episode reminders, not false
     alarms).
   - Wilson intervals for proportions, and subject-cluster bootstrap intervals
     for recall and false alarms per hour.
   - Strata by every condition key.
   - Frame-level confusion for labelled states.
5. **Report** (`python -m evaluation.run ... --out DIR`). `report.json` and
   `predictions.jsonl` record the commit, config sha256, overrides, manifest id,
   split, replay settings, backend actually used, and host.

## Rules the code enforces

- `--split test` needs `--frozen-test` and refuses `--set` overrides.
  Thresholds are tuned on `val` only.
- Evaluation needs licence class `own_collection`, `commercial_training` or
  `evaluation_only`. `research_only` data is refused.
- A file whose hash changed since the manifest was built is refused.
- Off the Pi, object detection is a **desktop surrogate** (ultralytics), not the
  Hailo `.hef`. The report says which ran. Shipping-path numbers come only from
  replay on a Pi 5 + Hailo-8L.
- Host per-frame cost is reported as the host's cost, never as Pi latency.

## Mission 10: leakage-proof, event-level workflow for in-cabin recordings

Built on the pipeline above (nothing replaced). Tests: `tests/test_eval_m10.py`.
Protocol additions and the data needed for first numbers:
`docs/research/M10_EVAL_READINESS.md`.

**Status:** ready and tested on synthetic annotations, predictions and clips.
**No in-cabin recording exists, so every report today prints NO DATA.**

| Piece | Where | What it does |
|---|---|---|
| Scenario coverage matrix | `evaluation/schemas/incabin_scenarios_v1.yaml`, `evaluation/scenarios.py` | 27 cells (occupants, seating, eyewear incl. IR-pass/IR-block sunglasses, lighting day/dusk/night IR/glare, phone hand/lap/ear/mounted/passenger, head orientation, vibration, occlusion hand/hat/lens, driver absence, blink vs acted microsleep vs sustained closure, mirror/cluster/passenger glances). Each cell: tags a clip must carry, positive / negative / boundary event types, minimum subjects/clips/events/exposure hours for a claim, behavioural cases (B1–B8), a recording checklist and safety notes. A matrix that weakens the safety block (stationary only, consenting adults, no children, no driving, acted drowsiness) is refused. |
| Annotation schema v2 | `evaluation/schemas/incabin_v2.json`, `evaluation/annotations_v2.py` | Adds clip identity (subject, session, vehicle, recording date, seat, other occupants), scenario cells and tags, annotator id, and `uncertain` (unlabelable) spans per event type. v1 files migrate from the manifest item's metadata, or are refused if it lacks vehicle or date. |
| Double annotation | `evaluation/agreement.py` | Event IoU pairing, agreement F1, onset difference p50/p90/max, state Cohen's κ; pass/fail thresholds in the schema. |
| Leak-free splits | `evaluation/leakage.py` | Split unit = subject, linked with every subject sharing a session, a vehicle-day or a clip (passengers). Refuses any of these crossing splits, near-duplicate frames (dHash over several frames per video), and overlap with a training manifest. |
| Test lock | `evaluation/leakage.py` | `lock` writes a sha256-stamped lock of every assignment and the test membership (once; never overwritten). Any later move, any new test item or any new item sharing a test person/vehicle-day is refused; removals (consent withdrawn) are allowed and reported. Scoring test needs `--frozen-test --purpose ...`; each access is appended to `<lock>.access.jsonl`. |
| Metric suite | `evaluation/event_metrics.py` | Per event type: recall, missed (abstained vs silent), latency p50/p90/max and latency minus the configured sustain time. Per alert type: false alerts per exposure hour (overall and by subject), precision, unknown rate (time the module reported the alert not evaluable), alerts during legitimate behaviour, boundary alerts. Subject-clustered bootstrap CIs (`metrics.cluster_bootstrap`) and Wilson intervals. A **power check** gates every metric. |
| Report | `python -m evaluation.report` | `eval_report.md` + `eval_report.json`: overall and per cell, NO DATA / INSUFFICIENT DATA printed explicitly; records commit, config sha256, model artifact sha256s (or "missing on this host"), split lock sha256, test-access record, matrix sha256, manifest id and the replay run's provenance and backend. |

Changes to the existing pieces:
- `replay.py` records, per processed frame, which alert types the module said it
  could not evaluate (`capability_status()` × `event_requirements()`: calibrating,
  detector not started, impaired). `ReplayResult.unavailable_spans()`.
- `run.py` writes `availability.jsonl` and, with `--split-lock`, verifies the lock
  and the leakage check. `--split test` now also needs `--split-lock` (CLI).

**Scoring rules** (from the matrix's `scoring` block):
- *Positive* events (long_closure, eyes_off_road, phone_use, driver_absent,
  camera_occluded) are caught by the first listed alert in
  `[start - early_s, end + late_s]`; latency is measured from the annotated
  onset. One alert may catch concurrent positives.
- *Negative* events (mirror, cluster, infotainment, passenger glances; a
  passenger's phone) are legitimate: an alert in their window is a false alert.
- *Boundary* events (short closures, phone at the ear, face occlusion, yawns,
  nods, eating) are outside the validated scope: alerts there are tallied
  separately, never as true or false.
- A missed event is *abstained* when every alert that could catch it was
  reported unavailable for at least half its span; otherwise it is a silent miss.
- Ignore spans, and uncertain spans for the relevant types, are neither errors
  nor exposure time.

**Power check.** A metric is printed only when its group has the cell's minimum
subjects, clips and events (or exposure hours): by default 30 subjects, 30
clips, 60 events per type, 60 negative events, 3.0 exposure hours. The
reasoning is in the matrix header. Below that the report prints
`INSUFFICIENT DATA (n/min ...)`; with nothing recorded, `NO DATA`.
`--include-diagnostics` adds the below-minimum numbers to the JSON only, as
`diagnostic_not_a_claim`. A 3–5 person pilot can never pass it.

```
python -m evaluation.scenarios                         # validate and summarise the matrix
python -m evaluation.report --out reports/nodata       # today's report: NO DATA everywhere
python -m evaluation.leakage assign --manifest m.json --out m2.json
python -m evaluation.leakage check  --manifest m2.json --root /data/ds --frames 5
python -m evaluation.leakage lock   --manifest m2.json --out split_lock.json
python -m evaluation.run --manifest m2.json --root /data/ds --split val --split-lock split_lock.json --out runs/r1
python -m evaluation.report --manifest m2.json --root /data/ds --split val --run runs/r1 \
    --split-lock split_lock.json --config config.yaml --out reports/r1
python -m evaluation.agreement --a labels/A1 --b labels/A2 --out agreement.json
```

## Not yet covered

- Pi-side replay: same harness, run on the device against a copied clip, after
  deploy approval.
- An annotation tool. The JSON format is simple enough to convert from CVAT or
  ASAM OpenLABEL exports.
- Variable-frame-rate clips: the timebase is index/fps. Re-mux VFR footage to
  constant fps, or pass `fps_override`.
- Runtime-level alert episodes (AlertPolicy) are not replayed; scoring is on
  module events.
- The M10 report scores the module's alert events. A module that would need
  hands, sunglasses-aware eye state or seat-region association to be evaluable
  is reported through its unknown rate, not hidden.
- An annotation converter from CVAT / Label Studio to v2 JSON is not written.
