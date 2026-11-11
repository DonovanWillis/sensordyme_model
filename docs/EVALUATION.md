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

## Not yet covered

- Pi-side replay: same harness, run on the device against a copied clip, after
  deploy approval.
- An annotation tool. The JSON format is simple enough to convert from CVAT or
  ASAM OpenLABEL exports.
- Variable-frame-rate clips: the timebase is index/fps. Re-mux VFR footage to
  constant fps, or pass `fps_override`.
- Runtime-level alert episodes (AlertPolicy) are not replayed; scoring is on
  module events.
