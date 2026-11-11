# Behaviour reasoning v2 (driver attention)

**Status: experimental, OFF by default, NOT validated.** Every test result
for this code is a *synthetic scenario*: hand-built landmarks, boxes, angles
and timestamps run through the real module. No number here is a measured
accuracy, and nothing here may be shown to a user as "validated" (see
[Before exposing to presets](#before-exposing-to-presets)).

Code: `core/behavior/` (pure components) and a flag-gated hook in
`modules/driver_attention.py`. Tests: `tests/test_behavior_*.py`.

## Why

Defects confirmed in the audit, cited at the release candidate `c01e43b`:

| Defect | Where (`modules/driver_attention.py` @ c01e43b) |
|---|---|
| `phone_use` = any phone box anywhere AND eyes off road. No hand, no relation between the gaze and the phone. A dash-mounted GPS plus a mirror check qualifies; so does a passenger's phone. | `:1941` `phone_use_condition = phone_visible and eyes_off_road`, `phone_visible` from `_last_phone_boxes` (`:1557`). Hand overlap only changes the detail and risk (`:1953`, `:1990`), and it is IoU (`:1490-1494`), which stays small when a large hand box covers a small phone. |
| `phone_detected` fires on the first frame a box appears; no persistence. | `:1524-1528`, emitted `:1956-1961` |
| `frequent_glancing` counts yaw only. Looking into the lap never counts; every mirror check counts. No cumulative eyes-off-road measure. | `:1897`, `:1909` |
| Face loss resets drowsiness. | `:1603-1606` |
| Closure threshold is 0.75 × the reference EAR, about the EAR drop of a downward look. | `:1783`, `:1810` |
| Head-pose intrinsics assume ~53° HFOV and zero distortion. | `:272-288`, `:297` |
| The hand tracker runs on wall time while the other trackers run on EvidenceClock. | `:1508` |

## How to enable

```yaml
modules:
  driver_attention:
    behavior_v2: true          # shipped: false
    behavior:                  # optional; every key optional
      max_frame_gap_s: 0.5     # default: max_evidence_gap_s
      face_loss_coast_s: 1.0
      eye_occlusion_check: true
      driver_region: null      # [x1, y1, x2, y2] in 0..1, or null
      phone: {presence_k: 3, presence_n: 5, ...}
      glance: {long_glance_s: 2.0, budget_s: 6.0, zones: {...}, legit_zones: [...]}
      eye: {down_head_pitch_deg: -10.0, ...}
```

`core/config_schema.py` validates the block (`core/behavior/config.py`
`validate_behavior`), so an unknown key or a wrong type stops startup. It
is a nested mapping on purpose. Presets override flat `driver_attention`
keys, and nothing in this block is validated enough to be a preset setting.

Sustain times default to the shipped keys, so turning v2 on does not quietly
change them: drowsiness uses `drowsy_seconds` and
`eye_closure_release_frames`, and phone use uses `phone_use_seconds`.

### What changes when it is on

| Event | Flag off (shipped) | Flag on |
|---|---|---|
| `drowsiness` | EAR < 0.75 × reference, sustained; resets on face loss | `DrowsinessEvidence` on the v2 eye state (below) |
| `phone_use` | phone box + eyes off road, sustained | `PhoneInteraction` reaches `in_use` |
| `phone_detected` | rising edge of any box | rising edge of k-of-n persistent presence |
| `frequent_glancing` | yaw-only episode count | `GlanceAnalyzer`, counted (non-legitimate) episodes, pitch included |
| `long_glance` | (none) | new, **log-only** |
| `glance_budget_exceeded` | (none) | new, **log-only** |
| PERCLOS | every low-EAR frame counts | a downward look is not closure; an unreadable eye is not a sample |
| hand tracker clock | `time.monotonic()` | evidence time |

Unchanged in both modes: `distraction`, `eyes_down`, `eyes_up`,
`attention_lost`, `fatigue_warning`'s threshold, `eating_drinking`,
`erratic_hand_movement`, and risk fusion's weights. Risk fusion reads the v2
`drowsy` and `phone_use` flags.

`long_glance` and `glance_budget_exceeded` are not in
`main.ALERT_EVENT_TYPES_PI`, so they are logged and never sound. Promoting
them is a product decision for after validation, and it also needs
`core/taxonomy.py`, the `alerts:` tiers and a `core/roles.py` floor review.

With the flag off, the only edits on the shipped path are naming the frame
time once, storing the person boxes and counting detector ticks. All 257
tests in the 18 existing driver/config test files pass unchanged (see
Tests).

## Components

All components live in `core/behavior/`. Each one is a small, pure class. No
clock is read: callers pass frame time.

### Shared rules (`evidence.py`)

- **Unknown is a state.** `Assessment(state, reason)`. `UNKNOWN` requires a
  reason. A component that cannot decide says so. It never answers "safe"
  (no event) or "unsafe" (an event) for want of an input.
- **Unobserved time is not evidence.** `FrameGapGuard` credits the time
  between frames only when the gap is at most `max_frame_gap_s`. A longer gap,
  or time going backwards, credits **nothing**. This is stricter than
  `EvidenceClock`, which still credits `max_gap_s` per stall. v2 receives the
  raw frame capture time for this reason.
- `KOfN(k, n)` is persistence over detector ticks. Only a tick on which the
  detector ran counts as an observation.
- `intersection_over_area(phone, hand)` is the fraction of the **phone's**
  area that the hand box covers.

### PhoneInteraction (`phone.py`)

The questions run in this order. The first one that cannot be answered gives
`unknown` with its reason:

1. Detector failed, not started, or result older than `max_detection_age_s` → `unknown`
2. Phone not seen on ≥ `presence_k` of the last `presence_n` ticks → `none`
3. Face not visible (cannot relate the phone to the driver) → `unknown`.
   With `driver_region` set and the phone outside it → `visible_only`.
4. Phone centre outside the driver reach region (± `reach_face_widths` around
   the face, from `reach_above_face_widths` above the face down) → `visible_only`
   ("outside driver reach region": a passenger's phone)
5. **held** (a hand covers ≥ `hold_overlap` of the phone's area on ≥ `hold_k` of
   the last `hold_n` ticks) or **near_face** (≥ `near_face_overlap` of the phone
   inside the face box grown by `face_margin_face_widths`). If neither:
   - static for `static_s` (centre within `static_tolerance` × box diagonal) → `visible_only` ("mounted?")
   - hand model unavailable → `unknown`
   - phone moving, no hand found → `unknown` (the hand detector's recall is low, see below)
6. Gaze toward the phone (the angle between the image-plane gaze `(yaw, -pitch)`
   and the face→phone direction is ≤ `gaze_tolerance_deg`, with gaze ≥
   `gaze_min_deg`; a phone within `in_line_face_widths` of the face centre is
   in the line of sight), accumulated over credited time to `use_sustain_s` →
   `in_use`. `phone_use` fires once per episode. The episode ends `release_s`
   after `in_use` stops.

Why step 5 does not fall back to "safe": offline, a hand-overlap gate cut
false phones on phone-free images from 56 to 6 when the hand boxes were human
labels. But MediaPipe Hands found a hand in only 8 of 152 web photos. "No hand
detected" is therefore weak evidence of "no hand".

| Parameter | Default | Needs in-cabin validation |
|---|---|---|
| `presence_k` / `presence_n` | 3 / 5 ticks | yes. At stride 3 on 30 fps this is 0.3 s. |
| `hold_k` / `hold_n` | 1 / 3 ticks | yes. Hand recall decides this. |
| `hold_overlap` | 0.3 of the phone's area | yes |
| `near_face_overlap`, `face_margin_face_widths` | 0.2, 0.5 | yes |
| `reach_face_widths`, `reach_above_face_widths` | 2.5, 0.5 | **yes. The value depends on the cabin and mount.** |
| `gaze_min_deg`, `gaze_tolerance_deg` | 10°, 45° | yes |
| `in_line_face_widths` | 0.3 | yes |
| `use_sustain_s` | `phone_use_seconds` (2.0) | yes |
| `static_s`, `static_tolerance` | 2.0 s, 0.10 | yes. A phone lying still in the lap reads as mounted. |
| `max_detection_age_s` | 1.0 s | yes |

### GlanceAnalyzer (`glance.py`)

The calibrated, fused gaze (yaw and pitch relative to the driver's own
neutral pose) is classified into zones. Each zone is `[yaw_min, yaw_max,
pitch_min, pitch_max]`, the first match wins, and anything unmatched is
`other`. The default geometry assumes a left-hand-drive car, an unmirrored
image and a camera roughly in front of the driver. In this convention, +yaw is
toward the frame's right, which is the driver's left.

| Zone | yaw | pitch | Default legit? |
|---|---|---|---|
| road | −20..20 | −8..12 | (on road) |
| cluster | −15..15 | −25..−8 | yes |
| infotainment | −45..−15 | −40..−8 | no |
| mirror_rear | −45..−15 | 5..30 | yes |
| mirror_left | 30..80 | −20..15 | yes |
| mirror_right | −90..−45 | −20..15 | yes |
| lap | −35..35 | −90..−25 | no |

A **glance** is a run of contiguous off-road time. It ends after `release_s`
back on the road, or after `unknown_end_s` of unknown gaze (no time is
credited while the gaze is unknown). A glance that stays entirely in
legitimate zones and ends within `legit_glance_max_s` is part of driving and
counts toward nothing. A glance that overstays, or enters any other zone, is
charged from its start.

- `long_glance`: one charged glance reaches `long_glance_s` (fires once per glance).
- `glance_budget_exceeded`: charged off-road seconds in the trailing
  `budget_window_s` exceed `budget_s`. It re-arms below half the budget.
- `frequent_glancing`: at least `repeat_min_count` charged glances in `repeat_window_s`.

| Parameter | Default | Note |
|---|---|---|
| `long_glance_s` | 2.0 s | Follows the shape of NHTSA's visual-manual guidance (2 s single glance). |
| `budget_s` / `budget_window_s` | 6.0 s / 30 s | **Not NHTSA's number.** NHTSA's 12 s is a total eyes-off-road time per *task*, not per sliding window. 6 s per 30 s is an unvalidated engineering default. |
| `legit_glance_max_s` | 1.6 s | Typical mirror checks are shorter. Needs in-cabin data. |
| `repeat_min_count` / `repeat_window_s` | 5 / 45 s | The shipped `glance_min_count` / `glance_window_s` values |
| `min_glance_s`, `release_s`, `unknown_end_s` | 0.15 s, 0.2 s, 1.0 s | Noise handling |
| `zones`, `legit_zones` | above | **Per vehicle and mount.** For a right-hand-drive car, the zones must be mirrored. |

### Eye state and drowsiness (`eye_state.py`)

`classify_eye_state` returns `open`, `closed`, `looking_down` or `unknown`:

- `unknown` if there is no face, both eyes are occluded (or have no landmarks),
  or there is no threshold yet. One occluded eye defers to the other.
- `open` if EAR ≥ the closure threshold (the same `effective_threshold` as the
  shipped path), or the shipped iris veto applies.
- `closed` if the lid covers the iris (iris-openness < `iris_closed`, or
  EAR < the PERCLOS-depth threshold when the iris is unmeasurable), **whatever
  the head does**. A head nodding off pitches down too.
- `looking_down` if EAR dropped, the iris is still visible, and the calibrated
  **head** pitch is ≤ `down_head_pitch_deg` or the calibrated **eye-in-head**
  pitch is ≤ `down_iris_pitch_deg`.
- `closed` otherwise (an EAR drop with no downward gaze).

Occlusion: the eye-patch heuristic `eye_patch_occluded` reports an eye as
occluded when the grayscale patch is darker than `occlusion_max_mean` **and**
flatter than `occlusion_max_std`. A sunglasses lens looks like this to a
visible-light camera. Turn it off with `eye_occlusion_check: false`.

`DrowsinessEvidence` credits closed time per frame, with no credit across
stale gaps. The episode ends after `eye_closure_release_frames` open or
looking-down frames. While the eye state is unknown, it **coasts**: the
accumulated closure is held (not credited, not reset) for `face_loss_coast_s`,
then dropped, and the state becomes `unknown`. Longer face loss is handled
exactly as shipped (`attention_lost`).

| Parameter | Default | Note |
|---|---|---|
| `down_head_pitch_deg` | −10° | Needs in-cabin validation |
| `down_iris_pitch_deg` | −8° | Depends on `eye_gaze_scale_deg`. Needs validation. |
| `iris_closed` | 0.2 | The value in `core/eye_signal.py` ("closed < 0.2") |
| `occlusion_max_mean` / `occlusion_max_std` | 40 / 8 (0–255) | **Unvalidated heuristic.** An IR-lit cabin sees through many lenses. A dark cabin without IR reads as occluded, which is reported as unknown. |
| `face_loss_coast_s` | 1.0 s | Needs validation |

### Monitoring uncertainty (`reasoner.py`)

`BehaviorReasoner.uncertainty()` (`DriverAttentionModule.behavior_uncertainty()`)
names every component that is currently `unknown`, with the reason. Possible
reasons include "face not detected", "eyes occluded (sunglasses or
obstruction)", "object detection failed: …", "object-detector result stale",
"hand model unavailable…" and "calibration not complete".

The same information appears in:

- `get_debug_state()["behavior"]`: the snapshot, every assessment, the `uncertain` map and the last glance.
- `get_telemetry_snapshot()`, under the `behavior_*` keys. The shared driver CSV
  has a fixed header and ignores them. A consumer that wants them reads the
  snapshot.

Uncertainty is **not** wired into `core/monitoring_state.py`. Doing that
could move the device's monitoring state (for example to DEGRADED) on an
unvalidated heuristic. That is a deliberate follow-up, not an omission.

### Calibrated head-pose intrinsics (`camera_model.py`)

`camera_intrinsics: {width, height, fx, fy, cx, cy, distortion}` comes from a
hardware calibration run (`cv2.calibrateCamera`). It is rescaled to the
processed frame and passed to solvePnP with its distortion. It is not part of
`behavior_v2` and applies whenever it is set. `null` (shipped) keeps the
matrix and the calibration file byte-identical. Set it **or**
`camera_hfov_deg`, not both. A stored head-pose baseline that was learned
under different intrinsics is rejected, and the device recalibrates. The
values themselves are left to hardware calibration.

## Tests

All results are **synthetic scenarios**.

```
PYTHONDONTWRITEBYTECODE=1 ~/.venvs/sensordyme-m01/bin/python -m pytest -p no:cacheprovider -q \
  tests/test_behavior_evidence.py tests/test_behavior_phone.py tests/test_behavior_glance.py \
  tests/test_behavior_eye_state.py tests/test_behavior_reasoner.py tests/test_behavior_config.py \
  tests/test_behavior_camera_model.py tests/test_behavior_integration.py
```

`tests/test_behavior_integration.py` drives the real module with the
geometric landmark generator from `tests/test_driver_attention_scenarios.py`,
a fake detector, fake MediaPipe Hands and a deterministic clock. Several
scenarios also run with the flag off, to pin the shipped behaviour they change:

| Scenario (synthetic) | Flag off | Flag on |
|---|---|---|
| mounted phone, attentive driver | — | no `phone_use`, `visible_only` |
| mounted phone, driver looks into the lap | `phone_use` | no `phone_use` |
| passenger holds a phone, driver glances that way | `phone_use` | no `phone_use` (outside reach) |
| driver holds the phone and looks at it (3 s feed) | — | `phone_use` once |
| one-tick phone flicker | — | no `phone_detected` |
| detector interruption | phone boxes cleared ("none") | phone `unknown`, reason given |
| six 0.8 s mirror checks | `frequent_glancing` | nothing |
| 1 s instrument-cluster glance | — | nothing |
| single 2.5 s console glance | nothing | `long_glance` |
| 4 × 1.8 s off-road within 30 s | — | `glance_budget_exceeded` |
| looking down at the lap 3 s, lids lowered | `drowsiness` | no `drowsiness` (`looking_down`) |
| eyes cast down with the head level | — | no `drowsiness` |
| real closure while nodding | — | `drowsiness` once |
| sunglasses (lenses dark, landmarks read "shut") | `drowsiness` | eye state `unknown`, uncertainty reported |
| one eye occluded | — | the other eye decides; a closure of it still fires |
| face lost 0.5 s mid-closure | no `drowsiness` | `drowsiness` once |
| face lost 10 s | — | `attention_lost`, uncertainty reported, closure evidence dropped |
| frames 1 s apart (stalls) | `drowsiness` (0.5 s credited per stall) | no evidence accrued |

Proof that the default is unchanged: with `behavior_v2: false`, all 257 tests
in these 18 existing files pass unchanged: `test_driver_attention_scenarios`,
`test_driver_attention_module`, `test_m01_role_floors`,
`test_driver_signal_conditioning`, `test_drowsiness`, `test_distraction`,
`test_phase1_driver_fixes`, `test_driver_hand_motion`,
`test_driver_review_fixes`, `test_risk_fusion`, `test_head_pose_geometry`,
`test_eye_gaze`, `test_calibration`, `test_ear`, `test_fatigue_pattern`,
`test_event_traceability`, `test_config` and `test_config_schema`.

## Before exposing to presets

v2 is a set of **rules** over model outputs. It is not a model, so it has no
card of its own in `core/model_registry.yaml`. Its claims can be no stronger
than the weakest link in the chain it reads. Today FaceMesh and Hands are
`unevaluated` (shipping exception), and the `driver_safety` chain's weakest
rung is `unevaluated` (docs/MODEL_REGISTRY.md). v2 also makes **Hands** a
required link for `phone_use`, where it was only corroboration before.

Each step below is required before the next one:

1. **In-domain, event-level evaluation** (`evaluated_in_domain` rung,
   docs/EVALUATION.md harness). Use consented SensorDyme recordings, with
   splits by subject, covering:
   - phone held and looked at, phone held but not looked at, mounted
     phone, passenger phone, phone at the ear
   - mirror and cluster checks, console and lap glances
   - real closures, downward looks, nodding
   - sunglasses (including under IR), partial occlusion
   - face loss, camera stalls
   - day, night and IR lighting; at least two vehicle geometries

   Report v1 and v2 side by side per event type: recall, precision, false
   alarms per hour, latency, with subject-bootstrap intervals. Tune
   parameters on a calibration split only, and report on held-out subjects.
2. **Measure the uncertainty rates.** For each condition, report how often
   each component is `unknown`. A rule that is right only when it is not
   unknown, and is unknown most of the time, is not protection. Record hand
   recall in the cabin, since it decides `hold_k` and the fallback paths.
3. **Run on device, event-level** (`validated_on_device`). Run on the Pi +
   Hailo package with the pinned artifacts, at the deployed frame rate and
   stride.
4. **Then make the product decisions.** Promote `long_glance` and
   `glance_budget_exceeded` (taxonomy, `ALERT_EVENT_TYPES_PI`, alert tiers),
   decide whether uncertainty feeds `core/monitoring_state.py`, review the
   `core/roles.py` floors, and only then consider a preset or a default flip.

Until all of this is done, `behavior_v2` stays `false` in `config.yaml`, no
preset sets it, and nothing user-facing may describe it as validated.

## Limitations

- Every default is an engineering guess. The tests show the logic does what
  it states on constructed inputs. They show nothing about real faces,
  lenses, cabins or detectors.
- Zone geometry and the reach region assume one cabin layout (left-hand
  drive, a frontal camera). Other layouts need configuration.
- Gaze-to-phone direction is compared in the image plane. Without
  camera-to-cabin geometry, the distance or angle to the phone is not known.
- When hands are not detected, a phone that is held but kept very still can
  read as mounted (`visible_only`). Hand recall is the binding constraint.
- A drowsy driver whose lids droop only partly while the head nods with the
  iris still visible reads as `looking_down`. Deep closures and PERCLOS still
  count.
- The occlusion heuristic is untested on real sunglasses and on IR images.
- Uncertainty is reported (debug/telemetry), not enforced (monitoring state).
- `frequent_glancing` under v2 has different semantics (pitch included,
  legitimate glances excluded) under the same event name.
