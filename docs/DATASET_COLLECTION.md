# SensorDyme dataset collection protocol

How to collect footage that can actually support an accuracy claim about *this*
product, on *this* camera, in *this* mounting geometry.

> **Legal note, stated once and meant throughout.** This is engineering guidance
> written by engineers. It is not legal advice. Filming infants and drivers engages
> real statutes that vary by jurisdiction — GDPR and its national implementations,
> COPPA, state biometric-privacy acts (Illinois BIPA in particular), and recording-consent
> law. Have the consent form in §8 reviewed by counsel before the first camera is
> switched on, not after.

---

## 1. Why this exists

Every accuracy number SensorDyme has today is domain-shifted, and the codebase says
so in its own words. `benchmarks/accuracy_state_farm.py`:

> State Farm's camera is mounted high on the PASSENGER side looking ACROSS the
> cabin. SensorDyme's recommended mount is dash-centre, DRIVER-FACING. These are
> therefore domain-shifted LOWER BOUNDS, not measurements of SensorDyme's operating
> accuracy.

The size of the shift is visible in the numbers themselves. From
`benchmarks/results/accuracy_state_farm_20260829_194634.json` (1000 images,
subject-disjoint val split, `yolo11n.pt`):

> **These numbers describe the CPU baseline, not the shipped detector.** They were
> measured on Ultralytics `yolo11n.pt`, which since Phase 4 runs only on developer
> machines. The device runs `damoyolo_tinynasL20_T.hef` on the Hailo-8L, which is
> differently calibrated in both directions — `docs/ACCELERATOR.md` measured 93.8%
> phone recall at best F1 for it against 25.4% for an NMS-on-chip `yolov11n.hef`.
> The rows below remain the right *domain-shift* evidence, which is what this
> section is about; they are not a statement of shipped accuracy. See
> `docs/DETECTION_THRESHOLDS.md` for the per-backend curves.

| Signal | Measured on State Farm | What it gates |
|---|---|---|
| Face availability (MediaPipe FaceMesh) | **39.7 %** | *every* drowsiness (EAR, PERCLOS) and gaze (yaw/pitch) detection |
| Person availability (YOLO) | 100.0 % | presence, zones, stillness, occupancy, baby's smallest-box heuristic |
| Person mean confidence | 0.778 | |
| Multi-person rate | 5.4 % | |
| Phone recall @ `conf 0.35` | 10.8 % (precision 100 %) | the old shipped threshold |
| Phone recall @ `conf 0.10` | **51.7 %** (precision 95.8 %) | the current shipped threshold |
| Phone recall @ `conf 0.05` | 67.8 % (precision 93.5 %) | best F1 |

A 39.7 % face-availability figure on a camera pointed across the cabin at a driver's
profile tells you almost nothing about a dash-centre camera pointed at a driver's
face — except that the true number is higher. **It bounds the signal; it does not
certify it.** And for the other two domains it does not even bound: State Farm has
no infants, no falls, and no room cameras. The baby and fall modules have **no
accuracy measurement of any kind**.

This protocol produces the dataset that closes that. It is written to be executed by
someone who was not in the room when it was designed.

---

## 2. Capture rig — fixed, and recorded with every clip

Domain shift is what this exercise exists to eliminate, so the rig must match the
product. Anything that varies must be *recorded*, not averaged over.

| Property | Value | Source |
|---|---|---|
| Camera | Raspberry Pi Camera Module 3 Wide NoIR (`imx708_wide_noir`) | the shipping part |
| Resolution | 640 × 480, RGB888 | `config.yaml` `camera.width/height` |
| Capture rate | 30 /s | `config.yaml` `camera.fps` |
| Host | Raspberry Pi 5, 4 GB | the shipping part |
| Rotation | record `rotate_180` state | `config.yaml` `camera.rotate_180` |
| Audio | **none** | see §8.6 |

**Record video at the product's own resolution.** Collecting at 1080p and
downsampling produces frames a 640×480 sensor pipeline would never have produced —
different noise, different ISP sharpening, different motion blur. If a higher-resolution
master is wanted for future re-use, capture *both* streams simultaneously and treat
the 640×480 stream as the one that is scored.

**NoIR means no infrared cut filter.** Two consequences that must be in the metadata:
colour rendition under daylight is materially off (skin tones shift red/magenta), and
IR illumination is usable at night. A night clip shot under a 850 nm illuminator and
a night clip shot under a dim household lamp are different domains; both are needed;
they must be distinguishable in the manifest (`illumination` field, §6).

### 2.1 Mounting geometry, per domain

Geometry is the variable that broke the existing numbers, so it is the variable that
gets measured — with a tape measure, into the metadata, on every session.

**DRIVING.** Dash-centre, driver-facing. Record: horizontal offset from the vehicle
centreline (cm, signed, positive toward the driver), height above the dash surface
(cm), height above the driver's seated eye level (cm, signed), and pitch angle
(degrees from horizontal, positive = tilted up). Also record the vehicle
make/model/class, because cabin geometry is the thing being sampled.

**BABY.** Room camera viewing a crib. Record: horizontal distance from camera to
crib centre (cm), camera height above floor (cm), pitch (degrees), and whether the
view is over the crib rail or through it. Whether the rail occludes the infant is a
first-order variable for `person_absent` and `unusual_stillness`, not a detail.

**FALLS.** Room camera. Record: camera height above floor (cm), pitch, and the
distance from camera to the centre of the area where falls occur (cm). Record the
room's longest sightline, because "fall away from camera" is only meaningful
relative to it.

**Collect at least three distinct geometries per domain.** A model validated at one
mounting position has been validated at one mounting position. Vary the offsets
deliberately, within the range a real installer would plausibly produce, and record
the range you covered.

---

## 3. Scenarios

Scenario codes are the `classname` column in the manifest (§7) and are stable
forever once assigned. Codes are prefixed by domain so the three domains can share
one namespace.

### 3.1 DRIVING (`d*`)

| Code | Scenario | What it tests |
|---|---|---|
| `d00` | Attentive driving, no phone, no food | the negative set — `phone_detected`, `distraction`, `eyes_down` must all stay silent |
| `d01` | Phone in hand, raised, at or near face height | `phone_detected`, `phone_use`, `hand_phone_iou_threshold` |
| `d02` | Phone in hand, near lap / low, driver looking down | `phone_use` (needs phone AND `eyes_down`), `pitch_threshold_deg` |
| `d03` | Phone in hand, near the wheel / at wheel rim height | the hardest phone case: small, low-contrast, hand-occluded |
| `d04` | Phone partially occluded — by hand, by wheel rim, by clothing | measures where the 51.7 % recall actually goes |
| `d05` | Phone mounted in a cradle, not touched | **negative for `phone_use`, positive for `phone_detected`.** `phone_detected` is log-only (no alert tier) for exactly this case |
| `d06` | Multiple occupants — passenger present, rear passengers | tests that the driver is the subject; `multi_person_rate` was 5.4 % on State Farm |
| `d07` | Passenger holding a phone, driver attentive | the false-positive case `d06` sets up |
| `d08` | Nighttime, ambient street/dash lighting only | face availability under low light |
| `d09` | Nighttime with IR illumination | the NoIR sensor's intended night mode |
| `d10` | Direct sun glare — low sun through windscreen, into the lens | the documented failure mode for EAR landmarks |
| `d11` | Sunglasses, clear glasses, glare on lenses | `iris_veto`, `ear_baseline_window_s`, `signal_median_window` all exist for this |
| `d12` | Varied clothing — high collar, hood, hat, scarf, headwear | face and torso segmentation |
| `d13` | Varied backgrounds — urban, rural, tunnel, parked, car wash | background clutter behind the driver's head |
| `d14` | Eating / drinking — cup, bottle, food in hand | `yolo_classes` maps 16 COCO classes to `eating_drinking` |
| `d15` | Genuine drowsiness signals — long closures, head nod, yawning | `drowsy_seconds`, `perclos_threshold`. **Acted, not induced — see §8.7** |
| `d16` | Mirror and blind-spot checks | the `mirror_check_yaw_deg: 55` / `rearview_tolerance_seconds: 5.0` path, which was measurably dead until 2026-08-29 and has never been validated on real head turns |
| `d17` | Ordinary off-centre glances — console, passenger, radio | `distraction_seconds: 3.0` must NOT fire below it |

### 3.2 BABY (`b*`)

| Code | Scenario | What it tests |
|---|---|---|
| `b00` | Infant alone in crib, awake, moving normally | the baseline negative |
| `b01` | Infant alone in crib, asleep, still | `unusual_stillness` (`still_warning_s: 120`) must fire; it is a **loud** type |
| `b02` | Caregiver near the crib, infant fully visible | presence with two people |
| `b03` | **Caregiver BLOCKING the infant** — body between camera and crib | the case that makes `person_absent` fire wrongly |
| `b04` | Infant partially occluded — blanket, toy, crib rail, sleep sack | the case `expect_presence: true` was changed for on 2026-08-29 |
| `b05` | Infant fully occluded — under a blanket, face-down in bedding | `person_absent` (`absence_warning_s: 30`) must fire. **The most important clip in this domain and the hardest to stage safely — see §8.7** |
| `b06` | Crib position varied — near/far, left/right/centre, rotated | geometry sampling |
| `b07` | Lighting varied — daylight, dim lamp, night-light, dark + IR | `imx708_wide_noir` across its whole range |
| `b08` | Infant holding a bottle | `holding_bottle`, `holding_bottle_confirm_s: 1.0` |
| `b09` | Infant holding a small object — toy, remote, cup | `holding_object`, `object_classes` |
| `b10` | **Object moving toward the infant's mouth** | `choking_hazard`, whose threshold was recalibrated to 0.30 on arithmetic alone and has never been scored against real footage |
| `b11` | Flailing / frantic arm movement | `flailing`, a **loud** type, `flailing_confirm_s: 0.5` |
| `b12` | Vigorous but ordinary play — waving, reaching, rolling | the negative for `b11`. `core/activity.py` distinguishes them on `FLAIL_MIN_REVERSALS: 4` vs `WAVE_MIN_SWINGS: 3` — an untested distinction |
| `b13` | Infant leaving a `safe` crib zone — climbing, being lifted out | `safe_zone_exit`, a **loud** type |
| `b14` | Two children in frame | the smallest-box heuristic |
| `b15` | Empty crib, room otherwise normal | long-duration negative for `person_absent` |

### 3.3 FALLS (`f*`)

| Code | Scenario | What it tests |
|---|---|---|
| `f00` | **Slow slump** — gradual descent, e.g. from a chair or against a wall | may never exceed `drop_velocity_threshold: 0.6`; the `fall_pose_recovery` path is the only route in |
| `f01` | **Sudden fall** — fast, from standing | the canonical case the drop threshold was written for |
| `f02` | **Fall toward the camera** | apparent size grows; hip *y* may barely change — the two-frame `avg_y` velocity is nearly blind to a fall along the optical axis |
| `f03` | **Fall away from the camera** | apparent size shrinks; landmark confidence drops fastest here |
| `f04` | Fall across the frame, laterally | the easiest case; the control |
| `f05` | **Furniture occlusion** — falls behind a sofa, table, bed | tests `occlusion_grace_s` and `fall_pose_recovery_max_gap_s: 2.5` |
| `f06` | **Fall then struggle** — falls, tries to get up, moves for 5–20 s | the case `fall_pending_timeout_s: 6.0` was rewritten for; the comment calls it "this product's actual use case, and the worse outcome" |
| `f07` | **Fall then stillness** — falls and does not move | the canonical confirm path: `stillness_duration_s: 1.5` |
| `f08` | Fall then recovery — falls, gets up within 10 s | must still alert; `fall_resolve_after_s: 4.0` |
| `f09` | **Repeat falls** — two falls within 30 s | `fall_cooldown_s` was 20.0 until 2026-08-29 and silently swallowed the second one |
| `f10` | Fall out of / beside a bed | overlaps `safe_zone_exit` bed monitoring |
| `f11` | NEGATIVE — sitting down normally, into chairs and onto beds/floor | the single most important negative |
| `f12` | NEGATIVE — **lying down deliberately**, onto a bed or a sofa | ends in `posture.LYING_DOWN` + stillness, i.e. every confirm condition except the drop |
| `f13` | NEGATIVE — bending over, kneeling, picking something off the floor | transient lying-adjacent posture |
| `f14` | NEGATIVE — exercise on the floor, stretching, yoga, playing with a pet | prolonged low postures |
| `f15` | NEGATIVE — walking, standing, ordinary room activity, long duration | `unusual_stillness` (`unusual_stillness_s: 600`) must NOT fire |
| `f16` | NEGATIVE — resting still on a sofa for > 10 minutes | `unusual_stillness` **must** fire here. Labelled a negative for `fall` and a **positive** for `unusual_stillness` |

`f11`–`f16` are not optional and are not filler. `fall` is a **critical**-severity,
push-notifying alert; its precision is the number that decides whether anyone keeps
the device switched on. A recall figure with no matched negative set is not a result.

---

## 4. Labelling protocol

The bar: **two annotators, working independently from this document alone, produce
the same labels on the same clip.** Every definition below is mechanical for that
reason. Where a definition is a choice rather than a fact, it says so.

### 4.1 Two levels, two files

* **Clip-level** — one row per clip. Scenario, subject, conditions, and the event
  timestamps in §4.3. This is what scores the temporal detections (`fall`,
  `flailing`, `unusual_stillness`, `person_absent`, `choking_hazard`).
* **Frame-level** — one row per extracted frame. Per-frame presence labels. This is
  what `benchmarks/accuracy_state_farm.py` consumes (§7).

Extract frames at **5 /s** from every clip (every 6th frame at 30 /s capture). That
is above the 4.17 /s absolute floor derived in `docs/PERFORMANCE_BUDGET.md` §2.2, so
a 5 /s frame sequence can still exercise the pose-derived detectors, and it keeps
adjacent frames from being near-duplicates. Around a labelled event window
(§4.3), extract at the **full 30 /s** — that is where the timing accuracy lives.

### 4.2 Annotate at a fixed playback state

Every temporal judgement below is made **frame-stepping, not at playback speed.**
Scrub controls, single-frame step, and a visible frame index are required tooling.
Two annotators watching at 1× will not agree on a fall onset frame and it is not
their fault.

### 4.3 The exact frame a fall starts

Three timestamps per fall, annotated in this order. The order matters: the easiest
one is annotated first and anchors the other two.

**`t_impact` — annotate this first.** The first frame in which any part of the
subject's **torso** (hip or shoulder) makes contact with the floor, or with the
furniture surface they come to rest on. Torso, not hand and not knee: a hand thrown
out to break a fall lands earlier and varies wildly between people, while the torso
landing is a single visible moment. If the subject comes to rest without the torso
contacting anything (slumping into a seated position against a wall), `t_impact` is
the first frame in which downward motion of the hip midpoint has **stopped** —
identified by stepping forward until the hip is at the same height for 3 consecutive
frames.

**`t_onset` — annotate second, by scrubbing backward from `t_impact`.** Step
backward until you find 2 consecutive frames in which the hip midpoint is **not**
descending. `t_onset` is the frame immediately after those. In words: the start of
the last uninterrupted downward movement that ends at `t_impact`.

Two clarifications that resolve most disagreements:

* A stumble followed by a recovered step followed by the fall has a non-descending
  interval in it, so `t_onset` is after the stumble. That is correct and deliberate:
  the stumble is not the fall.
* A slow slump (`f00`) may descend for 3 seconds. `t_onset` is still the start of
  that whole descent. The onset-to-impact duration is a **measured property of the
  clip**, recorded in `fall_duration_s`, not a thing to be normalised away.

**`t_settle` — annotate third.** The first frame at or after `t_impact` in which the
subject's hip midpoint stays within a **hip-width** of its position for a full
second. If the subject never settles within the clip (a `f06` struggle), record
`t_settle` as empty and set `settles: false`.

**Scoring window.** A `fall` event counts as a true positive if the device's event
timestamp falls within:

```
[ t_impact ,  t_impact + 6.0 s ]
```

6.0 s is not arbitrary: it is `fall_pending_timeout_s`. After it,
`modules/fall_detection.py` abandons the pending fall, so any `fall` event arriving
later **cannot have been caused by this fall** and is a false positive by
construction. Detections in `[t_onset, t_impact)` are also true positives — early is
not wrong — but record them separately, because a detector firing before impact is
detecting something different from the one the code is written around.

### 4.4 What counts as "phone in hand"

A frame is `phone_in_hand: true` when **all three** hold:

1. **Visible.** At least half of the phone's extent is inside the frame, and it is
   identifiable as a phone by shape and context without knowing what the clip is
   about.
2. **Contact.** The hand region and the phone region touch with no visible gap
   between them, *or* the phone is partially occluded by the hand. A hand hovering
   over a phone with visible background between them is not contact.
3. **Supported.** The hand bears the phone's weight. Fingers wrapped around it,
   under it, or gripping an edge.

Consequences, spelled out because they are where annotators diverge:

* Phone flat on the thigh, hand resting **beside** it → `phone_visible: true`,
  `phone_in_hand: false`. This is `d02` and it must not be labelled as held.
* Phone flat on the thigh, thumb on the screen, hand not supporting it →
  `phone_in_hand: **false**`, and set `phone_touched: true`. The product cannot see
  the difference, but the label set must, or the recall gap can never be diagnosed.
* Phone in a mount/cradle, hand on it → `phone_in_hand: false`,
  `phone_mounted: true`. `d05`.
* Phone at the ear, hand supporting → `phone_in_hand: true`.
* Only a corner of the phone visible past the hand → still true, if a person can
  identify it as a phone.

`phone_visible` and `phone_in_hand` are **separate columns and never collapsed.**
`modules/driver_attention.py` emits `phone_detected` on the rising edge of
*visibility* with no sustain gate, and `phone_use` on visibility **and** eyes-off-road
sustained for `phone_use_seconds: 2.0`. Those are two detections with two failure
modes; a single merged label cannot score them.

### 4.5 The other frame-level definitions

**`subject_present`** — any part of the subject's torso or head is visible. A hand
alone is not presence. Ground truth for `person_absent`.

**`occlusion`** — the fraction of the subject's torso hidden, in five buckets:
`0`, `1-33`, `34-66`, `67-99`, `100`. Buckets, not a percentage: two annotators will
agree on a bucket and will not agree on 41 % vs 47 %.

**`face_visible`** — both eyes and the nose are visible and unobstructed by a hand,
mask, or extreme head angle. Ground truth for the 39.7 % face-availability figure
this project most needs to replace. Sunglasses count as visible (the face is there;
whether FaceMesh finds it is the measurement).

**`eyes_closed`** — both eyelids fully closed. If the eyes are not resolvable,
`face_visible: false` and leave `eyes_closed` empty. Do **not** guess.

**`gaze_target`** — one of `road`, `mirror_rear`, `mirror_left`, `mirror_right`,
`instrument`, `console`, `lap`, `passenger`, `phone`, `other`, `unknown`. Judged
from head and eye direction together. `mirror_*` is separated from `other` because
`mirror_check_yaw_deg: 55` and `rearview_tolerance_seconds: 5.0` are a distinct code
path that has never been tested against a real head turn.

**`held_object`** — the `object_classes` label of anything the subject is holding,
by the §4.4 support test, or empty.

**`object_near_mouth`** — a held object whose nearest edge is within one hand-width
of the subject's mouth. Ground truth for `choking_hazard`.

### 4.6 Ambiguous cases

Every field accepts **`unsure`**. It is a first-class value, not a failure.

1. Annotator A labels the clip. Annotator B labels it independently, without seeing A's labels.
2. Fields where A and B agree: settled.
3. Fields where they disagree, or either said `unsure`: annotator C adjudicates.
4. If C cannot decide, the field is `ambiguous` and the clip enters the
   **`held_out_ambiguous`** pool.

`held_out_ambiguous` clips are **excluded from every recall and precision
denominator, and their count is reported alongside every result.** A protocol that
lets ambiguity be silently dropped is a protocol that lets accuracy be inflated by
discarding the hard cases — which are exactly the cases the product fails on.
"Recall 92 % (n=200, 31 ambiguous excluded)" is an honest number. "Recall 92 %" is
not.

**Track inter-annotator agreement and publish it.** Report Cohen's κ per field.
Below κ = 0.7 on a field, the *definition* is broken, not the annotators — rewrite
this section before collecting more data.

---

## 5. How many clips, and what that buys

### 5.1 A scenario with five clips cannot support a recall claim

If all *n* clips in a scenario are detected, the 95 % Clopper–Pearson lower bound on
true recall is `0.025^(1/n)`:

| *n*, all detected | 95 % lower bound on true recall | What may honestly be said |
|---|---|---|
| 5 | **47.8 %** | "the pipeline sees this at all". Nothing more. |
| 10 | 69.2 % | still nothing quotable |
| 20 | 83.2 % | "probably above 80 %" |
| 30 | 88.4 % | **the minimum for any per-scenario claim** |
| 50 | 92.9 % | supports a "> 90 %" claim |
| 100 | 96.4 % | supports a "> 95 %" claim |
| 200 | 98.2 % | supports a "> 98 %" claim |

Five clips, all detected, is consistent with a true recall of 48 %. That is a coin
flip on a fall alarm, and it is what "we tested it and it worked every time" means at
n = 5.

Realistically not everything is detected. Wilson 95 % intervals at an observed 90 %:

| *n* | Observed 90 % → 95 % CI | Half-width |
|---|---|---|
| 30 | 74.4 – 96.5 % | ± 11.1 pts |
| 50 | 78.6 – 95.7 % | ± 8.5 pts |
| 100 | 82.6 – 94.5 % | ± 6.0 pts |
| 200 | 85.1 – 93.4 % | ± 4.2 pts |

### 5.2 The counts

| Tier | Clips per scenario | Buys |
|---|---|---|
| **Pilot** | 5 | Smoke test only. Does the signal exist? Is the rig right? Is the labelling definition workable? **No claim of any kind.** |
| **Minimum viable** | **30** | A per-scenario recall claim with ± 11 pts. Enough to rank scenarios and find the failures. |
| **Target** | **100** | ± 6 pts per scenario. Enough to tune a threshold against without chasing noise. |
| **Domain headline** | **≥ 200 pooled** | The number that goes in a product claim. Pooled across the scenarios of one domain, weighted as the deployment weights them. |

**Run the pilot first, on every scenario, and stop.** Label the 5 clips with two
annotators and compute κ per §4.6. Fix the definitions. Only then collect the 30.
Collecting 100 clips against a definition two people read differently produces 100
clips that must be relabelled.

### 5.3 Subjects, not just clips

**Minimum 12 distinct subjects per domain; target 25.** Both numbers are
**RECOMMENDATIONS**. The reasoning is that State Farm's 26 drivers is what makes its
subject-disjoint split meaningful, and that 30 clips from 3 people is 3 samples of
inter-person variation dressed up as 30. Per-subject caps: **no subject contributes
more than 15 % of any scenario's clips**, and every subject appears in at least 5
scenarios.

Sample deliberately across the axes the models are known to be sensitive to: skin
tone (the whole Fitzpatrick range — face and hand detection accuracy varies across it
and an unbalanced set will hide that), body size, age, glasses, facial hair, and
head covering. For the baby domain, infant age spans a very wide range of body
proportion — sample it in bands (0–3, 3–6, 6–12, 12–24 months) and record the band.

### 5.4 Negatives are measured in hours, not clips

False-alarm rate is a rate, so it needs exposure time. With **zero** false alarms
observed in *T* hours, the 95 % upper bound on the rate is `3/T` per hour (the rule
of three). Inverting:

| Claim | Clean footage needed, with zero false alarms |
|---|---|
| ≤ 1 false `fall` per 24 h | **72 h** |
| ≤ 1 false `fall` per 7 days | **504 h** (21 days) |
| ≤ 1 false `phone_use` per 1 h drive | 3 h |
| ≤ 1 false `unusual_stillness` per night (8 h) | 24 h |

Two of these need continuous recording, not clips: `unusual_stillness`
(`still_warning_s: 120` for baby, `unusual_stillness_s: 600` for elderly) and
`person_absent` (`absence_warning_s: 30`) cannot be evaluated on a 20-second clip at
all. **Collect at least one full 8-hour overnight session per baby subject and at
least three full 8-hour sessions in the falls domain.** These are also the sessions
that answer the 48-hour memory-soak question in
`docs/HARDWARE_VERIFICATION.md` item 11 — one collection, two results.

---

## 6. File naming and metadata

### 6.1 Naming

Clips:

```
sd_<domain>_<scenario>_<subject>_<session>_t<take>.mp4

sd_fall_f00_s07_20260901a_t03.mp4
sd_baby_b05_s21_20260904b_t01.mp4
sd_drive_d03_s12_20260912a_t07.mp4
```

* `domain` — `drive` | `baby` | `fall`
* `scenario` — the code from §3
* `subject` — `s` + zero-padded id. **The id, never a name.** The id ↔ person
  mapping lives in one access-controlled file (§8.3) and nowhere else.
* `session` — `YYYYMMDD` + a letter for multiple sessions in a day. One session =
  one continuous setup: one rig geometry, one lighting condition, one location.
* `take` — `t` + zero-padded, unique within (subject, session, scenario)

Extracted frames keep the clip identity and add a zero-padded source frame index:

```
sd_fall_f00_s07_20260901a_t03_f000432.jpg
```

The index is the index **in the source clip at 30 /s**, not the index in the
extracted sequence. That keeps every frame traceable back to an exact moment and
lets a 5 /s extraction and a 30 /s extraction of the same clip coexist without
collision.

### 6.2 Clip metadata — `<clipid>.json`, one sidecar per clip

```json
{
  "clip_id": "sd_fall_f00_s07_20260901a_t03",
  "domain": "fall",
  "scenario": "f00",
  "subject": "s07",
  "session": "20260901a",
  "take": 3,
  "consent_id": "C-2026-0041",
  "recorded_at": "2026-09-01T14:22:10-04:00",
  "duration_s": 24.3,
  "fps": 30,
  "resolution": [640, 480],

  "rig": {
    "camera": "imx708_wide_noir",
    "rotate_180": false,
    "height_cm": 145,
    "pitch_deg": -12,
    "distance_to_subject_cm": 310,
    "geometry_variant": "g2"
  },

  "conditions": {
    "illumination": "daylight_indirect",
    "ir_illuminator": false,
    "lux_estimate": 180,
    "location_type": "living_room",
    "background_clutter": "moderate"
  },

  "subject_attrs": {
    "fitzpatrick": 4,
    "glasses": false,
    "head_covering": null,
    "clothing": "dark_long_sleeve",
    "age_band": "adult"
  },

  "events": [
    { "type": "fall", "t_onset_s": 8.40, "t_impact_s": 9.13,
      "t_settle_s": 10.90, "settles": true, "fall_duration_s": 0.73,
      "direction": "lateral", "occluded_at_impact": false }
  ],

  "negative_for": ["phone_use", "unusual_stillness"],

  "labelling": {
    "annotator_a": "A03", "annotator_b": "A07", "adjudicator": null,
    "ambiguous_fields": [], "labelled_at": "2026-09-08"
  }
}
```

`illumination` is a closed vocabulary — `daylight_direct`, `daylight_indirect`,
`overcast`, `indoor_bright`, `indoor_dim`, `night_lamp`, `night_dark`, `night_ir`,
`mixed_glare` — because free text cannot be grouped for analysis later.

`negative_for` is what makes negatives usable. A clip that is a `fall` positive is
simultaneously a negative for `phone_use` and `unusual_stillness`, and stating so
explicitly means the false-positive denominator is a real, countable set rather than
"everything we didn't label".

### 6.3 What the existing code already gives you, and what it does not

`core/dataset.py` has a working on-device `CaptureStore` — `training.capture` in
`config.yaml`, off by default. Its `Sample` dataclass carries `id`, `session`,
`captured_at`, `image`, `event_type`, `rule_id`, `confidence`, `module`,
`model_version`, `review`, `notes`, `detections`. It writes `manifest.jsonl` and
names files `{session}_{sample_id}.jpg`. Useful, and worth using for opportunistic
in-the-wild capture.

**It is not sufficient for this protocol, for one specific reason.**
`core/dataset.py::split_by_session` splits by **session**, not by **subject**:

```python
def split_by_session(samples, ratios=(0.7, 0.15, 0.15)):
    """Split into train/val/test by SESSION, never by frame."""
```

Session-disjoint is a real improvement over frame-random, and the docstring's
reasoning is right as far as it goes. But one subject recorded on Monday and again
on Thursday produces two sessions, and a session split will happily put one in train
and one in val — putting the *same person, same room, same clothing* on both sides.
`Sample` has no `subject` field at all, so the split has no way to do better.

**Fixes, in order of preference.** They are named here as the necessary work; they
are outside this document's file scope and are for whoever owns `core/`:

1. Add `subject` to `Sample` and add a `split_by_subject` alongside `split_by_session`.
2. Until then, enforce **one subject per session** as a collection rule, which makes
   session-disjoint imply subject-disjoint. This costs nothing at capture time and
   is the cheapest way to make the existing code correct for this dataset.

Adopt (2) now regardless of whether (1) happens.

---

## 7. Manifest format, and running the existing harness against it

### 7.1 What the harness actually reads

Verified against `benchmarks/accuracy_state_farm.py` and
`scripts/prepare_state_farm.py`:

* Manifests live at `DATA_ROOT/{split}_manifest.csv`, where
  `DATA_ROOT = <repo>/data/state_farm` and `split ∈ {train, val}` — those two names
  are the only values `--split` accepts (`accuracy_state_farm.py:422`).
* The writer declares exactly three columns
  (`prepare_state_farm.py:142`):

  ```python
  csv.DictWriter(f, fieldnames=["subject", "classname", "img"])
  ```

* The reader is `csv.DictReader` (`accuracy_state_farm.py:127`) and touches exactly
  three keys: `row["classname"]` (grouping and per-class sampling),
  `row["img"]`, and `row["subject"]` (the run header).
* Images resolve as (`accuracy_state_farm.py:143`):

  ```python
  DATA_ROOT / "train" / row["classname"] / row["img"]
  ```

  Note the literal `"train"` — **for both splits.** Images live under
  `<root>/train/<classname>/<img>` regardless of which manifest names them. Getting
  this wrong is the first `FileNotFoundError` anyone building a sibling dataset will
  hit.
* `PHONE_POSITIVE = ("c1","c2","c3","c4")` and `PHONE_NEGATIVE = ("c0","c9")` are
  module-level constants.

**`csv.DictReader` ignores columns it was not asked for.** So an extended manifest
with the three required columns plus everything else this protocol needs is
consumable by the existing harness **unmodified**. That is the integration point.

### 7.2 The SensorDyme manifest

```csv
subject,classname,img,clip_id,domain,scenario,session,src_frame,phone_visible,phone_in_hand,phone_mounted,phone_touched,subject_present,occlusion,face_visible,eyes_closed,gaze_target,held_object,object_near_mouth,illumination,geometry_variant,ambiguous
s12,c3,sd_drive_d03_s12_20260912a_t07_f000432.jpg,sd_drive_d03_s12_20260912a_t07,drive,d03,20260912a,432,1,1,0,0,1,0,1,0,phone,cell phone,0,daylight_indirect,g2,
```

Column contract:

* `subject`, `classname`, `img` — **required, exact names, first three columns.**
  This is the harness's contract; do not rename or reorder them.
* `classname` — for the DRIVING domain, use **State Farm's own `c0`–`c9`
  vocabulary**, mapped from the SensorDyme scenario code. That single decision is
  what lets `accuracy_state_farm.py` run against SensorDyme footage with no code
  change at all, and it lets the two datasets be scored side by side — which is the
  domain-shift measurement this whole exercise is for. Keep the SensorDyme scenario
  in the separate `scenario` column so nothing is lost.
* `img` — the filename only. Path is reconstructed as
  `DATA_ROOT/train/<classname>/<img>`.
* Everything after `img` — the §4.5 frame-level labels and enough clip identity to
  join back to the sidecar JSON. Ignored by the current harness; consumed by the
  baby and fall harnesses.
* `ambiguous` — a comma-separated list of field names that went to
  `held_out_ambiguous` (§4.6), empty otherwise.

Scenario → `classname` mapping for DRIVING (`c5`–`c8` deliberately unused; the
harness excludes them from the negative set because a phone may legitimately be in
frame):

| Scenario | `classname` | Rationale |
|---|---|---|
| `d00`, `d16`, `d17` | `c0` | safe driving — no phone. Harness negative. |
| `d06`, `d07` | `c9` | talking to passenger — no phone in the driver's hand. Harness negative. |
| `d01` | `c2` / `c4` | phone at the ear, right/left. Harness positive. |
| `d02`, `d03`, `d04` | `c1` / `c3` | texting, right/left. Harness positive. |
| `d05` | `c0` | mounted, untouched → not a phone-in-hand positive. `phone_mounted: 1` carries the real information. |
| `d14` | `c6` | drinking. Excluded from both sets, as State Farm's own is. |
| `d08`–`d13`, `d15` | inherit from the underlying behaviour | a nighttime texting clip is `c1`; conditions live in `illumination` |

### 7.3 Layout

```
data/sensordyme/
  train_manifest.csv
  val_manifest.csv
  train/                      # literal "train", for BOTH splits -- see §7.1
    c0/ ... c9/               # driving frames, State Farm vocabulary
    b00/ ... b15/             # baby frames, SensorDyme scenario codes
    f00/ ... f16/             # fall frames
  clips/
    <clip_id>.mp4
    <clip_id>.json
  consent/                    # see §8.3 -- access-controlled, separate volume
```

Run the existing harness against it by pointing `DATA_ROOT` at
`data/sensordyme`. The `--baseline` / `--compare` regression gate, the confidence
sweep, and the exit-code contract (`0` ok, `1` regression, `2` could not run) all
work unchanged.

**What the existing harness cannot do**, stated plainly: `PHONE_POSITIVE` and
`PHONE_NEGATIVE` are phone-specific, and its three metrics (phone PR, person recall,
face recall) are frame-level availability measures. Nothing in it can score a
*temporal* detection. `fall`, `flailing`, `unusual_stillness`, `person_absent` and
`choking_hazard` need a sibling harness that replays a clip through the module and
compares emitted events against the §4.3 timestamps and the §4.3 scoring window.
That harness does not exist. Build it against the same manifest schema, the same
subject-disjoint split, and the same `--baseline`/`--compare` exit-code contract, so
the two are interchangeable in CI.

---

## 8. Privacy and consent

This section is not a formality appended to a technical document. The dataset is
video of infants and of identifiable drivers. Handle it accordingly, and read §8.7
before staging anything.

### 8.1 Written consent, before any recording

One signed form per subject, before their first session. For an infant, the form is
signed by a parent or legal guardian, who must be **physically present for every
session**. The form states, in plain language:

* **Who** is recording, and how to reach a named human being.
* **What** is recorded: 640×480 video, **no audio**, of the described scenarios.
* **Why**: training and evaluating an automated safety-monitoring system.
* **Where it goes**: named storage locations, named people with access. A DPA-bound
  processor may appear here **only if named on the form the subject signed.**
* **Where it does not go**: no public dataset, no publication, no marketing, no
  model demo, no third party outside the named list, no generic-cloud upload.
* **How long**: the retention period in §8.4, as an explicit date.
* **Withdrawal**: how, to whom, with what effect — including the honest limit in §8.5.
* **Separate, separately-initialled opt-ins** for: (a) use in training; (b) use in
  evaluation; (c) retention beyond the default period; (d) any still frame appearing
  in an internal document. Bundled consent is not consent, and (d) is the one people
  most often want to decline.

Give every subject a signed copy. Log consent under `consent_id` in the clip
sidecar. **A clip with no `consent_id` does not enter the dataset**, and the
ingestion script should refuse it rather than warn.

For a driver: the form covers **everyone who will be in the vehicle**. A passenger
in `d06` is a subject and signs their own form.

### 8.2 Where footage may not go

Absolute, no exceptions without a fresh signed consent naming the destination:

* **No general-purpose cloud storage of raw footage.** No Drive, Dropbox, S3 bucket,
  personal NAS, or "just to share it quickly". Devices and one controlled store.
* **No third-party annotation service without an executed DPA** naming this dataset,
  specifying deletion on termination, prohibiting sub-processing without consent,
  and prohibiting use of the data to train the vendor's own models. Not a click-through
  ToS — a signed agreement. If procurement cannot get one, annotate in-house.
* **No frames through any third-party API**, including VLM APIs, for labelling
  assistance or any other purpose. Note that `config.yaml`'s `mac_max` profile ships
  `vlm.enabled: true` pointed at a hosted endpoint — that is a development profile
  and it must never be pointed at collected footage.
* **No footage in issue trackers, PRs, chat, or slide decks.** If a frame is needed
  to explain a bug, use a synthetic or a consenting team member's own clip.
* **No footage on a laptop that leaves the building** unless the volume is
  full-disk-encrypted and the copy is deleted on return.

The device itself is already built this way and the collection process should not be
the weak point: `config.yaml` records that annotated frames "stream in-memory over
HTTP to connected browsers only — never written to disk", that evidence snapshots
are opt-in and never uploaded, and that `training.capture` is off by default with no
network path out. **The rig is the risk, not the product.**

### 8.3 Storage and access

* One encrypted volume, one location. Full-disk encryption at rest.
* Named access list, reviewed monthly, minimum necessary.
* Access logged.
* The `subject_id → real identity` mapping is the crown jewel: **one file, on a
  separate encrypted volume, accessible to fewer people than the footage.** It is
  the only thing standing between "subject s07" and a named person, and it should
  never be in the same directory tree, backup set, or repository as the clips.
* **Never in git.** `.gitignore` the data root before the first clip lands — the
  State Farm dataset is already gitignored for size, and this one is gitignored for
  a much better reason.
* Backups inherit every rule above, including deletion. A backup that survives a
  withdrawal request is a withdrawal that did not happen.

### 8.4 Retention

**Default: 24 months from the recording date**, stated as an explicit date on the
consent form. Then deleted, unless the subject has separately opted in to an
extension and been asked again.

Set a calendar reminder at 23 months. Retention that depends on someone remembering
is retention that becomes indefinite.

Derived artefacts — extracted frames, crops, annotations — carry the **same** expiry
as the clip they came from. The ingestion tooling should be able to answer "what is
derived from clip X" in one command, because §8.5 depends on it.

### 8.5 Withdrawal

A subject (or a guardian) may withdraw at any time, for any reason, without giving
one, by contacting the named person on the form. On withdrawal:

1. **Acknowledge within 3 working days**, in writing, to a human.
2. **Within 30 days**: delete every clip, every extracted frame, every crop, every
   annotation row, and every manifest entry for that subject, from primary storage
   **and from every backup**. Remove the `subject_id` mapping row last.
3. Re-run the affected split (§9) — removing a subject changes a subject-disjoint
   split by definition, and a stale split silently reintroduces leakage.
4. **Confirm completion in writing**, listing what was deleted.
5. Record the withdrawal in a log that contains the `consent_id` and the date, and
   **no other identifying information**. That log is what proves the deletion
   happened; it must not itself become a record of who was filmed.

**Be honest about the limit.** A model already trained on a subject's footage cannot
be un-trained. Say this on the consent form, in advance, in plain words — do not
discover it at withdrawal time. The commitments that *can* be kept: the footage is
deleted, it is not used in any future training run, and any model retrained after
the withdrawal is trained without it. If the withdrawal is on serious grounds and a
shipping model was trained on the footage, retraining without it is the right call —
budget for it as a real possibility rather than treating it as unthinkable.

### 8.6 De-identification — and where it honestly cannot apply

Faces cannot be blurred in the driving set. `driver_attention` runs MediaPipe
FaceMesh on every frame; EAR, PERCLOS, yaw and pitch are **derived from the face**.
Blurring the face destroys the label and the signal together. Anyone claiming an
anonymised driver-monitoring dataset is either not measuring drowsiness or not
anonymising.

So the control for driving footage is **access, not transformation** (§8.3), and the
consent form must say so plainly: *your face is recorded, unblurred, and is the
thing being measured.*

Where de-identification does work, use it:

* **Baby domain.** The infant's face is not the signal — `baby_monitoring` reasons
  about person boxes, pose landmarks, motion energy and held objects. Blur faces on
  anything that leaves the core team.
* **Falls domain.** Same: `fall_detection` uses `core/posture.py` on pose landmarks.
  Faces are not needed and should be blurred on anything shared.
* **Any still frame in any document, ever.** Blur, or use a consenting team member.
* **Backgrounds.** Blur or crop anything identifying a location — house numbers,
  post, screens, number plates, prescription labels. `d13` deliberately varies
  backgrounds; vary them without capturing an address.
* **Never** put a name, a date of birth, or an address in a filename, a manifest
  column, or a sidecar. `subject_id` and `session` are the only identifiers, and
  neither should be derivable from a person.

### 8.7 Staging, and the scenarios that must not be staged as described

Some scenarios in §3 describe things nobody should be asked to do.

* **`b05` (infant fully occluded, face-down in bedding)** describes a suffocation
  risk. **Do not stage it with a real infant.** Use an appropriately sized,
  realistically weighted infant mannequin under real bedding, and label the clips
  `synthetic: true` in the sidecar so no accuracy claim silently rests on
  non-human footage. If the detector's behaviour on real infants under partial
  bedding is needed, `b04` — partial occlusion, adult present, continuously
  supervised — is the ceiling of what is acceptable.
* **`f00`–`f10` (falls).** Use trained stunt performers or trained volunteers, crash
  mats out of frame or dressed to look like flooring, and a spotter. **Do not
  collect falls from the elderly subjects the product is for.** The pose signal is
  what is being measured, and a younger performer's pose is close enough. Where
  older subjects can safely contribute, that is `f11`–`f16` — the negatives, which
  are where age-related gait and posture actually matter most and where the false-positive
  rate lives.
* **`d15` (drowsiness).** **Acted**, in a **stationary vehicle, engine off**. Do not
  induce fatigue, and do not record a genuinely drowsy person driving. Acted long
  closures and head nods exercise the same EAR and PERCLOS code paths.
* **Every driving scenario involving a phone** is recorded **stationary, engine
  off**, in a parked vehicle. Nothing in `d01`–`d17` requires the car to move; a
  static cabin with a rolling background is not what YOLO or FaceMesh is looking at.
  If moving footage is genuinely needed for motion blur and lighting transitions,
  use a closed course with a safety driver and **only** for `d00`, `d08`–`d13`,
  `d16`, `d17` — never for the distraction scenarios.
* **A guardian is present, in the room, for every session involving a child**, and
  may stop the recording at any moment without explanation. Give them the stop
  button physically.

**Anyone may stop at any time.** Build breaks into the schedule and mean it.

---

## 9. Splits

**Subject-disjoint. Always. No exceptions.**

This is not a preference; it is the single property that makes the existing State
Farm baseline trustworthy, and `scripts/prepare_state_farm.py` says why:

> The split is BY DRIVER (subject id), not by image: the dataset contains 26 drivers
> photographed in bursts, so a random per-image split leaks near-duplicate frames
> into validation.

SensorDyme's data is worse in this respect, not better. State Farm's bursts are
seconds apart; a 30 fps clip's adjacent frames are 33 ms apart and near-identical. A
frame-random split on this dataset would produce a model that scores beautifully and
detects nothing in a new room.

### 9.1 Policy

1. **Assign whole subjects to exactly one of train / val / test.** A subject never
   appears in two splits.
2. **Sessions inherit the subject's split**, so the same room, the same clothing,
   and the same rig geometry never straddle it.
3. **Target 70 / 15 / 15 by subject**, filled greedily over the sorted subject list
   to approximate the ratio by *frame count* — matching
   `prepare_state_farm.py::pick_val_subjects`, which fills to a fraction of images
   rather than a fraction of subjects. Exact ratios are impossible with whole
   subjects; that is the price and it is worth paying.
4. **Deterministic.** Sort subject ids, seed everything, write the chosen subject
   lists into the manifest directory. Re-running the split script must reproduce it
   byte for byte, or a "regression" in the `--compare` gate may just be a reshuffle.
5. **Test is opened once**, at the end, for the number that gets published. Every
   tuning pass runs against val. `config.yaml`'s `yolo_confidence_threshold` comment
   records a full threshold sweep against the val split — that is exactly the right
   use, and exactly why test must stay sealed.
6. **Stratify while respecting (1).** Every scenario should appear in all three
   splits, and the sensitive attributes of §5.3 should be spread rather than
   clustered. With 12 subjects this constrains hard; with 25 it is comfortable —
   another reason for the higher target.
7. **`held_out_ambiguous` clips (§4.6) are in no split.** They are reported as a
   count and excluded from every metric.

### 9.2 Verify the split, do not assume it

Write the check and gate on it:

```
assert set(train_subjects) & set(val_subjects) == set()
assert set(train_subjects) & set(test_subjects) == set()
assert set(val_subjects)   & set(test_subjects) == set()
assert every session appears in exactly one split
assert every clip_id appears in exactly one split
```

Run it in CI on every manifest change. A leaked split is invisible in the metrics —
it makes them *better* — which is precisely why it needs a machine to catch it.

---

## 10. What this protocol does not yet know

Stated up front so nobody mistakes the document's confidence for the data's.

* **How much the domain shift is actually worth.** The 39.7 % face availability is
  expected to rise substantially on a driver-facing camera. By how much is unknown
  until §3.1 is collected. Do not pre-announce a number.
* **Whether 30 clips per scenario is enough in practice.** The statistics in §5.1
  are exact; whether the *variance between scenarios* is larger than the variance
  within them is an empirical question the pilot answers.
* **The temporal-detection harness does not exist** (§7.3). Until it does, `fall`,
  `flailing`, `choking_hazard`, `person_absent` and `unusual_stillness` have no
  automated scoring path and this dataset cannot be turned into a number for any of
  them.
* **`choking_hazard` may be unmeasurable as configured.** Its threshold (0.30) was
  set by arithmetic against a *single* hardware observation of ~0.38 association
  (`config.yaml`, `hand_association_threshold`). If real footage shows association
  routinely below 0.30, the feature has no operating point and the fix is upstream
  of any threshold. `b10` is the scenario that finds out; treat its result as a
  design question, not a tuning question.
* **Infant body proportion across the age bands** may exceed what a COCO-trained
  `person` class handles. `confidence_threshold: 0.4` is already lowered for baby
  mode ("children are smaller/lower-conf") on intuition, not measurement. `b06` and
  the age bands in §5.3 are what turn that into a number.
* **The `mirror_check_yaw_deg` / `rearview_tolerance_seconds` path** was structurally
  dead until 2026-08-29 and has been exercised only by construction, never by a real
  head turn. `d16` is its first real test, and it should be treated as a first test
  rather than a regression check.
* **Nothing here is validated on the Hailo backend.** All baseline accuracy numbers
  above are CPU. `docs/HARDWARE_VERIFICATION.md` item 6 — Hailo/CPU detection parity
  within a 0.03 tolerance — is a prerequisite for this dataset's numbers meaning
  anything about the shipping device. Run it before the pilot, not after the target
  tier.
