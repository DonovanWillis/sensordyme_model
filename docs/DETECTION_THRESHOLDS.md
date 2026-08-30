# Detection confidence thresholds

> **PHASE 5 — READ THIS FIRST.** Every number in §§1-7 below was measured against
> a detector binary that is **no longer the one this product runs**. That binary
> (sha256 `4c71eab8…`) could not be reproduced — it matched no Hailo Model Zoo
> release. It was retired and replaced with a checksum-pinned build, and every
> threshold was re-derived against the replacement. §0 is the authoritative
> version; the sections below are kept because the reasoning in them is still
> correct and because deleting superseded evidence is how a project loses the
> ability to tell what changed.

## 0. The demo binary, and the thresholds measured on it

**Authoritative detector**

| | |
|---|---|
| model | `damoyolo_tinynasL20_T` (DAMO-YOLO tinynasL20_T, COCO-80) |
| build | Hailo Model Zoo **v2.15.0**, `hailo8l` |
| sha256 | `e69136d3ebf3cc8ebaf3da5417a59962b139f3d3d67d715e9a9b1ae10963960e` |
| size | 18,554,430 bytes |
| source | `https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/damoyolo_tinynasL20_T.hef` |
| structure | 6 raw output tensors, **no on-chip NMS** |
| licence | DAMO-YOLO Apache-2.0; Model Zoo build carries no copyleft |
| pinned in | `setup_pi.sh` (checksum-verified at install), `config.yaml` `accelerator.yolo_hef_sha256` |
| latency | mean 22.07 ms, p95 22.75, p99 22.85, max 26.47 (300 samples, Pi 5 + Hailo-8L) |

**Threshold expressibility on this binary.** This build is raw-decode: it emits 6
raw tensors and performs no NMS on chip, so `_nms_score_threshold` is `None` and
**every threshold below is expressible exactly as written**. The 0.20 floor
documented in §2b does *not* bind on this binary. It is retained as a portability
guard — swap in an NMS-on-chip `.hef` and anything under 0.20 silently becomes
0.20 — and `tests/test_phase4_detection_thresholds.py` enforces it so that swap
cannot happen quietly.

**Measured on the demo binary** — State Farm val split, subject-disjoint, seed 0,
600 images (60/class), backend `hailo`, via `benchmarks/accuracy_state_farm.py`.
Baseline written to `benchmarks/results/phase5_demo_binary.json`.

Phone (`cell phone`):

| conf | recall | FP rate | precision | F1 |
|---|---|---|---|---|
| 0.10 | 99.6% | 86.7% | 69.7% | 0.820 |
| 0.15 | 93.8% | 70.8% | 72.6% | 0.818 |
| **0.20** | **88.3%** | **52.5%** | **77.1%** | **0.823** ← shipped |
| 0.25 | 82.5% | 30.0% | 84.6% | 0.835 ← best F1 |
| 0.35 | 62.5% | 19.2% | 86.7% | 0.726 |
| 0.50 | 23.8% | 2.5% | 95.0% | 0.380 |

Person (COCO class 0), on the same 600 single-occupant images:

| conf | availability | spurious >1 box | old binary |
|---|---|---|---|
| 0.25 | 100.0% | 60.0% | 51.5% |
| 0.40 | 100.0% | 17.8% | 16.0% |
| **0.50** | **100.0%** | **4.0%** | 4.5% ← shipped |
| 0.60 | 99.8% | 0.8% | 0.5% |
| 0.70 | 94.7% | 0.5% | — |

**Regression gate vs the Phase 3 baseline** (`--compare`, tolerance 0.03):
`face_availability` 0.3900→0.3900, `person_availability` 1.0000→1.0000,
`phone_best_f1` 0.8475→**0.8354** (−0.0121). Exit 0, nothing regressed past the
gate. Note the harness's `phone_recall_at_shipped 0.6625→0.8833 "improved"` line
is **not** a model improvement — the Phase 3 baseline recorded `shipped_conf:
0.35` and the config now ships 0.20. At *matched* thresholds the new binary is
marginally **worse** on phone (0.35: 66.2%→62.5%) and marginally worse on
spurious person boxes at low confidence, and marginally **faster** (24.66→22.07
ms). All within the gate; none of it changes a threshold decision.

**Final demo thresholds**

| Module | Class | Threshold | Reason | Evidence | Status |
|---|---|---|---|---|---|
| driver_attention | cell phone | **0.20** | Near-max recall (88.3%) at acceptable precision (77.1%); a bare `phone_detected` is log-only by design, so FN is the worse error | Phase 5 sweep above | **MEASURED** |
| driver_attention | person | **0.50** | 100% availability at 4.0% spurious; capped at 0.50 not 0.60 so the dev/CI CPU path (YOLO11n, 86% availability at 0.60) stays correct too. Since Phase 4 that path is dev-only — a device has no CPU fallback — so the cap is worth re-deciding; see §2 | Phase 5 person sweep | **MEASURED** |
| baby_monitoring | person | **0.40** (held) | FP already defended structurally by `InfantIdentifier` (zone containment + size stability + temporal identity); FN is an undefended monitoring gap over a sleeping infant | Declined to raise — infant availability unmeasured | **HELD, JUDGEMENT** |
| baby_monitoring | held-object floor | **0.15** (held) | Choking is the highest-severity alert and has ~0.38 association ceiling against a 0.30 threshold; FN clearly worse | Phase 3 measurement | **HELD, JUDGEMENT** |
| occupancy | person | **0.50** (held) | Strongest FP absorption in the repo (confirm_frames 3, zone_confirm_s 2.0, PersonTracker); per-person recall at security range unmeasured | Declined to change | **HELD** |
| security presets | person | **0.40 / 0.35** (held) | Same reason as occupancy — the 100% availability figures are adults at close range and do not transfer to security-camera geometry | Declined to raise | **HELD** |
| fall_detection | — | **none** | Consumes no YOLO; MediaPipe Pose only. Deliberately has no detection gate in front of a fall alarm | `test_fall_detection_consumes_no_yolo_detections` | **N/A BY DESIGN** |

Two Phase 4 recommendations were **declined**: raising baby_monitoring to 0.50
and raising the security presets. Both rested on 100%-person-availability figures
measured on **adults at close range** in a car cabin, which do not transfer to an
infant in a crib or a person at security-camera distance. The missing
measurements are named in §6.

---


Every place in this repo that passes a `conf=` to an object detector, what it is
for, which failure mode is the dangerous one there, and what the value should be
now that the shipping detector is **DAMO-YOLO on a Hailo-8L NPU** rather than
YOLO11-nano on CPU.

Thresholds inherited from YOLO11n are wrong for DAMO-YOLO **in both directions**,
because the two models are calibrated differently. That is the whole reason this
document exists.

---

## 1. The evidence

Measured on a real Pi 5 + Hailo-8L over 600 subject-disjoint State Farm images.

**Person — spurious-box rate vs person availability**

| conf | damoyolo multi | damoyolo avail | yolov11n multi | yolov11n avail | yolov5s avail |
|------|---------------|----------------|----------------|----------------|---------------|
| 0.25 | 51.5% | 100.0% | 8.5% | 100.0% | 91.5% |
| 0.40 | 16.0% | 100.0% | 3.0% | 96.5% | 38.5% |
| 0.50 | 4.5%  | 100.0% | 1.0% | 94.0% | 9.5% |
| 0.60 | 0.5%  | 100.0% | 0.0% | 86.0% | 0.0% |

"multi" = more than one person box on a single-person image. CPU YOLO11n baseline
was 7.2% at 0.25.

**Phone (COCO `cell phone`), damoyolo on Hailo**

- recall 66.2% @ 0.35
- best F1 0.848 with 93.8% recall @ 0.20 (implied precision ≈ 77%, derived from
  F1 and recall — not measured directly)
- at conf 0.10 the raw-decode path emitted **14 phone boxes on a single image**

**Reading it.** DAMO-YOLO is markedly better calibrated: it holds 100% person
availability all the way to 0.60, where YOLO11n has collapsed to 86% and YOLOv5s
to zero. Between 0.25 and 0.60 DAMO-YOLO's spurious-box rate falls 100× at no
measured availability cost. On the person class that is close to a free lunch —
**but only on the DAMO-YOLO backend** (see §2).

**What this evidence is not.** State Farm is a passenger-side, across-the-cabin
view of a seated adult, close to the lens, in daylight. `benchmarks/accuracy_state_farm.py`
says so in its own header and calls its numbers domain-shifted lower bounds. It
is therefore evidence about **adults at close range**. It is not evidence about
an infant in a crib, a partially-covered infant, or a person at the far end of a
security-camera field of view. Everywhere below that those cases matter, this
document says the measurement is missing rather than extrapolating into it.

---

## 2. The constraint that caps every person threshold: the CPU path

`core/detectors.py:get_yolo_model` → `core/accel.py:maybe_hailo_yolo`. If
`accelerator.yolo_hef` is null, or Hailo init raises for any reason,
`accel.py:251-253` logs the reason and returns `None`, and `get_yolo_model` then
tries the CPU path — ultralytics YOLO11n.

**What changed in Phase 4, and what it does and does not change here.** This
section was written when `config.yaml` shipped `yolo_hef: null`, so the CPU
fallback was the *default* device configuration. It is not any more:

- `config.yaml` now ships `yolo_hef: models/damoyolo_tinynasL20_T.hef`, provisioned
  and checksum-verified by `setup_pi.sh`.
- `requirements-device.txt` ships no `ultralytics`, so **on a device there is no
  CPU fallback at all**. `get_yolo_model` raises and the service boots degraded
  (`monitoring_active: false`) rather than switching detectors.
- The CPU path survives on **developer machines and CI**, which install
  `requirements.txt`.

So the same config key still feeds two detectors with opposite calibration curves
— but they are now split by *destination* rather than by runtime luck. At 0.60
DAMO-YOLO is at 100% person availability and YOLO11n is at **86%**: one person in
seven not seen at all. A threshold tuned purely to the DAMO-YOLO column is
therefore correct on hardware and a 14% blind spot on a dev machine, instead of a
14% blind spot on hardware the moment the accelerator dropped out.

**Consequence: no person threshold in this document goes above 0.50**, which is
the last row where both backends are acceptable (DAMO 100% / 4.5% multi;
YOLO11n 94% / 1.0% multi). **This cap is kept, but its justification has narrowed
and is now worth re-deciding** — it no longer protects a shipping configuration,
only the dev/CI one. Unlocking 0.60 requires either making the threshold
backend-aware in code, or accepting that dev-machine person detection is
differently calibrated from the device's. Whoever owns thresholds should make that
call explicitly; this document is flagging it, not making it.

## 2b. The other constraint: the on-chip NMS floor at 0.200

`core/accel.py:398-430` distinguishes two kinds of `.hef`:

- **NMS-on-chip** (one output tensor): the score threshold is **compiled in** and
  cannot be changed at runtime. Measured on `yolov11n.hef`, its **0.200 floor
  caps phone recall at 25.4%** against CPU's 63.3%, because `conf` below 0.200 is
  *silently ignored*.
- **raw tensors** (several outputs, no NMS): decode is off-chip, `conf` is the
  only gate there is.

**Any configured value below 0.20 is a lie on one of the two builds.** It means
0.20 on an on-chip build and means itself on a raw-decode build — and on the
raw-decode build 0.10 is where 14 phone boxes appeared on one image. This is the
single strongest argument for the driver_attention change below: at 0.20 the
config means the same thing on both builds.

---

## 3. Threshold table

| Module | Class(es) | Current | Candidate | Evidence | FP risk | FN risk | **Final** |
|---|---|---|---|---|---|---|---|
| `driver_attention` (shared pass) | cell phone + 16 food/utensil classes + person | `yolo_confidence_threshold: 0.10` (config.yaml:306; code default `0.10`, `driver_attention.py:858`) | 0.20 | DAMO best F1 0.848 @0.20, 93.8% recall; 66.2% @0.35; 14 boxes/image @0.10; on-chip floor is 0.200 | Spurious `phone_detected` on the rising edge, with **no sustain gate** — the only genuinely unabsorbed FP path in the product | Missed phone = missed distraction; 0.35 misses one phone in three, and it is the module's headline claim | **0.20**, conditional on a `phone_detected` sustain gate (§5). Without that gate ship **0.25** as the interim. 0.10 must change either way — it is below the on-chip floor. |
| `driver_attention` | person (rides the same pass) | — no separate gate — | **new** `person_confidence_threshold: 0.50` | 0.50 → 4.5% multi (DAMO) / 94% avail (YOLO11n) | Extra person boxes extend face-loss patience from 2.5 s to 8.0 s (`face_lost_with_body_seconds`) and can move the "driver" pick in `_eating_boxes_near_face`, which takes the **largest** person box | A missed person shortens patience — the *safe* direction; person here is a corroboration signal, not an alert source | **0.50 via a new post-filter** (§4/§5). Person cannot ride a 0.20 gate; at 0.20 the DAMO multi-box rate is worse than the 51.5% measured at 0.25. |
| `baby_monitoring` | person, bottle, 17 `object_classes` | `confidence_threshold: 0.4` (config.yaml:446; code default 0.4, `baby_monitoring.py:816`) | 0.50 | 0.40 → 16.0% multi; 0.50 → 4.5%, both at 100% adult availability (DAMO) | A spurious box is a **candidate the InfantIdentifier must reject**. It no longer becomes "the baby" (positive identification replaced smallest-box-wins), but it still drives `min_margin` ties → `infant_unidentified` declines, i.e. a monitoring gap | A dropped infant box loses the subject for choking-hazard, flailing and stillness. **Infant availability at 0.50 is unmeasured** — the measured 100% is adults at close range | **0.50 — judgement call, not measured for this subject.** Rollback trigger stated in §6. |
| `occupancy` | person (+ vehicle/animal when zones configured) | `confidence_threshold: 0.5` (config.yaml:421; **required key**, `occupancy.py:82`) | 0.50 (no change) | 0.50 → 4.5% multi / 100% avail (DAMO), 1.0% / 94% (YOLO11n) | Phantom occupant → `occupancy_change`, false `zone_violation`, false `crowd`. But this module has the strongest FP absorption in the repo: `confirm_frames: 3`, `zone_confirm_s: 2.0`, `linger_s: 30`, `crowd_confirm_s: 5.0`, plus `PersonTracker` | A missed intruder is the product's entire purpose. **Per-person recall at security-camera range is unmeasured** — "availability" only says ≥1 box existed on a close-up single-subject image, which is a weak proxy for a wide multi-person scene | **0.50 — hold.** Raising it trades an unmeasured FN risk for FP headroom the temporal gates already provide. |
| `occupancy` preset `security_monitoring/away` | person | `confidence_threshold: 0.4` (presets.yaml:77) | 0.50 | Same rows as above | 16.0% multi on DAMO | The override exists to bias toward recall at range | **0.50** — judgement call. On DAMO the override buys nothing: availability is already 100% at 0.60, so dropping to 0.40 purchases spurious boxes, not recall. |
| `occupancy` preset `security_monitoring/package_watch` | person | `confidence_threshold: 0.35` (presets.yaml:80) | 0.50 | Between the 0.25 (51.5%) and 0.40 (16.0%) rows | Worst FP exposure of any shipped value on the person class | Same as above | **0.50** — judgement call, same reasoning. |
| `core/hand_perception.py` (via `predict_roi`, on a hand-sized crop) | `object_classes` + unknown | `unknown_object_confidence_floor: 0.15` (code default; config key `hand_unknown_object_floor` exists but is **absent from config.yaml**) | 0.15 (no change) | None. No measurement exists for a detector run on a hand-sized ROI crop; the 600-image sweep is full-frame | Spurious ROI objects feed choking-hazard risk. Partly absorbed: the score enters only as a multiplier, `0.7 + 0.3·conf` (`hand_perception.py:276`), and `hand_association_threshold: 0.35` is the real gate | **Choking hazard is the highest-severity alert in the product and has almost no headroom** — measured ceiling ~0.38 association against a 0.30 threshold. Suppressing evidence here is the worse error | **0.15 — hold**, but surface it in config.yaml with a comment that on an NMS-on-chip `.hef` it is silently 0.200. Missing measurement named in §6. |
| `core/hand_perception.py` | labelled-object bar | `detection_confidence: 0.35` (code default; config key `hand_detection_confidence`, **absent from config.yaml**) | 0.35 (no change) | None | This is a *labelling* bar (known class vs `unknown_object`), not a detection bar — a wrong label, not a phantom box | A real object degrades to `unknown_object`, which still scores | **0.35 — hold**, surface in config.yaml. |
| `driver_attention_max` (mac_max profile) | cell phone + 8 classes | `detector_confidence_threshold: 0.4` (config.yaml:687) | 0.4 (no change) | **The evidence does not apply.** yolo11x on MPS; an explicit `device=` pin bypasses the Hailo path entirely (`detectors.py` docstring) | — | — | **0.4 — no change.** Mac-only accuracy-ceiling profile, not the shipping product. |
| `driver_attention_openpilot` | cell phone | `phone_confidence_threshold: 0.35` (config.yaml:385) | 0.35 (no change) | **Does not apply**: `phone_device: mps` pins the device, bypassing Hailo | — | — | **0.35 — no change.** Dev comparison module. See the caveat in §4. |
| `driver_attention_skinnemoen` | cell phone | `phone_confidence_threshold: 0.35` (config.yaml:403) | 0.35 (no change) | Same as above | — | — | **0.35 — no change.** |
| `driver_attention_dashsentinel` | — | — | — | Loads no detector (`grep` finds no `get_yolo_model`); MediaPipe only | — | — | **No threshold. Not a YOLO consumer.** |
| `fall_detection` | — | — | — | **Consumes no YOLO detections at all.** Imports only `mediapipe`, `core.posture`, `core.activity`, `core.zones`; no `get_yolo_model`, no `predict(` | — | — | **No threshold assigned.** See the separate defect in §4. |

---

## 4. Hardcoded / structurally trapped values

Ordered by how much damage the trap can do.

**1. `driver_attention` gates the person class on the phone threshold. HIGH.**
`modules/driver_attention.py:1057-1060` issues one `predict()` for
`{*yolo_class_ids, _PERSON_CLASS}` at a single `conf`. config.yaml:324 already
flags this ("this threshold gates EVERY class in the same YOLO pass"). Under
YOLO11n the tension was tolerable; under DAMO-YOLO it is not, because the class
the threshold *must* be low for (phone: 0.20) and the class it *must* be high
for (person: 0.50) are now 100× apart in spurious-box rate. **Should become
configurable** — a separate `person_confidence_threshold`, applied as a post-hoc
filter on `box.conf` for `class_id == _PERSON_CLASS`, since a single `predict()`
call cannot carry per-class thresholds. Specified in §5.

**2. `core/accel.py:357` `_DEFAULT_NMS_IOU = 0.50` has no path from config. MEDIUM.**
`HailoYOLO.__init__` accepts `nms_iou`, but `maybe_hailo_yolo` calls
`HailoYOLO(hef_path)` with one argument (`accel.py:248`), so the parameter is
unreachable. **Should become configurable** under `accelerator:` for completeness
— but low urgency, and the code comment at `accel.py:343-357` is right about why:
the measured sweep (55.0% → 48.0% multi across IoU 0.70 → 0.40) shows IoU is a
weak lever because the extra boxes are spatially separate false positives, not
overlapping duplicates. Confidence is the lever.

**3. `hand_perception` and `InfantIdentifier` thresholds are configurable but
invisible. MEDIUM.** `hand_detection_confidence`, `hand_unknown_object_floor`,
`infant_zone_min_containment`, `infant_identification_min_confidence`,
`infant_identification_min_margin`, `infant_min_observation_s`,
`infant_reacquire_s` are all read via `config.get(...)` with a code default, and
none appears in config.yaml. They are tunable only by someone who has read the
source. **Should be surfaced in config.yaml at their current defaults** — no
behaviour change, but they stop being invisible.

**4. `occupancy` uses `config["confidence_threshold"]` — required, no default. LOW.**
`occupancy.py:82`. Every other module uses `.get(key, default)`. A config missing
the key raises `KeyError` at construction rather than degrading. Arguably the
better contract, but the inconsistency is worth knowing about; if it stays,
`baby_monitoring` and `driver_attention` should match it rather than the reverse
— an invisible default on a safety threshold is worse than a startup failure.

**5. `predict_roi(conf=0.25)` and `HailoYOLO.predict(conf=0.25)` defaults. LOW.**
`core/detectors.py:73` and `core/accel.py:459`. Both are hardcoded defaults that
**no caller currently uses** — `hand_perception.py:389` passes
`conf=self._unknown_floor`, and every module passes `conf=` explicitly. Harmless
today, a trap for the next caller who omits it. Not worth making configurable;
worth making them non-defaulting (required keyword) so omission is a type error
rather than a silent 0.25.

**6. `fall_detection.py:622` reports a fabricated detection confidence. Not a
threshold, but adjacent and worth fixing.** `Detection(label="person", confidence=0.9)`
— the 0.9 is invented. `core/detections.py` says `DetectionFrame` is what
AI-configured rules evaluate against, and `RuleEngine` may compare confidences.
A rule reading fall_detection's detections is comparing against a literal that
means nothing. Either derive it from the posture confidence already available on
the next line, or make it explicit that this module reports presence without a
confidence.

---

## 5. Regression tests worth having

I do not own `.py` files; these are specified precisely enough to write directly.
Suggested home: `tests/test_detection_thresholds.py` for the config-level guards,
with the module-behaviour tests going alongside their existing module test files.

**T1 — `test_no_shipped_threshold_below_hailo_onchip_floor`** (config-level, highest value)
Load `config.yaml` and `presets.yaml`. Collect every YOLO confidence value:
`modules.driver_attention.yolo_confidence_threshold`,
`modules.baby_monitoring.confidence_threshold`,
`modules.occupancy.confidence_threshold`,
`modules.driver_attention_openpilot.phone_confidence_threshold`,
`modules.driver_attention_skinnemoen.phone_confidence_threshold`,
`mac_max.detector_confidence_threshold`, and every `confidence_threshold` inside
a `presets.yaml` `overrides:` block.
**Assert every one is `>= 0.20`.** Failure message must cite `core/accel.py:398-410`:
below 0.200 an NMS-on-chip `.hef` silently ignores the value, so the config does
not mean what it says. This is the test that would have caught the current 0.10.

**T2 — `test_person_thresholds_safe_on_cpu_fallback`** (config-level)
For the person-consuming keys (`baby_monitoring.confidence_threshold`,
`occupancy.confidence_threshold`, the new `driver_attention.person_confidence_threshold`,
and both `occupancy` preset overrides), **assert `0.20 <= value <= 0.50`.**
Failure message must cite the YOLO11n column: 86% person availability at 0.60,
and `accelerator.required: false` means the fallback is automatic. This is the
guard against a future "DAMO holds 100% to 0.60, so let's use 0.60" change.

**T3 — `test_driver_attention_passes_configured_conf_to_detector`** (behaviour)
Build `DriverAttentionModule` with a stub model whose `predict` records its
kwargs. Drive enough frames to hit a stride tick (`yolo_every_n_frames`).
**Assert `predict` was called with `conf == config["yolo_confidence_threshold"]`
exactly** — not a rounded or re-derived value — and with `classes` containing
`_PERSON_CLASS` (0) and the cell-phone class (67). Guards against the threshold
being silently decoupled from config during the person-split refactor.

**T4 — `test_person_gate_boundary`** (behaviour; only meaningful after the §4.1 split)
With `yolo_confidence_threshold: 0.20` and `person_confidence_threshold: 0.50`,
feed a stub result containing person boxes at `conf` 0.49 and 0.51.
- Assert the 0.51 box is retained and `_last_person_visible is True`.
- Then feed only a 0.49 box: assert it is dropped and `_last_person_visible is False`.
- Then feed a cell-phone box at 0.21: assert it **is** retained in
  `_last_phone_boxes` — i.e. the person gate must not also gate the phone class.
The third assertion is the point of the test: it fails if someone "simplifies"
the split back into one threshold.

**T5 — `test_baby_monitoring_detect_uses_configured_conf`** (behaviour)
Stub the model on `BabyMonitoringModule`; call `_detect_people`.
**Assert `predict` received `conf == config["confidence_threshold"]`** and
`classes == sorted({0, 39, *object_class_ids})`. Pair with a config-level
assertion that the shipped value is `0.50`, whose failure message names the
missing measurement (infant availability on crib footage) so a future change
back to 0.40 is a deliberate act with a reason attached.

**T6 — `test_infant_candidate_set_rejects_spurious_box`** (behaviour, boundary either side)
Drive `InfantIdentifier.update` with (a) one stable small box for >3 s, then
(b) the same box plus a flickering second box present on only ~1 frame in 6.
**Assert the identification result is unchanged** — same `track_id`, same
confidence tier. This is the test that says the 0.40 → 0.50 change is defence in
depth rather than the only thing standing between the product and a wrong-subject
score.

**T7 — `test_fall_detection_consumes_no_yolo`** (structural, cheap, worth it)
**Assert `modules/fall_detection.py` imports nothing from `core.detectors`.**
One line. It pins the finding in this document so nobody later "unifies" the
person-detection path across modules and silently gives fall detection a
threshold it was deliberately never assigned.

**T8 — benchmark gate (not a unit test).** After any change here, run
`python benchmarks/accuracy_state_farm.py --baseline out.json` once, then wire
`--compare out.json` into CI. It already exits nonzero on regression beyond a
0.03 tolerance. That is the only thing in the repo that measures whether a
threshold change actually helped.

---

## 6. Missing measurements, named

Where the evidence does not exist, this is what would have to be measured rather
than guessed.

1. **Infant / small-subject person availability on DAMO-YOLO at 0.40 vs 0.50 vs 0.60.**
   Crib footage, including a partially-covered infant and a face-down infant.
   Gates the `baby_monitoring` 0.40 → 0.50 recommendation, which is currently a
   judgement call resting on adult-at-close-range data.
   *Rollback trigger until it exists:* if the soak test shows `infant_unidentified`
   or `person_absent` rates rising after the change, revert to 0.40 — the
   InfantIdentifier declining to score is a monitoring gap, and 0.50 is not worth
   one.
2. **Per-person recall at security-camera range and geometry**, DAMO-YOLO, 0.35
   through 0.60. "Availability" on single-subject close-ups does not answer the
   question `occupancy` actually asks ("how many people"). Gates any future change
   to `occupancy.confidence_threshold` and to the two preset overrides.
3. **Phone precision at 0.20 as a per-frame rate on driver-facing geometry**, and
   the resulting `phone_detected` rising-edge event rate per minute. The implied
   ~77% precision is derived arithmetic from a reported F1, not a measurement, and
   it is the number that decides whether 0.20 needs the sustain gate or merely
   benefits from it.
4. **`predict_roi` behaviour on hand-sized crops on the Hailo raw-decode path** at
   `conf` 0.15. The 14-boxes-at-0.10 observation was full-frame; a crop changes
   both the object's relative scale and the box count. Gates `hand_unknown_object_floor`.
5. **DAMO-YOLO food/utensil-class calibration.** Sixteen of the eighteen classes
   in `driver_attention.yolo_classes` are food and utensils, and nothing here
   measures them at all. They are spatially gated to the driver's upper region
   (`_eating_boxes_near_face`), which absorbs a lot, and `eating_risk` is a
   RiskFusion input rather than a standalone alert — so this is the lowest-priority
   gap of the five, but it is a gap.

---

## 7. Summary of recommended config changes

Ordered by safety importance. See the SendMessage to the integration lead for the
exact key/value form.

1. `modules.driver_attention.yolo_confidence_threshold`: **0.10 → 0.20** (0.25 if
   the sustain gate does not land this cycle). *Measured.* Below 0.20 the value is
   silently ignored on an NMS-on-chip build and FP-explosive on a raw-decode one.
2. `modules.driver_attention.person_confidence_threshold`: **new key, 0.50**,
   plus the post-filter in `_run_object_detections`. *Measured* for the value;
   *structural* for the split.
3. `modules.baby_monitoring.confidence_threshold`: **0.4 → 0.50.** *Judgement
   call* — measured on adults, not infants; rollback trigger in §6.1.
4. `presets.yaml` `security_monitoring/away` and `/package_watch`
   `confidence_threshold`: **0.4 and 0.35 → 0.50.** *Judgement call* — on DAMO-YOLO
   the recall these overrides were buying no longer exists to buy.
5. `modules.occupancy.confidence_threshold`: **hold at 0.5.** *Measured* that it
   is safe; *unmeasured* whether it is optimal at range.
6. Surface `hand_detection_confidence: 0.35`, `hand_unknown_object_floor: 0.15`
   and the five `infant_*` keys in config.yaml **at their current defaults**. No
   behaviour change; they stop being invisible.
7. Consider `accelerator.required: true` on any deployment that adopts a person
   threshold above 0.50 later — it is what makes the DAMO-YOLO column the only
   column that matters.
