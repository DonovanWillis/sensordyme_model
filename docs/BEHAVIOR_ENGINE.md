# Behaviour engine (M8.4, ALPHA)

The driver-safety pipeline from camera frame to buzzer, the assumptions it
stands on, and what M8.4 changed: owner decisions A (experimental alert
families are observations) and B (a visible phone is not risk), a typed
evidence layer with explicit abstention, shadow verdicts for the validated
alerts, driver association, conditional hand inference, and a scripted
regression and evaluation harness.

**Everything measured here is synthetic regression evidence** from scripted
scenarios through the real module (`sim/perception.py` `run_cabin`). None of
it is real-world accuracy. Nothing here was run on the Pi, the Hailo or the
camera (the device arrives 2026-11-15).

Line numbers in sections 1-9 refer to this branch at the M8.4 ALPHA commits;
section 10's refer to the M8.5 phone-handling commit (M8.5 shifted
`modules/driver_attention.py` by about 70 lines).

## 1. Pipeline map (frame -> alert)

| Stage | Where | What happens |
|---|---|---|
| Frame + time | `modules/driver_attention.py:2117` `process()`; `:2124` | capture time from the runtime; `StallAwareEvidenceClock` (`:935`) credits one frame interval across a stall |
| Object perception (strided) | `:2138`, `_run_object_detections` `:1723`, predict `:1733` | detector every `yolo_every_n_frames` (3): person (>= 0.50), phone/food (>= 0.20) |
| Hands (scheduled, B5) | `:1771` `_should_run_hands` | MediaPipe Hands on ticks with a phone box, else at most every 0.5 s |
| Driver relation | `phones_for_driver` `:1050`, `_eating_boxes_near_face` `:1668` | largest person box = "the driver" (shipped proxy) |
| Phone held? | `PhoneHoldEvidence` `:982`, update `:1799` | held / not_held (still >= 2 s, hands seen elsewhere) / unknown |
| Face signals | FaceMesh `:2142`; EAR `:2325`; head pose `:2352`; iris gaze `:2383` | every frame |
| Calibration | `AttentionCalibrator`, observe `:2389` | per-driver neutral pose + open-eye EAR over the first 10 s |
| Eye threshold | adaptive reference `:2398`; closure test `:2455` | iris veto, looking-down attribution (M8 D2) |
| Temporal aggregation | drowsiness `:2462`, PERCLOS `:2486`, distraction `:2555`, eyes_down `:2571`, phone_use `:2599`, face lost `:2249`, face-loss coast `:2186`, erratic hands `:2154` | `GracedConditionTracker` family (`modules/base.py`): sustain + release grace |
| Confidence | `core/confidence.py` (`margin_confidence`, `joint`) | how far past the threshold, in units of measured noise |
| **Evidence + shadow (M8.4)** | `_engine_step` `:2270` / `:2654` -> `core/behavior/engine.py` | typed evidence, abstention, phone level, driver association, shadow verdicts |
| Risk | `phone_risk_for` `:2680`; `RiskFusion` `:1070`, update `:2681` | weighted EMA score -> `elevated_risk` / `high_risk` (EXPERIMENTAL observations) |
| Events | `Event.now(...)` per tracker | one event per episode onset |
| Alert policy | `services/runtime.py:2749` `_dispatch_module_events` -> `core/alerts.py:832` `notify` | `permits()` `:794` refuses experimental families; episodes repeat/escalate (`observe` `:909`) |
| Severity / uplink | `services/runtime.py:2779`, `:2793`; `core/uplink.py:501` | experimental -> severity `log`, `experimental: true` |
| Backend | `cloud/app.py:4948` ingest, `:5208` notify loop, `:4678` row field | stored `log`, never notified, `experimental` on the events API |

### Unsupported assumptions (stated, not fixed unless noted)

| Assumption | Where it lives | Status after M8.4 |
|---|---|---|
| The largest person box is the driver | `phones_for_driver`, eating gate, face-loss corroboration | **M8.5:** production phone_use now relates a phone to the person `associate_driver` picks (seat region, else the box holding the measured face, else the largest box); eating gate and face-loss corroboration unchanged. The engine uses the same association and reports disagreement with the largest box (`agrees_with_largest_person`). Unevaluated. |
| A phone box near the driver + eyes off road = phone use | `phone_use_condition` | **M8.5:** production needs the phone HANDLED (held, or a handling cue with holding unobserved; section 10). Engine: needs a phone HELD (hand overlap); unheld -> ABSTAIN, mounted -> NOT_SUPPORTED. Risk no longer counts a visible phone (decision B). |
| Head orientation = gaze | fused head pose + iris offset | Iris offset is fused; no gaze-target geometry. `gaze_toward` compares directions only. |
| Glance vs sustained | trackers' sustain times | Engine reports `glance` vs `sustained` explicitly; same bars. |
| Missing face = risk / no eye state = open | face-lost risk; drowsiness resets on face loss | Engine: eye state / gaze MISSING -> ABSTAIN, bounded coast (1 s), never "open"/"on road". |
| Occlusion vs behaviour | sunglasses read whatever EAR the landmarks guess | Engine: dark, flat eye patch -> eye state MISSING -> drowsiness ABSTAIN. A dark cabin without IR also reads occluded (reported, honest). |
| Camera misalignment vs behaviour | calibration taken once | Engine: a near-motionless off-road pose for 30 s -> ABSTAIN "camera may have moved (recalibrate)". Production keeps alerting. |
| Device performance failure vs safe driving | stalls | Production credits one frame interval per stall; engine credits nothing across a stall; detector failure -> phone evidence INSUFFICIENT. |
| Empty vehicle vs absent driver | attention_lost repeats | Engine: nobody for 30 s -> ABSTAIN "vehicle may be empty"; blocked lens named as the cause. |

## 2. Decision A: experimental families are observations

`high_risk`, `elevated_risk`, `erratic_hand_movement` (`core/behavior/experimental.py`).

| Layer | Invariant | Code | Test |
|---|---|---|---|
| Alert policy | never actuates, opens no episode, records no delivery, unless `experimental_audible=True` | `core/alerts.py` `permits`, `notify` | `test_the_policy_never_actuates_an_experimental_family_by_default` |
| Opt-in | only the boolean `alerts.experimental_alerts_audible: true` | `main.build_alert_manager` | `test_only_the_explicit_boolean_opt_in_makes_them_audible` |
| Deployed configs | lists still naming them are inert | `AlertManager.__init__` strips them from fixed levels/categories | `test_a_deployed_config_still_listing_them_does_not_sound_them` |
| Runtime | severity `log`; no episode re-open | `services/runtime.py` | `test_runtime_tags_them_log_uploads_them_as_observations_and_sounds_nothing` |
| Uplink | top-level `experimental: true`, `severity: "log"`; absent otherwise | `core/uplink.py` | `test_the_uplink_payload_carries_experimental_true_and_severity_log` |
| Backend | stored `log`, column `experimental` 1/0 (NULL history), never notified | `cloud/app.py` | `test_backend_stores_them_as_log_with_experimental_true_and_never_notifies` |
| Presets/role | in no tone; role no longer requires `high_risk` loud; every validated floor kept | `presets.yaml` (v4), `core/roles.py` | `test_no_preset_pattern_or_variant_makes_them_audible`, `test_the_driver_role_...` |
| Module | text starts "EXPERIMENTAL observation"; no full-screen flash | `_risk_event`, `is_alert_active` | `test_experimental_events_are_labelled_observations`, `test_the_module_no_longer_flashes_...` |

**Safe migration.** `scripts/migrate_config.py` only adds keys. It now adds
`alerts.experimental_alerts_audible: false`. A unit configured before M8.4
keeps `high_risk` in `loud_types` and `elevated_risk` / `erratic_hand_movement`
in `soft_types`. Those entries are inert: the policy refuses the types, and the
runtime and uplink tag them `log`, unless the opt-in is the boolean `true`. So
deleting the key, a config reset, a reboot, a preset or variant switch, or an
owner setting all end up "not audible". The previous loud-type invariant now
compares the effective (gated) lists
(`tests/test_phase1_config_invariant.py`).

**App contract (IOS stream).** An event row with `experimental == true` and
`severity == "log"` is shown as "Experimental observation", never as an alert.
Older rows have no column value and come back as `experimental: false`.

## 3. Decision B: phone evidence states

`core/behavior/phone_evidence.py`. The levels are ordered:
`no_phone < phone_visible < phone_mounted < phone_near_driver < phone_held <
possible_interaction < active_distraction_evidence`. `insufficient_evidence`
means there is no answer.

| Level | Meaning | Risk term (was) |
|---|---|---|
| insufficient_evidence | detector failed/stale/not started; no driver region | 0 (0.3 / 0.5) |
| phone_visible | outside the driver's region | 0 (0) |
| phone_mounted | still, hands seen elsewhere, none on it | 0 (0) |
| phone_near_driver | driver's region, holding not established | **0** (0.3; 0.5 face lost; 1.0 while phone_use active) |
| phone_held | hand overlaps the phone | 0.6 with eyes off road, else **0** (0.6 always) |
| possible_interaction | held + gaze toward the phone | 0.6 |
| active_distraction_evidence | held + eyes off road >= `phone_use_seconds` | 1.0 (1.0 for phone_use active) |

Changed constants (risk fusion only, which is experimental): the phone risk
term values above. Weights, thresholds and EMA are unchanged.

## 4. Evidence representation and abstention

`core/behavior/signals.py`. `SignalEvidence` carries: `value` (None = missing,
with a required `missing_reason`), `observed_at`, `now`, `freshness_s`,
`source` (capability), `provenance` (stage), `confidence`, `uncertainty`
(measured noise), `continuity_s`, `calibrated`, `capability_valid`. Signals:
face, eye_state, gaze, phone, hand, person, frame (thumbnail quality:
ok/dark/blocked), driver. `Verdict` is SUPPORTED / NOT_SUPPORTED / ABSTAIN, and
an ABSTAIN must give its reason. The module exposes `behavior_abstentions()`,
`get_debug_state()["engine"]` and flat `engine_*` telemetry fields.

## 5. Temporal reasoning and shadow mode

`core/behavior/engine.py` judges drowsiness, distraction, eyes_down, phone_use
and attention_lost using the shipped config keys and bars. Its accumulator
follows the shipped `GracedConditionTracker`: time runs from the first true
frame, through false frames covered by the release grace. On top of that it
adds:

- no credit across a stall;
- a bounded 1 s coast on missing evidence, then the claim is dropped;
- the mirror-check patience;
- misalignment and empty-vehicle abstention.

Shadow-only constants (no production effect): `coast_s 1.0`,
`misalignment_s 30`, `misalignment_spread_deg 3`, `empty_after_s 30`,
`match_window_s 3`, frame blocked if thumbnail mean < 12 and std < 6, dark if
mean < 40.

Shadow vs production on the scripted set, all 24 scenarios
(`eval_after.json` `comparison`):

- On clean evidence, every onset matched with a 0.0 s difference
  (drowsiness, distraction, eyes_down, phone_use, attention_lost).
- Production-only `phone_use` in four cases:
  - `phone_mounted_look_before_latch`: mounted, hands visible, look starts
    before the set-down latch;
  - `phone_mounted_look_down`: mounted, hands unseen;
  - `passenger_leaning_with_phone`: the passenger's box is the largest;
  - `hands_out_of_view_phone`: a lap phone with no hand found. Truth is
    ambiguous; the shadow abstains.
- No shadow-only episodes.
- Promoting the shadow phone gate to production would change a validated
  alert. That needs an owner decision; it is not made here.
- **M8.5 (owner decision taken, section 10):** production no longer fires
  `phone_use` in the first three cases above; in `hands_out_of_view_phone` it
  keeps it on the lap-zone handling cue (production-only by design: the shadow
  still requires a hand). The shadow gate itself was not promoted.

## 6. Driver association

`core/behavior/association.py`: `configured_seat_region` (installer's
`behavior.driver_region`), `face_anchored_person`, `assumed_largest_person`,
`ambiguous`, `no_person_detected`. There is no first-boot seat calibration in
the codebase. The accuracy of every method is **unevaluated**.

## 7. Regression scenarios and evaluation (synthetic)

`sim/scenario.py` `CABIN_SCENARIOS` (24). `scripts/eval_behavior.py` writes the
JSON. `tests/test_m84_behavior_scenarios.py` pins every validated alert
(type and time) to its c9029fa value.

Evidence classes: synthetic 24 scenarios, human-labelled 0, model-generated 0.

| | production (before = after) | shadow |
|---|---|---|
| TP / FP / FN | 11 / 4 / 0 | 11 / 1 / 0 |
| precision (Wilson 95%) | 0.733 (0.48-0.89) | 0.917 (0.65-0.99) |
| recall (Wilson 95%) | 1.0 (0.74-1.0) | 1.0 (0.74-1.0) |
| false alerts / scripted hour | 27.96 | 6.99 |
| detection delay mean / max | 2.7 s / 3.2 s | 2.7 s / 3.2 s |

Abstention rate (frames): drowsiness 0.33, distraction 0.30, eyes_down 0.24,
attention_lost 0.12, phone_use 0.04. Missing-evidence rate: eye_state 0.33,
gaze 0.24, person/phone/driver 0.02. Experimental observations: 7 before,
4 after. The ones that disappeared were `elevated_risk` raised by an unheld,
mounted or passenger phone.

## 8. Conditional hand inference (B5)

Under `hands_policy: phone_visible` (the default), Hands runs on every detector
tick that has a phone box. Otherwise it runs only when `hands_max_interval_s`
(0.5 s) of evidence time has passed since its last run. That liveness sample
stays under the perception engine's 1.0 s freshness bar
(`core/perception.py` `DEFAULT_MAX_AGE_S`), so the hands capability never
reads as stale just because it was not scheduled.

- The phone_use set-down skip (`PhoneHoldEvidence`) only reads hands on ticks
  that have a phone, so phone_use decisions cannot change.
- On a skipped tick, hand evidence is stale (`hands_ran_last_tick: false`,
  engine `hand` signal), never "no hand".
- The experimental erratic-hand trajectory restarts after a skipped tick. In
  practice `erratic_hand_movement` is now judged only while a phone is in view.
- Hand speed now uses the frame's capture time, not the wall clock at
  processing time.
- `every_tick` restores the pre-M8.4 behaviour. The existing erratic detector
  tests pin it.

Host cost on this CPU, measured with real MediaPipe Hands on scripted frames
at the device's 30 fps and stride 3. The data is in `bench_hands_policy.json`.
These are host numbers, not Pi numbers:

| scenario | every_tick: module ms/frame (Hands calls / detector ticks) | phone_visible: ms/frame (calls) |
|---|---|---|
| sustained_look_away | 10.1 (220 / 220) | 2.6 (37) |
| mirror_checks | 10.9 (320 / 320) | 2.6 (54) |
| phone_mounted_unused | 10.8 (370 / 370) | 7.6 (270) |
| held_phone_eyes_off_road | 10.8 (220 / 220) | 6.3 (120) |

That works out to about 27 ms per Hands call on this host (M8.3 measured a
p50 of 32 ms). Validated events were identical under both policies in all 24
scenarios.

- At the 10 fps the scenario harness uses, Hands calls fell from 2548 to 1522
  across 2549 detector ticks. At 10 fps a detector tick only comes every
  0.3 s.
- A phone in a mount keeps Hands running on every tick.

## 9. Hardware-only (not claimed)

- Pi 5 cost of FaceMesh/Hands under each policy (`hwv_stage.py latency`).
- In-cabin NoIR behaviour of the occlusion heuristic: sunglasses vs a dark
  cabin vs IR.
- Real passenger/driver geometry for the face-anchored association.
- Real mount-phone stillness and hand recall: when does the not-held latch
  form?
- The buzzer: no physical sound is verified by any test here.

## 10. M8.5 phone handling vs off-road attention

Owner-requested fix. On 06dbca3 / 87d91b3 production fired the validated
`phone_use` on a long look at a phone in a dash/vent mount when (a) the look
began before `PhoneHoldEvidence` latched "not held" (about 2 s of stillness with
hands seen elsewhere), or (b) hand tracking saw no hands, so the hold stayed
"unknown". Decision B already gave that phone 0 risk, and the shadow engine
never called it phone use. M8.5 separates the two questions in production:

- **Is the driver's phone being handled?** That decides `phone_use`.
- **Are the eyes off the road?** That is judged by `eyes_down`, `distraction`
  and `eyes_up` with their existing rules, constants and timings. M8.5 does
  not touch them.

A mounted or unheld phone therefore never produces `phone_use`. A long look at
it still gets the off-road alert the look warrants. Nothing is silenced.

### 10.1 The rule

`phone_use` = `phone_use_seconds` (2.0 s, unchanged) of sustained eyes-off-road
(the same OR of yaw / pitch-down / pitch-up conditions, same tracker, same
release grace, same face-loss coast) **while** `supports_phone_use(level)` holds
for the driver's phone. The level comes from the shared M8.4 phone-evidence
classifier, `core/behavior/phone_evidence.classify_handling`, which the shadow
engine's `classify_phone` also calls. One source of truth:

| hold (`PhoneHoldEvidence`, unchanged) | handling cue | level | `supports_phone_use` |
|---|---|---|---|
| `held` (hand box overlaps the phone on one of the last 3 ticks, or on this tick) | not consulted | `phone_held` | **yes** (unchanged: same time, same detail) |
| `not_held` (still >= `phone_static_s`, hands seen elsewhere, none on it) | not consulted | `phone_mounted` | no (unchanged) |
| `unknown` (no hand found, hand model off or not run, or latch not formed yet) | `phone_moving` / `phone_at_face` / `phone_in_lap_zone` | `phone_near_driver` + cue | **yes**, detail says "possible handling: ... (unvalidated phone-handling evidence)" |
| `unknown` | none | `phone_near_driver` | **no** (was yes): the look is left to eyes_down / distraction |
| no driver phone: none in frame, the passenger's, an ambiguous driver, or detector failed | – | – | no |

The handling cues (`handling_cue`, first match wins) are positive evidence that
the phone may be in a hand. None of them is a hand.

| cue | evidence | constant (all reused, none new) |
|---|---|---|
| `phone_moving` | `PhonePlacement`: the phone box centre moved more than 10 % of its diagonal from where it was first seen or last came to rest, within the last 2 s of evidence time. The first sighting is not a move. Detector flicker is remembered for 2 s, so a phone reappearing in place is not a move and one reappearing elsewhere is | `phone_static_s` 2.0 s and 0.10 x diagonal: `PhoneHoldEvidence`'s own stillness bar, so "moving" is exactly "not still" in the latch's sense |
| `phone_at_face` | phone centre vertically within the measured face box and horizontally within the face box widened by half a face width per side (phone to the ear, or held up in front of the face) | 0.5 face widths = `PhoneConfig.face_margin_face_widths` |
| `phone_in_lap_zone` | phone centre below the chin AND inside the face's horizontal span, AND the eyes_down condition holds this frame (calibrated pitch < `pitch_threshold_deg`) | none: the face box and the existing pitch threshold |

Where the cue's face comes from: the current face box. During a face-loss
coast it is the last face box seen, and the coasted eyes_down condition stands
in for the gaze.

### 10.2 Code (this branch)

- `core/behavior/phone_evidence.py`: `CUE_*`, `STILL_S` / `STILL_TOLERANCE` /
  `AT_FACE_MARGIN_FW`, `PhonePlacement`, `handling_cue`, `classify_handling`,
  `supports_phone_use`; `PhoneEvidenceInput.handling_cue`;
  `PhoneEvidence.handling_cue`, emitted in `to_dict` only when set.
  `classify_phone` delegates its unheld branch to `classify_handling`. Without a
  cue, levels and reasons are byte-identical to M8.4. `phone_risk_for` is
  unchanged: a cue adds **no** risk (Decision B stands).
- `modules/driver_attention.py`:
  - `phones_for_associated_driver` (`:1076`);
  - on each detector tick, `associate_driver` relates phones to the driver
    (`:1870`) and `PhonePlacement` (built at `:1507`) is fed;
  - `_phone_handling_now` (`:1747`) is the per-frame gate, called at the
    phone_use decision (`:2694`) and in the face-loss coast (`:2319`);
  - `_phone_use_detail` states the cue;
  - `get_debug_state()["phone_handling"]` (`:2929`) gives level, cue,
    `supports_phone_use` and `evidence: "unvalidated"`;
  - the engine is handed the same hold and cue only when its driver box is the
    one production associated (`:2147`; was `agrees_with_largest`).
- `presets.yaml` `driver_safety` **version 4 -> 5**: phone_use
  `temporal_policies` text, a new "Unvalidated phone-handling evidence (M8.5)"
  limitation, the hand-tracking limitation no longer says "a phone with no hand
  found is still judged as in use", and the driver-association limitation
  covers phone use.

### 10.3 Changed rules (complete list)

1. A phone with hold `unknown` and no handling cue no longer arms phone_use.
   This was the old "UNKNOWN keeps the old behaviour" clause of
   `PhoneHoldEvidence`.
2. Production relates a phone to the driver by `associate_driver`, not by the
   largest person box. With no person box every phone counts, as before. With
   an AMBIGUOUS association none does.
3. A hand overlapping the phone on the current tick counts as held for the
   gate. It always did for the event detail; in real runs it implies `held`
   anyway.

No threshold, sustain time, release grace, coast, or validated non-phone rule
changed. No new tunable was added.

### 10.4 Scenarios, before (87d91b3) vs after

Scripted, deterministic, through the real module (`run_cabin`, 10 fps). All
validated alerts (`type@scenario-seconds`). Evidence:
`~/sensordyme-audit/2026-11-10/m85-evidence/phone/scenarios_{before_87d91b3,after}.json`.
The "before" run is 87d91b3's module and core with this branch's scenario files.

| scenario | before | after |
|---|---|---|
| phone_mounted_unused (hands visible, eyes on road, 1.2 s glance) | – | – |
| phone_mounted_hands_unseen (eyes on road) | – | – |
| m85_mount_unseen_short_glance (1.5 s look) | – | – |
| phone_mounted_look_before_latch | phone_use@2.2, eyes_down@2.7 | **eyes_down@2.7** |
| phone_mounted_look_down (hands unseen) | phone_use@7.2, eyes_down@7.7 | **eyes_down@7.7** |
| m85_mount_unseen_long_look_side (40 deg, 4 s) | phone_use@7.2, distraction@8.2 | **distraction@8.2** |
| phone_mounted_long_look_latched | eyes_down@12.7 | eyes_down@12.7 |
| m85_mount_flicker_long_look (detector recall 60 %) | eyes_down@7.7, phone_use@8.2 | **eyes_down@7.7** |
| m85_sunglasses_mount_long_look | phone_use@7.2, eyes_down@7.7 | **eyes_down@7.7** |
| m85_stall_mount_long_look (1 s stalls) | phone_use@12.9, eyes_down@14.4 | **eyes_down@14.4** |
| m85_mount_detector_fails_mid_look | eyes_down@7.7 | eyes_down@7.7 |
| detector_failure_look_down | eyes_down@4.7 | eyes_down@4.7 |
| passenger_holding_phone | – | – |
| passenger_leaning_with_phone (passenger box largest) | phone_use@7.2, distraction@8.2 | **distraction@8.2** |
| held_phone_eyes_off_road | phone_use@4.2, eyes_down@4.7 | phone_use@4.2, eyes_down@4.7 |
| m85_held_phone_eyes_on_road | – | – |
| m85_sunglasses_held_phone | phone_use@4.2, eyes_down@4.7 | phone_use@4.2, eyes_down@4.7 |
| m85_stall_held_phone | phone_use@9.0, eyes_down@11.5 | phone_use@9.0, eyes_down@11.5 |
| m85_mount_then_picked_up (hands visible) | phone_use@7.2, eyes_down@7.7 | phone_use@7.2, eyes_down@7.7 |
| m85_mount_then_picked_up_hands_unseen | phone_use@7.2, eyes_down@7.7 | phone_use@7.2 (moved, then lap zone), eyes_down@7.7 |
| m85_mount_look_then_pick_up (moved mount -> hand mid-look) | eyes_down@7.7, phone_use@8.5 | eyes_down@7.7, phone_use@8.5 |
| m85_held_then_mounted_mid_look (hand -> mount mid-look) | phone_use@4.2, eyes_down@4.7 | phone_use@4.2 (moving), eyes_down@4.7 |
| hands_out_of_view_phone (lap, hands unseen, looking down) | phone_use@4.2, eyes_down@4.7 | phone_use@4.2 (lap zone), eyes_down@4.7 |
| m85_moving_phone_hands_unseen (beside the lap, moving) | phone_use@5.2, eyes_down@5.7 | phone_use@5.2 (moving), eyes_down@5.7 |
| m85_phone_at_ear_look_away (call, 40 deg look) | phone_use@6.2, distraction@7.2 | phone_use@6.2 (at face), distraction@7.2 |

The other 19 scenarios (no phone) are identical. Across all 39, every
non-phone validated event is identical before and after: type, time and
detail. So are the experimental observations and the risk peaks.

**Lap phone, hands unseen, looking down: `phone_use` (decided).** The phone is
directly below the face, inside the face's horizontal span, and the driver is
looking down at it. That is where a hand-held phone is read, not where a dash
or vent mount sits relative to a driver-facing camera. The hand model's recall
is too low for "no hand found" to clear it: MediaPipe Hands found a hand in
8/152 web images (docs/research/M8_1_CV_FOLLOWUP.md). `eyes_down` fires as
well, 0.5 s later. The shadow still abstains ("holding unknown") because it
requires a hand. This is the one production-only phone_use left, and it is
deliberate.

### 10.5 Evaluation (synthetic, labelled by the scripts)

`scripts/eval_behavior.py` now runs both sets (`--set m84|m85|all`, default
all). Evidence: `eval_{before_87d91b3,after}_{m84,all}.json`. Evidence classes:
synthetic 39, human-labelled 0, model-generated 0. **Not real-world accuracy.**

| production | before (87d91b3) | after |
|---|---|---|
| M8.4 set (24): TP / FP / FN | 11 / 4 / 0 | 11 / 1 / 0 |
| M8.4 set: precision (Wilson 95 %) | 0.733 (0.48-0.89) | 0.917 (0.65-0.99) |
| M8.4 set: false alerts / scripted hour | 27.96 | 6.99 |
| all (39): TP / FP / FN | 31 / 8 / 0 | 31 / 1 / 0 |
| all: precision (Wilson 95 %) | 0.795 (0.64-0.89) | 0.969 (0.84-0.99) |
| all: recall (Wilson 95 %) | 1.0 (0.89-1.0) | 1.0 (0.89-1.0) |
| all: false alerts / scripted hour | 40.06 | 5.01 |
| all: detection delay mean / max | 3.2 s / 9.5 s | 3.2 s / 9.5 s |

- The remaining false positive is `camera_shifted`'s distraction, unrelated to
  phones.
- The shadow is unchanged on the 39 scenarios: 26 / 2 / 5. Its 5 misses are the
  stall and no-hand cases, where it abstains by design.
- Experimental observations: 8 before, 8 after.

### 10.6 Limits (stated, not fixed)

- **A phone held still away from below the face, with no hand seen, reads as
  mount-like.** The look then gets `eyes_down` at 2.5 s or `distraction` at
  3.0 s instead of `phone_use` at 2.0 s. Examples: beside the wheel, or on the
  thigh to one side. This is the recall price of "prefer the attention alert
  when the evidence is insufficient". It is never silence.
- A mount whose image position is directly below the face, or at face height
  within half a face width of it, reads as a lap / face phone. That gives
  `phone_use`, the pre-M8.5 behaviour.
- Mount vibration or detector box jitter larger than 10 % of the box diagonal
  reads as `phone_moving`, so a look can be `phone_use`, the pre-M8.5
  behaviour. The same jitter already stops the not-held latch from forming.
- All cues are 2-D image geometry from one camera. There is no depth and no
  cabin model, and they assume a driver-facing camera.
- Eyes-off-road that alternates between down and sideways without either
  condition sustaining used to be caught by phone_use's OR when a phone was
  visible. With an unhandled phone it is now caught only by the individual
  rules. This gap already exists with no phone.
- The face-anchored driver association is unevaluated, like every association
  method (section 6).

### 10.7 Hardware-only questions (Sunday)

- Mount stillness on the Pi + Camera Module 3: does a phone in a dash/vent
  mount stay within 10 % of its box diagonal, tick to tick, with the engine on?
  Does it pass 2 s still?
- Where do a real dash mount, a vent mount and a lap phone fall relative to
  the face box in the installed camera's image? Is the mount ever directly
  below the face?
- Hand recall on a gripping hand in the cabin (B4 with the hand hidden vs
  visible). Which detail string appears?
- B3b: the look at the mount right after placing it, and with hands out of
  view. Expected `eyes_down` / `distraction` only.
