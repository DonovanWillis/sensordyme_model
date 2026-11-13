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

Line numbers refer to this branch at the M8.4 ALPHA commits.

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
| The largest person box is the driver | `phones_for_driver`, eating gate, face-loss corroboration | Unchanged for production (validated phone_use). The engine uses the person box containing the **measured face** and reports when that disagrees with the largest box (`agrees_with_largest_person`). Unevaluated. |
| A phone box near the driver + eyes off road = phone use | `phone_use_condition` `:2599` | Production unchanged (validated). Engine: needs a phone HELD (hand overlap); unheld -> ABSTAIN, mounted -> NOT_SUPPORTED. Risk no longer counts a visible phone (decision B). |
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
