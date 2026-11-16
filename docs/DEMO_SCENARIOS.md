# M13 demonstration scenarios: the 10 flagship scenarios and the Jupiter demo

**Evidence status: SIMULATED only.** Every result below comes from the real
product code on constructed inputs:

- Scenarios 1–8 use the module harness. This is the real `DriverAttentionModule`
  at the shipped `config.yaml` thresholds, with scripted FaceMesh, object
  detector and hand-model outputs (`sim.perception.run_cabin`).
- Scenarios 9–10 and the Jupiter demo use the virtual device: the real runtime,
  uplink and backend on one simulated clock (`sim.harness.SimStack`).

None of this has been measured on the Pi, the Hailo, the Camera Module 3, a
real face or a real cabin. It shows what the shipped logic does when it is given
these inputs. It is not an accuracy figure.

All scenarios are **stationary**. Never rehearse them while driving.

| What | Where |
|---|---|
| The scenarios and the checks | `sim/flagship.py`, `sim/jupiter_demo.py` |
| Pinned (pytest) | `tests/test_m13_flagship_scenarios.py`, `tests/test_m13_jupiter_demo.py`, `tests/test_m13_overlay_story.py` |
| Rehearsal (table / steps) | `scripts/demo/flagship_rehearsal.py [--json] [--only 3,6] [--fps 15]`, `scripts/demo/jupiter_rehearsal.py [--json] [--prompt ...]` |

Run them from the repo root with the project venv. For example,
`.venv/bin/python scripts/demo/flagship_rehearsal.py`. Both scripts stamp every
line SIMULATED. `flagship_rehearsal.py` exits 0 when every FAIL is a recorded
known gap, and 1 on any unexpected FAIL. **Current result: `SIMULATED -- 10 PASS,
0 FAIL on known gaps only, 0 FAIL unexpected, of 10`** (no open gaps). `jupiter_rehearsal.py` exits 0 when
every step held.

## 1. What every scenario must show

For each scenario the system must:

1. Alert only when the configured policy is actually satisfied. The alert types
   are exactly the expected ones. The latency must fall within the shipped
   sustain time plus 4 frames.
2. Distinguish the behaviour. The overlay's behaviour phases
   (`overlay_extras()` → `core/overlay.py`) must tell its story, and must never
   show a behaviour that is not happening.
3. Never attribute a passenger's phone to the driver.
4. Report insufficient evidence where it has insufficient evidence.
5. Preserve the `driver_safety` role's mandatory safety policy. The shipped
   config passes the role floor, and every mandatory alert is loud.
6. Say why the event occurred, in the event's own `contributing_signals`.

A requirement the product does not meet is a **FAIL tagged with a known gap**
(section 4). The test pins it as a strict xfail. All three gaps found in M13
(G1, G2, J1) are now fixed, and their xfails are ordinary passing tests. Nothing was tuned to make a
row pass.

## 2. The 10 flagship scenarios

Scenarios 1–8 ran at 10 fps, the rate of the M8.4/M8.5 cabin suites. The test
also runs them at 15 and 30 fps with the same outcome, and 5 fps gave the same
result in a manual run. Times are seconds after the behaviour starts.
"Developing" means the condition is true but not yet sustained. "Confirmed"
means it is the module's alert condition.

| # | Scenario | Setup (scripted) | Expected | Measured (SIMULATED) | Verdict |
|---|---|---|---|---|---|
| 1 | Normal forward attention | 20 s attentive, hands on the wheel | no alert; face measured every frame; driver identified | no alert; no behaviour phase; face measured every frame; driver `identified` | PASS |
| 2 | Short ordinary glance | one 1.5 s glance, 40° sideways | distraction developing, never confirmed; no alert | `distraction:developing` only (never reached 3.0 s); no alert | PASS |
| 3 | Prolonged attention away | head turned 40° for 6 s | developing → confirmed; one loud distraction at ~3.0 s | `distraction` at +3.2 s; developing → confirmed; signals "turned 40 deg right of forward (limit 35 deg) + held 3.0 s" | PASS (G2 fixed) |
| 4 | Phone visible in a mount | dash-mounted phone, hands on the wheel, eyes on the road, one 1.2 s glance down | no alert; never phone_use developing | no alert; only `eyes_down:developing` for the glance; phone role `driver` (in the driver's space, not in use); `phone_detected` logged, never alerted | PASS |
| 5 | Phone held | phone in the lap with a hand on it, looked at for 6 s | phone_use at ~2.0 s, eyes_down at ~2.5 s, both loud | `phone_use` +2.2 s ("hand overlapping the phone (experimental hand tracking)"); `eyes_down` +2.7 s ("looking 30 deg below forward (limit 20 deg) + held 2.5 s") | PASS (G2 fixed) |
| 6 | Passenger using a phone | seated passenger holding a phone; driver watches the road, one 1 s glance across | no alert; the phone is never the driver's (role `other`) | no alert; never a driver phone box; never phone_use developing; overlay phone role `other` | PASS (G1 fixed) |
| 7 | Driver's eyes closed | eyes closed for 4 s | drowsiness at ~2.0 s, loud (critical for the role) | `drowsiness` +2.2 s ("eyes closed 2.0 s + eye openness 0.07 under 0.30"); `fatigue_warning` +2.8 s (soft, PERCLOS over a short session, allowed once) | PASS (G2 fixed) |
| 8 | Driver briefly occluded | face hidden for 3 s, body still detected | no alert; face reported not visible; face rules abstain | no alert (`attention_lost` waits 8 s with a body); overlay `face: {visible: false}` for 3.0 s; engine abstained on distraction, drowsiness and eyes_down; driver still `identified` | PASS |
| 9 | Camera unavailable | virtual device: camera frozen 20 s | FAILED / not protecting, naming the camera; no alert from stale frames; the driver is told | device FAILED +2.25 s, app `not_protecting` +2.25 s, reason "camera: no new frame…"; `monitoring_impaired` event and impaired tone; no driver alert | PASS |
| 10 | Monitoring restored | camera back; 12 s later eyes closed 4 s | READY / protecting without recalibration; the buzzer alert path resumes | device READY +0.25 s, app protecting +0.25 s; `drowsiness` loud and alerted, buzzer at +2.25 s; same event_key in the device's `/api/events`; signals "eyes closed 2.0 s + eye openness 0.07 under 0.30" | PASS (G2 fixed) |

`flagship_rehearsal.py` prints the same table with these columns: events,
phases, attribution, evidence, why, latency and verdict.

### Fixed in M13

These were overlay-story defects found by these scenarios. Each is fixed with
regression tests in `tests/test_m13_overlay_story.py`, and each of those tests
fails on the old code:

| Id | Defect (before) | Fix |
|---|---|---|
| F1 | `overlay_extras()` showed **phone_use "developing" for any visible phone**: a mounted one (scenario 4) or the passenger's (scenario 6). The app would have drawn "phone use developing" for a passenger's phone. | The phase now uses the condition the phone_use tracker is actually fed. |
| F2 | It showed **drowsiness "developing" during a downward look**, even though the module itself attributes the lowered lids to looking down (no closure is being timed). | The phase now uses the drowsiness condition itself. |
| F3 | It sent the **raw Euler head angles under `calibrated: true`**. An upright head read pitch ±180 and a 30° look down read +150. `docs/LIVE_OVERLAY.md` says calibrated angles are relative to the driver's forward pose. | The head pose now comes from this frame's perception signals: relative once calibrated, and raw with `calibrated: false` before that. |

The same conditions now also drive `get_debug_state()`'s phone_use and drowsy
phases. No event, threshold or alert decision changed. The M10, M10.1 and M8.5
suites pass unchanged.

## 3. The Jupiter "second wow" demo

**Instruction:** "Alert me when a person remains in this area for 10 seconds."
The deterministic parser (`jupiter/policy.py`) supports **the exact phrasing**,
with no Ollama and no language model. The run proves it by making any model
call fail.

| Step | The app's call | Measured (SIMULATED) |
|---|---|---|
| 0 | boot, claim | general-purpose device (`role: general`, `module: shared_perception`), ready |
| 1 | `POST /v1/devices/{id}/zones` | area `entrance` drawn (left half, floor level), rejected 0 |
| 2 | `POST /v1/ai/configure` | `source: policy_parser`; 1 rule: `person` / `lingering` / zone `entrance` / `dwell_s 10`; nothing unsupported. "This area" resolves to the one drawn area. With none drawn it is `marked area`, which waits for one and is never applied to the whole frame. |
| 3 | review (`policies`) | `lingering ['person']`, zone drawn, `temporal {dwell_s: 10}`, status ready, needs camera / inference / objects. The policy also says what it does **not** watch: no face recognition, no intent, and so on. The rule is data only. |
| 4 | `POST /v1/devices/{id}/preset {preset: "", rules, draft_version}` | HTTP 200, `rules_active 1` |
| 5 | (uplink) | mode `ai_custom`; the rule engine runs `('lingering', 'person', 'entrance', 10.0)` on the shared detector and tracker |
| 6 | scripted people | 9 s inside the area: nothing. 15 s outside it: nothing. 15 s inside: **one alert at +10.25 s** (device clock, 0.25 s sampling). |
| 7 | `GET /v1/devices/{id}/events`, device `GET /api/events` | backend: `person`, signals "A person remains in this area for 10 seconds", duration 10.1 s (the time in the area, J1 fixed), severity soft (warning), alert `alerted` / `simulated`. Device: the same `event_key`, sync `sent`, read through a sd-local-v1 session. |
| 8 | `POST /v1/ai/configure` "…a stranger remains…" | refused (`identity`); the running rule is kept |

**Other phrasings the parser reads to the same policy:**

- "alert me if someone stays in this area for 10 seconds"
- "Let me know when a person stays in the marked area for more than 10 seconds"
- "Tell me if anyone lingers in this area for 10s"
- "Alert me when someone is in this area for ten seconds"
- "… stays in this area longer than 10 seconds"
- "… for 10 seconds or more"

**Phrasings it does not claim:**

- "Notify me when a person **has been** in this area for 10 seconds"
- "… **is standing** in this area …"

These go to the language model. Without one, the backend answers honestly that
it cannot interpret them and changes nothing. For the demo, use the exact
phrasing above.

## 4. Known gaps (measured, not tuned away)

**All three are fixed** on `feat/m13-showstopper` (commit `61d0f59`), each with
the formerly strict-xfail test now passing. The table is kept as the record of
what the scenarios found. No threshold, timing or alert decision changed; the
fixes are labelling (G1), the free-text reason (G2) and a reported number (J1).

* **G1 fix:** a phone is `driver` when it matches one of the driver's phones,
  else `other` whenever the driver is known.
* **G2 fix:** the module writes one measured line per event, joined with " + ":
  for example "turned 40 deg right of forward (limit 35 deg) + held 3.0 s",
  "looking 30 deg below forward (limit 20 deg) + held 2.5 s", "eyes closed
  2.0 s + eye openness 0.07 under 0.30". The wording is the engineering default,
  and the owner may still revise it.
* **J1 fix:** a dwell alert's `duration_s` is the dwell plus the confirmation
  step: 10.1 s in the demo.

| Id | Gap (as found) | Measured | Classification | Smallest fix |
|---|---|---|---|---|
| G1 | A passenger's phone has no overlay role | `core/overlay.py` assigns a phone role only when the driver has a phone of their own. In scenario 6 the passenger's phone is sent with **no** `role`. It is not mis-attributed to the driver, but it is not marked `other` either, so the app cannot tell "not the driver's" from "undecided". | Defect (overlay labelling only; no alert or attribution decision is affected) | `core/overlay.py`: `elif d.label == "cell phone" and (driver_phones or driver_box is not None):`. Not made here, because `core/overlay.py` was outside M13's write scope. |
| G2 | `distraction`, `drowsiness` and `eyes_down` carry an empty `contributing_signals` | The ordinary events say why only through their type and duration (the sustained seconds). The overlay's "why" line (`last_event.signals`) and the history detail are blank. Only phone_use and the special paths (a wide-angle turn, a face-lost extrapolation, an uncalibrated closure) write a reason. | Limitation (the alert is correct and identified by type + duration; the free-text reason is missing) | The module writes one measured line, for example "head turned 40° from your forward pose for 3.0 s" or "eyes closed 2.0 s". This needs an owner decision on the wording, because `jupiter/explain.py` splits module signals into facts and the app shows them. |
| J1 | A dwell alert's `duration` is not the time in the area | The `person` event reads `duration 0.1` s: the rule engine's next-frame confirmation after the dwell watcher was satisfied. The person had been in the area 10 s. The rule's own sentence in `signals` still says why. | Defect (a misleading number in the history; timing and firing are correct) | `core/rules.py`, the dwell branch of `observe_frame`: report `rule.dwell_s + transition.duration_s` (the subject's continuous time in the area) as `Alert.duration_s`. |

**Expected limitations (not gaps):**

- **A camera fault is noticed after about 2 s.** For about 2.2 s after the
  camera stops, the app can still read ready/protecting. The device declares a
  frame overdue after 2 s, and the monitoring check and uplink each run every
  2 s. The measured value was 2.25 s to FAILED and to the app's
  "not protecting". The bound pinned in the test is 4.5 s on the device and
  6.5 s in the app.
- **`phone_detected` is log-only.** It is recorded on the rising edge of *any*
  phone in the frame, including the passenger's (scenario 6) and a mounted one
  (scenario 4). It is never alerted, never uplinked, and makes no claim about
  whose phone it is.
- **The module harness reads the driver's region from calibration.**
  Attribution in scenario 6 relies on the calibrated seat region, from M10
  driver association. A passenger box larger than the driver's, leaning in, is
  pinned separately (`passenger_leaning_with_phone` in
  `tests/test_m85_phone_scenarios.py`).

## 5. Rehearsing on the real Pi

**Safety (from `evaluation/schemas/incabin_scenarios_v1.yaml`):**

- The vehicle is parked, with the parking brake on.
- One **consenting adult** takes part. No children are in the vehicle.
- Drowsiness is **acted**.
- An operator outside the driver's seat runs the session.
- Anyone may stop at any time.
- Never while driving, and never on a public road.

The Pi run is diagnostics, not an accuracy claim. Accuracy needs the verified
labels and power check described in `scripts/hw_validation/cabin_session.py`.

Use the existing tools. Set `V=~/hwval/sensordyme_model/scripts/hw_validation`
on the Pi.

- **Cued session.** Print the operator sheet with
  `python3 $V/cabin_session.py plan --preset core --out ~/hwval/cabin/S01_demo`.
  Then run it with `run --plan …`, and score it with `score --session …`.
  Run it **on the Pi**, so the cue clock and the event log share one clock.
  Add `--fake` for a dry run of the script (SIMULATED).
- **What the app should show.** Watch it with
  `python3 $V/check_live_status.py --device http://localhost:8899 --backend <url> --samples N`.
  This reads `/api/health` and `GET /v1/devices` side by side.
- **The in-cabin output.** Check it with
  `python3 $V/buzzer_check.py run --out ~/hwval/buzzer_$(date +%s).json`.

The `cabin_session.py` block column names a block in
`evaluation/schemas/cued_session_v1.yaml`. Pass it with `plan --blocks …`, or
use `plan --preset core` / `--preset passenger`. The matrix column names a
cell in `evaluation/schemas/incabin_scenarios_v1.yaml`. Rows with no block are
run by hand with the operator's own cue, and are watched with
`check_live_status.py`.

| # | On the Pi (stationary) | `cabin_session.py` block | Matrix cell | Watch for |
|---|---|---|---|---|
| 1 | Sit normally and look at the road after calibration | `calibrate`, `baseline` | `N1_baseline_attentive` | no alert; app READY / protecting |
| 2 | Mirror glances under 2 s | `mirror_glances` | `G1_mirror_glances` | no alert; the overlay shows head-turn "developing", which clears |
| 3 | Look away and hold | `off_road_prolonged` | `H1_head_orientation` | distraction ~3 s, loud tone; the overlay goes developing → confirmed |
| 4 | Phone in a dash mount, hands on the wheel | `phone_mounted` | `P4_phone_mounted` | no phone_use; a long look down at it is eyes_down, never phone_use |
| 5 | Pick the phone up and look at it | `phone_handheld` | `P1_phone_in_hand`, `P2_phone_in_lap` | phone_use ~2 s, then eyes_down |
| 6 | The passenger uses a (prop) phone | `passenger_present` (`--preset passenger`; a consenting adult passenger with their own form) | `P5_passenger_phone` | no alert; the driver's phone highlight never lands on it (overlay role `other`) |
| 7 | **Acted** eye closure for 3–4 s | `closures`, `look_down_vs_closure` | `E3_sustained_closure` | drowsiness ~2 s, loud (critical) tone |
| 8 | Pass a hand in front of the face for 2–3 s | none (by hand) | `O1_occlusion_hand` | no alert; the overlay shows face not visible, then the face again |
| 9 | Cover the lens for 15 s | none (by hand) | `O3_lens_occlusion` | the app says monitoring cannot run, with no driver alert. **Do not unplug the camera ribbon while powered.** A true camera fault (no frames at all) is rehearsed on the virtual device only (scenario 9 here). |
| 10 | Uncover; after READY, an acted closure | `closures` after `O3` | `O3` recovery, `E3_sustained_closure` | READY / protecting again (no recalibration), then the loud drowsiness tone |

**Jupiter on hardware.** The Pi ships with the `driver_safety` role. On that
role, AI rules are **additive**: they run beside driver safety and never
replace it. The simulated run here used a general-purpose device config
(`role: general`, `module: shared_perception`). A room demo on the Pi needs that
config. A dwell rule beside driver safety on the Pi was **not** exercised here.
In both cases, draw the area in the app first, then type the exact instruction.
`scripts/demo/jupiter_rehearsal.py` shows what each step should return.
