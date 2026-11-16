# Temporal reasoning (M10)

How the shipped driver monitor (`modules/driver_attention.py`) turns frames into
sustained conditions and events, what happens when frames or signals go
missing, and the deterministic regression suite that pins it.

Everything below is measured on **synthetic timelines** (constructed landmarks
at injected frame times, `tests/m10_temporal_harness.py`) with the shipped
`config.yaml` values. It says what the temporal logic does. It is not an
accuracy measurement on real drivers, faces, lenses or cabins.

## 1. Clocks

Every sustain timer measures "how long has this been true" as a difference of
timestamps. Which timestamps decides what counts as evidence.

| Clock | Advances on | Used by | Across a gap |
|---|---|---|---|
| frame time | every frame (capture time from the runtime, `set_frame_time`) | hand speed, behaviour v2, the shadow engine | raw |
| evidence clock (`_evidence_clock`) | every processed frame | drowsiness, attention_lost, erratic hands, risk fusion, detector freshness | a gap over `max_evidence_gap_s` (0.5 s) credits the median of the last 15 frame intervals (one frame); a stream slow throughout keeps 0.5 s per frame |
| gaze clock (`_gaze_clock`, M10) | frames that feed the gaze trackers: a usable calibrated head pose, a pre-calibration pose (fed False), or a face-lost frame (coasted or False) | distraction, eyes_down, eyes_up, glance counter, phone_use | as above, so a stretch with the face found but no usable head pose counts one frame interval |
| eye clock (`_eye_clock`, M10) | frames that sample PERCLOS (face found; with v2, eye state known) | PERCLOS | as above, so a face loss counts one frame interval, neither open nor closed |
| shadow (`FrameGapGuard`) | every frame | `core/behavior/` engine and v2 | credits nothing over 0.5 s |

All three production clocks are `StallAwareEvidenceClock`
(`modules/driver_attention.py`). The rule is the same for a camera stall and
for a signal that could not be read while frames kept coming: what was not
observed is not evidence, and a condition needs fresh frames to confirm.

## 2. Semantics per signal

"Accrues" is what counts toward the sustain time; "pauses" holds progress
without adding to it; "resets" drops it.

| Signal / event | Tracker, clock | Accrues | Pauses | Resets / ends | Face lost | Gaze unreadable (face found) | Camera stall |
|---|---|---|---|---|---|---|---|
| `drowsiness` (2.0 s) | `GracedConditionTracker(count_true_time_only, release_frames=2)`, evidence | closed time only: each closed frame owns the interval to the next frame | up to 2 consecutive open frames (glitch bridging) | the 3rd consecutive open frame | fed "not closed": reset after 3 frames (see open item O1) | n/a (EAR still read) | one frame interval |
| `fatigue_warning` (PERCLOS > 0.15) | `PerclosTracker`, eye clock | sample-and-hold closed fraction over 60 s of eye time; 0 until 10 s observed | face loss (one frame interval) | re-arms below 0.8 x threshold | paused (M10; was open time, or closed time between two closed samples) | n/a | one frame interval |
| `distraction` (3.0 s; 5.0 s if the turn reaches 55 deg) | `PeakCorroboratedGraceTracker`, gaze | elapsed since onset, bridging up to 0.4 s of on-road frames | gaze unreadable (M10) | on-road for 0.4 s | with a person detected: the last reading is coasted (credited; event confidence capped at 0.5 by `_extrapolated`); without one: fed False | paused (M10; was credited in full on the next readable frame) | one frame interval |
| `eyes_down` / `eyes_up` (2.5 s) | `GracedConditionTracker`, gaze | as distraction | gaze unreadable (M10) | 0.4 s back in range | as distraction | paused (M10) | one frame interval |
| `frequent_glancing` (5 in 45 s) | `GlanceCounter`, gaze | a glance = a rising edge of \|yaw\| > 35 | gaze unreadable | a glance ends after 0.4 s observed back on centre, first to last centred frame (M10; was last off-centre to next off-centre frame) | not fed | not fed | one frame interval |
| `phone_use` (2.0 s) | `GracedConditionTracker`, gaze | phone HANDLED and eyes off road | gaze unreadable (M10; was fed "on road") | 0.4 s not satisfied | coasted like the gaze it rests on | paused (M10) | one frame interval |
| `attention_lost` (2.5 s; 8.0 s while a person is detected the whole time) | `CorroboratedGraceTracker`, evidence | elapsed with no face | n/a | face back for 0.4 s; corroboration lost mid-episode falls back to 2.5 s from the episode start | this IS the face-lost signal | n/a | one frame interval: a stall cannot create face-lost time |
| `elevated_risk` / `high_risk` (experimental, never audible) | `RiskFusion`, evidence | time-based EMA, `SustainedConditionTracker` 1.0 s | - | - | face-lost weight | - | EMA step of one interval |

Before calibration the gaze trackers are fed False (nothing can be measured
against a neutral pose that does not exist yet). Drowsiness runs on the
fallback / session EAR threshold and, since M10, says so on the event.

### Eye closure vs looking down

A downward look lowers the upper lid by about the closure ratio (0.75 x open
EAR). Three protections keep it from reading as drowsiness, all exercised by the
suite:

- **pitch / iris gating** (`_closure_is_looking_down`, `core/behavior/eye_state.classify_eye_state`):
  an EAR drop with the calibrated head pitched 10 deg+ down or the iris 8 deg+
  down, while the iris is still visible (iris-openness >= 0.2), is
  `looking_down`, not closure. Scenario `look_down_lowered_lids`: eyes_down at
  +2.6 s, never drowsiness, at every rate.
- **a lid over the iris is closure whatever the head does**: `nod_off_head_down`
  (head 15 deg down, lids shut) fires drowsiness at +2.1 s.
- **PERCLOS counts only deep closure** (EAR < 0.4 x the open reference), so a
  lowered lid never enters PERCLOS. The iris veto suppresses an EAR "closure"
  the iris contradicts (glasses glare).

### Hysteresis and repetition

- Every tracker fires once per episode and re-arms only after its release.
  Noise shorter than the release grace (0.4 s, or 2 frames for eye closure)
  neither restarts the clock nor drops a live alert. `fatigue_warning` re-arms
  below 0.8 x its threshold.
- Alert episodes (`core/alerts.py`, unchanged): a tracked condition's episode
  stays open while the module reports it active, repeats every `repeat_s`
  (critical 4 s, warning 5 s), escalates after `escalate_after_s` (warning
  10 s), and ends when the condition clears. `tests/test_m10_temporal_alerts.py`
  pins: a 25 s distraction is one episode, `double, double, urgent, urgent,
  urgent`; a closure with landmark glitches is one critical episode; 0.2 s
  flicks back to the road do not split an alert; a 3 s camera stall inside a
  closure keeps one episode (no actuation during the stall, repeats resume
  after); a face loss is one attention_lost episode ending when the face
  returns.

## 3. Stall handling

A stall is a gap between frames longer than `max_evidence_gap_s` (0.5 s): a
frozen camera, a blocked inference loop, a dropped run of frames. The audit
finding "stall time counts as evidence" (F4) was fixed before M10 (M8,
`StallAwareEvidenceClock`). M10 re-verified it in the suite, for every timer:

- `stall_three_closed_frames`: closed frames 3 s apart never fire drowsiness.
- `stall_mid_closure_short`: 1.0 s closed, 3 s stall, 0.5 s closed: nothing
  (1.5 s observed). `stall_mid_closure_long`: 1.5 s + 3 s stall + more closure
  fires 0.6-0.8 s after the camera comes back, on fresh frames.
- `stall_mid_face_lost`: 1.5 s face lost, 5 s stall, 0.5 s face lost: no
  attention_lost. A stall cannot create face-lost time.
- `stall_mid_distraction`: 1.5 s turned, 5 s stall, 0.3 s turned: nothing.
- `frame_rate_drop_30_to_5` / `_30_to_2`: a closure that starts at 30 fps
  and continues at 5 fps or 2 fps fires at +2.2 s / +2.5 s. 0.5 s frames are
  still credited in full.

M10 found the same defect class one level down. A signal that could not be read
while frames kept arriving was still credited (T1, T2 below). Those signals now
have their own evidence clocks.

The runtime keeps an alert episode open through a stall (no frames means no
`observe()`, so no repeats). Monitoring health ends module episodes if the
stall turns into a FAILED state (`services/runtime.py`).

## 4. Uncertainty

- Graded events carry `confidence` = how far past the threshold the reading
  was, in units of that signal's own measured noise (`core/confidence.py`).
  Duration or count rules (attention_lost, frequent_glancing) carry `None`.
- Evidence the module did not see is lowered or marked, never reported as
  certain:
  - coasted (face-lost) distraction / eyes_down / eyes_up / phone_use:
    confidence capped at 0.5, "may or may not still hold" (`_extrapolated`),
    detail "extrapolated from last known gaze";
  - phone_use without a hand on the phone: detail says what holding evidence
    there was ("hand tracking unavailable, holding not checked", "no hand
    confirmed on it; possible handling: ... (unvalidated ...)");
  - attention_lost with a body still seen: detail "person still in frame";
  - **M10:** drowsiness before calibration: detail "driver not calibrated yet:
    closure judged against an uncalibrated EAR threshold". The rule still
    fires. Only the claim is qualified, to match the frame's
    `eye_state.closed = None`;
  - **M10:** gaze events no longer report full confidence for time nobody
    measured (T1).
- A stale or failed detector clears its boxes (phone, person, hands), so phone
  evidence cannot outlive the detector. The shadow engine reports it as
  `INSUFFICIENT` / ABSTAIN.
- The shadow engine (`core/behavior/engine.py`, never audible) ABSTAINs on
  missing evidence. **M10:** it also ABSTAINs on the frame after a gap when the
  claim's condition holds, instead of saying NOT_SUPPORTED.

## 5. Defects found in M10 (each failing test first)

`tests/test_m10_temporal_defects.py`: **19 of 24 fail on base 4dfd761, 24/24
pass after.**

| | Defect | Evidence on base | Fix |
|---|---|---|---|
| T1 | Face found, head pose unreadable: the gaze trackers were not updated, so the next readable frame was credited the whole stretch. The same frames fed phone_use "eyes on the road". | 1.5 s turned + 5 s unreadable + turned: distraction on the first frame back (6.4 s, confidence 1.0). It also fired when the driver had looked back at the road meanwhile. eyes_down likewise. A 1 s pose failure dropped a live phone_use alert, which re-fired 2 s later. | gaze evidence clock. The trackers pause, and phone_use pauses too. |
| T2 | PERCLOS charged unobserved time: a face loss between open samples counted as open time, and between closed samples as closure. | The same 20 % blinking read PERCLOS 0.061 with a 30 s look away vs 0.147 without (30 fps). Two blinks around a 20 s look away fired fatigue_warning. | eye evidence clock |
| T3 | Drowsiness before calibration reported its margin confidence unmarked, while `eye_state.closed` was None. | event detail empty | detail names the uncalibrated threshold |
| T4 | Shadow engine: NOT_SUPPORTED ("closed 0.0s") for eyes shut on every frame of a 1 fps stream | verdict `not_supported` | ABSTAIN "frame gap" when the condition holds after a gap |
| T5 | `GlanceCounter` measured a glance's end from the last off-centre frame to the next one, not over observed centred frames. At 5 fps one centred frame (0.2 s) "lasted" 0.4 s. | yaw chatter at the threshold, 5 fps: frequent_glancing (5 glances in 3.4 s) AND distraction for one look | first-to-last centred frame, as `GracedConditionTracker` measures a release |

No threshold, sustain time, role floor, alert type, category or episode rule
was changed. T1 and T2 only stop crediting time nobody observed, the principle
of audit F4. T5 changes only the soft `frequent_glancing` notice. T4 changes
only the non-audible shadow.

## 6. Scenario table, before / after

`tests/test_m10_temporal_scenarios.py`: 33 named scenarios x rates = 163 cases.
The suite runs at 30, 10 and 5 fps, plus 30 and 10 fps with +/-30 % frame
jitter (`j`). Latencies are measured from the condition's onset segment, and
each case allows the sustain time plus 4 frames of filter and sampling lag. Only
alertable, non-experimental events are listed. **Base 4dfd761: 147/163.
m10/temporal: 163/163.**

| Scenario | Rates | Before (4dfd761) | After (m10/temporal) |
|---|---|---|---|
| `blinks_normal` | 30/10/5/30j/10j | 5/5 pass. none | 5/5 pass. same |
| `microsleep_1_2s` | 30/10/5/30j/10j | 5/5 pass. none (drowsy_seconds is 2.0) | 5/5 pass. same |
| `sustained_closure` | 30/10/5/30j/10j | 5/5 pass. drowsiness +2.07 / +2.20 / +2.40 / +2.11 / +2.42 s; fatigue_warning +2.2-2.8 s | 5/5 pass. same |
| `heavy_long_blinks` | 30/10/5/30j/10j | 5/5 pass. fatigue_warning +8.7 / +16.6 / none / +8.6 / +18.6 s | 5/5 pass. same (5 fps: open item O2) |
| `look_down_lowered_lids` | 30/10/5/30j/10j | 5/5 pass. eyes_down +2.57 / +2.70 / +3.00 / +2.63 / +2.89 s, never drowsiness | 5/5 pass. same |
| `nod_off_head_down` | 30/10/5/30j/10j | 5/5 pass. drowsiness +2.07 / +2.20 / +2.40 / +2.11 / +2.34 s | 5/5 pass. same |
| `closure_with_one_frame_glitches` | 30/10/5/30j/10j | 5/5 pass. one drowsiness, +2.57 / +2.70 / +2.60 / +2.69 / +2.89 s (each glitch frame pauses the clock) | 5/5 pass. same |
| `ear_chatter_at_threshold` | 30/10/5/30j/10j | 5/5 pass. at most one drowsiness (10, 5 fps) | 5/5 pass. same |
| `mirror_checks` | 30/10/5/30j/10j | 5/5 pass. none | 5/5 pass. same |
| `blind_spot_check_4s` | 30/10/5/30j/10j | 5/5 pass. none (rearview tolerance 5 s) | 5/5 pass. same |
| `blind_spot_check_7s` | 30/10/5/30j/10j | 5/5 pass. distraction +5.10 / +5.30 / +5.40 / +5.14 / +5.37 s | 5/5 pass. same |
| `distraction_sustained` | 30/10/5/30j/10j | 5/5 pass. distraction +3.07 / +3.20 / +3.40 / +3.12 / +3.31 s | 5/5 pass. same |
| `distraction_recovery` | 30/10/5/30j/10j | 5/5 pass. none (a 1 s return resets: no partial credit) | 5/5 pass. same |
| `distraction_brief_return` | 30/10/5/30j/10j | 5/5 pass. distraction +3.07 / +3.20 / +3.40 / +3.12 / +3.31 s (0.2 s flick bridged) | 5/5 pass. same |
| `frequent_glancing` | 30/10/5/30j/10j | 5/5 pass. at the 5th glance, +0.07 / +0.20 / +0.40 / +0.09 / +0.23 s | 5/5 pass. same |
| `look_up` | 30/10/5/30j/10j | 5/5 pass. eyes_up +2.57 / +2.70 / +3.00 / +2.63 / +2.89 s | 5/5 pass. same |
| `yaw_chatter_at_threshold` | 30/10/5/30j/10j | **4/5.** 5 fps: frequent_glancing +3.40 s AND distraction +4.20 s for one look | 5/5 pass. 5 fps: distraction +4.20 s only (T5) |
| `yaw_hover_at_threshold_1hz` | 30/10/5/30j/10j | 5/5 pass. 30: frequent_glancing +4.07; 10: distraction +3.20, frequent_glancing +4.20; 5: distraction +3.40; 10j: frequent_glancing +4.31 s | 5/5 pass. 10j: frequent_glancing +7.25 s, rest same (open item O3) |
| `phone_handled` | 30/10/5/30j/10j | 5/5 pass. phone_use +2.07 / +2.20 / +2.40 / +2.11 / +2.34 s, eyes_down +0.5 s later | 5/5 pass. same |
| `phone_mounted` | 30/10/5/30j/10j | 5/5 pass. eyes_down only | 5/5 pass. same |
| `phone_use_through_pose_failure` | 30/10/5/30j/10j | **0/5.** phone_use fired twice (+2.1 s and again ~+6.5 s): the live alert dropped during a 1 s pose failure | 5/5 pass. one phone_use (T1) |
| `face_lost_brief` | 30/10/5/30j/10j | 5/5 pass. none | 5/5 pass. same |
| `face_lost_long_no_body` | 30/10/5/30j/10j | 5/5 pass. attention_lost +2.50 / +2.50 / +2.60 / +2.56 / +2.65 s | 5/5 pass. same |
| `face_lost_long_with_body` | 30/10/5/30j/10j | 5/5 pass. attention_lost +8.03 / +8.10 / +8.00 / +8.05 / +8.10 s | 5/5 pass. same |
| `gaze_unknown_mid_turn` | 30/10/5/30j/10j | **0/5.** distraction +0.00-0.10 s after the pose returned (5 s unobserved credited) | 5/5 pass. +1.57 / +1.70 / +2.10 / +1.58 / +1.75 s of fresh frames (T1) |
| `gaze_unknown_look_back` | 30/10/5/30j/10j | **0/5.** distraction at +6.5 s for a 1.5 s + 0.4 s turn | 5/5 pass. none (T1) |
| `face_lost_brief_mid_closure` | 30/10/5/30j/10j | 5/5 pass. none, except 5 fps: drowsiness +2.60 s | 5/5 pass. same (open item O1) |
| `stall_three_closed_frames` | 30/10/5/30j/10j | 5/5 pass. none | 5/5 pass. same |
| `stall_mid_closure_short` | 30/10/5/30j/10j | 5/5 pass. none | 5/5 pass. same |
| `stall_mid_closure_long` | 30/10/5/30j/10j | 5/5 pass. drowsiness +0.57 / +0.70 / +0.80 / +0.61 / +0.79 s after the stall | 5/5 pass. same |
| `stall_mid_face_lost` | 30/10/5/30j/10j | 5/5 pass. none | 5/5 pass. same |
| `stall_mid_distraction` | 30/10/5/30j/10j | 5/5 pass. none | 5/5 pass. same |
| `frame_rate_drop_30_to_5` | 30/30j | 2/2 pass. drowsiness +2.20 / +2.11 s | 2/2 pass. same |
| `frame_rate_drop_30_to_2` | 30 | 1/1 pass. drowsiness +2.50 s | 1/1 pass. same |

The two `T2` scenarios (PERCLOS around a face loss) are in the defect file, not
in this table. The alert-episode file passes 6/6 on both.

Reproduce: `pytest tests/test_m10_temporal_scenarios.py tests/test_m10_temporal_defects.py tests/test_m10_temporal_alerts.py`.
For the "before" column, run the same three files and `tests/m10_temporal_harness.py`
on a checkout of 4dfd761.

## 7. Open items (measured, not changed)

- **O1 (RESOLVED in M10.1, section 8) Drowsiness resets on a face loss, and how fast depends on the frame rate.**
  The closure tracker is fed "not closed" while the face is lost and resets
  after `eye_closure_release_frames` (2) frames. That is 0.07 s at 30 fps and
  0.4 s at 5 fps, so `face_lost_brief_mid_closure` (1.5 s shut, 0.3 s lost, 1 s
  shut) fires only at 5 fps. A nodding driver whose mesh drops briefly restarts
  at 30 fps. The shadow engine already coasts closure through a loss of up to
  1 s (`DrowsinessEvidence`). Promoting that is an owner decision, because it
  changes a mandatory rule.
- **O2 (RESOLVED in M10.1, section 8: the cause was the iris veto, not the median) The median filters are sized in frames, not seconds.** `signal_median_window`
  is 5 frames, which is 0.17 s at 30 fps and 1 s at 5 fps. The same 20 %
  long-blink pattern reads PERCLOS 0.147 / 0.101 / 0.057 at 30 / 10 / 5 fps, so
  `heavy_long_blinks` raises no fatigue_warning at 5 fps. Onset latency grows
  from +0.07 s to +0.4 s. A time-based window is a tuning change for
  validation, not for this mission.
- **O3 No hysteresis band on the gaze thresholds.** A head hovering at
  33 / 37 deg against the 35 deg threshold, 0.5 s on each side, has no
  frame-rate-independent answer: 30 fps counts five glances
  (frequent_glancing), and 5-10 fps bridges them into one distraction. Both
  are within the rules as written, because 0.5 s below the threshold is just
  past the 0.4 s release. An enter/exit band would be a new policy and needs
  real-cabin data.
- **O4 (REPRODUCED in M10.1; policy proposed, OFF pending owner approval: docs/ESCALATION_POLICY.md) Escalation does not carry across a real recovery.** Two distractions 2 s
  apart are two episodes, and the second one starts again at the onset pattern
  (`test_a_real_recovery_ends_the_episode_and_the_next_one_starts_fresh`). A
  driver who keeps relapsing never hears `urgent`. This is `core/alerts.py`
  policy.
- **O5 A live gaze alert holds while the gaze is unreadable.** With T1 a
  distraction, eyes_down or phone_use alert that is already active stays
  active through a head-pose failure, because the condition is not confirmed
  over. The hold has no bound while the face stays found. A face loss hands
  over to attention_lost.
- **O6 A slumped driver.** Closure followed by a face loss with the body still
  seen waits for the 8 s corroborated attention_lost bar. The closure just
  before the loss is not used as evidence against the "turned to a mirror"
  reading. That would be a new policy, so test it in shadow first.
- **O7 (RESOLVED in M10.1, section 8) The per-frame iris veto works on raw frames, the EAR median on
  smoothed ones.** One open glitch frame in five pauses closure accrual (2 s
  of closure takes about 2.5 s) without resetting it. One episode, but later.
- **O8 A stream slower than 2 fps.** After a drop to 1 fps the stall-aware
  clock credits one old-rate interval per frame until 8 of its last 15
  intervals are slow, then 0.5 s per frame. Alerts still fire, seconds later
  in wall time; the shadow abstains on every frame where a condition holds (T4).
- All of the above is synthetic. Thresholds and timings need in-cabin
  validation (docs/BEHAVIOUR_REASONING.md).

## 8. M10.1: frame-rate-independent eye evidence

**Root cause of O2 (measured, not assumed).** With the iris veto off, or the
median at 1 frame, the same 0.7 s long blinks measure the full 10.5 s of
closure at every rate and fatigue fires at 5 fps. The median itself does not
shrink a closure; it delays it by 2 frames. The iris veto compared the RAW
frame's iris openness with that delayed EAR, so on every reopening the smoothed
EAR still read "closed" for 2 frames while the raw iris already read "open", and
the veto cancelled those frames: 0.07 s per closure at 30 fps, 0.4 s at 5 fps.

**Fixes** (no threshold changed: drowsy_seconds 2.0, perclos_threshold 0.15,
perclos_closure_ratio 0.4, signal_median_window 5, eye_closure_release_frames 2):

1. The iris openness gets the same 5-frame median over the same frames as the
   EAR it vetoes (`core/eye_signal.AlignedOptionalSmoother`); the looking-down
   attribution uses it too. Gaze gating and the baselines keep the raw value.
2. A frame with no face is not an "open" frame. The closure episode is HELD,
   crediting nothing, for `eye_closure_face_loss_hold_s` (0.4 s, the same grace
   `signal_release_s` gives every other condition), then reset -- the same time
   at every rate (`GracedConditionTracker.hold`).
3. New derived capability `eye_timing` (`eye_timing_min_fps` 4.5): below it,
   drowsiness and fatigue_warning are reported unavailable and monitoring is
   DEGRADED with the measured rate, instead of being implied. 5 fps is the
   lowest rate the suite verifies; 4.5 allows timing jitter. It is an analysis
   capability on the device, the backend and the app (withdrawn with the camera,
   paused while stopped).

**Before (629f906) / after, same harness and scenarios** (first onset, s from the
scenario's segment; PERCLOS at the end of the run):

| scenario | 30 fps | 15 fps | 10 fps | 5 fps |
|---|---|---|---|---|
| heavy_long_blinks, fatigue_warning | +8.7 / +8.3 | +10.6 / +6.8 | +16.6 / +8.6 | **none** / +7.0 |
| heavy_long_blinks, PERCLOS (truth 0.263) | 0.226 / 0.250 | 0.214 / 0.261 | 0.178 / 0.249 | 0.142 / 0.283 |
| face_lost_brief_mid_closure, drowsiness | **none** / +0.57 | **none** / +0.67 | **none** / +0.70 | +0.80 / +0.80 |
| closure_with_one_frame_glitches, drowsiness | +2.57 / +2.07 | +2.53 / +2.13 | +2.70 / +2.20 | +2.60 / +2.40 |
| sustained 3 s closure, drowsiness | +2.07 / +2.07 | +2.13 / +2.13 | +2.20 / +2.20 | +2.40 / +2.40 |

What changed at 30 fps, stated plainly: PERCLOS reads closer to the truth (the
2-frame cut is gone, +0.024 on this pattern), a closure with one landmark glitch
per 5 frames fires at the sustain time instead of ~0.5 s later, and a face loss
of up to 0.4 s inside a closure pauses it instead of restarting it. Nothing
fires on less observed closure than before: normal blinking (0.25 s every 3 s)
and sub-threshold long blinks (0.4 s every 4 s, 10 %) raise nothing at any rate,
jittered or not (`tests/test_m101_frame_rate.py`). The 5 fps PERCLOS (0.283)
sits above the truth because a 0.7 s closure is sampled as 3-4 frames of 0.2 s.

Rates now in the suite: 30, 15, 10, 5 fps, plus 30/10/5 fps with +/-30 %
jitter (227 cases, `tests/test_m10_temporal_scenarios.py`), and 56 M10.1 cases
(`tests/test_m101_frame_rate.py`: capability at 1-4 fps, face loss held vs
reset, partial observation with the head pose unreadable, hold semantics).

**Escalation (O4).** Reproduced as a strict-xfail regression
(`tests/test_m101_escalation.py`): six 6 s look-aways 2 s apart are six
episodes of ~3.4 s, ~20 s alerted in 48 s, and only `double` is heard. The
bounded relapse policy is implemented and OFF; docs/ESCALATION_POLICY.md has
the decision.
