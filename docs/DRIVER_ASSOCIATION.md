# Driver association and driver identity (M10)

Which face the eye / gaze / head-pose signals are read from, which person box
is "the driver" for phones and food, and how sure the device is about either.

**Evidence status: synthetic only.** Every behaviour below is pinned by
deterministic scenarios through the real `DriverAttentionModule`
(`tests/test_m10_driver_scenarios.py`, `tests/test_m10_driver_identity.py`).
None of it has been measured in a real cabin, on the Pi, or with the Camera
Module 3 Wide lens. The accuracy of every method is **unevaluated**.

Code: `core/behavior/driver_identity.py` (region calibration, face selection,
facade), `core/behavior/association.py` (person association),
`modules/driver_attention.py` (`_select_driver_face`, the `self._identity`
call sites).

## 1. The problem it closes

Before M10:

- FaceMesh ran with `max_num_faces=1`. With two people in view, the face whose
  eyes and gaze drove drowsiness / distraction / eyes_down could be the
  passenger's. The person-box association was a separate decision.
- With several people and no face to anchor to, the largest box was "the
  driver" (an assumption); food was always related to the largest box.
- There was no installation-time seat region; only an installer-configured
  `behavior.driver_region`.

## 2. Method

### 2.1 Driver region, by precedence

1. **Configured** `modules.driver_attention.behavior.driver_region`
   (`[x1, y1, x2, y2]`, 0..1): drawn by the installer. It should cover the
   driver's seat **head to lap** (both the driver's face centre and person-box
   centre must fall in it).
2. **Installation-calibrated** (`data/calibration.driver_region.json`), derived
   during the existing calibration window (section 4).
3. None: no region (behaviour as before M10 for face selection).

### 2.2 Face selection (whose eyes are measured)

Active only when a region is known **when the module is built** (FaceMesh's
face budget cannot change under a built graph) and
`behavior.driver_identity.face_selection` is `auto` (default). FaceMesh then
runs with `max_num_faces=2` and `DriverFaceSelector` picks, in order:

1. **Continuity** – the face within 1.5 face widths of the driver's last face,
   seen within 1.0 s. A passenger leaning into the region does not take over
   while the driver's face is tracked; a driver slumping out of the region
   stays the driver.
2. **In the region** – the face whose centre is in the driver region. One face:
   `identified`. Several: the one nearest the calibrated driver face (position
   and size), or without a calibrated anchor the largest; reported `uncertain`.
3. **All faces outside the region** –
   - a face on the person box centred in the driver region (a leaning or
     slumping driver): kept, `uncertain`;
   - else a face on **no** detected person (the person was missed, or the
     detector cannot currently say): kept, `uncertain` – there is no evidence
     it is not the driver's, and dropping it could only delay a driver alert;
   - else (every face is on a person box outside the region: a passenger's)
     **no driver face this frame**.

"On a person" means the face centre lies in the box's upper 60% (its head
part), so a rear passenger's box that reaches down behind the driver's head
does not claim the driver's face.

**M10.1: two position tracks (no face is recognised, nothing is enrolled).**

- **The driver's body.** This is the person box the selected driver face was
  on. It is followed box to box by IoU (≥ 0.3) while the face is not seen, for
  up to 2 s without a match.
  - While that body is still in the seat, a face that is on another person's
    box and not on the driver's body is that person's (a passenger leaning in).
  - Such a face is excluded from steps 2 and 3 wherever it is. If it was the
    only face in the region, the result is **no driver face this frame**.
- **The other occupants' faces seen in the last 1 s.** Each sighting claims
  only its nearest face: one person has one face.
  - Re-acquisition needs all of these at once:
    - the driver's face is lost, and every face is on a passenger's box;
    - one face is NEW (it continues no other occupant's track);
    - it sits on a box that also carries a tracked occupant's face;
    - it is within 3 face widths of where the driver's face was last seen, at
      most 5 s ago.
  - When they all hold, that face is the driver slumped toward the passenger,
    measured as `uncertain`. Anything less clear stays not measured, with the
    reason.

When no face is the driver's, the module takes the **face-lost path** exactly
as if FaceMesh had found no face: drowsiness resets (closed eyes cannot be
measured), distraction / eyes_down / eyes_up / phone_use coast on the driver's
last calibrated gaze (as for a fast head turn), and `attention_lost` runs
(`face_lost_seconds`, or `face_lost_with_body_seconds` with a person in view).
A passenger's eyes are never measured as the driver's.

Without a region, or with `face_selection: off`, FaceMesh keeps one face and
its face is used, exactly as before M10.

### 2.3 Person association (phones and food)

`associate_driver` (`core/behavior/association.py`), in order:

| method | when | driver box | identity |
|---|---|---|---|
| `configured_seat_region` / `calibrated_seat_region` | exactly one person centred in the region | that person | identified |
| same | several people centred in the region, the driver's (selected) face settles it | the one whose head is at the face | identified |
| `ambiguous` | several people centred in the region and the face does not settle it (or no trusted face) | **none** | uncertain |
| `face_anchored_person` | region known, nobody centred in it, the *selected* driver face is on one person (a driver leaning out) | that person | uncertain |
| `driver_not_in_region` | region known, nobody in it and no selected driver face on anyone | **none** | not_visible |
| `face_anchored_person` | no region: the box containing the measured face | that person | identified if one person, else uncertain |
| `ambiguous` | no region: the face lies in several boxes that disagree on the largest | **none** | uncertain |
| `assumed_largest_person` | no region, no usable face, exactly one person | that person | identified |
| `unresolved` | no region, no usable face, several people | **none** | uncertain |
| `no_person_detected` | no person box | none | unknown |

With a region known but face selection not active (a region derived in this
session, see 4.3), the measured face is **not trusted** to override the
region: it may be a passenger's.

`phones_for_associated_driver` relates a phone to the driver only through the
driver box; **no box, no phone** (as for `ambiguous` since M8.5). Food uses
the same box (`_eating_boxes_near_face` now receives only the associated
driver's box). With no person box at all every phone still counts (unchanged).

## 3. Uncertainty semantics

Telemetry (`get_telemetry_snapshot`, logged by `core/telemetry.py`):

| field | values |
|---|---|
| `driver_identity` | `identified` / `uncertain` / `not_visible` / `unknown` |
| `driver_identity_method` | the association method above (`unknown` when the detector is stale) |
| `driver_faces` | faces FaceMesh returned this frame |
| `driver_region_source` | `configured` / `calibrated` / `none` |

`get_debug_state()["driver_identity"]` adds the reason, the face choice, the
person association, the face-selection mode and budget, the region, and the
calibrated region's status (why it was or was not derived / loaded).

- `identified`: one candidate, or the region / continuity settles it.
- `uncertain`: several people / faces and nothing settles it, or a fallback
  (leaning driver, face on nobody, detector down). **Alerts still fire** from
  the selected face; phones / food are related only to a box that is not
  ambiguous or unresolved.
- `not_visible`: no face is the driver's (or the region is empty). The
  face-lost path runs for the driver; `attention_lost` fires if it persists.
- `unknown`: people cannot be counted (detector stale / failed / not started)
  and the face was not verified by selection.

Exactly what happens to each alert:

| situation | drowsiness / PERCLOS | distraction, eyes_down/up | phone_use | eating | attention_lost |
|---|---|---|---|---|---|
| identified | selected face | selected face | driver's box | driver's box | as before |
| uncertain, a face selected | selected face (not dropped) | selected face | driver's box if any; none if ambiguous/unresolved | same | as before |
| not_visible (only passengers' faces) | resets (cannot measure) | coast on the driver's last gaze | coasts on the driver's box's phone (none if the region is empty) | driver's box, if any | runs (8 s with a person in view, 2.5 s without) |

## 4. Installation calibration

### 4.1 What is recorded

During the existing `AttentionCalibrator` window (driver seated, looking at
the road; `calibration_duration_s`, trailing window), on each face frame: the
measured face box (from 9 landmarks spanning the face) and, on each detector
tick, the person boxes. When the attention calibration completes, the region
is derived from the trailing window:

- needs >= 5 face samples and >= 3 detector ticks;
- **exactly one person on >= 80% of detector ticks** (else: "calibrate with
  the driver alone in view" – a region learned with a passenger in view could
  be the passenger's seat);
- the median face centre must lie on the median person box;
- region = union(median face box, median person box) + half a face width
  (x) / height (y), clipped to the frame;
- rejected if wider than 70% of the frame (could not tell seats apart) or the
  face is narrower than 3% of the frame.

The face centre and width are stored as the **anchor** used to rank several
faces in the region.

### 4.2 Persistence

`<calibration_path stem>.driver_region.json` (default
`data/calibration.driver_region.json`), `version: 1`, written atomically,
always (with `region: null` and the reason when not derived). It is loaded
only when the attention calibration itself loaded (a stale / missing / forced
attention calibration means a fresh window, which rewrites both), and ignored
when unreadable, in another format, implausible (same checks as derivation,
anchor inside the region), or when the frame's aspect ratio differs from the
calibration's by more than 2%.

### 4.3 First boot and `--recalibrate`

`main.py --recalibrate` ignores both stored files and runs a fresh window. The
new region is used for person association **immediately**; face selection
(two FaceMesh faces) starts at the **next module build** (restart, preset
switch). Until then FaceMesh keeps one face and the identity reason says so.

### 4.4 Procedure (README Phase 4)

1. Mount and aim the camera; nothing else in the cabin should move it later
   (moving the camera invalidates the region – run step 2 again).
2. With **only the driver** in view (no passenger, nobody leaning in, rear
   seats empty or out of frame), the driver seated normally and looking at the
   road, start `main.py --recalibrate` and wait ~10 s (up to 30 s if the head
   is not still) until the log shows calibration complete.
3. Check `data/calibration.driver_region.json`: `region` is not null; if it is,
   `status` says why (usually another person in view). Repeat step 2.
4. Restart the service once so face selection uses two faces.
5. Optional: an installer can instead set `behavior.driver_region` (head to
   lap of the driver's seat); a configured region always wins.
6. Sanity check with a passenger seated: `get_debug_state()["driver_identity"]`
   (`--debug`) should read `identified` and `face.selected` should point at
   the driver.

## 5. Configuration

```yaml
modules:
  driver_attention:
    behavior:
      driver_region: null          # [x1, y1, x2, y2] 0..1; wins over the calibrated region
      driver_identity:
        face_selection: auto       # auto | off   (off = one FaceMesh face, as before M10)
        calibrated_region: true    # derive / persist / use the installation-calibrated region
```

Validated by `core/behavior/config.validate_behavior` (unknown keys, bad
values are config errors). Constants (margins, continuity, plausibility) are in
`core/behavior/driver_identity.py`; they are not tuned on cabin data.

## 6. Before / after (synthetic scenarios)

Same scenarios run against base `4dfd761` and this branch
(`tests/test_m10_driver_scenarios.py`; install session with the driver alone,
then a drive session). FaceMesh returns the **passenger's face first** unless
stated: what a one-face FaceMesh does when it locks onto the passenger. Times
are the first onset (s from the start of the drive session) of each driver
alert (drowsiness, distraction, eyes_down, eyes_up, phone_use, attention_lost).

| scenario | truth | before (4dfd761) | after (M10) | identity after |
|---|---|---|---|---|
| driver alone, attentive | nothing | nothing | nothing | identified |
| driver alone, eyes shut 3 s | drowsiness | drowsiness 3.15 | drowsiness 3.15 | identified |
| larger (leaning) passenger, driver attentive | nothing | nothing | nothing | identified |
| larger passenger looks down at phone in hand, driver attentive | nothing | **phone_use 2.05, eyes_down 2.55** | nothing | identified |
| same, configured `driver_region` | nothing | **eyes_down 2.55** | nothing | identified |
| driver looks down at phone in hand, passenger present | phone_use | **nothing (missed)** | phone_use 2.05, eyes_down 2.55 | identified |
| same, no region, driver's face first | phone_use | phone_use 2.05, eyes_down 2.55 | phone_use 2.10, eyes_down 2.55 (1) | uncertain |
| passenger leans into the driver region, turned, phone in hand | nothing | **phone_use 3.15, distraction 4.15** | nothing | identified |
| driver leaves, passenger stays with eyes shut 10 s | driver absent | **drowsiness 4.15** (passenger's eyes) | attention_lost 10.0 | not_visible |
| rear passenger asleep, phone in hand; driver attentive | nothing | **drowsiness 2.05** | nothing | identified |
| seat swap after calibration; new driver's eyes shut | drowsiness | **nothing (missed)** | drowsiness 5.65 | identified |
| passenger asleep, driver attentive | nothing | **drowsiness 2.05** | nothing | identified |
| driver's eyes shut, passenger attentive | drowsiness | **nothing (missed)** | drowsiness 3.15 | identified |
| driver turns 45 deg then fully away 4 s, passenger visible | distraction | **nothing (missed)** | distraction 4.15 | not_visible |
| driver alone slumps out of the region, eyes shut (continuous) | drowsiness | drowsiness 2.05 | drowsiness 2.05 | identified |
| same, face lost 1.5 s then re-found outside the region | drowsiness | drowsiness 4.05 | drowsiness 4.05 | uncertain |
| same, detector box misses the head (face on nobody) | drowsiness | drowsiness 4.05 | drowsiness 4.05 | uncertain |

**M10.1** (`tests/test_m101_driver_edges.py`; "before" measured on `629f906`):

| scenario | truth | before (629f906) | after (M10.1) | identity after |
|---|---|---|---|---|
| passenger leans into the region (own box outside it), looking ahead, while the driver is turned away | distraction | **nothing (masked)**, identified | distraction 4.15 | not_visible |
| same, the passenger's box centred in the region too | distraction | **nothing (masked)**, identified | distraction 4.15 | not_visible |
| driver, eyes shut, slumps onto the passenger's box (face lost 1.5 s); passenger's face visible | drowsiness | **nothing (missed)**, not_visible | drowsiness 5.05 | uncertain |
| same, the passenger's own face NOT visible | (ambiguous) | nothing | nothing: not measured, by design | not_visible |
| passenger beside, eyes shut, driver turned away | nothing | nothing | nothing | not_visible |
| driver's face occluded 0.8 s, passenger visible, then eyes shut 3 s | drowsiness | drowsiness | drowsiness | identified |
| driver's face lost 3 s (body in the seat), back, eyes shut | drowsiness | drowsiness | drowsiness | identified |
| seat change: driver leaves 5 s, another person sits, eyes shut | drowsiness | drowsiness | drowsiness | identified |

Every M10 row above is unchanged (`tests/test_m10_driver_scenarios.py`).

(1) With no face on the first frame of a session and two people, the phone is
no longer lent to the largest person (`unresolved`), so `phone_detected` is
logged on that frame and `phone_use` starts one frame (50 ms) later.

Food (module level, `tests/test_m10_driver_identity.py`, two people, no
region, the larger passenger's box first):

| scenario | before | after |
|---|---|---|
| bottle at the larger passenger's face, driver's face measured | **eating_drinking** | nothing |
| bottle at the driver's face | **nothing (missed)** | eating_drinking |

Unit-level change: `associate_driver([driver, passenger])` with no face and no
region was `assumed_largest_person` (the larger box); it is now `unresolved`
with no driver box (the M8.4 / M8.5 tests that pinned the old answer are
updated).

Driver-alert timing for the driver alone is unchanged; with a passenger
present the driver's drowsiness onset equals the driver-alone onset
(`test_driver_drowsiness_alert_timing_is_unchanged_by_a_passenger`). The M8.4
cabin regression pins (`tests/test_m84_behavior_scenarios.py`,
`tests/test_m85_phone_scenarios.py`) are unchanged except the engine's stated
association method, now `calibrated_seat_region` (the cabin lead-in is the
calibration window).

## 7. CPU cost

Mac micro-benchmark (Apple silicon, MediaPipe FaceMesh `refine_landmarks=True`,
video mode, 640x480, median of 400 calls; real faces from matplotlib's
`grace_hopper.jpg` pasted once or twice; **a Mac estimate, not a Pi
measurement**):

| frame | `max_num_faces=1` | `max_num_faces=2` | delta |
|---|---|---|---|
| no face | 1.97 ms | 1.97 ms | 0 |
| one face | 3.21 ms | 5.16 ms | +1.95 ms (+61%): the face detector runs every frame while fewer than 2 faces are tracked |
| two faces | 3.23 ms | 5.87 ms | +2.64 ms (+82%): a second landmark pass |

Selection itself: 9 landmarks per face for the box (reading all 478 protobuf
landmarks of two faces cost ~0.5 ms on the Mac) and ~4 µs of selector logic.
On the Pi 5 CPU the FaceMesh delta is expected to be several times larger
(MediaPipe face landmarks run on the CPU there, not the Hailo): budget roughly
+6-10 ms per frame when a region is known, to be measured on the device. With
no region the cost is unchanged (one face).

## 8. Limitations

- **Unvalidated.** Margins (half a face), continuity (1.5 face widths, 1 s),
  the 80% single-person rule, the 60% head part and the 70% width cap are
  reasoned, not fitted. Synthetic faces are rigid translations / scalings of
  one generator face.
- RESOLVED in M10.1 (body track): a passenger leaning into the driver region
  while the driver's face is not tracked was taken as the driver's face. It is
  still taken when the detector has no box for the driver's body (stale, or
  the driver's box lost for > 2 s), or when the face lies on both boxes.
- RESOLVED in M10.1 (re-acquisition), with one limit: a driver who slumps
  onto a passenger's box is measured (`uncertain`) only while the passenger's
  own face is visible beside it. Without it the face could be the passenger's,
  so it stays not measured and attention_lost covers it, later than drowsiness
  would.
- New in M10.1: if the detector's box for a returning driver shifts so that the
  driver's own face lands on a passenger's box only, that face is rejected until
  continuity or the region settles it. This is reasoned, not measured, and it
  needs the in-cabin data in section 9.
- First boot: face selection starts at the next module build (4.3).
- The calibrated region is per seat, not per person: after a seat swap the new
  driver is selected correctly, but the EAR / neutral-gaze baselines are still
  the calibrating driver's (as before M10).
- Face-lost corroboration (`face_lost_with_body_seconds`) still counts any
  person in view, including a passenger: with the driver gone and a passenger
  present, attention_lost takes 8 s, not 2.5 s.
- FaceMesh is shared (core/perception.py): with two faces, another consumer of
  the shared face result that takes face 0 may read the passenger's face.
- Moving the camera after calibration invalidates the region; nothing detects
  that yet beyond the existing misalignment heuristic.

## 9. What needs in-cabin data

- Driver / passenger / rear-passenger face and person-box positions on the
  Camera Module 3 Wide at the real mount, to set the margins and verify the
  70% cap and the head-part rule (left- and right-hand drive).
- How often FaceMesh (two faces) swaps, drops or merges faces with two
  occupants, and whether 1 s / 1.5 face widths of continuity holds under
  real motion, glasses, IR at night.
- Person-detector boxes for leaning / slumped occupants (does the driver's box
  follow the head?), and for rear passengers.
- Real CPU cost of two faces on the Pi 5 at the deployed resolution and frame
  rate.
- Labelled multi-occupant drives to measure identity accuracy and the alert
  changes above against ground truth.
