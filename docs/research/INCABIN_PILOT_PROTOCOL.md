# In-cabin pilot recording protocol (DRAFT, not approved)

**Status:** draft for review. No recording may start until:
- this protocol, the consent form and the data path are approved in writing;
- counsel has reviewed biometric/privacy law for where participants live (BIPA, CUBI, CCPA/CPRA, GDPR if any EU/UK participant).

**Purpose:**
- Produce the first small in-domain clips from SensorDyme's own optics, so the harness (`evaluation/`, `docs/EVALUATION.md`) can measure what no public dataset can:
  - Camera Module 3 Wide NoIR geometry;
  - day and 850 nm infrared appearance;
  - legitimate glances vs distraction;
  - phone handling.
- A 3–5 person pilot is for **finding failure modes, debugging labels and timing the procedure**. It is NOT evidence of population-level accuracy. Its numbers are reported per subject, with that caveat, and are never quoted as product accuracy.

## 1. Scope and safety

- **Stationary vehicle only.** Parked, engine off (or accessory power), parking brake on, on private property or a quiet car park. No driving of any kind in the pilot.
- No participant is ever asked to drive while distracted, drowsy or impaired. Drowsiness is only *acted* (long eye closures, nods) while parked.
- Participants: 3–5 consenting adults (18+), not in a supervisory relationship with the operator that could make consent less than free.
- Sessions last about 60 minutes including breaks. Anyone may stop at any time without giving a reason, and keeps any compensation.

## 2. Equipment

| Item | Notes |
|---|---|
| Raspberry Pi 5 + Hailo-8L + Camera Module 3 **Wide NoIR** | The production stack. Recording uses `rpicam-vid` directly; SensorDyme software is not needed and not deployed for recording |
| 850 nm IR illuminator | As in `hardware/enclosure`. Log its power setting. Eye-safety: use the illuminator's rated working distance, never closer than its spec |
| Mounts | (A) dash centre / steering column, primary; (B) rear-view-mirror base, secondary. Both record **at the same time**, so one labelling pass covers two geometries |
| Second camera, optional | Any phone camera on a tripod outside the driver's face line, recording an overview of hands for annotation only (never a training input) |
| Calibration targets | A 9×6 checkerboard (25 mm squares) for lens intrinsics per mount. Printed gaze targets: windscreen centre, left/right mirror, cluster, infotainment, lap, passenger |
| Cue logger | A laptop or tablet the operator taps at the start and end of every scripted action. It writes a CSV of UTC times |
| Time sync | Pi clock synced (chrony/NTP) before the session. A clapperboard frame (operator shows a phone stopwatch to each camera) at the start and end |
| Props | A phone (screen on/off), a phone mount, a water bottle, a snack bar, sunglasses (IR-pass and IR-block if available), a cap |
| Power | 12 V → 5 V 5 A supply or USB-C PD bank, so the car battery is not drained |
| Storage | An encrypted USB SSD (LUKS) for raw video. Nothing on SD cards beyond the OS |

**Recording settings:**
- 1536×864 or 1280×720 at 30 fps, H.264 at high bitrate, plus a 640×480 stream matching the device.
- Auto-exposure on (the device's real behaviour).
- For each clip, log `rpicam-vid` arguments, tuning file, sensor mode, illuminator state and power, and mount ID.

## 3. Conditions

| Condition | How |
|---|---|
| Daylight | Daytime, sun through side window (log direction) |
| Low light | Dusk, or a covered windscreen/garage |
| Night IR | Dark, 850 nm illuminator on, cabin lights off |
| Night IR + interference | Torch sweeps imitating headlights, the phone screen lit toward the face |

Each participant does the full script in **daylight** and **night IR** at minimum. Low light is added if time allows.

## 4. Session script (per condition, about 20 min)

| Block | Action | Duration | Labels produced |
|---|---|---|---|
| Setup / calibration | Checkerboard in 15 poses per mount. Participant looks at the windscreen centre for 10 s, then normal posture for 20 s (the device's calibration window) | 3 min | `ignore` span (calibration) |
| Baseline | Attentive posture, natural blinking, hands on wheel | 2 min | negatives (false-alarm time) |
| Legitimate glances | Cued: left mirror, right mirror, rear-view mirror, cluster, infotainment. Each 0.5–1.5 s, ×5 | 3 min | `glance_<target>` events. These must NOT alert |
| Long off-road glances | Cued: look at infotainment, lap or passenger for 2, 3 and 4 s, ×3 each | 3 min | `eyes_off_road` events with duration |
| Phone visible, not used | Phone in mount, then on passenger seat, then in cupholder, screen on | 2 min | `phone_visible_only` (negatives for phone_use) |
| Phone handling | Pick up and put down; hold at ear left/right (call); hold low and look at it (text) 3–5 s; hold in lap and look; look at mounted phone 2 s | 4 min | states `phone_in_hand`, `phone_near_face`; events `phone_use` (in hand AND eyes on phone) |
| Other hand activity | Drink, eat, adjust hair or glasses, scratch face, wave | 2 min | negatives / `eating_drinking` |
| Acted eye closure | Cued closures of 0.5, 1, 2 and 3 s ×3; slow blinks; yawns with and without a hand; head nods | 2 min | states `eyes_closed`; events `long_closure` (≥1.5 s), `yawn`, `nod` |
| Occlusion | Cap brim down, sunglasses (each type), hand over mouth, visor down | 2 min | occlusion flags |

The operator taps the cue logger at the start and end of each action. Cue times are a **guide** for annotators, not ground truth.

## 5. Consent and withdrawal

The consent form, reviewed by counsel before use, states:
- **Purpose:** developing and evaluating in-vehicle monitoring AI for products sold by SensorDyme.
- **What is recorded:** face, upper body, hands, NIR and visible video. No audio. Demographics are optional, self-reported and stored separately.
- **Separate opt-ins**, each defaulting to NO:
  - (a) evaluation of SensorDyme models;
  - (b) training of SensorDyme models;
  - (c) named annotation contractors under a data processing agreement;
  - (d) any still image in documents or presentations.
- **Biometric notice:** face landmarks and face geometry are computed from the video, with the purpose and retention period stated.
- **Retention:** raw video is deleted at the earliest of (i) the end of the pilot analysis plus 12 months, (ii) withdrawal, or (iii) 3 years.
- **Withdrawal:** by email, at any time. Raw video, crops and annotations are deleted within 30 days and the deletion is confirmed. Models already trained cannot be "untrained", but the person is excluded from all future training and evaluation manifests. Say this plainly.
- **Contact:** the data protection lead, and the participant's own copy of the signed form.

Consent records are stored separately from the video and keyed by pseudonym. The pseudonym-to-name key is held offline by the data protection lead only.

## 6. Privacy protections

- **No face blurring of the participant.** The task needs the face. Protection comes from consent, minimisation and access control.
- **Bystanders and identifiers:**
  - Mask or crop out bystanders through windows.
  - Remove anything identifying in the frame (number plates, documents, the screen content of the prop phone). Use a blank prop phone where possible.
- **No identity recognition:** no face recognition or re-identification is ever run on this data.
- **No casual copies:** clips are never shown outside the annotation team, and never posted to chat, issue trackers or pull requests.

## 7. Data path and storage (needs approval before collection)

1. **On site:** write to the encrypted SSD, and verify checksums at the end of the session.
2. **Transfer:** move to `$SDJOB_DATA/incabin-pilot/` on a compute worker. Options, to be approved:
   - (a) attach the SSD to the worker;
   - (b) stream Pi → worker over SSH without the Mac storing anything.

   The worker's data volume must be encrypted at rest (BitLocker on quantcompute or oryn's drive). **Verify this before the first session.**
3. **Manifest:** build a dataset manifest with `licence.class = own_collection`, `consent.protocol` = the form version, and `consent.retention_until` = the deletion date. `evaluation.manifest` refuses data past its retention date.
4. **Access:** the operator and the annotators only. Raw video never goes to the MacBook, to git, or to cloud storage.
5. **Deletion:** a script deletes by pseudonym across raw files, crops, caches and annotation exports. Log it, and verify quarterly.

## 8. Annotation procedure

- **Format:** `evaluation/annotations.py` JSON per clip, with `events`, binary `states`, and `ignore` spans (calibration, setup, operator in frame).
- **Tool:** any tool that exports intervals (CVAT, Label Studio video). A converter maps them to this JSON.
- **Labels:**
  - **Events:** `phone_use`, `eyes_off_road` (≥2 s), `long_closure` (≥1.5 s), `yawn`, `nod`, `glance_<target>`, `eating_drinking`.
  - **States:** `eyes_closed` (per eye if feasible), `phone_in_hand`, `phone_near_face`, `face_visible`.
  - **Attributes:** `eyewear`, `occlusion`.
- **Definitions** are written down before annotation starts. For example, `eyes_closed` means both eyelids cover the pupil; a glance starts at the first frame the eyes leave the road region.
- **Pilot quality control:**
  - Two annotators label 100% of the clips independently.
  - Report agreement: Cohen's κ for states, and onset difference for events.
  - A third person adjudicates.
  - Revise the definitions before any larger collection.

## 9. Splits and use

- Pilot subjects are **validation/development only**. They never enter a frozen test set and never train a shipped model unless opt-in (b) was given.
- Split by subject: all of a subject's clips stay together.
- **Reporting:** per subject, with counts. No pooled accuracy claims. Use it to calibrate the head-pose rig (HFOV/distortion), test the glance and phone rules, and size the real collection (target ≥30 subjects per stratum; see the Mission 4 dataset plan).

## 10. Approvals required before the first session

1. This protocol and the consent form (counsel reviewed).
2. Pi use for recording (it touches the device).
3. Transfer and storage route, plus encryption at rest on the target worker.
4. Annotation tool and annotators (data processing agreement if external).
