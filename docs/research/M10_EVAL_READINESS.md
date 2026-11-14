# Mission 10 evaluation readiness: in-cabin event-level workflow

**Status (2026-11-14):** the workflow is built and tested on synthetic data
only. **No in-cabin recording exists.** Every report run today prints
**NO DATA**. Nothing in this document is an accuracy figure.

**Read with:**
- `docs/EVALUATION.md`: the harness and the M10 pieces;
- `docs/research/INCABIN_PILOT_PROTOCOL.md` (DRAFT): safety, consent, data path;
- `docs/research/INCABIN_READINESS.md`: gates G1–G7, session checklist, transfer;
- `docs/SUNDAY_DEVICE_ACTIVATION.md` §5: behavioural cases B1–B8.

The additions below are **proposed amendments to the draft protocol**. They
take effect only when the protocol is approved (gate G2). No recording may start
before gates G1–G6 are closed.

## 1. What is ready

| Piece | Command / file | Tested by (`tests/test_eval_m10.py`) |
|---|---|---|
| Scenario coverage matrix v1 (27 cells) | `evaluation/schemas/incabin_scenarios_v1.yaml`; `python -m evaluation.scenarios` | required scenarios covered; unsafe or inconsistent matrices refused; clip tags checked against cells |
| Annotation schema v2, v1 migration | `evaluation/schemas/incabin_v2.json`, `evaluation/annotations_v2.py` | 12 refusal cases; v1 migration from manifest metadata, or refusal; annotation and manifest must agree |
| Double-annotation agreement | `python -m evaluation.agreement --a A1/ --b A2/ --out a.json` | IoU, F1, onset and κ known answers; pass/fail; same-annotator refusal |
| Leak-free subject splits | `python -m evaluation.leakage assign / check` | subject, session, vehicle-day and co-occupant crossings refused; linked groups placed together, deterministically; multi-frame near-duplicates; training-manifest overlap |
| Locked test split + access log | `python -m evaluation.leakage lock / verify` | hash tamper, moved item, grown test set, sneaky re-entry refused; withdrawal allowed; `--frozen-test` + purpose required; accesses numbered |
| Event-level metric suite + power check | `evaluation/event_metrics.py` | every outcome counted with known answers (caught, repeat, false, false on legitimate behaviour, boundary, ignored, uncertain, abstained, unscored); latency and latency minus sustain; per-subject false alerts/h; bootstrap; NO DATA / INSUFFICIENT / ok |
| Report generator | `python -m evaluation.report ...` → `eval_report.md`, `eval_report.json` | NO DATA everywhere today; provenance recorded; claims withheld below minimum; eight refusal paths; test access recorded |
| Replay availability | `evaluation/replay.py`, `evaluation/run.py` (`availability.jsonl`) | end-to-end replay of a synthetic clip through the real `DriverAttentionModule` (face mesh fed geometric landmarks): calibration window and the not-started detector appear as unknown time; the head-down event is caught at the eyes_down sustain time |

Test count: 43 in `tests/test_eval_m10.py`, plus the existing 14 in
`tests/test_eval_harness.py` (unchanged, passing).

## 2. Protocol additions (proposed)

### 2.1 Per-clip metadata (session CSV → manifest `cond_*` columns)

Required for every clip. Without them the split check cannot prove there is no
leakage, and the manifest is refused.

| Column | Example | Why |
|---|---|---|
| `subject` | `S01` | the split unit |
| `session` | `S01-2026-12-05-a` | one continuous recording |
| `cond_vehicle` | `V01` | vehicle-day leakage |
| `cond_recorded_on` | `2026-12-05` | vehicle-day leakage |
| `cond_seat` | `driver` | annotation v2 `clip.seat` |
| `cond_occupants` | `S07` (`;`-separated) | anyone else visible is linked to the subject's split |
| `cond_mount`, `cond_lighting`, `cond_eyewear`, `cond_occupancy` | `dash_centre`, `night_ir`, `none`, `driver_only` | required matrix tags (strata) |
| other `cond_<axis>` | `phone=handheld_lap` | the clip's value on each matrix axis it exercises |
| `cond_scenarios` | `P2_phone_in_lap;L3_night_ir` | the cells the clip counts toward |

### 2.2 Recording checklist per scenario cell

Every cell inherits the matrix safety block: stationary vehicle, parking brake
on, engine off or accessory power; consenting adults only; no children in the
vehicle; nobody drives; drowsiness is acted; an operator is present; anyone may
stop at any time. Per-cell checklists and extra safety notes are in the matrix
file (`checklist`, `safety`). In summary:

| Cell | Record | Labels it must carry | Cases |
|---|---|---|---|
| N1 baseline | 2 min attentive; 30 s hands in lap | mirror glances (negatives) | B5, B7 |
| M1 multiple people | passenger moves and talks; the driver does cued off-road looks and closures | eyes_off_road, long_closure, glance_passenger | |
| SP1 seating | forward/high, rearward/low, reclined; recalibrate after each change | eyes_off_road, long_closure, glances | B2 |
| EW1 / EW2 / EW3 eyewear | clear glasses; IR-pass sunglasses; IR-block sunglasses (expect abstention) | long_closure, eyes_off_road (+ face_occluded for IR-block) | B1, B6 |
| L1–L4 lighting | day; dusk with illuminator off; night with 850 nm IR; direct sun or glare | long_closure, eyes_off_road, phone_use, glances | B1, B2, B4, B5 |
| P1 / P2 phone hand / lap | blank prop phone, look 3–5 s ×5; repeat with the hand hidden | phone_use, eyes_off_road, phone_handling | B4 |
| P3 phone at ear | 10 s each ear ×3 | phone_handling (`attrs.position: ear`) | |
| P4 mounted phone | 60 s unused; 4 s looks ×3 | eyes_off_road (`attrs.target: mounted_phone`) | B3 |
| P5 passenger phone | passenger looks at a prop phone 10 s ×5 | passenger_phone_use (negative) | |
| H1 head orientation | sweeps to targets at 15/30/45/60°, held 1 s and 4 s | eyes_off_road, glances, nod | B2, B5 |
| V1 vibration | door closes and mount taps every 10 s; engine idle only if the approved protocol allows it, outdoors | long_closure, glances | B3 |
| O1 / O2 / O3 occlusion | hand over mouth/eyes; cap brim / visor; card over the lens (full, partial) | face_occluded, camera_occluded | B8 |
| A1 absence | lean out of frame 5 s ×5; step out and back once (kerb side) | driver_absent | B8 |
| E1 / E2 / E3 eye closure | natural blinking 2 min; acted 0.5 s and 1 s closures and nods; acted 2 s and 3 s closures | eyes_closed state; short_closure, nod (boundary); long_closure | B1 |
| G1 / G2 / G3 glances | mirrors, cluster/infotainment, passenger: cued, under 2 s, ×5 | glance_* (negatives); eyes_off_road for the 3–4 s looks | B2, B5 |

### 2.3 Annotation

- Use schema v2 (`evaluation/schemas/incabin_v2.json`). An event's `start_s` is its
  **onset** (the first frame the behaviour begins), so latency is measured from
  what the driver did, not from when a duration threshold was crossed.
- Mark time that cannot be labelled as `uncertain` with a reason and the
  affected types. Do not drop it and do not guess. The report shows the
  unlabelable fraction.
- Two annotators label 100% of the pilot independently (protocol §8). Run
  `python -m evaluation.agreement`. Thresholds: event IoU ≥ 0.5, agreement
  F1 ≥ 0.8, onset p90 ≤ 0.5 s, state κ ≥ 0.7. A failing clip pair goes to the
  adjudicator; only the adjudicated file is referenced from the manifest.

### 2.4 Split and lock order (once per dataset version, before any scoring)

1. Build the manifest (`python -m evaluation.manifest build`), with the columns in §2.1.
2. `python -m evaluation.leakage assign` (split by linked subject group), then
   `check --root ... --frames 5`.
3. `python -m evaluation.leakage lock --out split_lock.json`. Store the lock with
   the manifest. It is never regenerated: a new lock means a new test set, and it
   must be reported as one.
4. Tune only on `val`. Score `test` only with `--frozen-test --purpose "..."`.
   Every access is appended to `split_lock.json.access.jsonl`, and the report
   prints the access number.
5. Withdrawal: delete the participant's items, rebuild the manifest, and re-verify.
   The lock allows removals and lists them in the report.

### 2.5 Consent and data handling

No change to the draft protocol. Pointers:
- consent and opt-ins: protocol §5 and the readiness consent workflow. Opt-in (a),
  evaluation, is required for any clip in a manifest;
- a passenger is a participant with their own form and pseudonym (`cond_occupants`);
- privacy: protocol §6. Data path, encryption and transfer: protocol §7 and gates
  G4/G5. Only manifests, lock files and reports reach the Mac; raw clips never do;
- retention: `consent.retention_until` (enforced by `evaluation.manifest.validate`).

## 3. What data is needed for first numbers

| Stage | Data | What the report can show |
|---|---|---|
| Today | none | NO DATA everywhere (the command works today) |
| Pilot (protocol §9) | 3–5 consenting adults, day + night IR, the §4 script, double-annotated | INSUFFICIENT DATA in every cell, by design. `--include-diagnostics` adds per-subject diagnostics to the JSON (never claims). These are useful for label definitions, annotator agreement, abstention and latency sanity, and finding failure modes |
| First claimable cell | in the split being reported: ≥ 30 subjects and ≥ 30 clips in the cell, ≥ 60 events per positive type, ≥ 60 events per negative type, ≥ 3.0 exposure hours per alert type | that cell's numbers, with Wilson and subject-bootstrap intervals |

**Sizing note.** Minimums apply to the split being reported. A final claim is
made on the locked test split, so the collection needs ≥ 30 test subjects per
claimed cell:
- with the default 70/15/15 split, that is about 200 subjects;
- if in-cabin data is used for **evaluation only** (no training on it, the
  current plan), assign 0/50/50 (`--split 0,0.5,0.5`), and about 60 subjects
  give 30 for tuning (val) and 30 for the final measurement (test).

The cheapest first claims are the cells that one session covers many times over:
- E3 sustained closure: 3 reps × 2 durations per subject means ≥ 60 events from 10 subjects. The subject minimum (30) is the binding constraint;
- G1 mirror glances: 3 mirrors × 5 per subject;
- N1 baseline exposure: 3 h over 30 subjects is 6 min each.

## 4. Open items

- **The protocol additions above need G2 approval.** The protocol is still DRAFT, and gates G1–G6 are open.
- **Minimums are a proposal.** Changing them means bumping the matrix version, and the report records the sha256.
- **Phone use is never evaluable on the Mac surrogate.** It needs the object detector. When the detector has not run, `phone_use` shows as 100% unknown, as the end-to-end test demonstrates. Shipping-path numbers need replay on a Pi 5 + Hailo-8L.
- **No annotation converter.** One is needed from CVAT / Label Studio exports to v2 JSON.
- **Alert episodes are not replayed.** Runtime AlertPolicy episodes (escalation, repeats) are not part of replay. Scoring is on module events.
- **Some sustain times have no key.** `frequent_glancing` and other soft alerts are not scored; they are listed as unscored module events.
- **Two definitions differ.** v2 `eyes_off_road` is ≥ 2 s; the product's `distraction_seconds` is 3.0. Recall will show the gap honestly; it is not a labelling error.
