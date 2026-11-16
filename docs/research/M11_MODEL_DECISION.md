# M11 model decision: release vs experimental, what each must pass on the Pi

**Evidence level of every number here: L1.** Float model, Open Images val (1,585 images), the
product's own letterbox, decode and NMS on a simulated chip. **Not Hailo INT8, not in-cabin, not NoIR,
not a driving result.** Nothing here changes the production `.hef`, decoder or thresholds.

## 1. Decision

| Configuration | Status | Runs where |
|---|---|---|
| **A: shipping `.hef` + `legacy` decoder** (`e69136d3…960e`, 18,554,430 bytes) | **RELEASE** (installed, unchanged) | every device |
| **B: shipping `.hef` + `reference` decoder** (`accelerator.yolo_decoder: reference`, opt-in, exists) | **first candidate**. Promote only per §4, to a development unit first | nowhere by default |
| **C: EXP-08 three-class** (`damo_yolo_t_exp08_person_phone_hand`, `{80: hand}`) | **EXPERIMENTAL**: no compiled `.hef` exists | research only |
| **D: B person + C phone/hand** (two networks) | **EXPERIMENTAL, not implemented** (no `.hef`, no two-network runtime; +0.008 large-person R@P0.80 is under the 0.02 bar) | research only |
| EXP-09 (head-preserving fine-tune) | **REJECTED** (M8.4 §6) | — |
| Relapse escalation | implemented, **OFF** (owner decision: off until false-alert data exists) | — |

**The strongest immediate experiment is B:** the same network and Hailo cost, decoded correctly. It
needs only measurements on the Pi, not training (M10.1 H1: `hwv_stage.py parity --decoder both`).

## 2. New in M11: the product's own thresholds under each decoder (fleet job `20261114-182049-0c1222`)

The product gates with raw scores:
- `person_confidence_threshold` **0.50** for person;
- `yolo_confidence_threshold` **0.20** for every other class.

The decoders score differently, so the same raw number is a different operating point under each.
Scoring only, re-run on the stored M10 predictions:
- inputs: `zoo_legacy_val` `7a7b1888…`, `zoo_reference_val` `93d911bf…`, `exp08_reference_val` `9657f377…`;
- manifest `b878e8429d221f3f`;
- evidence at `~/sensordyme-audit/2026-11-10/m11-evidence/product_thresholds-0c1222/`.

| Model / decoder | Class | AP50 | **At the product gate: precision / recall** | False boxes per image | False-positive localisation / background | Large-person R or false-phone images | P0.80: thr / R | Threshold matching shipping's precision at its gate: thr / R (gain) |
|---|---|---|---|---|---|---|---|---|
| **A** shipping, legacy | person (0.50) | 0.660 | **0.902 / 0.536** | 0.162 | 168 / 65 | large R 0.650 | 0.378 / 0.605 | — |
| **A** shipping, legacy | phone (0.20) | 0.728 | **0.526 / 0.805** | 0.633 | 16 / 77 | **25 / 42** phone-free images flagged | 0.609 / 0.594 | — |
| **B** shipping, reference | person (0.50) | 0.718 | **0.930 / 0.553** | 0.116 | 107 / 61 | large R 0.661 | 0.332 / 0.661 | 0.456 / 0.585 (**+0.049**) |
| **B** shipping, reference | phone (0.20) | 0.795 | **0.527 / 0.914** | 0.714 | 19 / 86 | **28 / 42** | 0.609 / 0.594 | 0.199 / 0.922 (**+0.117**) |
| C EXP-08, reference | person (0.50) | 0.697 | 0.951 / 0.444 | 0.064 | 58 / 34 | large R 0.583 | 0.295 / 0.630 | 0.391 / 0.555 (+0.019) |
| C EXP-08, reference | phone (0.20) | 0.837 | 0.543 / 0.891 | 0.653 | 27 / 69 | 27 / 42 | 0.441 / 0.789 | 0.188 / 0.906 (+0.102) |
| C EXP-08, reference | hand (0.20) | 0.723 | 0.498 / 0.768 | 1.752 | 652 / 983 | — | 0.344 / 0.638 | — |

**Reading:**
1. **Switching A → B with the thresholds unchanged loses no precision in float:**
   - person: precision 0.902 → 0.930 and recall +0.017, with **65 fewer** false person boxes (233 → 168), most of them localisation errors (the M5 anchor-offset defect);
   - phone: precision 0.526 → 0.527 and recall +0.109.

   So B needs no threshold change in float. The person gate could even go to 0.456 for +0.049 recall at A's precision. **That is a proposal for after INT8 parity, not a change.**
2. **The phone gate (0.20) is deliberately permissive and noisy:**
   - 25 of 42 verified phone-free images (A) get a false phone box; 28 of 42 under B (+3, n = 42, not significant).
   - This is why a visible phone is **log-only** (`phone_detected`), and `phone_use` needs a handling cue (hand, motion, at-face, lap-and-gaze-down) **plus** a sustain time.
   - Raising the phone gate is a safety trade (missed phones) for the owner, with in-cabin data. It is **not** made here.
   - The look-alikes behind false phones are remote, clock, mouse and cup (M8.4 §5.3).
3. **Recall by attribute at the product gate:**

   | Object | A | B |
   |---|---|---|
   | Large person | 0.650 | 0.661 |
   | Occluded person | 0.415 | 0.446 |
   | Truncated person | 0.619 | 0.633 |
   | Small person | 0.012 | 0.062 (irrelevant at driver distance) |
   | Truncated phone | 0.667 | 0.833 |
   | Large phone | 0.795 | 0.915 |

## 3. What already exists (not repeated)

- **Matched-precision evaluation:** every model and class at its own precision-0.80 threshold, with paired group bootstraps (M10 job `625fa0`, `docs/research/M10_MODEL_CANDIDATES.md` §3).
- **Class-specific error anatomy:** localisation, duplicate, cross-class, verified-absent and background errors, plus look-alikes (M8.4 §5.3). The new localisation/background split at the product gate is in §2.
- **Known-good fixtures:**
  - `scripts/hw_validation/known_good/`: 6 images (`kg_*.png`, `SOURCES.json`);
  - `expected_float.json` (legacy), reproduced exactly by job `20261114-142158-db510c`;
  - `expected_float_reference.json` (reference, product decoder `DEVICE_REFERENCE`);
  - the parity gate's self-test (`20261114-143113-fc4b32`) matched exactly.
- **Quantisation sensitivity:** host-simulated INT8 costs person about 0.03 AP50 and about 0.04 R@P0.80, while 8-bit outputs alone cost nothing (M8.4 §5.5, **not Hailo**). The Pi's H1 parity replaces this estimate with a measurement.
- **Provenance:**
  - production `.hef` pin `e69136d3ebf3cc8ebaf3da5417a59962b139f3d3d67d715e9a9b1ae10963960e` (`core/model_registry.yaml` `damo_yolo_tinynasl20_t_coco80`; `setup_pi.sh` and `config.yaml` agree, which `hwv_stage.py pins` checks);
  - MediaPipe `.tflite` pins, verified at every boot;
  - the zoo ONNX `02d7ef7f…` for float references;
  - EXP-08: registry `status: unevaluated`, export only, **no `.hef`**.
- **Calibration requirements (for any future compile, C):**
  - at least 1,024 in-cabin NoIR frames through the device letterbox (`research/m10/hailo_compile/calib.py`, byte-identical to `predict`);
  - collected only under the consent protocol;
  - **thresholds re-derived on INT8 scores** on the Pi.

## 4. Promotion criteria (no switch is automatic; owner approval each time)

**B (decoder only):**
1. **H1 `parity --decoder both`:** B's INT8-vs-float agreement on the known-good images is no worse than A's.
2. **`latency`:** `predict()` p95 inside the 100 ms stride budget with < 1 % late frames. Pi decode cost recorded (+0.9 ms on x86).
3. **`thermal`:** 30 minutes, never throttled, fps within 10 %.
4. **In-cabin sessions with verified labels** (`docs/REAL_DATA_EVALUATION_PROTOCOL.md`):
   - B's false alerts per hour no worse than A's;
   - `phone_use` false alerts on mounted and passenger phones no worse.
5. **Thresholds:**
   - keep 0.50 / 0.20 (float shows no precision loss, §2);
   - any change (e.g. person 0.456) only after INT8 scores from the Pi confirm it.
6. Development unit first, never straight to customers.

**C/D:** the same gates, plus the DFC compile (owner approval), INT8 cost ≤ 0.02 person R@P0.80 and ≤ 0.03 hand AP50 vs float, and for D ≥ 29.5 fps with two networks.

## 5. Expected Pi tests for each candidate

| Candidate | Test (command) | Expected |
|---|---|---|
| A | `hwv_stage.py model / parse-hef / capture / parity --decoder legacy / latency / thermal` | the baseline every comparison uses; legacy parity production == offline |
| B | the same capture, `parity --decoder reference` (offline from the same INT8 tensors; production untouched) | agreement no worse than A; nothing switched |
| C | `hwv_stage.py arch --second-hef <path> --second-hef-sha256 <sha> --second-extra-classes 80:hand` | BLOCKED until a compiled `.hef` exists (by design) |
| D | — | not runnable: no two-network runtime |
