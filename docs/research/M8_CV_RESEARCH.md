# M8 CV research: person regression, phone interaction, thresholds, and the hybrid architecture

Mission 8, Checkpoint G and section 9. **No training was run.** Nothing in production changed: the
`.hef`, the decoder default (`legacy`), `config.yaml` and product code are untouched.
- **Data.** Open Images subset `oi7-sd-v1`, manifest `b878e8429d221f3f`. Val split, plus the
  **train split for the pretrained COCO model only**: it never saw Open Images, so train is a fair
  extra test set for it, with 2.6× the phones of val. Train is not fair for EXP-08, which trained on
  it, and no EXP-08 number below comes from train. The frozen test split was not touched; the tools
  refuse it.
- **Code.** `research/m8/`:
  - `coco80_preds.py`: all 80 COCO classes from the shipping float ONNX, both product decoders;
  - `analysis.py`: `ops`, `paired` and `phone`;
  - `platt_from_bins.py`;
  - `qc_coco80.py` and `oryn_m8.sh`: job drivers.
- **Evidence.** `~/sensordyme-audit/2026-11-10/m8-evidence/cv/`, one folder per job.
- **Compute.**
  - quantcompute (RX 7700S, DirectML; provider asserted, `DmlExecutionProvider` first).
  - oryn was offline (Tailscale) for the whole session. Every EXP-08 prediction file lives there, so
    the new EXP-08 analyses are scripted (`research/m8/oryn_m8.sh`) but **not run**. The EXP-08 numbers
    below come from the M7 artifacts already on the MacBook.

| Job | Target | What | Outcome |
|---|---|---|---|
| `20261111-233453-0e0d3f` | quantcompute | inventory of `$SDJOB_DATA` and packages | no models, no saved predictions; ORT 1.24.4, DML; no MediaPipe, no OpenCV |
| `20261111-234104-7b590f` | quantcompute | first prediction run | failed: float32 not JSON-serialisable (fixed in `6cd7901`) |
| `20261111-234153-084592` | quantcompute | all-class predictions, val (1,585) and train (4,883), legacy and reference decoders; `ops`, `paired`, `phone` | succeeded; 414 s |
| `20261111-235004-026a1f` | quantcompute | `phone` re-run, adding scoring on hand-verified images only | succeeded |
| `20261111-235044-b2261f` | quantcompute | `ops` re-run, adding recall by size at fixed precision | succeeded |

**Reproduction check.** The DirectML val numbers reproduce Mission 5's CUDA bench exactly:
- person AP50: 0.656 legacy, 0.713 reference;
- phone AP50: 0.728 legacy, 0.795 reference;
- the same precision-at-recall points.

The model was fetched from the Hailo zoo URL and verified against the pinned sha256 `02d7ef7f…`.
DirectML inference is p50 9.7 ms (`coco80_val/coco80_report.json`). That is not a Pi or Hailo figure.

## A. Person regression and Arch B

### A1. The regression is small where a product operates, large only at the knee

Person, val, 4,008 people. Sources:
- pretrained: job `234153-084592` (`m8_ops_val.json`), identical to M7;
- EXP-08 at NMS 0.5: M7 job `20261111-154932-148be2` (`person_regression.json`).

| | Pretrained, legacy (shipping) | Pretrained, reference | EXP-08, NMS 0.5 |
|---|---|---|---|
| AP50 | 0.656 | **0.713** | 0.686 |
| Recall at precision 0.80 (threshold) | 0.604 (0.379) | **0.661** (0.332) | 0.627 (0.300) |
| Recall at precision 0.90 (threshold) | 0.537 (0.498) | **0.591** (0.451) | — (not in M7 output) |
| Precision at recall 0.70 | 0.446 | **0.695** | 0.548 |

- **At precision 0.80, EXP-08 costs 3.4 recall points.** That precision is roughly where the shipped
  person threshold (0.50, precision 0.90 on val) sits. The M7 headline (0.55 vs 0.695 at recall 0.70)
  is measured at the knee, where both curves fall steeply; the pretrained model never reaches recall
  0.80 at all.
- **EXP-08 is still above the shipping configuration.** It beats the legacy decoder the device
  actually runs on every row: AP50 0.686 vs 0.656, recall at precision 0.80 0.627 vs 0.604.
- **Significance is unmeasured.** The paired test pretrained-reference vs EXP-08 is scripted
  (`oryn_m8.sh`) but not run (oryn offline). For scale: the same tool's paired photographer-group
  bootstrap of reference vs legacy (job `234153-084592`, `m8_paired_val.json`, 300 resamples, 1,060
  groups) gives these intervals:
  - recall at precision 0.80: +0.057 (95% CI 0.045–0.072);
  - AP50: +0.058 (0.047–0.070);
  - precision at recall 0.70: +0.25 (0.16–0.42).

  The interval half-width for recall at precision 0.80 is therefore about 0.013, which makes a
  0.034 gap probably real. This is an estimate, not a test.

### A2. In-cabin people are large, and large people are where both models are strongest

Pretrained reference decoder, recall by size at the precision-0.80 threshold (`m8_ops_*.json`,
job `235044-b2261f`):

| Split | Large | Medium | Small |
|---|---|---|---|
| Val (n 2,649 / 712 / 322) | **0.733** | 0.594 | 0.177 |
| Train (n 8,431 / 2,153 / 1,003) | **0.748** | 0.658 | 0.253 |

On the legacy decoder, small-person recall collapses (0.034 val, 0.046 train), while large is 0.702
val and 0.722 train.

- **A driver fills a large share of a 640×480 cabin frame**, so large-person recall is the number a
  driver monitor depends on. The same split for EXP-08 is scripted but not run (oryn).
- **M7's miss breakdown says the models differ mostly on large people.** Of the people one model
  finds and the other misses, 256 of 346 and 255 of 341 are large, and the counts cancel.

### A3. Head-interference evidence: box quality drops for every class

Source: M7 score files (`20261111-153317-2c1ead`, `20261111-154932-148be2`). "Mean TP IoU" is the
mean IoU of true-positive boxes at score 0.3.

| | Person AP50:95 / AP50 | Person mean TP IoU | Phone AP50:95 | Phone mean TP IoU |
|---|---|---|---|---|
| Pretrained, reference | 0.544 / 0.713 (0.76) | 0.871 | 0.707 | 0.943 |
| EXP-08 epoch 10 | 0.351 / 0.610 (0.58) | 0.778 | 0.448 | 0.856 |
| EXP-08 epoch 50 | 0.433 / 0.678 (0.64) | 0.808 | 0.650 | 0.900 |
| EXP-08 epoch 60 | 0.489 / 0.690 (0.71) | 0.841 | 0.688 | 0.917 |

- **Phone box quality also dropped**, even though phone AP50 held or improved (0.795 → 0.798–0.807).
  A class-specific interference effect would hurt person only. A worse **class-agnostic** box
  regressor hurts every class.
- **Both curves were still climbing at epoch 60**, and both jump in the no-augmentation phase.
- **This supports the M7 hypothesis** that DAMO's `load_pretrain_detector` re-initialised the
  `gfl_reg` box branch, over class competition in `gfl_cls`. It explains the doubled misplaced
  false positives (IoU 0.1–0.5).
- **Hand confusion stays excluded.** Background false positives on a hand box are 1.3–3.3% (M7).
- **EXP-09 is still the decisive test.** It is the head-preserving fine-tune proposed in M7 and not run.

### A4. Calibration: one monotone map per model and class makes thresholds portable

Cross-fitted Platt scaling: a logistic fit on logit(score), fitted on half the photographer groups
and scored on the other half (`m8_ops_*.json`, job `235044-b2261f`).

| Model, class | ECE raw → cross-fit | Raw score where a detection is right 50 / 70 / 80 / 90% of the time |
|---|---|---|
| Pretrained, person (val) | 0.099 → 0.002 | 0.416 / 0.507 / 0.565 / 0.648 |
| Pretrained, phone (val, reference) | 0.069 → 0.028 | 0.513 / 0.641 / 0.714 / 0.805 |
| Pretrained, person (train) | 0.102 → 0.004 | 0.400 / 0.487 / 0.543 / 0.625 |
| Pretrained, phone (train, reference) | 0.075 → 0.016 | 0.500 / 0.619 / 0.689 / 0.779 |
| EXP-08, person ‡ | 0.081 | 0.357 / 0.436 / 0.488 / 0.567 |
| EXP-08, phone ‡ | 0.082 | 0.486 / 0.592 / 0.656 / 0.742 |
| EXP-08, hand ‡ | 0.086 | 0.367 / 0.442 / 0.491 / 0.566 |

‡ Fitted to the M7 score files' reliability bins (`research/m8/platt_from_bins.py`; evidence
`platt_from_bins_m7_scores.jsonl`). The bin fit reproduces the per-detection fit on the pretrained
person class: 0.566 vs 0.565 at hit rate 0.8.

- **Both models are under-confident at high scores and over-confident at low ones.** One map per
  model and class removes that.
- **EXP-08 scores sit 0.06–0.08 lower for the same reliability** (person 0.488 vs 0.565, phone 0.656 vs 0.714 at hit rate 0.8). A shared threshold therefore does
  not transfer between models (see C).

### A5. Is Arch B the right operating point?

**Partly, and not yet worth a second network.**
- **What B buys.** +0.034 recall at precision 0.80 and +0.027 AP50 on person. Phone and hand are
  identical to A.
- **What B costs.** A second `.hef`, about 2× accelerator time, and network switching on the
  Hailo-8L (unmeasured). The product code has a single detector slot (`core/perception.py`) and holds
  one activated network group (`core/accel.py`).
- **Where the gap sits.** On the class (person) and the operating region (the knee) that matter
  least for driver monitoring, where the driver is a large person.
- **Decision.** Prefer A conditionally. Run EXP-09 first (about 3.1 h, needs approval), and the
  scripted large-person split on oryn. Adopt B only if both of these hold:
  - EXP-09 fails to bring person recall at precision 0.80 within 0.01 of pretrained;
  - the large-person recall gap at precision 0.80 is above 0.02.

  B also needs the Pi's two-network throughput (section E) to be acceptable.

## B. Phone interaction

### B1. Open Images phones are mostly not in hands

| | Val | Train |
|---|---|---|
| Phone-verified images | 147 | 390 |
| ↳ phone-free | 42 | 137 |
| Plain phone boxes | 125 | 325 |
| ↳ on images where "hand" was verified | 46 | 122 |
| ↳ overlapping a human hand box (≥10% of the phone; "in hand") | 21 | 55 |

Source: `m8_phone_*.json`, job `235004-026a1f`.

**Open Images phones are mostly product and scene shots**, so any gate requiring a hand or person
will look bad on all-phone recall. The fair quantities are recall on in-hand phones and false phones
on phone-free images.

### B2. The human-hand gate removes about half the false phones, not 90%

Mission 5 reported that a human-hand gate cut false phones on phone-free images from 56 to 6. That
gate can only pass where "hand" was verified, so it also silently deleted every false phone on
hand-unverified images. Scored on hand-verified images only (new `gates_on_hand_verified_images`):

| Legacy decoder, phone ≥ 0.2 | No gate | Phone overlaps a human hand box | Phone inside a person box |
|---|---|---|---|
| Train: phone-free images with a false phone (of 34) | 73.5% (34 FPs) | **38.2%** (16 FPs) | 41.2% (16 FPs) |
| Train: recall of in-hand phones (n 55) | 1.000 | **0.982** | 0.673 |
| Train: recall of not-in-hand phones (n 67) | 0.851 | 0.000 | 0.000 |
| Train: precision (95% CI) | 0.657 (0.571–0.743) | 0.692 (0.582–0.800) | 0.607 |
| Val: phone-free images with a false phone (of 12) | 75.0% | 33.3% | 25.0% |
| Val: recall of in-hand phones (n 21) | 0.952 | 0.952 | 0.762 |

- **A hand-overlap rule is a good *phone-held* detector.** It keeps 95–100% of in-hand phones while
  dropping phones lying elsewhere.
- **It halves, but does not remove, false phones.** About 47% of the false phones on phone-free,
  hand-verified images (16 of 34, train) sit on a human hand box. They are things held in hands.
- **These are oracle (human) hands, so they are an upper bound.** Predicted hands are worse.
  MediaPipe Hands found a hand in only 8 of 152 val images (M5). The EXP-08 predicted-hand gate is
  scripted (`oryn_m8.sh`, tags `exp08_own_hands`, `archB`, `legacy_with_exp08_hands`) but not run.
- **Person-box containment is not a substitute.** It keeps only 67–81% of in-hand phones.

### B3. Look-alikes: cross-class suppression does not work

Every phone detection was paired with the strongest overlapping (IoU ≥ 0.3) detection of remote, tv,
laptop, mouse, keyboard, book, clock, cup and so on (`feature_distributions@0.3`).

| At phone ≥ 0.3 (train, reference decoder) | True phones (n 283) | False phones on phone-free images (n 96) |
|---|---|---|
| Overlapping look-alike score: q25 / median / q75 | 0.172 / 0.259 / 0.391 | 0.191 / 0.320 / 0.430 |
| Most frequent look-alike | remote 110, laptop 89, tv 31, book 28 | remote 40, laptop 16, tv 15, mouse 8, clock 6 |

- **Remote and laptop co-fire on real phones** almost as strongly as on false ones.
- **"Drop the phone if a look-alike scores at least r × the phone score" trades recall for little
  precision.** Train, reference decoder, threshold 0.5:
  - r = 0.5: precision 0.788 → 0.837 but recall 0.709 → 0.527;
  - r = 1.0: nothing changes.

  The legacy decoder (argmax per anchor) shows the same.
- **Look-alike false phones are a training-data problem, not a decoder rule.** EXP-08, which trained
  with Open Images negatives, has the lowest false-phone rate at matched recall (see C1).

### B4. What temporal context can buy (design only; there is no video data)

Nothing below is measured. Image-level numbers are **not** driving-event accuracy.
- **Two kinds of false phone.** Frame-to-frame flicker is roughly independent across frames, and
  k-of-n voting suppresses it. Static look-alikes, such as a remote on the seat or a phone in a
  cradle, persist, and voting does **not** remove them.
- **The held signal targets the persistent kind.** The B2 hand-overlap rule (≥10% of the phone box
  on a hand box) plus motion coherence (the phone box moving with the hand box over a short window)
  addresses persistent look-alikes and parked phones.
- **Measuring it needs consented in-cabin clips with event labels**:
  `docs/research/INCABIN_PILOT_PROTOCOL.md` gates G1–G6 (all open).
- **Inputs to hand to DRIVER/NODES:**
  - per-frame calibrated phone probability (A4);
  - hand-overlap fraction;
  - phone–hand centre gap in phone-box units (`analysis.features`);
  - track id from NODES.

## C. Thresholds and NMS: measured operating points

Val unless marked. NMS IoU 0.5, class-aware, for every model; M7 showed 0.5 also suits the 3-class
model (phone 0.807 vs 0.798, hand 0.725 vs 0.706). "False-phone images" is the share of verified
phone-free images with at least one false box.

| Model / decoder | Class | Threshold | Precision | Recall | False-box images | Source |
|---|---|---|---|---|---|---|
| Shipping (legacy) | person | **0.50 (shipped)** | 0.902 | 0.535 | 7.5% of 40 | `235044-b2261f` |
| Shipping (legacy) | person | 0.40 | 0.820 | 0.596 | 17.5% | same |
| Shipping (legacy) | phone | **0.20 (shipped)** | 0.525 | 0.805 | **59.5%** of 42 | same |
| Shipping (legacy) | phone | 0.40 | 0.684 | 0.727 | 45.2% | same |
| Shipping (legacy) | phone | 0.40, train | 0.753 | 0.776 | 40.9% of 137 | same |
| Reference decoder | person | 0.50 | 0.930 | 0.553 | 7.5% | same |
| Reference decoder | phone | 0.20 | 0.527 | 0.914 | 66.7% | same |
| Reference decoder | phone | 0.40 | 0.686 | 0.750 | 47.6% | same |
| EXP-08, NMS 0.5 | person | 0.50 (shipped value) | 0.952 | **0.428** | 15.0% of 40 | M7 `154932-148be2` |
| EXP-08, NMS 0.5 | person | **0.30** | 0.800 | 0.627 | 32.5% | same |
| EXP-08, NMS 0.5 | phone | **0.40** | 0.783 | 0.789 | **33.3%** of 42 | same |
| EXP-08, NMS 0.5 | phone | 0.20 | 0.534 | 0.867 | 61.9% | same |
| EXP-08, NMS 0.5 | hand | **0.30** | 0.730 | 0.683 | 19.7% of 61 | same |
| EXP-08, NMS 0.5 | hand | 0.40 | 0.869 | 0.570 | 11.5% | same |

Recommendations, as evidence for the product streams. Nothing here is a config change by this stream.

1. **The shipped thresholds do not transfer to the 3-class model.** At person 0.50 its recall falls
   to 0.428. Any 3-class `.hef` needs its own thresholds: person 0.30, phone 0.40, hand 0.30–0.35.
   Re-derive them on INT8 output, because quantisation shifts scores.
2. **Express thresholds as calibrated precision targets, not raw scores.** Use the A4 maps. That
   keeps them stable across models, decoders and INT8.
3. **The shipping phone threshold 0.20 puts a false phone on 59.5% of phone-free images.** Raising it
   to 0.40 costs 7.8 recall points (val; 6.0 on train) and removes about a quarter of those images.
   EXP-08 at 0.40 matches the shipping recall with half the false-phone images (33% vs 60%).
   Single-frame phone thresholds should be paired with B4's temporal and held logic rather than
   tuned alone.
4. **Box quality vs decoder.** Mean TP IoU for person is 0.78 legacy vs 0.90 reference at 0.5. The
   reference decoder matters for anything geometric: hand–phone overlap, tracking IoU. It stays
   experiment-only until INT8 parity (runbook H1).

## D/E. Inputs for attention fusion and tracking (NODES / DRIVER)

| Signal | Measured property | Use |
|---|---|---|
| Calibrated person probability | ECE ≤ 0.007 after the map | driver-present prior; large-person recall 0.73 at precision 0.80 (val) |
| Calibrated phone probability | ECE 0.016–0.028 after the map | per-frame evidence, not an alert |
| Phone–hand overlap ≥ 10% of the phone | keeps 95–100% of in-hand phones; halves false-phone images (oracle hands) | "phone held" vs "phone present" |
| Phone inside a person box | keeps 67–81% of in-hand phones | weak; do not gate on it |
| Look-alike co-score | not discriminative | do not use |
| Box IoU for tracking | reference 0.86–0.95 mean TP IoU; legacy 0.75–0.83 | the legacy +0.5-cell offset hurts association at small sizes |

## Section 9: hybrid architecture comparison

Labels: **M** = measured (source given); **E** = estimate; **U** = unmeasured.

| Criterion | (1) COCO DAMO-T + MediaPipe face/eyes/hands (shipping) | (2) One 3-class DAMO-T (person/phone/hand) + MediaPipe face/eyes | (3) Arch B: COCO DAMO-T person + 3-class phone/hand | (4) General detector + specialised hand model + face/eye landmarks + temporal |
|---|---|---|---|---|
| Person (val AP50 / recall at precision 0.80) | M 0.656 / 0.604 (legacy); 0.713 / 0.661 (reference) | M 0.686 / 0.627 | M 0.713 / 0.661 | as (1) |
| Phone (val AP50 / false-phone images at about recall 0.79) | M 0.728 / 50% at 0.3 (legacy) | M 0.807 / 33% at 0.4 | M same as (2) | as (1) |
| Hand boxes | M MediaPipe found a hand in 8 of 152 web images (M5); U in cabin | M AP50 0.725, precision 0.73 / recall 0.68 at 0.3 | M same as (2) | U: zoo `hand_landmark_lite` needs a palm detector; no accuracy evidence |
| Class interference | none (COCO head) | M person −0.034 recall at precision 0.80; class-agnostic box loss (A3); EXP-09 untested | none for person | none |
| Hailo-8L compile | M shipping `.hef` (zoo v2.15.0), 6 raw outputs | U: needs the DFC (x86 Linux, not on the fleet); head subgraph structurally matches the zoo graph (M, EXP-08 export) | U: as (2), plus two network groups | U: extra zoo nets (landmark nets "not a MediaPipe drop-in") |
| Accelerator time per detector frame | M 22.07 ms mean (PHASE5) | E ≈ 22 ms (same backbone; 3 vs 80 class channels) | E ≈ 44 ms + switch (U) when both run on a detector frame; E ≈ 29 ms average if the 3-class net runs on every 3rd detector frame | E 22 ms + landmark nets (U) |
| Multi-model scheduling | one network group, activated once and held (`core/accel.py`) | same as (1) | U: needs a HailoRT multi-network scheduler and a second detector slot in `core/perception.py` (code change) | U: as (3) |
| Pi 5 CPU | M 29.5 fps pipeline, 414 MB RSS (PHASE5); U per-component FaceMesh/Hands ms | E lower than (1): Hands moves to the accelerator; 3-class decode is cheaper than 80-class (legacy 2.6 / reference 5.9 ms on the worker CPU) | E (2) + a second decode | E highest (extra landmark pipelines plus temporal state) |
| Memory | M 414 MB RSS; `.hef` 18.6 MB | E ≈ (1) | E + a second `.hef` (≈18.6 MB file) | E (1) + landmark nets |
| Power | M ≥ 3.27 W PMIC lower bound | E ≈ (1) | E above (1) (more NPU duty) | E above (1) |
| Provenance | M pinned sha256 `e69136d3…`, registry `shipping` | M checkpoint `baad3842…`, `experiment_only`; `commercial_status: unreviewed` | as (2) plus the shipping pin | U per new net |
| Licensing | Apache-2.0 (DAMO); MediaPipe Apache-2.0, "not for life-critical" | Apache-2.0 base; fine-tune on OI (CC BY 2.0 images, CC BY 4.0 labels); review pending | as (2) | hand/face zoo nets: licence not recorded; SCRFD/RetinaFace/tddfa are non-commercial (excluded) |
| Temporal reasoning | belongs to DRIVER/NODES for every option | same | same | same |

**Recommendation: (2), a single 3-class DAMO-YOLO-T on the Hailo-8L**, with MediaPipe Face Mesh on
the CPU for eyes and head pose, and the temporal and held logic (B4) in DRIVER/NODES. Conditions:
- it must pass an INT8 compile and the Pi gates below;
- EXP-09 should first try to recover person;
- the shipping COCO `.hef` remains the production model until then.

Why:
1. **It is the only option with a measured hand-box source of usable quality** (hand AP50 0.725),
   and hand geometry is what separates phone-held from phone-present (B2).
2. **Same network count and accelerator time as today.** It avoids the unmeasured Hailo-8L
   multi-network scheduling and the product-code change that (3) needs.
3. **It halves false-phone images at the shipping recall** (33% vs 60%, val, 42 images).
4. **Its person cost is small where it matters** (A1, A2), has a plausible cause (A3) and a
   ≈3.1 h fix to try (EXP-09).

Fallbacks:
- **(3)** only under the A5 conditions.
- **(4)'s specialised hand model** only if the in-cabin pilot shows the 3-class hand class failing
  on NoIR/IR images.

### What the real Pi must verify

1. **INT8 accuracy of the 3-class `.hef`.**
   - DFC compile on x86 Linux with ≥1,024 in-domain NoIR calibration frames (`FINETUNE_EXP08_SPEC.md`).
   - INT8 vs float AP50 drop ≤ 1.5 per class on OI val.
   - Re-derive thresholds from the A4 maps on INT8 scores.
2. **Accelerator latency and throughput.** The `.hef` per-frame time against the 22.07 ms DAMO-T
   baseline, and the end-to-end processed fps p50/p95 and stalls (runbook G3/H2).
3. **Host cost.** Measure, per frame on the Pi 5 CPU:
   - 3-class decode time, legacy vs reference geometry;
   - FaceMesh ms per frame;
   - Hands ms per frame (unmeasured today: `compute.measured: []`).
4. **Only if (3) is still in play.** Two network groups under the HailoRT scheduler on one VDevice:
   - per-switch overhead;
   - fps with the 3-class net every frame and every 3rd frame;
   - whether the current single-activation design can host it.
5. **In-cabin NoIR/IR accuracy.**
   - Driver large-person recall at the chosen threshold.
   - Hand recall: 3-class vs MediaPipe Hands.
   - Recall of phones in hand.
   - False phones from cradle phones, remotes and similar hand-held objects.
   - Event-level accuracy with temporal logic (pilot G1–G6).
6. **Thermals and power.** A 30-minute soak (runbook J) with the chosen architecture, plus the
   PMIC power delta against today.

## Not done, and why

- **oryn offline (Tailscale) all session.** Not run, all scripted in `research/m8/oryn_m8.sh`, CPU
  only, no checkpoint loading:
  - every analysis that needs EXP-08 per-image predictions: paired pretrained-vs-EXP-08 CIs, the EXP-08
    large-person split and Platt cross-fit, the predicted-hand phone gate, the held-vs-present
    classifier, and Arch B's realistic gate.
- **No EXP-08 runs on quantcompute.** Its ONNX export and checkpoint exist only on oryn. No
  checkpoint was copied to the MacBook.
- **No EXP-09 and no training.** EXP-09 needs approval, and oryn is offline.
- **No MediaPipe evaluation.** quantcompute has no MediaPipe; no new hand model was downloaded.
- **No Hailo or DFC measurements.** None exist on the fleet. Every Hailo figure above is either the
  PHASE5 device measurement or labelled E/U.
- **No video.** B4 is design only.
