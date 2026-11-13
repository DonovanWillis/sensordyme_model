# M8.4 model audit: baselines, data, failure modes, quantisation, EXP-09 and Hailo preparation

Mission 8.4 "Operation Leapfrog", BETA stream (model research, edge-AI optimisation). Base
`c9029fa`. **Nothing in production changed**: the `.hef` (sha256 `e69136d3…960e`), the decoder
default (`legacy`), `config.yaml` and product code are untouched, and no experimental weights were
promoted. Every number below cites a fleet job; artifacts are in
`~/sensordyme-audit/2026-11-10/m84-evidence/beta/<job>-<name>/`.

**All accuracy here is image-level Open Images V7 val (`oi7-sd-v1`). It is not in-cabin accuracy and
not distracted-driving (event) accuracy.** The frozen test split (1,584 images) was not exported,
scored or looked at.

## 0. Evidence levels (used everywhere below)

| Level | Meaning | Status for the shipping model | Status for a 3-class candidate |
|---|---|---|---|
| L1 float | float32 ONNX/PyTorch on x86 | measured (M5, M8.3, here) | measured (EXP-08, EXP-09, here) |
| L2 host-simulated quantisation | ONNX Runtime QDQ INT8 on x86 (§5) | **measured here** | measured here (§8) |
| L3 compiled-artifact validation | a `.hef` from the Hailo Dataflow Compiler, `parse-hef`, INT8 emulator | the zoo `.hef` exists (compiled by Hailo) | **not done**: no DFC on the fleet (§8.1) |
| L4 Hailo runtime | the `.hef` on a Pi 5 + Hailo-8L | PHASE5 latency only; INT8 accuracy unmeasured | **not done** |

An L2 number is never an L3/L4 number: the DFC uses its own quantisation algorithms
(equalisation, bias correction, optional fine-tuning) and calibration set.

## 1. Baselines (val, Open Images protocol, `research/detector_bench`)

| Model / path | Person AP50 [95% CI] | Person AP50:95 | Person R@P0.80 | Phone AP50 [CI] | Phone AP50:95 | Hand AP50 | Source |
|---|---|---|---|---|---|---|---|
| **Shipping**: DAMO-T COCO, legacy decoder (what runs) | 0.656 [0.61–0.69] | 0.352 | 0.604 | 0.728 [0.62–0.80] | 0.452 | — | M5 `022003-26d0f7`; reproduced here `085405-4d4ad4` (float) |
| Same, device path through `core/accel.py` predict() | 0.660 | 0.353 | — | 0.728 | 0.452 | — | M8.3 `20261113-043857-f182d1` |
| Same weights, reference decoder (opt-in candidate) | 0.713 [0.68–0.75] | 0.544 | 0.661 | 0.795 [0.71–0.87] | 0.707 | — | M5; M8.1 `152527-afe36f`; reproduced `085405-4d4ad4` |
| EXP-08 (3-class, 60 ep, stock loader), NMS 0.5 | 0.686 [0.65–0.72] | 0.489* | 0.627 | 0.807 [0.74–0.88] | 0.688* | 0.725 [0.70–0.75] | M7 `154932-148be2`; M8.1 |
| EXP-09 (head-preserving) | §6 | | | | | | `20261113-084342-27ed4d` |

\* EXP-08 AP50:95 at DAMO's NMS 0.7 (`EXP08_FULL_RESULTS.md`).
R@P0.80 = recall at precision 0.80, the product-relevant operating point (M8). Paired group bootstraps
(M8.1): pretrained-reference − EXP-08 person R@P0.80 = +0.033 [0.018, 0.052].

## 2. Candidates

| Candidate | What it adds | What it costs | Status |
|---|---|---|---|
| Reference decoder (anchor offset 0, multilabel) | +0.057 person / +0.067 phone AP50; box IoU 0.78→0.90 | decode 2.7 vs 1.7 ms p50 on x86 (M8.3) | opt-in, experiment-only until INT8 parity (runbook H1) |
| EXP-08 3-class (person/phone/hand) | the only measured hand-box source (AP50 0.725); own-hand phone gate: false-phone images 33%→2.4% (M8.1) | person R@P0.80 −0.033 (significant) | rejected as production; research checkpoint |
| **EXP-09** head-preserving fine-tune | tests whether the person loss is the re-initialised regressor | one 3.1 h run | run here, §6 |
| Arch B (two detectors) | pretrained person + 3-class phone/hand | 2× accelerator time; multi-network scheduling unmeasured | not justified (M8.1: large-person gap 0.019 < 0.02) |

## 3. Data, splits, leakage, label noise

- **Subset** `oi7-sd-v1`, manifest `b878e8429d221f3f`: 8,052 images (train 4,883 / val 1,585 /
  test 1,584), 5,718 photographer groups, built identically on oryn and quantcompute.
  Split unit = photographer group (`subject`); 9 near-duplicate pairs were grouped at build.
- **Leakage check (new, `research/m84/leakage_check.py`).** On the manifest (job
  `20261113-084342-27ed4d`, gate 2, and the local copy): **0** image ids, **0** photographer groups
  and **0** file sha256 shared between train/val/test. The trainer's own COCO export is re-checked
  after EXP-09 (`leakage_export.json`, §6).
- **Label noise.** Open Images boxes are exhaustive only for human-verified classes. Train: phone
  never verified on 4,486 / 4,883 images (92%), hand on 1,906 (39%) (`EXP08_RESULTS.md`); 4,788 images
  have at least one unverified class. Verified-present-without-box: person 117, hand 48, phone 14
  (build summary). The scorer does not count detections on images where a class is unverified.
- **Evaluation power.** Val has 4,008 person boxes but only **128 phones** (125 plain) and 42 verified
  phone-free images; phone differences under ~0.08 AP50 are not resolvable (CIs, §6 and M8.1).

## 4. Licences (unchanged facts; no new data or models were downloaded)

| Asset | Licence / restriction |
|---|---|
| DAMO-YOLO code + T weights | Apache-2.0 (authors' Drive mirror; Alibaba OSS bucket gone) |
| Hailo zoo `damoyolo_tinynasL20_T` ONNX / `.hef` | zoo terms; float ONNX sha `02d7ef7f…6657` |
| Open Images V7 | annotations CC BY 4.0; images CC BY 2.0 per file (attribution in the manifest) |
| State Farm | Kaggle competition use only: not usable for product numbers |
| Public driver datasets (DMD etc.) | research-only / CC BY-NC-ND |
| Hailo zoo SCRFD / RetinaFace / tddfa | non-commercial |
| Ultralytics | AGPL-3.0 (excluded) |
| MediaPipe | Apache-2.0; model cards say not for life-critical decisions |

## 5. B2: detector characterisation (val only)

Tools: `research/m84/characterise.py` (slices, error anatomy, mining), `research/m84/hostquant.py`.
Models: pretrained reference/legacy decoders, EXP-08 (NMS 0.5), and EXP-09 (added in §6).
Tables: `docs/research/m84_figures/*.csv`; figures `b2_pr.png`, `b2_reliability.png`, `b2_luma.png`,
`b2_hostquant.png` (made by `research/m84/figures.py`).

### 5.1 Size, occlusion, truncation, depiction: recall at each model's precision-0.80 threshold

Job `20261113-085455-3e932f` (oryn CPU). n = plain (non-group) boxes.

| Model | Class | all | large | medium | small | occluded | truncated | depiction |
|---|---|---|---|---|---|---|---|---|
| pretrained reference | person (n 3,683 / 2,649 / 712 / 322 / 1,898 / 1,505 / 246) | 0.658 | **0.734** | 0.594 | 0.177 | 0.580 | 0.714 | 0.423 |
| pretrained legacy | person | 0.598 | 0.702 | 0.469 | **0.034** | 0.498 | 0.677 | 0.358 |
| EXP-08 | person | 0.627 | 0.714 | 0.510 | 0.165 | 0.531 | 0.690 | 0.532 |
| EXP-08 | hand (n 2,109 / 627 / 1,005 / 477 / 847 / 301 / 63) | 0.636 | 0.844 | 0.696 | 0.237 | 0.545 | 0.757 | 0.365 |
| pretrained reference | phone (n 125 / 117 / 8 / 0 / 3 / 18 / 12) | 0.584 | 0.598 | 0.375 | — | — | 0.444 | 0.417 |
| EXP-08 | phone | 0.784 | 0.829 | 0.125 | — | — | 0.889 | 0.667 |

- **Small objects are the dominant miss for every model**: person recall drops from 0.73 (large) to
  0.18 (small, < 0.25% of the image); legacy decoding collapses small people to 0.034 (the stride/2
  anchor shift is a larger fraction of a small box). A driver in a 640×480 cabin frame is large; a
  phone in a hand is medium-to-small and almost absent from OI val (8 medium, 0 small phones), so
  **small-phone performance is unmeasured**.
- **Occlusion costs 0.08–0.15 recall** (person 0.580 vs 0.734 large); truncation costs little.
- Phone attribute cells have n ≤ 18 and are descriptive only.

### 5.2 Low light (PROXY: mean Rec.601 luma of web photos)

Open Images has no illumination label and no NIR imagery. Images were ranked by mean luma (0–255,
from the pixels; `stats_val.csv`); quartile edges 90.3 / 113.9 / 137.6; 106 images have mean luma < 60,
38 < 40. **A dark web photo is not a NoIR cabin frame at night**; this only asks whether accuracy
falls with darkness within this data.

| Model | Class | Q1 darkest AP50 [CI] (npos) | Q2 | Q3 | Q4 brightest |
|---|---|---|---|---|---|
| pretrained reference | person | 0.704 [0.60–0.78] (969) | 0.724 | 0.695 | 0.740 |
| pretrained legacy | person | 0.654 [0.52–0.74] | 0.664 | 0.627 | 0.688 |
| EXP-08 | person | 0.675 [0.58–0.74] | 0.693 | 0.665 | 0.720 |
| EXP-08 | hand | 0.728 [0.68–0.77] (579) | 0.712 | 0.711 | 0.762 |
| pretrained reference | phone | 0.807 [0.65–0.93] (23) | 0.862 | 0.835 | 0.769 |

**No measurable low-light degradation across quartiles** (all CIs overlap; Q1 is within 0.04 of Q4
for every model/class). The absolute-darkness slices (< 60, < 40) are in §6's re-run table.
This says nothing about IR illumination, sensor noise at night or NoIR colour response: those need
in-cabin data (INCABIN_PILOT_PROTOCOL gates G1–G6).

### 5.3 Error anatomy and class confusion (fixed operating points: person 0.4, phone 0.4, hand 0.3)

Jobs `20261113-085455-3e932f` (oryn) and `20261113-091502-563014` (quantcompute; all-80-class
predictions from M8 job `234153-084592`, so look-alike classes are visible).

| Model | Class | FPs | localisation (IoU 0.1–0.5) | duplicate | on another class's box | on a verified-absent image | background |
|---|---|---|---|---|---|---|---|
| pretrained reference | person | 403 | 217 | 12 | 7 (hand 6, phone 1) | 8 | 159 |
| pretrained legacy | person | 523 | 319 | 9 | 7 | 8 | 180 |
| EXP-08 | person | 237 | 147 | 3 | 2 | 20 | 65 |
| pretrained reference | phone | 44 | 5 | 0 | 3 (hand) | 29 | 7 |
| EXP-08 | phone | 28 | 4 | 0 | 1 (hand) | 17 | 6 |
| EXP-08 | hand | 533 | 228 | 15 | 20 (person) | 6 | 264 |

- **Person errors are mostly mislocalisation** (54% reference, 61% legacy): the legacy decoder adds
  ~100 misplaced boxes at the same threshold, the M5 anchor-offset defect in error form.
- **Phone false positives are mostly on verified phone-free images** (29 of 44). The look-alike that
  co-fires (IoU ≥ 0.5) on false phones (pretrained, reference): **remote 25**, clock 4, parking
  meter 3, mouse 3, suitcase 2. False persons co-fire with chair 41, car 19, bed 15, dog 10.
- **Cross-class confusion among the three target classes is rare**: ≤ 7 person FPs and ≤ 3 phone
  FPs sit on a hand box; 20 of EXP-08's 533 hand FPs sit on a person box (IoU ≥ 0.5, i.e. a hand
  box that is really a whole person). Hand "background" FPs (264) are partly unboxed hands: the OI
  hand annotation is the noisiest of the three (39% of train never verified).
- **Hard negatives (train only, for a future run).** `hardneg_phone_train.csv` (job
  `091502-563014`): 557 pretrained-model phone detections ≥ 0.3 not on a phone box, on 449 train
  images. Only **136 are on phone-verified images** (true hard negatives); 421 are on images where
  phone was never verified and may be unlabelled phones. Look-alikes: remote 193, cup 43, book 41,
  laptop 40, toothbrush 30, bottle 28, tv 23, mouse 22. 200 overlap a human hand box (IoA ≥ 0.3):
  hand-held non-phones, the exact confusion a phone-use rule must survive.
- Top false positives per model/class: `b2_top_fps.csv` (image ids, scores, category, luma).

### 5.4 Calibration and PR operating points

Per-class reliability bins and PR curves for every model: `b2_reliability.png/.csv`,
`b2_pr.png/.csv`. ECE (raw score as precision): person 0.099 pretrained / 0.081 EXP-08; phone
0.069 / 0.083 (M7/M8). M8 A4's cross-fitted Platt maps (person ECE 0.099 → 0.002) remain the way
to express thresholds as precision targets; the thresholds that give precision 0.80 here:
person 0.332 (reference) / 0.379 (legacy) / 0.300 (EXP-08); phone 0.609 / 0.609 / 0.409.

### 5.5 Quantisation sensitivity: HOST-SIMULATED INT8 of the shipping heads graph (NOT Hailo)

Job `20261113-085405-4d4ad4` (quantcompute, ONNX Runtime 1.24.4 CPU EP, QDQ, weights int8
symmetric, activations uint8; calibration 256 **train** images, seed 84; ids in
`calibration_ids.txt`). The float run reproduces M5 exactly (0.656/0.728 legacy, 0.713/0.795
reference), so the harness is sound. Paired group bootstrap, 300 resamples, Δ = float − variant.

| Variant | Decoder | Person AP50 (Δ [CI]) | Person R@P0.80 (Δ [CI]) | Phone AP50 (Δ [CI]) |
|---|---|---|---|---|
| float | reference | 0.713 | 0.661 | 0.795 |
| uint8 outputs only | reference | 0.714 (−0.000 [−0.001, 0.000]) | 0.662 | 0.795 (0.000) |
| per-channel W8A8, MinMax | reference | 0.682 (**+0.032 [0.024, 0.040]**) | 0.623 (**+0.038 [0.026, 0.053]**) | 0.783 (+0.012 [−0.055, 0.075]) |
| per-channel W8A8 + uint8 out | reference | 0.681 (+0.032 [0.025, 0.040]) | 0.624 (+0.037 [0.025, 0.053]) | 0.782 (+0.013 [−0.055, 0.075]) |
| per-tensor W8A8 + uint8 out | reference | 0.697 (+0.017 [0.009, 0.023]) | 0.642 (+0.018 [0.008, 0.029]) | 0.811 (−0.016 [−0.046, 0.011]) |
| per-channel, percentile 99.999 (32 imgs) + uint8 out | reference | 0.679 (+0.034 [0.027, 0.043]) | 0.619 (+0.042 [0.029, 0.056]) | 0.780 (+0.015 [−0.051, 0.074]) |
| per-channel W8A8 + uint8 out | legacy | 0.623 (+0.032 [0.024, 0.041]) | 0.568 (+0.036 [0.025, 0.053]) | 0.722 (+0.007 [−0.063, 0.077]) |

Head-level error (anchors where float ≥ 0.05): class-score |Δ| mean 0.035, p99 0.23 for per-channel
MinMax; DFL box distance |Δ| mean 4.4 px, p99 67 px (anchors with float score ≥ 0.20). uint8
outputs alone: score |Δ| p99 0.0017, box 0.45 px p99.

Reading:
- **Rounding the six outputs to 8 bits costs nothing measurable.** The INT8 risk is in the body.
- **Naive post-training INT8 costs person ~0.03 AP50 and ~0.04 R@P0.80 (significant), 4× larger
  in precision at recall 0.70** (0.695 → 0.557). Phone changes are inside the noise (128 phones).
- Per-tensor weights did *better* than per-channel here; we have no verified explanation (it may be
  how ORT's QDQ treats the fused RepConv/DFL convs), so neither number predicts the DFC result.
- **Implication for Monday (runbook H1):** an INT8 person drop of ~0.03 AP50 at the DFC's defaults
  would be unsurprising; the thresholds must be re-derived on INT8 scores (score shifts are large:
  over val, 20,627 (anchor, class) scores crossed 0.40 in one direction or the other, against 46,113
  that are at or above 0.40 in float); and the parity stage should
  compare detections at matched *precision*, not at a fixed raw score.

## 6. EXP-09: head-preserving fine-tune (REJECTED)

**Run.** Oryn job `20261113-084342-27ed4d`: 60 epochs, 3.2 h wall, seed applied (`SD_SEED_FIX=1`),
head-preserving load (rows 0/67/-1). Output checkpoint sha256 `d07e89ee2491d563…`, which stays on oryn.

**Verdict.** The integrator ran the pre-registered criteria with `research/m84/exp09_verdict.py`, because the
BETA stream was stopped before it reported. Evidence: `m84-evidence/beta/exp09/verdict.txt`. Val split only.

| Criterion | Result | |
|---|---|---|
| G1: person R@P0.80 gap to pretrained ≤ 0.010 | 0.614 vs 0.661: **+0.047 [0.030, 0.066]** | **FAIL** |
| G2: phone AP50 ≥ 0.79 | 0.842 [0.772, 0.896] | pass |
| G3: hand AP50 ≥ 0.70 | 0.724 [0.697, 0.754] | pass |
| S1 person P@R0.70 ≥ 0.66 / S2 person AP50 ≥ 0.703 | 0.552 / 0.682 | fail / fail |
| S3 phone R@P0.80 ≥ 0.75 / S4 false-phone images ≤ pretrained | 0.781 / 0.357 vs 0.476 | pass / pass |

**Compared with EXP-08:**
- person R@P0.80: −0.013 [−0.027, −0.000];
- phone AP50: +0.035 [−0.003, +0.067];
- hand: unchanged.

**Large people, recall at precision 0.80:**

| Model | Recall |
|---|---|
| Pretrained | 0.733 |
| EXP-08 | 0.714 |
| EXP-09 | 0.703 |

EXP-09's gap is 0.030, over the 0.02 bar.

**Decision: reject.** Keeping the pretrained head did not close the person regression, and EXP-09 is no better
than EXP-08. The head re-initialisation (§7) is therefore **not** the main cause. Implications:
- The single 3-class network does not meet the person bar.
- By the M8 rule, Arch B (pretrained person + 3-class phone/hand) is now the candidate route. It still needs the
  Pi INT8 and two-network throughput checks, and a decoder change for the hand channel (§8).
- The production `.hef` is unchanged and is not affected.

## 7. Prior attempts and what M8.4 found in them

- EXP-08a/c/d pilots (2 epochs; `EXP08_RESULTS.md`): stock labels most balanced; binary ignore masks
  hurt; same-seed reruns differed by 0.06 phone AP50.
- **New: the DAMO-YOLO trainer never applies its seed.** `damo/apis/detector_trainer.py:98` has
  `# set_seed(cfg.miscs.seed)` commented out at the pinned commit `319572e`. Every EXP-08 run recorded
  "seed 1234" but was unseeded, which explains the same-seed run-to-run spread. EXP-09 applies it
  (`SD_SEED_FIX=1`); cuDNN non-determinism remains, so EXP-09 is still one draw.
- **New: the head re-initialisation is confirmed in code**, not just inferred:
  `load_pretrain_detector` copies every tensor except names containing `head`, i.e. `gfl_cls`,
  `gfl_reg` and the per-level `scales` all start from random init (`damo/detectors/detector.py:44–49`).
- EXP-08 export (`20261111-090242-3ab280`): structurally matches the zoo heads; 5.8 ms CUDA / 47.8 ms
  CPU on oryn (not Pi).

## 8. B4: Hailo deployment preparation

Filled by the integrator from the committed tooling and fleet artifacts, because the BETA stream was stopped
before it reported.

- **Tooling probe (oryn job `20261113-090503-dc1833`).** No Hailo SDK, Dataflow Compiler,
  `hailo_platform` or zoo on the fleet: only `onnxruntime` and `onnx`, on x86_64.
  - **Compiled-artifact validation:** NOT DONE.
  - **Hailo runtime:** NOT DONE, needs the Pi.
  - Installing the proprietary DFC needs owner approval.
- **Device-layout export of EXP-08 (oryn job `20261113-090349-6d22e9`, `research/m84/device_export.py`).** It
  writes 6 heads in the device's layout (80/40/20 grids, 68-channel box heads), with person, phone and hand in
  class slots 0/67/80 of an 81-channel class head:
  - ONNX sha `f591a6b6…a609`, heads81 sha `a0327a25…aa1`.
  - **Parity:** torch vs ONNX |Δ| max 6.5e-5 (box) and 1.5e-6 (cls); heads81 vs heads3 identical; unused
    channels exactly 0.
  - **Product decoders drop channel 80 (hand).** Both `legacy` and `reference` keep 80 COCO classes, so a 3-class
    model on the device would report person and phone but never hand until the decoder is changed. That is a
    product-code change, not made here.
  - **The job was a smoke run (`--limit 30`).** Its `score_exp08_device_*_val.json` files score 30 images
    against the full val ground truth (4,008 persons), so their "AP50" values (person 0.011, phone 0.039) are
    an artefact of coverage. **They are not accuracy and must not be cited.**
  - A full-val device-path run of the export was not completed.
- **Float fixtures** for a future compiled `.hef`: `exp08_known_good_expected_float.json` and
  `exp08_oi_val_expected_float.json`, in the job's artifacts (`~/sensordyme-fleet/runs/20261113-090349-6d22e9/`).

## 9. Artifacts (hashes; large files stay on the workers)

| Artifact | sha256 | Location |
|---|---|---|
| Shipping `.hef` | `e69136d3ebf3cc8e…960e` | Pi / zoo v2.15.0 (registry) |
| Shipping float ONNX (zoo 2022-12-19) | `02d7ef7f35da6abf…6657` | both workers `$SDJOB_DATA/models` |
| Shipping heads graph (zoo cut, ORT) | `de27bd7cab8d391e…` | quantcompute `$SDJOB_DATA/models` |
| DAMO-T COCO checkpoint | `a5ee63820e0e02a8…5f95` | oryn `$SDJOB_DATA/models/drive` |
| EXP-08 best_ckpt | `baad3842943b6ebc…f37b` | oryn `$SDJOB_DATA/exp08_full60/workdirs` |
| EXP-09 best_ckpt | §6 | oryn `$SDJOB_DATA/exp09_headpres/workdirs` |
| Host-simulated INT8 graphs | `hostquant_report.json` `quant_sha256` | quantcompute `$SDJOB_DATA/m84_hostquant` |
| Val luminance stats | `stats_val.csv` (evidence) | oryn `$SDJOB_DATA/m84` |

## 10. Experiments ranked by expected value (for the product decision, not for a paper)

| Rank | Experiment | Why it is worth it | Cost | Gate |
|---|---|---|---|---|
| 1 | **INT8 parity of the shipping `.hef` on the Pi** (runbook H1, known-good + OI fixtures) | every threshold and the reference-decoder switch wait on it; §5.5 says a ~0.03 person drop is plausible | 1 h on the Pi | hardware session |
| 2 | **EXP-09** | decides one-network (A) vs two (B) for the 3-class route | 3.1 h oryn | done here (§6) |
| 3 | Calibrated thresholds (Platt maps per model × decoder × INT8) | removes per-model threshold retuning; needs only scores | hours, CPU | after 1 |
| 4 | Evaluation power: OI V7 extension for phone / hand / verified negatives (val-side) | 128 phones cannot resolve < 0.08 AP50; every phone decision is underpowered | ~3–5 GB download | owner approval |
| 5 | In-cabin consented NoIR pilot | the only data that measures the product (domain, IR, posture, real phones-in-hand) | weeks | G1–G6 |
| 6 | Compile the chosen 3-class candidate with the DFC (≥ 1,024 NoIR calibration frames) and run §8's fixtures | L3/L4 evidence for the hand class | x86 Linux + SDK licence | owner approval |
| 7 | Hard-negative fine-tune with the 136 verified train hard negatives (+ hand-held look-alikes) | targets remote/cup/book confusions | 3.1 h | after 4 (otherwise unmeasurable) |
| 8 | Seed variance of the winning recipe (3 seeds) | one run is one draw; needed before any claim | 9 h | after 2 |
| 9 | QAT / quantisation-aware fine-tune | only if 1 shows an unacceptable INT8 drop | 3–6 h + DFC | after 1, 6 |
| 10 | Larger backbone (DAMO-S) | Hailo-8L latency budget; no evidence T underfits | 6 h + compile | low |
