# M10 model candidates: hand-class decoding, the export contract, and architectures A / B / C

Mission 10 "Accuracy first", MODEL stream. Base `4dfd761`, branch `m10/model`. **Nothing in production
changed**: the `.hef` (sha256 `e69136d3…960e`), the decoder default (`legacy`) and `config.yaml` are
untouched, no model was promoted, nothing was trained, nothing was downloaded. The frozen test split
(1,584 images) was not exported, scored or looked at.

**Evidence levels** (as in `M84_MODEL_AUDIT.md` §0): every accuracy number here is **L1** (float32 ONNX on
x86, oryn CPU). There is no L2 (host-simulated INT8), L3 (compiled `.hef`) or L4 (Pi + Hailo-8L) number
for any candidate in this document. Latency numbers are oryn desktop CPU / RTX 4060 and are **not Pi
numbers**. All accuracy is image-level Open Images V7 val (`oi7-sd-v1`, manifest `b878e8429d221f3f`,
1,585 images): not in-cabin, not event-level.

## Fleet jobs (all oryn; artifacts in `~/sensordyme-fleet/runs/<job>/artifacts/`, sha256-verified)

| Job | What | Commit | Cite? |
|---|---|---|---|
| `20261114-034527-625fa0` | device-path val run of every candidate + A/B/C table + paired bootstraps (1,000 resamples) | `4cf3142` | **yes** (main result) |
| `20261114-040135-b69aa0` | desktop latency, uncontended | `0869e3f` | yes (desktop only) |
| `20261114-035312-ff8c48` | export contract incl. swap-R/B on full val | `978d423` | yes for the colour test; its class-map rows were a tool bug (fixed in `86d9f1a`) |
| `20261114-035955-9ffec6` | export contract re-run (fixed class-head filter, DAMO config quotes; no colour pass) | `0869e3f` | yes |
| `20261114-034422-c20223` | 30-image smoke of the device-path job | `4cf3142` | **no** (coverage artefact) |
| `20261114-034919-d2ff0e` | latency while `625fa0` held 4 CPUs | `95d5b0c` | no (CPU contended; superseded by `b69aa0`) |
| `20261114-034537-965b9a` | latency, input file missing | `4cf3142` | no (failed before measuring) |

Inputs (sha256, verified in-job): zoo heads graph `de27bd7cab8d391e36d1ac1b84850f88d474cd3307e6b2b990e9b3c49ce850ba`
(the M8.3 graph); EXP-08 `heads81` `a0327a25da034e942bb1f6529b7747cb5ddef8c821dc9545ac40ccf394ed7aa1` (M8.4 job
`20261113-090349-6d22e9`, from best_ckpt `baad3842943b6ebc…f37b`); stored predictions M5 legacy `0f55bf8d…c4bf`,
M5 reference `725d8b2b…8b03`, M7 EXP-08 NMS 0.5 `ba6b0ad1…5710`. `core/accel.py` in the job:
`11df3c131efba422b7a8e31059b9508b64b6d519e1a280c684f6ad096201424a` (this branch).

## 1. Task 1: hand-class decoding (`accelerator.yolo_extra_classes`)

Both decoders kept only the 80 COCO class channels (`core/accel.py`), so the EXP-08 device-layout export
(person → 0, phone → 67, hand → 80 of an 81-channel class head) reported person and phone but never hand
(M8.4 §8).

- **Change.** An opt-in, typed map `accelerator.yolo_extra_classes: {80: hand}`, validated by
  `core/config_schema.parse_yolo_extra_classes` (refuses channels < 80, non-integers, empty, duplicate or
  COCO-colliding names) and honoured by both decoders: a mapped channel is one more class column (it
  competes in the legacy argmax, it is a candidate in the reference multilabel selection), goes through
  the same class-wise NMS, and is named in `model.names`.
- **Default unchanged, bit for bit.** Absent/empty map = the old `[:, :80]` slice. Checked against the
  pre-M10 `core/accel.py` (from `4dfd761`) on 20 random head sets × 2 decoders × 2 thresholds: identical
  bytes for all 291,205 boxes (one-off check, not committed); in-repo by
  `test_the_default_is_bit_for_bit_the_pre_m10_decode` and the unchanged existing decoder suites
  (`test_hailo_decoder.py`, `test_phase3_raw_decode.py`, `test_m83_inference_path.py`, 159 passed with
  `test_config_schema.py`, `test_accel*.py`, `test_hailo_lifecycle.py`).
- **Fail closed at load**, before the exclusive chip is opened: a class head narrower than the highest
  mapped channel + 1, an on-chip-NMS `.hef`, and the shipping COCO-80 `.hef` (its channel 80 is the
  network's own background slot: max 0.0079 over val in this run, 0 of 1,585 images ≥ 0.20).
- **Proof on fixtures** (`tests/test_m10_hand_class_decode.py`, 37 tests): synthetic device-layout heads
  (80/40/20 grids, 68-channel DFL, 81-channel class) with planted peaks for channels 0, 67 and 80 at known
  cells and strides; exact class, score and box after decode + NMS for both decoders (expected values by
  hand arithmetic); a hand under the phone's anchor is kept by `reference` and not by `legacy` (argmax);
  a duplicate hand is suppressed; the default drops channel 80; the refusals above; a 640×480 frame
  through the fake chip returns the hand in frame pixels.
- **Not wired into the product.** No module asks for hands; `docs/DECODER_CANDIDATE.md` documents how a
  consumer would (look the id up by name in `model.names`, `predict(..., classes=[hand_id])`).

## 2. Task 3(a): EXP-08 device path on full val, hand proven end to end

Job `625fa0`: `heads81` through `HailoYOLO.predict` on the fake chip (letterbox, BGR→RGB, uint8 NHWC,
decode, NMS 0.50, map back), 1,585 val images, map `{80: hand}`.

| EXP-08 path | Person AP50 | Phone AP50 | **Hand AP50 [95% CI]** | Hand R@P0.80 |
|---|---|---|---|---|
| device path, `reference` decoder | 0.697 | 0.837 | **0.723 [0.698, 0.750]** | 0.638 |
| device path, `legacy` decoder | 0.655 | 0.829 | 0.619 [0.587, 0.656] | 0.547 |
| same heads, DAMO stretch input (diagnostic) | 0.691 | 0.811 | 0.722 [0.697, 0.751] | 0.623 |
| DAMO's own pipeline, re-NMS 0.5 (M7 preds, re-scored here) | 0.686 | 0.807 | 0.725 [0.700, 0.755] | 0.635 |

Paired group bootstrap (device-path reference minus the other row):

| Δ | AP50 [95% CI] | R@P0.80 [95% CI] |
|---|---|---|
| hand: device vs DAMO pipeline | −0.0025 [−0.0135, +0.0089] | +0.003 [−0.020, +0.021] |
| hand: letterbox vs stretch | +0.0007 [−0.0097, +0.0116] | +0.015 [−0.010, +0.031] |
| hand: reference vs legacy decoder | **+0.104 [0.085, 0.124]** | **+0.091 [0.066, 0.124]** |
| person: device vs DAMO pipeline | +0.010 [0.002, 0.018] | +0.002 [−0.010, +0.016] |
| phone: device vs DAMO pipeline | +0.030 [−0.002, +0.061] | 0.000 [−0.099, +0.095] |

**Reading.** The hand class survives the device path: 0.723 vs the float 0.725, a gap of −0.0025 whose
interval contains 0. The letterbox (the device) vs stretch (EXP-08's training transform, `keep_ratio:
False`) makes no measurable hand difference. **The legacy decoder costs hand 0.10 AP50**: its +stride/2
anchor shift is a large fraction of a hand box (most hands are medium/small), the same mechanism that
collapsed small-person recall in M8.4 §5.1. A 3-class model on the device needs the reference decoder.
The device path is slightly *better* than DAMO's pipeline for person (+0.010), which is consistent with
exact NMS 0.5 on raw outputs vs M7's re-suppression of NMS-0.7 output (an approximation, M7 §2).
The 30-image M8.4 smoke numbers (`6d22e9`) remain uncitable; these replace them.

## 3. Task 3(b): A / B / C at equal operating conditions (val, job `625fa0`)

**Conditions, identical for every row unless marked:** same 1,585 images, PIL decode fitted to a 640
max side, the product's own letterbox / decode / class-wise NMS 0.50 (`core/accel.py` on the fake chip),
floor 0.05, the Open Images scorer (`research/detector_bench/oid_eval.py`). **Operating point: each
model and class at its own precision-0.80 threshold** (equal precision, not equal raw score).

**Arch B (fusion, exact):** per image, the union of the shipping model's `person` detections (reference
decoder) and EXP-08's `phone` and `hand` detections (reference decoder, map `{80: hand}`), scores and
boxes unchanged, no cross-model suppression (disjoint classes, each already class-wise NMS'd at 0.50).
`B_legacy_person` takes person from the legacy decoder instead (= today's detector plus a second network).

| Arch | Model / decoder | Person AP50 [CI] | Person R@P0.80 (thr) | Large-person R@P0.80 | Phone AP50 [CI] | Phone R@P0.80 (thr) | False-phone images @P0.80 [Wilson] | Hand AP50 [CI] |
|---|---|---|---|---|---|---|---|---|
| **C** what runs | shipping, `legacy` | 0.660 [0.617, 0.696] | 0.605 (0.378) | 0.703 | 0.728 [0.616, 0.804] | 0.594 (0.609) | 10/42 = 0.238 [0.135, 0.385] | — |
| C | shipping, `reference` (opt-in candidate) | 0.718 [0.684, 0.752] | 0.661 (0.332) | 0.733 | 0.795 [0.705, 0.869] | 0.594 (0.609) | 10/42 = 0.238 [0.135, 0.385] | — |
| **A** | EXP-08, `reference`, map | 0.697 [0.658, 0.730] | 0.630 (0.295) | 0.725 | 0.837 [0.777, 0.894] | 0.789 (0.440) | 14/42 = 0.333 [0.210, 0.485] | 0.723 [0.698, 0.750] |
| A | EXP-08, `legacy`, map | 0.655 [0.609, 0.691] | 0.592 (0.318) | 0.700 | 0.829 [0.760, 0.888] | 0.781 (0.442) | 14/42 = 0.333 | 0.619 [0.587, 0.656] |
| A | EXP-09 (recorded verdict; **not** equal conditions: DAMO pipeline, re-NMS 0.5) | 0.682 | 0.614 | 0.703 | 0.842 [0.772, 0.896] | 0.781 | 0.357 (at its own op. point, M8.4 §6) | 0.724 [0.697, 0.754] |
| **B** | shipping `reference` person + EXP-08 phone/hand | 0.718 [0.684, 0.752] | 0.661 (0.332) | 0.733 | 0.837 [0.777, 0.894] | 0.789 (0.440) | 14/42 = 0.333 [0.210, 0.485] | 0.723 [0.698, 0.750] |
| B | shipping `legacy` person + EXP-08 phone/hand | 0.660 | 0.605 (0.378) | 0.703 | 0.837 | 0.789 | 14/42 | 0.723 |

n: person 4,008 boxes (2,649 large plain boxes, area ≥ 2.25% of the image); phone 128 (42 verified
phone-free images); hand 2,113. AP50 CIs: 300 photographer-group resamples.

**Reproduction checks (same scorer):** the stored M5 prediction files re-score to 0.656 / 0.728 (legacy)
and 0.713 / 0.795 (reference) person / phone AP50, as published. The device-path `legacy` predictions
here are **byte-identical** to M8.3's (`7a7b1888…`, job `20261113-043857-f182d1`, person 0.660). The
device-path `reference` person AP50 is 0.718 vs M5's 0.713; the difference is the input (M5 fed native
resolution into the letterbox's cv2 INTER_LINEAR downscale; here PIL antialiased fit-to-640 first, as M8.3
did). Phone is identical. Within this table every row uses the same input.

### Paired deltas (group bootstrap, 1,000 resamples, A − B form)

| Comparison | Person AP50 | Person R@P0.80 | Large-person R@P0.80 | Phone AP50 | Phone R@P0.80 | False-phone-image rate |
|---|---|---|---|---|---|---|
| **B − A** (person) | +0.021 [0.009, 0.033] | **+0.031 [0.014, 0.047]** | +0.008 [−0.003, +0.020] | = (same source) | = | = |
| **A − C legacy** | **+0.037 [0.022, 0.051]** | **+0.025 [0.009, 0.043]** | **+0.023 [0.010, 0.036]** | **+0.109 [0.022, 0.202]** | +0.195 [−0.032, +0.475] | +0.095 [−0.056, +0.255] |
| A − C reference | — | — | — | +0.042 [−0.038, +0.126] | +0.195 [−0.036, +0.475] | +0.095 [−0.056, +0.255] |
| C reference − C legacy (= B − C legacy, person) | +0.058 [0.046, 0.071] | +0.056 [0.044, 0.071] | +0.031 [0.024, 0.038] | (M5: +0.067) | 0 | 0 |

**Reading.**
- **Person:** the pretrained weights with the reference decoder (C-ref, and therefore B) are best. A
  trails them by 0.031 R@P0.80 overall (significant, as M8.1 found under DAMO's pipeline: 0.033), but on
  **large people, the driver case, by only 0.008 [−0.003, 0.020]**: smaller than M8.1's 0.019 (which had
  no interval) and not significant. A beats what ships today (C-legacy) on every person metric.
- **Phone:** A/B phone AP50 is 0.109 above what ships (significant) and 0.042 above C-ref (not
  significant: 128 phones). Phone R@P0.80 is +0.195 for A/B (p = 0.057 against either C). At equal
  precision A raises **more** verified phone-free images (14/42 vs 10/42; +0.095 [−0.056, 0.255], not
  significant): the phone-free set is 42 images, so this is undecided either way. The hand gate (M8.1 §3)
  is the lever for false phones, and it now runs on device-decoded hands (Task 1).
- **Hand:** only A/B have it: 0.723 with the reference decoder; 0.619 with legacy.

## 4. Task 3(c): cost

### Desktop latency (job `b69aa0`, oryn; **NOT a Raspberry Pi, NOT Hailo-8L**)

Head graphs only (zoo cut `4b0550f7…f84` at the zoo's own end nodes; EXP-08 `heads81` `a0327a25…aa1`),
batch 1, 640×640, ONNX Runtime 1.26.0, 20 warm-up + 200 timed runs, p50 (p90) ms.

| Provider | Shipping (A/C network) | EXP-08 | B: both back to back |
|---|---|---|---|
| CPU, 1 thread | 138.8 (143.5) | 136.7 (140.3) | 276.9 (287.3) |
| CPU, 4 threads | 48.7 (60.4) | 44.5 (55.8) | 133.7 (168.7)* |
| CUDA, RTX 4060 | 6.28 (6.95) | 6.26 (6.89) | 12.62 (13.37) |

\* Two 4-thread sessions alternating in one process cost more than the sum of the two; most likely ORT
thread-pool spinning between sessions (not investigated). `nvidia-smi` showed 13% GPU utilisation before
the job (another user may have been active). The two networks have identical compute (same 77 Conv
kernels, M6.5/M8.4), so per-network cost is the same; B is 2× the network work by construction.

Host decode per detector frame at the product threshold 0.20 (x86 CPU, job `625fa0`, 300 images; **not
a Pi**): shipping legacy 1.04 ms p50 (1.87 p90), shipping reference 1.95 (2.88), EXP-08 legacy 1.80
(2.46), EXP-08 reference 2.04 (2.66). So A costs ~2.0 ms and B ~4.0 ms of x86 decode per frame vs 1.0 ms
today; B could decode person-only on the first network (`classes=[0]`), which is cheaper, not measured.
The Pi 5's Cortex-A76 cost is **unmeasured**; `scripts/hw_validation/stage_timing.py` measures it.

### Hailo-8L fit (desk analysis; nothing here is measured on Hailo)

- **Operators.** Op histogram of `heads81` vs the zoo graph the shipping `.hef` was compiled from
  (contract job `9ffec6`): identical Conv 77 / Relu 69 / Add 20 / Concat 12 / MaxPool 3 / Resize 2 /
  Mul 5 / Sigmoid 5; the export adds `Cast` 3 and `Constant` 28 and expresses the Focus-stem slicing with
  6 instead of 8 `Slice`. Every compute op has already been compiled by Hailo for Hailo-8L (the shipping
  `.hef`); the extra `Cast`/`Constant` nodes are a **parser risk** to constant-fold first
  (`research/m10/hailo_compile/README.md`). Output layout is the shipping one (six heads, 68/81 channels).
- **Quantisation risk specific to EXP-08.** Box-head logits on the six known-good frames span −20.4…33.0
  vs −6.4…11.6 for the shipping weights (~3× the range for the same 8-bit output). Class probabilities
  are in [0, 1] for both. Untested at L2/L3 for EXP-08.
- **Two networks on one Hailo-8L (B).** HailoRT can configure several network groups on one device and
  time-share them (its scheduler), or the DFC can compile two models into one `.hef`; the switch cost on
  Hailo-8L for two DAMO-T-sized networks is **unknown** and is exactly what the Pi must measure. The
  **product cannot run B today**: `core/accel.py` opens one `VDevice` per `HailoYOLO`, caches one model
  per `.hef`, and the chip is exclusive per process ("a second VDevice in the same process fails"), so a
  second `.hef` would be refused. B needs a shared-VDevice/two-network-group change in `core/accel.py`
  (not made). Single-network reference points: Pi 5 + Hailo-8L, shipping `.hef`, 22.07 ms mean inference
  and 29.5–29.9 fps pipeline (`docs/PHASE5_DEMO_READINESS.md` §4); zoo-published 124 fps on an x86 host.
  B's naive sum (2 × 22.07 ms, no switching cost) is arithmetic, not a measurement.
- **Schedule lever.** `driver_attention` already runs the detector on 1 of every 3 frames
  (`yolo_every_n_frames: 3`). B could run the 3-class network less often than the person network, at
  a recall cost on brief phone use that would have to be measured on in-cabin clips.

## 5. Task 2: export / deployment contract (`research/m10/export_contract.py`)

Jobs `ff8c48` (with the swap-R/B pass) and `9ffec6`. Product side read from source; model side from the
ONNX graphs and the DAMO tree at `319572e`.

| Item | Device (`core/camera.py`, `core/accel.py`, `config.yaml`) | Model side | Verdict |
|---|---|---|---|
| Normalisation | none: raw 0..255 uint8 to the chip | DAMO `image_mean [0,0,0]`, `image_std [1,1,1]` (`damo/config/augmentations.py:25-26`); no arithmetic between graph input and first Conv in any of the four graphs (only the Focus-stem `Slice`/`Concat`) | **match** |
| Colour | camera delivers BGR (picamera2 `RGB888` = B,G,R bytes; device-verified 0.998 corr); `predict` converts BGR→RGB | not established from source (search of the DAMO tree found no decode site); **empirical, full val, device path (EXP-08):** RGB person/phone/hand AP50 0.697/0.837/0.723 vs R/B swapped 0.664/0.814/0.556; shipping model (M8.3): 0.660 vs 0.635 | **match** (RGB) |
| Resize | letterbox 640×480 → 640×640: scale 1.0, bands top/bottom 80 px, pad 114, `cv2.INTER_LINEAR` | DAMO trains with stretch (`keep_ratio: False`, `image_max_range (640, 640)`) | **mismatch, measured harmless**: EXP-08 hand +0.0007 [−0.010, 0.012], person/phone letterbox ≥ stretch (§2); shipping model M5 |
| Tensor layout | `.hef` input NHWC uint8 (1,640,640,3); outputs FLOAT32 NHWC | ONNX NCHW float32 (1,3,640,640) | match (the DFC converts; raw 0..255 makes uint8 ≡ float) |
| Outputs | 6 heads, paired by grid and channel count (68 = box), strides 8/16/32 | `heads81` and zoo cut: 3 × (68-ch `Mul` + 81-ch `Sigmoid`) at 80/40/20 | match. The native EXP-08 ONNX ends its box heads in `Cast` and its class heads have 3 channels: compile `heads81`, not the native graph |
| DFL | `_DFL_BINS` 17 → reg_max 16; softmax expectation × stride | DAMO `reg_max: 16` | match |
| Class map | 0..79; 80+ dropped unless `yolo_extra_classes` | zoo: 81 live rows, ch. 80 = background slot; `heads81`: live rows exactly [0, 67, 80], dead rows bias −20 | zoo match; EXP-08 **mismatch unless the opt-in map is set** (Task 1) |
| NMS | class-wise IoU 0.50, max 500 (reference) | DAMO `nms_iou_thre 0.7`, `nms_conf_thre 0.05` | deliberate difference (0.50 measured better, `DECODER_CANDIDATE.md`) |
| Thresholds | `driver_attention.yolo_confidence_threshold 0.20`, `person_confidence_threshold 0.50` | precision-0.80 thresholds (float, §3): shipping legacy person 0.378 / phone 0.609; EXP-08 reference person 0.295 / phone 0.440 / hand 0.344 | **mismatch for any non-shipping model**: product thresholds are raw-score, model- and decoder-specific, and INT8 shifts scores (M8.4 §5.5) |
| Calibration | — | spec: ≥ 1,024 in-cabin NoIR frames through the device letterbox (`research/m10/hailo_compile/calib.py`, byte-identical to `predict`, test-pinned) | spec only; no in-cabin data exists |
| Hashes | `.hef` `e69136d3…960e` pinned | zoo ONNX `02d7ef7f…6657`, zoo cut `4b0550f7…f84`, zoo heads (M8.3) `de27bd7c…50ba`, EXP-08 native `f591a6b6…a609`, `heads81` `a0327a25…aa1`, ckpt `baad3842…f37b` | recorded |

No production preprocessing was changed. The one real mismatch for a candidate (thresholds) is a
re-derivation task, not a code fix; the resize mismatch is measured and harmless on this data.

**Reproducible compile process:** `research/m10/hailo_compile/` (`README.md`, `calib.py`,
`compile.py`): pinned Model Zoo v2.15.0 / hailo8l, parse → optimise (≥ 1,024 NoIR calibration frames,
device preprocessing) → compile → `parse-hef` contract check → L3 emulator parity on the known-good and
48 OI fixtures → L4 Pi capture + `compare_float_vs_int8.py`. **Not executed**: no Dataflow Compiler on the
fleet (M8.4 job `20261113-090503-dc1833`). The SDK-free parts are tested (`tests/test_m10_hailo_compile.py`).

## 6. Verdict

**Criteria** (pre-existing; restated, none invented after seeing these numbers): M8.4 EXP-09 gates
G1 person R@P0.80 within 0.010 of the pretrained model (reference decoder), G2 phone AP50 ≥ 0.79,
G3 hand AP50 ≥ 0.70; M8.1's Arch B trigger (large-person gap > 0.02 and EXP-09 failing); and the
standing rule that nothing deploys before Pi INT8 parity (runbook H1).

| Arch | G1 person (gap to pretrained) | G2 phone | G3 hand | Large-person gap | Networks | Product change needed |
|---|---|---|---|---|---|---|
| A (EXP-08, ref + map) | **fail**: 0.031 [0.014, 0.047] | pass 0.837 | pass 0.723 | 0.008 [−0.003, 0.020] | 1 | map + reference decoder + new `.hef` |
| A (EXP-09) | **fail**: 0.047 [0.030, 0.066] (M8.4) | pass | pass | 0.030 | 1 | — rejected (M8.4) |
| B | pass (0 by construction) | pass 0.837 | pass 0.723 | 0 | 2 | map + reference decoder + new `.hef` + two-network support in `core/accel.py` |
| C reference | pass | pass 0.795 | no hand | 0 | 1 | decoder switch only |

- **Production stays** (`legacy`, shipping `.hef`): no candidate has any L2–L4 evidence.
- **B is the deployment candidate on accuracy**, pending Pi INT8 parity: it is the only architecture
  that passes G1–G3 at equal operating conditions. This confirms M8.4's route.
- **But B is not proven necessary for the driver.** Under equal conditions A's large-person deficit is
  0.008 with an interval that stays under M8.1's 0.02 bar, so the M8.1 trigger is **not** met by this
  measurement (it was met for EXP-09, not for EXP-08). A remains the fallback if B's Pi cost fails, and
  A already beats what ships on every class. The choice between them is a Pi question, not an accuracy
  question.
- **Independent of A vs B:** the reference decoder is required for any 3-class model (hand −0.10 AP50
  with legacy), and C-reference alone is worth +0.056 person R@P0.80 over what runs, at zero network cost.

**What Pi evidence decides it** (in order):
1. H1 INT8 parity of the shipping `.hef` (both decoders): if the reference decoder's gain survives INT8,
   switch C to reference on a development unit first.
2. Compile EXP-08 `heads81` (`research/m10/hailo_compile/`), then INT8 parity on the Pi: B and A both need
   it. Fail if INT8 costs > 0.02 person R@P0.80 or > 0.03 hand AP50 vs float on the same pipeline.
3. Two-network throughput on Pi 5 + Hailo-8L after the shared-VDevice change: B is acceptable if the
   driver_attention pipeline keeps ≥ 29.5 fps (PHASE5 baseline) with the detector at its current cadence
   (every 3rd frame), or at a measured reduced 3-class cadence whose phone recall loss on in-cabin clips is
   accepted. If not, deploy A, provided its INT8 large-person R@P0.80 is within 0.02 of the INT8
   shipping model's.
4. In-cabin NoIR validation (G1–G6) before any threshold is final; thresholds re-derived on INT8 scores.

## 7. Open items

- Compile and every L3/L4 step (needs the DFC, owner approval, a Pi session).
- `compare_float_vs_int8.py` / `hwv_decode.py` decode `NUM_CLASSES = 80`: tensor parity covers channel 80,
  detection-level hand agreement needs a class-count option.
- `capture_hailo_outputs.py` constructs `HailoYOLO` without the class map: captured tensors are complete,
  but its production-decoder boxes omit hand.
- Two-network support in `core/accel.py` (shared VDevice / scheduler) does not exist; B cannot run on the
  device until it does.
- EXP-08 host-simulated INT8 (L2, `research/m84/hostquant.py`) was not run; the wider box-logit range
  makes it worth running before the DFC.
- Phone and false-phone comparisons are underpowered (128 phones, 42 phone-free images).
- Colour order of DAMO training is established empirically (swap test), not from source.
