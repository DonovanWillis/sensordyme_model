# M7 detector analysis: the EXP-08 person regression, and architectures A vs B

Mission 7 Checkpoint G. **No training was run.**
- Open Images val split only (`b878e8429d221f3f`); the test split is untouched.
- The EXP-08 checkpoint `baad3842…` is unchanged.
- Tools: `research/pilot/person_regression.py` (operating points, failure breakdown, `merge`, `renms`), `m7_detector_analysis.sh`, `m7_nms_check.sh`.
- oryn jobs: `20261111-153317-2c1ead` (epoch sweep) and `20261111-154932-148be2` (NMS-matched comparison and architecture B).
- Evidence: `~/sensordyme-audit/2026-11-10/m7-evidence/detector/`.

## 1. The 3-class model finds the same people but adds about twice the false positives

Person class, both models at NMS IoU 0.5:

| | Pretrained, reference decoder | EXP-08 full60 |
|---|---|---|
| AP50 / AP50:95 | **0.713** / **0.544** | 0.686 / 0.470 |
| Precision at recall 0.6 / 0.7 / 0.8 | **0.89** / **0.695** / not reached | 0.84 / 0.55 / 0.12 |
| Recall at precision 0.7 / 0.8 | **0.698** / **0.661** | 0.668 / 0.627 |
| Missed people (of 4,008) | 1,117 | 1,122 |
| False positives at recall 0.70 | 1,230 | **2,312** |
| ↳ duplicates of a found person | 65 | 118 |
| ↳ misplaced (IoU 0.1–0.5 with a person) | 560 | 1,089 |
| ↳ background | 605 | 1,105 |
| False positives on verified person-free images | 32 | 68 |
| Background false positives on a hand box | 3.3% | 1.3% |
| ECE | 0.099 | 0.081 |

- **Recall is not the problem.** Misses are almost identical, and each model finds about 340 people the other misses, with similar size, occlusion and truncation profiles.
- **The extra false positives are spread across types.** Misplaced and background errors both roughly double, which points to worse ranking and box quality overall rather than one failure mode.
- **It is not person/hand confusion.** Only 1–3% of background false positives sit on a hand, and 2% of misses overlap one.
- **Calibration is no worse.** ECE is 0.081 vs 0.099.

## 2. NMS mismatch: real, not the cause

- **The earlier comparison was not like for like.** `EXP08_FULL_RESULTS.md` scored the pretrained model through the bench's NMS at IoU 0.5, and EXP-08 through DAMO's postprocess at IoU 0.7. Duplicates were 798 of 2,241 false positives.
- **Re-suppressing at 0.5** (`renms`) cut duplicates to 118 but left precision at recall 0.70 unchanged (0.556 → 0.548).
- **The pretrained model behaves the same:** person AP50 is 0.713 at NMS 0.5 and 0.714 at 0.7.
- **Side result: NMS 0.5 is better for the 3-class model on phone and hand** (phone AP50 0.798 → 0.807, hand 0.706 → 0.725, person 0.690 → 0.686).
- **Caveat:** `renms` re-suppresses already-suppressed output, so it only approximates NMS 0.5 on raw outputs.

## 3. Most likely cause: fine-tuning reinitialised the whole head

- **What the code does.** EXP-08 starts from `damoyolo_tinynasL20_T_418.pth` ("Finetune from …" in the training log). DAMO-YOLO's loader (`damo/detectors/detector.py:46`) copies every tensor except those whose name contains `head`.
- **What that means here.** In DAMO-YOLO-T the head is two layers, `gfl_cls` (class scores) and `gfl_reg` (box distributions). So the box regressor was retrained from scratch on 4,883 images, even though its shape doesn't depend on the class count. The backbone and neck kept their pretrained weights.

The per-epoch person curve fits this and rules out forgetting (a forgetting curve would peak early and then fall):

| Epoch | 10 | 20 | 30 | 40 | 50 | 55 | 58 | 60 |
|---|---|---|---|---|---|---|---|---|
| Person AP50 | 0.610 | 0.666 | 0.678 | 0.680 | 0.678 | 0.679 | 0.688 | 0.690 |
| Person AP50:95 | 0.351 | 0.410 | 0.424 | 0.431 | 0.433 | 0.433 | 0.472 | 0.489 |
| Person precision at recall 0.70 | 0.31 | 0.46 | 0.50 | 0.51 | 0.52 | 0.53 | 0.55 | 0.56 |

- **Still improving at the end.** Box quality (AP50:95) and precision are still rising at epoch 60.
- **The late jump follows augmentation switching off.** The rise after epoch 55 matches the 5 no-augmentation epochs at minimum learning rate.
- **The largest remaining gap measures box fit.** That gap is AP50:95 (0.47 vs 0.544), which points at the box regressor.
- **Not proven.** This is the explanation most consistent with the evidence, not a proof. Confirming it needs the training run in §6.

## 4. Checkpoint selection

Epoch 60 is the best epoch for person on every metric. An earlier checkpoint would not have avoided the regression.

## 5. Architecture A (one 3-class detector) vs B (pretrained person plus 3-class phone and hand)

| | A: EXP-08, NMS 0.5 | B: pretrained person + EXP-08 phone/hand |
|---|---|---|
| Person AP50 (bootstrap 95% CI) | 0.686 (0.651–0.716) | **0.713** (0.679–0.748) |
| Person precision at recall 0.70 | 0.55 | **0.695** |
| Phone AP50 / recall at precision 0.8 | 0.807 / **0.789** | same as A |
| Hand AP50 / precision at recall 0.70 | 0.725 / 0.704 | same as A |
| Networks per frame | 1 | 2 |
| GPU time per frame (oryn, PyTorch p50) | ~11.7 ms | ~2× |

- **B must take phone from EXP-08, not the pretrained model.** Pretrained phone recall at precision 0.8 is only 0.594, against 0.789 for EXP-08.
- **The AP difference is not settled.** The person AP intervals overlap and no paired test was run.
- **The precision difference is large but unquantified.** The gap at the operating point (0.55 vs 0.695) is large, but no interval has been computed for it.

Product cost of B:

| Factor | A | B |
|---|---|---|
| Hailo `.hef` files | 1 | 2 (the shipping COCO `.hef` plus a new 3-class one) |
| Accelerator time | 1× | about 2×, plus context switching between two networks on the Hailo-8L (not measured) |
| Host memory | one model | two models (about 8.2 M parameters and 17 GFLOPs each at 640) |
| Pipeline | one decode path | two decode configurations; the shared perception engine already supports per-component scheduling |
| Lower-cost variant | — | run the 3-class model only every N frames, or only when a phone or hand is plausible |

Unknown until the hardware session: Hailo-8L FPS for one or two DAMO-T networks, the INT8 accuracy of either, and the Pi 5 host cost.

## 6. Recommendation and proposed experiment (needs approval; not run)

1. **No-cost change, research checkpoints only.** Score the 3-class model at NMS 0.5. Production decoding and the `.hef` are unchanged.
2. **Before paying for B's second network, run one head-preserving fine-tune, EXP-09.**

**EXP-09 head-preserving fine-tune** (one run; needs your approval):
- **Change.** Same data, seed and schedule as `exp08_full60` (max 60 epochs, patience 10, warmup 2, no-aug 5, batch 8, 640). Only the start changes:
  - load `gfl_reg` (box distributions; class-independent) from the pretrained checkpoint;
  - start the `gfl_cls` person and phone rows from COCO "person" (0) and "cell phone" (67);
  - initialise the hand row as DAMO-YOLO does today.
- **Implementation.** A small exact-match patch to the loader, in the style of `research/pilot/early_stop_patch.py`, plus a sanity check that the copied rows equal the pretrained tensors before training. Score at NMS 0.5 on val only.
- **Cost.** About 3.1 h on oryn's GPU (the EXP-08 run took 11,198 s); about 5 GB card memory. Run only while the owner's GPU is free.
- **Success (all must hold, val):**
  - person precision at recall 0.70 ≥ 0.66;
  - person AP50 ≥ 0.703;
  - hand AP50 ≥ 0.70;
  - phone recall at precision 0.8 ≥ 0.75;
  - false phones on phone-free images ≤ the pretrained model's.
- **Decision.**
  - If it succeeds, A (one network) is preferred.
  - If person precision stays below about 0.62, adopt B, pending Hailo throughput measurements on Monday's hardware.
- **Still experiment-only either way.** Neither outcome changes the production `.hef` or the decoder default; both still need INT8 and on-device evidence.
