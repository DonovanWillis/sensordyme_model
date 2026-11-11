# DAMO-YOLO decoder candidate (`accelerator.yolo_decoder: reference`)

**Status: CANDIDATE.**
- Implemented and selectable; the default stays `legacy`.
- Verified on float ONNX against Open Images val.
- **Not yet validated on the INT8 `.hef` on a Pi.** That comes through the hardware package on branch `feat/m5-hw-validation`.
- Do not switch a fielded device until that passes.

## The two defects (reproduced in `tests/test_hailo_decoder.py`)

| Defect | Legacy (shipped) | Reference (DAMO-YOLO and the Hailo Model Zoo) |
|---|---|---|
| Anchor point | `(cell + 0.5) * stride`: every box is shifted right and down by stride/2 (4, 8 or 16 px at 640) | `cell * stride` (`zero_head.py` `get_single_level_center_priors`; zoo `nanodet_split` offset 0.0) |
| Class selection | Only each anchor's argmax class | Every (anchor, class) at or above `conf` (`boxes.py` `multiclass_nms`) |

**Unchanged in the candidate:**
- tensor pairing by grid shape;
- 80 class channels, with channel 81 dropped as padding;
- DFL softmax expectation × stride;
- letterbox inverse and clipping;
- class-aware NMS at the configured IoU (0.50).

The reference keeps up to 500 boxes. The tests also check the candidate against an independent NumPy transcription of DAMO-YOLO's decode, on random heads under three letterbox geometries.

## Measured effect

**Setup:** Open Images val; manifest `b878e8429d221f3f`; 1,585 images; float ONNX (Hailo Model Zoo 2022-12-19, the source of the shipping `.hef`); CUDA; job `20261111-022003-26d0f7`.
- The decoder is the **product's own code** (`core/accel.py` at `998d3b4`, overlaid inside the job and checked by sha256).
- It produced metrics identical to the independent research decode.
- Brackets are 95% intervals from a bootstrap over photographer groups.

| | Phone AP50 | Phone AP50:95 | Phone mean TP IoU | Person AP50 | Person AP50:95 | Person mean TP IoU |
|---|---|---|---|---|---|---|
| legacy | 0.728 [0.62–0.80] | 0.452 | 0.817 | 0.656 [0.61–0.69] | 0.352 | 0.777 |
| **reference** | **0.795** [0.71–0.87] | **0.707** | **0.930** | **0.713** [0.68–0.75] | **0.544** | **0.897** |
| reference, stretch preprocessing | 0.779 | 0.689 | 0.925 | 0.708 | 0.530 | 0.896 |
| reference, top-left letterbox, pad 0 | 0.797 | 0.711 | 0.934 | 0.713 | 0.543 | 0.897 |
| reference, NMS IoU 0.7 | 0.786 | 0.707 | 0.934 | 0.714 | 0.557 | 0.897 |

**Conclusions:**
- The anchor fix is mostly **localisation**: mean TP IoU 0.82 → 0.93 for phones and 0.78 → 0.90 for people. Box tightness is what zone, dwell and hand–phone overlap rules depend on.
- Keep the production letterbox. DAMO's own stretch, which it was trained with, is not better here.
- Keep NMS IoU 0.50. At 0.7 phone precision at 0.20 falls from 0.527 to 0.481.

## Why multilabel adds phone boxes on phone-free images

Measured on the 42 verified phone-free val images, at scores ≥ 0.05:
- 172 phone boxes; 68 of them exist only because of multilabel (phone was not the anchor's top class).
- The anchors' top classes were phone-like rectangular objects: remote 13, clock 9, tv 8, laptop 5, keyboard 5, book 4, scissors 4, and others.
- **59 of the 68 score below 0.20, 9 score 0.20 or above, and 1 scores 0.40 or above.**

So multilabel's extra false boxes are low-confidence look-alikes. At an operating threshold of 0.4 or more they almost vanish (phone sweeps for legacy and reference are identical from 0.5 up). What multilabel adds where it matters is recall at low thresholds (phone recall at 0.20: 0.805 → 0.914).

## Threshold trade-off (val only; the test split is untouched)

Reference decoder, DAMO-T. "Phone-free images with a false phone" counts out of 42 verified phone-free images, so each point is about 2.4% and the intervals are wide.

| Phone threshold | Recall | Precision | Phone-free images with a false phone |
|---|---|---|---|
| 0.20 (shipped) | 0.91 | 0.53 | 67% |
| 0.30 | 0.84 | 0.60 | 55% |
| 0.40 | 0.75 | 0.69 | 48% |
| 0.50 | 0.70 | 0.77 | 31% |
| 0.60 | 0.59 | 0.79 | 24% |
| 0.70 | 0.43 | 0.83 | 12% |

No single threshold gives both high recall and few false phones on these web photos. **A phone box is not phone use.**
- **Hand gate:** the hand-gate study (`research/detector_bench/interaction.py`) shows human-labelled hand overlap would cut false phones on phone-free images from 56 to 6 at 0.20 while keeping every in-hand phone.
- **MediaPipe can't supply it:** MediaPipe Hands found a hand in only 8 of 152 of these photos, which is why a learned `hand` class is being piloted (EXP-08a).
- **Thresholds and persistence need our own footage:** final thresholds, persistence (k-of-n frames) and gaze consistency must be chosen on in-cabin validation video, not on still web photos.

## Compatibility with the Hailo `.hef` (what is known, and what is not)

**Known:**
- The candidate reads the **same six tensors** as the legacy path, paired by shape.
- The shipping `.hef` is compiled from exactly these ONNX end nodes (`Mul_224`, `Sigmoid_244`, … in the zoo's `damoyolo_tinynasL20_T.yaml`).
- The `HailoYOLO` notes record, from the real `.hef`, 68-channel raw logits and 81-channel sigmoid probabilities.
- Hailo's own evaluation of this network decodes with offset 0.

**Not known:** INT8 numeric parity, meaning whether quantisation error moves scores near the thresholds or changes the box DFL argmax.
- Float parity does not imply INT8 parity.
- Run `scripts/hw_validation/capture_hailo_outputs.py` on a Pi, then `compare_float_vs_int8.py` on a worker, for both decoders on identical inputs.

**Pi CPU cost:** host decode measured 2.6 ms (legacy) vs 5.9 ms (reference) on the worker CPU at a 0.05 floor. The device decodes at the module threshold (0.20+), so fewer candidates, but the Pi CPU is slower. Measure it with `stage_timing.py`.

## To promote (all required)

1. INT8 parity and agreement on the Pi (hardware package §capture/compare) within the stated tolerances.
2. Measured Pi decode time, and processed fps no worse than with legacy.
3. Event-level replay on in-cabin validation clips: phone-use and person-dependent alerts no worse on false alarms per hour.
4. Owner approval, then set `accelerator.yolo_decoder: reference` on a **development** unit first.
