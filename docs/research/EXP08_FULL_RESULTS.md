# EXP-08 full run: DAMO-YOLO-T, person / phone / hand, 60 epochs

The one long run authorised for Mission 6.5. **Experiment only**: it replaces nothing, the
production `.hef` and the decoder setting are unchanged, and the registry records it as
`experiment_only` (`hand_box_detection.exp08_full60`).

## Run

| Item | Value |
|---|---|
| Job | oryn `20261111-055506-564ff3`, source `536d786`, `research/pilot/exp08.sh` |
| Settings | `EXP=exp08_full60` stock labels (weight 1), seed 1234, max 60 epochs, warmup 2, no-aug (min LR, mosaic off) 5, batch 8, 640 |
| Early stopping | patience 10 val evaluations on COCO AP50:95; budget 20,700 s inside a 21,600 s job timeout |
| Base weights | DAMO-YOLO-T 41.8 (Drive mirror), sha256 `a5ee6382…` verified, loaded `weights_only=True` |
| Data | Open Images subset `oi7-sd-v1`, manifest `b878e8429d221f3f`: train 4,883 images, val 1,585. The test split was not exported or used |
| Duration | 05:55–09:01 EST, 11,198 s wall, 60/60 epochs (~3.0 min/epoch including val) |
| Stop | completed the schedule; no early stop (patience peaked at 9/10, epoch 45) |
| Best checkpoint | epoch 60, `best_ckpt.pth` sha256 `baad3842943b6ebc4cd58ecb6eff0a6615e9238d92d115cee50ddca94f4ef37b` (68.95 MB, on oryn under `$SDJOB_DATA/exp08_full60/workdirs/`) |
| Latest checkpoint | `85b09e88a0621bc7…` (same epoch; written separately at the end) |
| Resources | iteration 0.255 s median; PyTorch memory 2.3 GB; card memory peak 4.9 GB (the owner's desktop holds 1.3 GB); GPU utilisation mean 58%, max 100%; GPU temperature max 64 °C; host memory peak 5.7 GB of the 8 GB cap |
| Owner's GPU | not in use at launch or during the run (card memory never exceeded the run's own 4.9 GB) |

Loss fell from 2.5 to 0.6. Val AP50:95 (DAMO's COCO evaluation, 3 classes) rose from 0.166
(epoch 1) to 0.453 by epoch 32, plateaued at 0.453–0.459 for epochs 32–55, then rose in the
no-aug phase: 0.468, 0.480, 0.492, 0.500, **0.505** at epoch 60, still rising. The
per-epoch log is `sd_epochs.jsonl` in the job artifacts.

## Val results (Open Images protocol, `research/detector_bench`)

`research/pilot/compare_checkpoints.py`, val split only.
- **Pretrained-corrected:** the shipping model's float ONNX with the reference decoder (job `022003-26d0f7`).
- **Pretrained-legacy:** the same ONNX with the shipping decoder.
- **2-epoch:** pilot `exp08a_rerun`.

| | Pretrained-corrected | Pretrained-legacy | 2-epoch | **60-epoch best** |
|---|---|---|---|---|
| Phone AP50 (95% CI) | 0.795 (0.705–0.869) | 0.728 | 0.721 | **0.798** (0.728–0.872) |
| Phone AP50:95 | 0.707 | 0.452 | 0.464 | 0.688 |
| Phone best precision at recall ≥ 0.70 | 0.686 | 0.684 | 0.556 | **0.771** |
| Phone P / R @0.5 | 0.767 / 0.695 | 0.767 / 0.695 | 0.721 / 0.688 | 0.837 / 0.680 |
| Phone-free images with a false phone @0.5 (of 42) | 0.31 | 0.31 | 0.33 | **0.26** |
| Phone-free images with a false phone @0.2 | 0.67 | 0.60 | 0.79 | 0.62 |
| Person AP50 (95% CI) | **0.713** (0.679–0.748) | 0.656 | 0.638 | 0.690 (0.655–0.721) |
| Person AP50:95 | 0.544 | 0.352 | 0.363 | 0.489 |
| Person best precision at recall ≥ 0.70 | **0.568** | 0.254 | 0.336 | 0.430 |
| Person-free images with a false person @0.3 (of 40) | 0.25 | 0.25 | 0.60 | 0.33 |
| Hand AP50 (95% CI) | — | — | 0.471 | **0.706** (0.680–0.736) |
| Hand AP50:95 | — | — | 0.218 | 0.425 |
| Hand P / R @0.3 | — | — | 0.481 / 0.507 | 0.650 / 0.690 |
| Hand-free images with a false hand @0.3 (of 61) | — | — | 0.30 | 0.20 |
| ECE, phone (all / score ≥ 0.3) | 0.069 / 0.074 | 0.069 / 0.087 | 0.091 / 0.206 | 0.083 / 0.119 |

The pretrained rows were scored from the float ONNX at native resolution through the
product's decoder. The trained models were scored from PyTorch at 640 through DAMO-YOLO's
own post-processing. Same images, same ground truth and the same scorer, but not the same
inference path: differences of a point or two are within that and within the CIs.

## Against the pre-registered criteria (`EXP08_RESULTS.md`)

| Criterion | Result | Met |
|---|---|---|
| Hand AP50 ≥ 0.60 | 0.706 | yes |
| Phone precision at recall 0.70 ≥ the decoder-fixed pretrained model's (0.686) | 0.771 | yes (7-point threshold sweep; 128 phones) |
| Person AP50 within 2 points of 0.713 (≥ 0.693) | 0.690 | **no** (by 0.003; CIs overlap) |
| False-phone rate on phone-free images not above the pretrained model's | 0.26 vs 0.31 @0.5 | yes |

**Verdict: does not meet its success criteria** (person preservation missed narrowly; person
precision at high recall is clearly lower: 0.430 vs 0.568). It is the first model with a
usable hand detector (AP50 0.706 vs 0.471 at 2 epochs) and matches the pretrained model on
phones. The evidence power limits from the pilots still apply (42 phone-free images, 128
phones).

## Hailo compatibility and cost (job `20261111-090242-3ab280`, `research/pilot/export_compare.py`)

- **Export.** `best_ckpt.pth` was exported the way the zoo graph behind the shipping `.hef`
  was: RepConv fused, raw heads, opset 11, 640×640. Export sha256 `f591a6b6eca94baf…`.
- **Structure** (job `20261111-090525-b99173`). The subgraph feeding the six head outputs
  matches the zoo `.heads` graph: 77 Conv with identical kernel shapes (26 1×1, 51 3×3); the
  same Relu, Add, Concat, Mul, Sigmoid, Resize and MaxPool counts. The differences are
  exporter artefacts: 28 Constant, 3 Cast, and Slice 6 vs 8. Head outputs are class maps
  `[1,3,H,W]` (80 in the zoo) and box maps `[1,68,H,W]` at strides 8/16/32.
- **This is not a Hailo compile.** No Hailo Dataflow Compiler is available, INT8
  quantisation accuracy is unmeasured, and the product decoder would need a 3-class
  configuration. Nothing was compiled and the `.hef` was not touched.
- **Size.** 8.17 M parameters, 17.1 GFLOPs at 640, the same as DAMO-YOLO-T apart from the
  head's class channels.
- **Latency** (onnxruntime, batch 1, 640, 4 threads; the zoo graph includes host-side decode).

  | Backend | New p50 / p90 | Zoo p50 / p90 |
  |---|---|---|
  | CUDA, RTX 4060 | 5.8 / 6.4 ms | 6.6 / 7.2 ms |
  | CPU, oryn | 47.8 / 54.5 ms | 56.3 / 64.6 ms |

  Neither figure is the Pi or Hailo-8L.

## Limitations

- **Single seed, single run.** The 2-epoch pilots showed run-to-run phone AP noise of about
  ±0.08 at 2 epochs; at 60 epochs it is unmeasured.
- **Still rising at the end.** Val AP was still rising at epoch 60. A longer no-aug phase
  might help, but that would be another long run, which needs separate approval.
- **Person loss.** The person drop is consistent with fine-tuning on 4,883 images; the
  COCO-trained head saw far more people.
- **Not production.** The evidence stays image-level. No video, device or Hailo evidence
  exists for this model. It is not a production candidate.
