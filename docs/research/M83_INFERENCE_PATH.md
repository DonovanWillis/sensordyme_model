# M8.3: the device inference path, validated without the Pi

**Status:** measured on the fleet (x86 CPU, float ONNX) and on the Mac (CI fixtures).
**Nothing here ran on a Raspberry Pi or a Hailo-8L.** No INT8 number, Hailo latency,
accelerator utilisation or thermal figure is reported. Those are Monday measurements
(`docs/MONDAY_HARDWARE_SESSION.md`).

## What was exercised

The product's own `core/accel.py` `HailoYOLO`, unmodified, with only the chip replaced:

| Layer | What ran | Source |
|---|---|---|
| HEF parsing, head pairing | `HailoYOLO.__init__` / `_group_raw_outputs` against a stand-in `hailo_platform` whose `.hef` has the `parse-hef` layout: six NHWC float32 outputs, grids 80/40/20, a 68-channel box head and an 81-channel class head per grid, listed in a non-finest-first order | `scripts/hw_validation/fake_hailo.py` |
| Preprocessing | `predict()`: letterbox (scale, centred 114 bars), BGR→RGB, UINT8 NHWC `(1, 640, 640, 3)` | product code |
| "Chip" | float ONNX of the shipping model (Hailo Model Zoo `damoyolo_tinynasL20_T`, sha256 `02d7ef7f…6657`), heads cut at `Mul_224 … Sigmoid_324`, fed the exact uint8 tensor `predict()` built, outputs transposed NCHW→NHWC | `fake_hailo.onnx_backend` |
| Decode, NMS, class map, frame mapping | `_decode_raw` (legacy, the default) and `_decode_reference` (opt-in candidate) | product code |
| Independent reference | input: a numpy letterbox written without any `core` import; decode: `scripts/hw_validation/hwv_decode.py` (separately written numpy); on CI fixtures also a deliberately naive per-anchor loop decoder with an explicit IoU matrix | `research/m83/inference_path_val.py`, `tests/test_m83_inference_path.py` |

## Results on Open Images val (fleet job `20261113-043857-f182d1`, oryn, CPU, commit `989ecfd`)

Manifest `b878e8429d221f3f`, 1,585 val images (the frozen test split was not touched). Each image is fitted to a 640 longer side with PIL bilinear (the Mission 4/5 bench input) and handed to `predict()` as a BGR frame, so `predict()` letterboxes at scale 1.0. ONNX Runtime 1.27.0 CPU, OpenCV 4.11.0, numpy 1.26.4 (the device pins OpenCV 4.10.0.84 and numpy 1.26.4).

| Check | Result |
|---|---|
| Input tensor `predict()` sent == independent letterbox, bit for bit | **1585 / 1585** |
| Legacy decode: `predict()` detections (frame px) == `hwv_decode` LEGACY + unmap, same class, |Δ| ≤ 1e-3 | **1585 / 1585**, max |Δ| 3.05e-5 (float32 rounding) |
| Candidate (`reference`) decode == `hwv_decode` with its semantics (offset 0, multilabel, IoU 0.5, cap 500, inclusive threshold) | **1585 / 1585**, max |Δ| 3.05e-5 |
| Detections at the 0.05 scoring floor / at the product's 0.20 | 151,829 / 20,737 |
| Class head range (80 real channels) | 1.2e-7 … 0.935: sigmoid probabilities |
| Box head range | −7.16 … 17.69: raw DFL logits |
| 81st class channel | **present in the float ONNX too** (`(1, 81, G, G)`), max 0.0079 over all 1,585 images: the network's own near-zero background slot, not Hailo compiler padding. Every decoder drops it. |

Accuracy of the device path (`research/detector_bench/bench.py score`, Open Images protocol; brackets: 95 % group bootstrap):

| Variant | Person AP50 | Person AP50:95 | Phone AP50 | Phone AP50:95 |
|---|---|---|---|---|
| **device path, legacy (shipping)** | **0.660** [0.617–0.696] | 0.353 | **0.728** [0.616–0.804] | 0.452 |
| Mission 5 reference for the same weights + decoder (`docs/DECODER_CANDIDATE.md`, job `20261111-022003-26d0f7`, CUDA, research letterbox + `_decode_raw`) | 0.656 | 0.352 | 0.728 | 0.452 |
| R and B swapped at the chip (a deliberate fault) | 0.635 [0.594–0.674] | 0.336 | 0.707 [0.604–0.788] | 0.420 |
| full-resolution image straight into `predict()` (cv2 `INTER_LINEAR` downscale, no antialias) | 0.661 [0.618–0.699] | 0.353 | 0.739 [0.633–0.819] | 0.460 |

Reading:
- The full device path reproduces the Mission 5 numbers, which were measured by calling `_decode_raw` directly on research-built tensors. So the parts Mission 5 bypassed (`predict()`'s cv2 letterbox, BGR→RGB, frame mapping, HEF-shaped head pairing) change nothing measurable.
- **Channel order:** RGB scores higher than swapped on both classes, consistent with the model expecting RGB (what `predict()` sends, from the camera's BGR; `core/camera.py` documents the BGR as measured on the device). The margin is small (0.025 / 0.021 AP50) and the intervals overlap, so this alone is weak evidence; it agrees with Hailo's RGB convention and the Mission 4/5 setup.
- **Larger camera modes:** `predict()` downscaling a large frame with `INTER_LINEAR` did not lose accuracy here. At the shipping 640×480 there is no resize at all.

## Bugs found and fixed (commit `ee9f6e6`)

The fault states were driven through `predict()` on the stand-in chip. Before the fix, measured on a planted phone at frame (300, 200)–(364, 328):

| Fault in the device result | Before (legacy and reference alike) | After |
|---|---|---|
| an output missing | bare `KeyError` with the tensor name | `RuntimeError` naming the head, the grid and the outputs present |
| box head NCHW `(1, 68, 40, 40)` instead of NHWC | **a wrong box, silently**: (200, 136)–(456, 392) | `RuntimeError`: layout must be NHWC `(1, G, G, C)` |
| output without the batch axis | **no detections, silently** | `RuntimeError` |
| a class head from the wrong grid | **no detections, silently** | `RuntimeError` |
| NaN in a class score | read as "below threshold" | `RuntimeError` (non-finite) |
| inf / NaN in the box logits of a detected anchor | **NaN coordinates** returned to callers (`int(nan)` then crashed `driver_attention`) | `RuntimeError` (non-finite) |
| a `.hef` grid that does not divide 640 | loaded; stride truncated (640 // 21 = 30) | refused at load |
| two heads of one role at a grid | loaded; one silently overwrote the other | refused at load |
| class head narrower than 80 | loaded; class ids silently shifted (legacy path) | refused at load |
| a non-uint8 frame | passed to the UINT8 input vstream | `TypeError` |

An empty result remains a valid answer ("nothing there"); a fault can no longer produce one. The checks reuse values the decoder already reduces (per-anchor maxima, the kept anchors' distances; a single sum on the reference path), so they add no full pass over the 1.25 M output floats. No threshold, decoder default or decode arithmetic changed; the existing raw-decode, decoder, lifecycle and M5 tests pass unchanged.

## Confirmed as designed (not changed)

- **Class mapping:** `names[0] = "person"`, `names[67] = "cell phone"`. Every `yolo_classes` name in `config.yaml` is a COCO name, and `driver_attention`'s name→id derivation maps `cell phone` to `phone_detected`.
- **Confidence:** inclusive (`score >= conf`), on the raw-decode path the only gate there is. INT8 class scores are discrete steps, so a score can sit exactly on 0.20; inclusive is the device's semantics. DAMO-YOLO's own postprocess is strict (`>`).
- **NMS:** class-wise greedy, IoU 0.50, a box at exactly IoU 0.50 survives. A phone inside a person box is never suppressed by it.
- **Known decoder difference (documented, opt-in only):** on identical tensors the DAMO convention places every box exactly stride/2 up and left of the legacy decode, and multilabel keeps a phone that loses its anchor's argmax. `docs/DECODER_CANDIDATE.md`; the default stays `legacy`.

## Host CPU baselines (labelled; never a Pi or Hailo number)

| Stage | fleet x86, OI val person photos, 400 frames (`20261113-050231-7108be`) | Mac M-series, known-good images, 300 frames |
|---|---|---|
| FaceMesh, every frame | p50 3.03 / p95 7.33 ms | p50 4.57 / p95 12.65 ms |
| Hands, detector frames | p50 32.4 / p95 53.2 ms | p50 37.5 / p95 88.6 ms |
| `predict()` preprocess | p50 0.35 / p95 0.48 ms | p50 0.41 / p95 1.03 ms |
| decode (legacy, conf 0.20) incl. NMS | p50 1.77 / p95 2.97 ms | p50 1.91 / p95 5.00 ms (planted heads) |
| module logic (rules, trackers, fusion) | p50 0.19 / p95 1.27 ms | p50 0.29 / p95 1.39 ms |
| "infer" (stand-in) | 64.8 ms = x86 float ONNX, **not Hailo** | ~0.7 ms = planted, **not Hailo** |

On all 1,585 val images (`20261113-043857-f182d1`): legacy decode at 0.20 p50 1.66 / p95 3.19 ms, reference decode p50 2.73 / p95 4.28 ms, preprocess p50 0.37 ms.

What transfers: the ordering. MediaPipe Hands on detector frames is the largest host cost, then FaceMesh; decode and SensorDyme's own logic are small. The Pi 5's Cortex-A76 is slower than either host, so the Pi numbers must be measured (`hwv_stage.py latency`, runbook H2, M5 step 7).

## Reproduce

```bash
# CI (Mac): fixtures only, no model files
~/.venvs/sensordyme-m01/bin/python -m pytest -p no:cacheprovider -q tests/test_m83_inference_path.py tests/test_m83_hw_package.py
# fleet (oryn, CPU): the commit must be committed; data stays on the worker
./sdrun submit -t oryn --repo <repo> --include core --include research --include evaluation \
  --include scripts/hw_validation --name m83-infpath-val --no-gpu --cpus 4 --mem 4G --timeout 3600 -- \
  bash -c 'D=$SDJOB_DATA; $D/venvs/sensordyme-a779b4d/bin/python -u -m research.m83.inference_path_val \
    --data $D/openimages/oi7-sd-v1 --onnx $D/models/damoyolo_tinynasL20_T.onnx \
    --heads-onnx $D/models/damoyolo_tinynasL20_T.heads.onnx --preds $D/m83_inference_path/$SDJOB_ID --out $SDJOB_OUT'
```

Evidence (small files): `~/sensordyme-audit/2026-11-10/m83-evidence/hardware/`.
