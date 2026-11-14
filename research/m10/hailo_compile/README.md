# Reproducible Hailo-8L compile of a SensorDyme candidate detector

**Status: written, never executed.** The fleet has no Hailo Dataflow Compiler (DFC), no
`hailo_sdk_client`, no Model Zoo and no `hailo_platform` (M8.4 tooling probe, oryn job
`20261113-090503-dc1833`). Installing the proprietary DFC needs owner approval and a Hailo Developer
Zone account. Nothing here produces or claims an L3 (compiled `.hef`) or L4 (Pi runtime) number.

The production `.hef` (`models/damoyolo_tinynasL20_T.hef`, sha256
`e69136d3ebf3cc8ebaf3da5417a59962b139f3d3d67d715e9a9b1ae10963960e`, Model Zoo v2.15.0 hailo8l) is not
an input or an output of this process. `compile.py` refuses any output named like it, any output under
the repo's `models/`, and any existing file with its hash.

## What gets compiled

The EXP-08 device-layout export (`research/m84/device_export.py`, oryn job `20261113-090349-6d22e9`):

| Input | sha256 | Where |
|---|---|---|
| EXP-08 best_ckpt (60 ep, research checkpoint) | `baad3842943b6ebc4cd58ecb6eff0a6615e9238d92d115cee50ddca94f4ef37b` | oryn `$SDJOB_DATA/exp08_full60/workdirs/exp08_full60/best_ckpt.pth` |
| native ONNX (opset 11, RepConv fused) | `f591a6b6eca94bafc8f71f8cc1026b0b0ec06fa37c0424f465befa6c6040a609` | oryn `$SDJOB_DATA/m84/export_smoke/exp08.onnx` |
| **heads81 ONNX (compile this)** | `a0327a25da034e942bb1f6529b7747cb5ddef8c821dc9545ac40ccf394ed7aa1` | oryn `$SDJOB_DATA/m84/export_smoke/exp08.heads81.onnx` |

`heads81` is the native graph cut at the six head end nodes (`/head/scales.{0,1,2}/Mul`,
`/head/Sigmoid{,_1,_2}`), with the 3-row class convs embedded in 81-row convs: person → channel 0,
phone → 67, hand → 80; every other row has weight 0 and bias −20. Its six outputs therefore have the
shipping `.hef`'s layout (80/40/20 grids, 68-channel DFL box heads, 81-channel class heads), and
`core/accel.py` loads it unchanged; the hand channel is read only with the opt-in
`accelerator.yolo_extra_classes: {80: hand}` (M10).

**Parse risk to check first.** Op histograms (M10 contract job `20261114-035955-9ffec6`) of `heads81`
and of the zoo graph cut at its own end nodes (`zoo.heads.extracted.onnx`, sha256 `4b0550f7…f84`, the
graph behind the shipping `.hef`): identical Conv 77, Relu 69, Add 20, Concat 12, MaxPool 3, Resize 2,
Mul 5, Sigmoid 5; the export additionally has `Cast` 3 and `Constant` 28 and has `Slice` 6 vs 8 (the
Focus-stem slicing is expressed differently). If the DFC parser rejects them, constant-fold the graph
(e.g. `onnxsim`) and re-verify float parity with ONNX Runtime (max |Δ| on the known-good frames) before
going on; record the new sha256.

**Quantisation risk to watch.** On the six known-good frames the EXP-08 box heads span
−20.4…33.0 logits against −6.4…11.6 for the shipping weights (same job): about 3× the range for the
same 8-bit output, so coarser DFL steps if the outputs are quantised per tensor. M8.4's host-simulated
"uint8 outputs only" variant was harmless for the shipping weights (§5.5); it has not been run for EXP-08.

## Pins

| Item | Pin | Why |
|---|---|---|
| Hailo Model Zoo | **v2.15.0**, target `hailo8l` | the production `.hef` was built by it (`config.yaml` `accelerator.yolo_hef_sha256` note); a candidate built by the same toolchain loads in the Pi's existing HailoRT |
| Dataflow Compiler, HailoRT | the versions the Model Zoo v2.15.0 release names as its requirements | not re-stated here: read them from the release notes at the tag and record them (`compile.py` writes `tool_versions()` into the manifest) |
| Host | x86_64 Linux supported by that DFC release | the DFC is x86_64 Linux only |
| Model script | the zoo's `damoyolo_tinynasL20_T` `.alls` at tag v2.15.0, copied verbatim | same quantisation recipe as the shipping build; `compile.py check` refuses `nms_postprocess` (the product decodes raw heads; an on-chip NMS build compiles its threshold in) and any normalisation other than mean 0 / std 1 |
| Repo | the commit that holds this README | `compile.py`, `calib.py` and the decoder that reads the result |

Before the first real run, check every DFC call in `compile.py` (`ClientRunner`,
`translate_onnx_model`, `load_model_script`, `optimize`, `compile`, `save_har`, `infer_context`,
`InferenceContext.SDK_QUANTIZED`) against that DFC version's user guide and record any change here.

## Calibration set (the part that decides INT8 accuracy)

- **≥ 1,024 in-cabin NoIR frames** from the product camera (Camera Module 3 Wide NoIR) through
  `core/camera.py` at 640×480, saved losslessly (PNG) or as `capture_hailo_outputs.py` bundles.
  Cover: day and night (IR illumination), every seat position the unit will watch, phone in hand /
  phone near ear / no phone, hands on and off the wheel, glasses and sunglasses, empty cabin.
- Consent and storage under `docs/research/INCABIN_PILOT_PROTOCOL.md` (gates G1–G6). Calibration
  sessions/subjects must be disjoint from any in-cabin evaluation set. Never use Open Images val/test.
- **Preprocessing identical to the device**: `calib.py` runs each frame through
  `scripts/hw_validation/hwv_common.letterbox_bgr_to_rgb`, which a test pins byte for byte to what
  `HailoYOLO.predict` hands the chip (`tests/test_m10_hailo_compile.py`): letterbox to 640×640 with
  `cv2.INTER_LINEAR`, pad 114, centred (640×480 → 80 px bands top and bottom), BGR → RGB, uint8 NHWC.
  **No normalisation**: the network takes raw 0..255 (`research/m10/export_contract.py`), so the DFC
  receives the device tensor cast to float32.

```bash
python research/m10/hailo_compile/calib.py --frames ~/incabin/calib_frames --out ~/m10_compile/calib
# -> calib_uint8_nhwc.npy (N,640,640,3) uint8 + calib_manifest.json (per-frame sha256, luma, release_grade)
```

## Steps

```bash
R=~/sensordyme_model                       # this repo at the pinned commit
W=~/m10_compile; mkdir -p $W
ONNX=$W/exp08.heads81.onnx                 # sha256 a0327a25...aa1 (copy from oryn)
ALLS=$W/damoyolo_tinynasL20_T.alls         # the zoo v2.15.0 model script, verbatim
OUT=$W/out/exp08_heads81_hailo8l.hef       # never models/damoyolo_tinynasL20_T.hef

# 0. no SDK needed: hashes, six heads, calibration (>= 1,024, release grade), model script, output path
python $R/research/m10/hailo_compile/compile.py check --onnx $ONNX --calib $W/calib --model-script $ALLS --out-hef $OUT

# 1. parse -> optimise -> compile (writes .parsed.har, .optimized.har, the .hef, *.compile_manifest.json)
python $R/research/m10/hailo_compile/compile.py compile --onnx $ONNX --calib $W/calib --model-script $ALLS --out-hef $OUT

# 2. structure: one UINT8 640x640x3 input, six outputs (80/40/20 x 68/81), no NMS
hailortcli parse-hef $OUT | tee $W/out/parse_hef.txt
python $R/research/m10/hailo_compile/compile.py parse-hef $W/out/parse_hef.txt

# 3. L3: quantised emulator vs float ONNX, per head and after the product decoder (class map {80: hand})
python $R/research/m10/hailo_compile/compile.py emulate --har $W/out/sensordyme_exp08_heads81.optimized.har \
    --onnx $ONNX --images $R/scripts/hw_validation/known_good --report $W/out/emulate_known_good.json
```

Compare step 3's float detections with the float fixtures M8.4 already made for this export
(`exp08_known_good_expected_float.json`, `exp08_oi_val_expected_float.json`: 6 known-good frames and 48
Open Images val images with people / phones / hands; in `~/sensordyme-fleet/runs/20261113-090349-6d22e9/artifacts/`)
before reading the INT8 numbers: a float mismatch means the inputs differ, not the quantisation.

### 4. L4 on the Pi (hardware session; runbook H1 style)

1. Copy the candidate `.hef` to a **separate** path on the Pi (e.g. `~/hwval/candidates/`); never replace
   `models/damoyolo_tinynasL20_T.hef`. Use a development config whose `accelerator.yolo_hef` points at
   the candidate with its own `yolo_hef_sha256` and `yolo_extra_classes: {80: hand}`.
2. `scripts/hw_validation/capture_hailo_outputs.py --images ... --out ...` on the fixture images and
   on in-cabin validation frames, then on a worker
   `scripts/hw_validation/compare_float_vs_int8.py --bundle ... --onnx exp08.heads81.onnx --end-nodes <its six output names>`.
3. **Open item:** `compare_float_vs_int8.py` / `hwv_decode.py` decode `NUM_CLASSES = 80`, so their
   detection-agreement tables do not include hand; their per-tensor parity does cover channel 80. Add a
   class-count option (81) before relying on detection-level hand agreement.

## Proposed acceptance (to pre-register before the run; not yet validated anywhere)

| Check | Criterion |
|---|---|
| parse-hef | step 2 passes |
| float reproducibility | step 3 float detections equal the M8.4 float fixtures (same tensors) |
| per-head INT8 parity (L3 and L4) | class-head cosine and box-head cosine reported per head; any head below the shipping `.hef`'s own H1 parity on the same frames is a finding |
| accuracy | person R@P0.80, phone AP50, hand AP50 re-measured at INT8 on the same scorer; drops > 0.02 person R@P0.80 or > 0.03 hand AP50 vs float fail. For scale: host-simulated INT8 (L2, not Hailo) cost the shipping model 0.032 person AP50 (M8.4 §5.5) |
| thresholds | every product threshold re-derived on INT8 scores (precision targets, not raw scores) |
| latency | Pi + Hailo-8L latency and fps of the candidate, measured with `stage_timing.py`; the shipping `.hef`'s 22.07 ms mean (Pi, PHASE5) is the reference |

## Reproducibility notes

- `compile.py` records the sha256 of the ONNX, calibration array, model script, both `.har`s and the
  `.hef`, plus tool versions and timestamps, in `<net>.compile_manifest.json`.
- DFC optimisation is not guaranteed bit-reproducible across machines or versions; compile twice and
  record whether the `.hef` hashes agree. If they do not, the manifest (inputs + versions) is the
  reproducibility unit, and parity must be measured on the exact `.hef` that ships.
