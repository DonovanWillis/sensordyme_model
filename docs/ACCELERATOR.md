# Using the AI HAT+ (Hailo) for inference

The AI HAT+ 13 TOPS carries a **Hailo-8L**. It cannot run stock `.pt` or
`.onnx` checkpoints — models must be compiled ahead of time to a `.hef` with
Hailo's Dataflow Compiler. This page is the honest map of what runs where.

## What offloads today

| Model | Backend | Notes |
|---|---|---|
| YOLO object detection (phone/food/person — occupancy, driver_attention, baby_monitoring) | **Hailo** on a device; ultralytics-on-CPU on a developer Mac | via `core/accel.py` → `core/detectors.py` |
| Comparison modules' phone check (openpilot, skinnemoen) | **MPS/CPU** as configured (`phone_device: mps`) | an explicit device pin bypasses the accelerator; only `phone_device: null` routes it to Hailo |
| MediaPipe Face Mesh (EAR/yaw/pitch/PERCLOS) | **CPU today** | see the correction below — a Hailo build of the *landmark* net does exist |
| openpilot dmonitoring_model.onnx | **CPU** (onnxruntime) | would need its own Hailo compilation; not done |

**As shipped, `accelerator.yolo_hef` is `models/damoyolo_tinynasL20_T.hef` and
`setup_pi.sh` provisions that file from a pinned URL, verified by sha256.** The AI
HAT+ does all object detection on a device.

*This paragraph used to read "`accelerator.yolo_hef` is `null` … the AI HAT+
contributes nothing and every inference runs on the Pi CPU". That was true before
Phase 4 and it is the reason the default was changed: with no `.hef`, the device
fell through to the CPU path, and the CPU path was AGPL-3.0 `ultralytics`. A
`null` default read as safe and neutral and was neither.*

**There is no CPU fallback on a device.** `requirements-device.txt` ships neither
`ultralytics` nor `torch`, so if the accelerator or the model is unavailable there
is no second detector to degrade into. `core/detectors.py` raises rather than
returning a stub, and the service boots into a **degraded** state — it serves
`/api/health` and the dashboard and reports `monitoring_active: false` with the
reason — instead of looking alive while detecting nothing. The CPU path still
exists on a developer machine, where `requirements.txt` pins `ultralytics` and
nothing is conveyed to anyone.

### Correction (2026-08-29): you do not need the Dataflow Compiler to start

An earlier version of this page said "no Hailo build of mediapipe exists; this is
the main CPU consumer and it stays", and implied that using the HAT required
compiling your own model. Both were wrong, and the second one is why the HAT has
been idle. Hailo publish **prebuilt Hailo-8L `.hef` files** you can download
directly — no compiler, no calibration dataset, no x86 machine:

```
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/<version>/hailo8l/<name>.hef
```

> **The version in that path matters and is not interchangeable.** SensorDyme
> ships the **v2.15.0** build of `damoyolo_tinynasL20_T`
> (sha256 `e69136d3…`, 18,554,430 bytes). The v2.19.0 build of the same name is a
> *different binary* and will fail `setup_pi.sh`'s checksum. The exact pinned URL
> lives in `setup_pi.sh` — take it from there, not from this template. The
> template is for *exploring* the zoo; it is not an install instruction.

Verified reachable (HTTP 200, sizes as served) on 2026-08-29, at `v2.19.0`:

| File | Size | Relevance |
|---|---|---|
| `yolov11n.hef` | 11 MB | **AGPL-3.0 weights** (Ultralytics-derived) — do not ship; NMS-on-chip, floors `conf` at 0.200 |
| `yolov8n.hef` | 11 MB | **AGPL-3.0 weights** (Ultralytics-derived) — do not ship |
| `damoyolo_tinynasL20_T.hef` | 19 MB (v2.19.0 build) | **THE SHIPPED DETECTOR — but ship the v2.15.0 build, not this one.** Apache-2.0, raw-tensor. Shipped: Model Zoo **v2.15.0**, sha256 `e69136d3…`, 18,554,430 bytes, pinned in `setup_pi.sh` |
| `scrfd_2.5g.hef` | 5 MB | face **detection** |
| `face_landmarks_lite.hef` | 4 MB | face **landmarks** — the MediaPipe-equivalent net |
| `hand_landmark_lite.hef` | 3 MB | hand landmarks |
| `yolov8s_pose.hef` | 26 MB | body pose (fall detection, flailing) |
| `tddfa_mobilenet_v1.hef` | 5 MB | 3D head pose |
| `r3d_18.hef` | 55 MB | spatio-temporal video classifier (16-frame clips) |

Two honest caveats before anyone plans around this:

* **The landmark nets are not a MediaPipe drop-in.** MediaPipe ships a detector,
  a landmark net, and a good deal of tracking/post-processing glue. The zoo gives
  you the *landmark net only* — you would pair it with `scrfd_2.5g` for detection
  and reimplement the post-processing yourself. That is a real project, not a
  config change. What is now false is the claim that it is *impossible*.
* **Every FPS figure Hailo publishes was measured with an x86 host driving the
  NPU.** Throughput on a Pi 5 host is UNVERIFIED and the host CPU may well be the
  bottleneck for the pre/post-processing around the accelerated part.

Compiling a *custom* model (your own fine-tuned weights) still needs the Hailo
Dataflow Compiler, which is **x86-64 Linux only** — it will not run on an Apple
Silicon Mac. That is a real constraint on any fine-tuning roadmap: plan for an
x86 Linux box or a cloud instance.

## The single most important thing to know about a .hef: does it do NMS on-chip?

Measured on the real device, 2026-08-29. This decides whether the detector can
see a phone, and it is invisible unless you look for it.

A Model Zoo `.hef` comes in one of two shapes, and `hailortcli parse-hef` tells
you which:

* **On-chip NMS** — one output, already decoded and suppressed. The score
  threshold and IoU are **compiled in**. `predict(conf=...)` below that
  threshold is a **silent no-op**: the chip never emits the detection, so no
  amount of tuning downstream can recover it.
* **Raw tensors** — several outputs per FPN scale, no NMS. Decoding is ours
  (`core/accel.py`'s `_decode_raw`), which costs host CPU and buys back complete
  control of the threshold.

What that costs, on the same 600 subject-disjoint driver images:

| .hef | NMS | Baked threshold | Phone recall (best F1) | Latency |
|---|---|---|---|---|
| `yolov11n` | on-chip | **0.200** | 25.4% | 15.2 ms |
| `yolov5s` | on-chip | 0.001 | 90.0% | 19.0 ms |
| `damoyolo_tinynasL20_T` | **none** | n/a — ours | **93.8%** | 23.6 ms |

The `yolov11n` curve is flat at 25.4% for every requested `conf` from 0.05 to
0.20, then falls — the exact signature of a 0.200 floor. Phase 1 lowered the
shipped threshold from 0.35 to 0.10 on measured evidence and tripled CPU phone
recall; on a 0.200-floored `.hef` **that entire gain is unavailable**.

So: check the threshold before adopting a model, not after benchmarking it.

```bash
hailortcli parse-hef models/<name>.hef | grep -iE 'Score threshold|Output'
```

A second, subtler trap: a **better-calibrated** model looks *worse* if you score
it at someone else's confidence. `damoyolo` produced spurious extra person boxes
on 55% of single-person images at `conf=0.25` — and 0.5% at `conf=0.60`, while
still finding every person. Tightening NMS IoU barely moved it (55%→48%),
because the extra boxes were spatially separate false positives rather than
overlapping duplicates. **Confidence is the lever for those; IoU is not.** That
matters beyond tidiness: `baby_monitoring` treats the smallest person box as the
infant, so a spurious box is a wrong-subject hazard.

## Licensing is not a property of the .hef

The Hailo Model Zoo is MIT, but that covers Hailo's *wrapper* — the tooling and
compile recipe. Hailo cannot relicense weights they do not own, so the upstream
model's licence flows through to the compiled artifact. `yolov11n.hef` and
`yolov5s.hef` are still Ultralytics-copyleft; `damoyolo_tinynasL20_T.hef` is
Apache-2.0 because DAMO-YOLO is. See `docs/LICENSING.md` — that document, not
the Model Zoo's own LICENSE file, is the one that governs what may ship.

## Setup on the Pi

```bash
sudo apt install hailo-all        # driver + HailoRT + tools (setup_pi.sh does this)
sudo reboot
hailortcli fw-control identify    # should print the Hailo-8L's serial
ls /dev/hailo0                    # device node must exist
```

For full PCIe bandwidth on the Pi 5 add to /boot/firmware/config.txt:

```
dtparam=pciex1_gen=3
```

## Getting a .hef

**You do not need to.** `setup_pi.sh` provisions the shipped model, pinned by URL
and verified by sha256, and refuses to install a mismatch. The shipped default is
already correct:

```yaml
# config.yaml — what ships
accelerator:
  enabled: true
  yolo_hef: models/damoyolo_tinynasL20_T.hef
  yolo_hef_sha256: e69136d3ebf3cc8ebaf3da5417a59962b139f3d3d67d715e9a9b1ae10963960e
  required: false
```

If you are evaluating a *different* model, two rules — both learned the hard way,
and both the opposite of what this section used to advise:

**1. Prefer a RAW-TENSOR build, not NMS-on-chip.** An NMS-on-chip `.hef` has its
score threshold **compiled in** and silently ignores any lower `conf` you pass —
measured as a 0.200 floor on `yolov11n.hef`, which capped phone recall at 25.4%
against CPU's 63.3%. Nothing errors; `config.yaml` simply stops describing the
running detector. `core/accel.py` handles **both** formats (it decides from the
HEF's own output count, not the filename), so raw costs you nothing but a little
CPU and buys back the threshold. The shipped DAMO build is raw — 6 output
tensors, no on-chip NMS — which is why the confidence values in `config.yaml`
mean exactly what they say.

**2. Check the licence before the mAP.** `yolov11n.hef`, `yolov8n.hef` and
`yolov8s_h8l.hef` are Ultralytics-derived: the weights carry **AGPL-3.0**, in a
different wrapper. Putting one in `models/` puts copyleft weights on a device you
convey. `make deploy` excludes `*.hef` (and `*.pt`) precisely so a model cannot
reach a device from someone's working tree — but that guard does not make an
AGPL model safe to *run*, only harder to ship by accident.

The `_h8l` suffix means Hailo-8L, which is what the 13 TOPS HAT needs; plain
`_h8` files are for the 26 TOPS HAT and won't run.

Restart and verify:

```bash
sudo systemctl restart sensordyme
curl -s localhost:8899/api/health | python3 -m json.tool | grep -A3 accelerator
# "Hailo in use for object detection (...)" = working
```

Compiling your own (custom weights) requires Hailo's Dataflow Compiler on an
x86 machine with a calibration dataset — see hailo.ai's model zoo repo. That is
an offline build step, not something the device does.

## Caveat

`core/accel.py`'s `HailoYOLO` was written against the documented HailoRT Python
API and Model Zoo output layouts on a Mac, and **has since been executed on real
Hailo hardware** (Phase 3/4 — `docs/PHASE5_DEMO_READINESS.md` is the record; the
23.6 ms and 93.8% figures above are device measurements). This section previously
said it had never run; that is no longer true.

What is still worth doing on a fresh board: validate with
`journalctl -u sensordyme | grep accel` and a look at the overlay boxes. Confirm
the log line naming **six raw output tensors and no on-chip NMS** — that is the
line that tells you the confidence thresholds in `config.yaml` mean what they say.

### Open question: anchor offset in the raw decode (unverified)

`HailoYOLO._decode_raw` places each anchor at the cell **centre**,
`(i + 0.5) * stride`. That is an assumption nobody has checked against a
reference. Upstream DAMO-YOLO (`ZeroHead.get_single_level_center_priors`, the
NanoDet-Plus convention) and, as recalled, the Model Zoo's damoyolo postprocess
config (`nanodet_split`, `scale_factors: [0.0, 0.0]`) put priors at the cell's
**top-left** corner, `i * stride`. If they are right, every box this decoder
returns is shifted right and down by stride/2 — +4 / +8 / +16 px at strides
8 / 16 / 32 in letterbox pixels (the same in frame pixels on a 640×480 camera).

None of the device measurements above can see this: phone recall, person
availability and spurious-box rates are presence metrics, and a constant shift
does not change presence. It does bias zone membership, the near-mouth region,
hand–phone IoU and crib containment.

**The check that settles it:** take ~50 frames with one clearly visible person,
decode them with this `.hef` and with a reference — the DAMO-YOLO PyTorch
checkpoint of the same model, or the Model Zoo's own eval postprocess run on this
`.hef` — and compare the mean (dx, dy) between matched person boxes. A constant
bias near +stride/2 means change the anchor to `xs * stride` / `ys * stride` and
update the two hand-computed expectations in `tests/test_phase3_raw_decode.py`;
a mean near zero means the current code is right and this note can go.

**An init failure does not fall back to CPU.** It did before Phase 4, and that
fallback was the AGPL path; the device no longer has `ultralytics` installed, so
there is nothing to fall back to. `/api/health` reports the failure and the
service runs **degraded** with `monitoring_active: false`. Treat a degraded boot
as an outage, not as a slower mode.
