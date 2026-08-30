# Using the AI HAT+ (Hailo) for inference

The AI HAT+ 13 TOPS carries a **Hailo-8L**. It cannot run stock `.pt` or
`.onnx` checkpoints — models must be compiled ahead of time to a `.hef` with
Hailo's Dataflow Compiler. This page is the honest map of what runs where.

## What offloads today

| Model | Backend | Notes |
|---|---|---|
| YOLO object detection (phone/food/person — occupancy, driver_attention, openpilot phone check) | **Hailo when `accelerator.yolo_hef` is set**; CPU otherwise | via `core/accel.py` → `core/detectors.py`; automatic fallback |
| MediaPipe Face Mesh (EAR/yaw/pitch/PERCLOS) | **CPU today** | see the correction below — a Hailo build of the *landmark* net does exist |
| openpilot dmonitoring_model.onnx | **CPU** (onnxruntime) | would need its own Hailo compilation; not done |

**As shipped, `accelerator.yolo_hef` is `null` and no `.hef` exists in this repo,
so the AI HAT+ contributes nothing and every inference runs on the Pi CPU.**
`/api/health` reports this honestly ("Hailo present and runtime OK, but no
compiled model configured").

### Correction (2026-08-29): you do not need the Dataflow Compiler to start

An earlier version of this page said "no Hailo build of mediapipe exists; this is
the main CPU consumer and it stays", and implied that using the HAT required
compiling your own model. Both were wrong, and the second one is why the HAT has
been idle. Hailo publish **prebuilt Hailo-8L `.hef` files** you can download
directly — no compiler, no calibration dataset, no x86 machine:

```
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/<name>.hef
```

Verified reachable (HTTP 200, sizes as served) on 2026-08-29:

| File | Size | Relevance |
|---|---|---|
| `yolov11n.hef` | 11 MB | drop-in for the current `yolo11n.pt` detection path |
| `yolov8n.hef` | 11 MB | alternative detector |
| `damoyolo_tinynasL20_T.hef` | 19 MB | higher mAP than yolov11n **and Apache-2.0** — see the licensing note in docs/HUGGINGFACE.md |
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

Easiest: download a precompiled YOLO from the Hailo Model Zoo (e.g.
`yolov8s_h8l.hef` — the `_h8l` suffix means Hailo-8L, which is what the
13 TOPS HAT needs; plain `_h8` files are for the 26 TOPS HAT and won't run).
Choose an **NMS-on-chip** build — `core/accel.py`'s output parsing assumes the
Model Zoo NMS postprocess format (per-class `[ymin, xmin, ymax, xmax, score]`,
normalized).

Then:

```yaml
# config.yaml
accelerator:
  enabled: true
  yolo_hef: /home/pi/sensordyme/models/yolov8s_h8l.hef
  required: false   # true once you trust it: refuse to boot without the HAT
```

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

`core/accel.py`'s `HailoYOLO` was written against the documented HailoRT
Python API and Model Zoo NMS output layout, but has **not been executed on real
Hailo hardware** (this repo is developed on a Mac). First run on the Pi should
be validated with `journalctl -u sensordyme | grep accel` and a look at the
overlay boxes; any init failure automatically falls back to CPU and reports
through /api/health.
