# Using the AI HAT+ (Hailo) for inference

The AI HAT+ 13 TOPS carries a **Hailo-8L**. It cannot run stock `.pt` or
`.onnx` checkpoints — models must be compiled ahead of time to a `.hef` with
Hailo's Dataflow Compiler. This page is the honest map of what runs where.

## What offloads today

| Model | Backend | Notes |
|---|---|---|
| YOLO object detection (phone/food/person — occupancy, driver_attention, openpilot phone check) | **Hailo when `accelerator.yolo_hef` is set**; CPU otherwise | via `core/accel.py` → `core/detectors.py`; automatic fallback |
| MediaPipe Face Mesh (EAR/yaw/pitch/PERCLOS) | **CPU always** | no Hailo build of mediapipe exists; this is the main CPU consumer and it stays |
| openpilot dmonitoring_model.onnx | **CPU** (onnxruntime) | would need its own Hailo compilation; not done |

So: the HAT accelerates the object-detection half of the pipeline. Face
landmarks stay on CPU regardless — plan capacity accordingly.

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
