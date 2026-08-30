# Hardware verification checklist

Everything in this file is **UNVERIFIED**. It is the set of claims that cannot be
established from a development Mac, with the exact command or procedure that would
establish each one on real hardware.

Why it exists: a large fraction of this codebase's most important paths — the
camera, the accelerator, BLE provisioning, GPIO, thermal behaviour, and sustained
frame rate — are Pi-only. They are read and reasoned about here and executed
nowhere. Several defects found in the 2026-08-29 audit lived in exactly that gap,
including one (`Camera.release()` never calling `close()`) that had been sitting in
the recovery ladder since the beginning. Anything on this list should be treated as
"believed correct, never observed" until someone ticks it off on a device.

Dependencies absent from the dev Mac, confirmed: `picamera2`, `hailo_platform`,
`dbus`, `gi`, `gpiozero`, `lgpio`, `serial`.

---

## How to use this

Run these on the Pi, in order. Record the actual output — not "looks fine" — in the
Result column and commit it. A tick with no pasted output is not a verification.

| # | Claim | Status | Result |
|---|---|---|---|
| 1 | Camera opens and produces a real frame | UNVERIFIED | |
| 2 | `release()` genuinely frees the device; `reopen()` succeeds repeatedly | UNVERIFIED | |
| 3 | Recovery ladder step 1 (retry → reopen) works under a real read failure | UNVERIFIED | |
| 4 | Hailo device present and HailoRT usable | UNVERIFIED | |
| 5 | A downloaded `.hef` actually runs and returns boxes | UNVERIFIED | |
| 6 | Hailo vs CPU detection parity | UNVERIFIED | |
| 7 | Sustained FPS under real load, per module | UNVERIFIED | |
| 8 | Thermal / undervoltage reporting is real | UNVERIFIED | |
| 9 | BLE advertises and accepts provisioning | UNVERIFIED | |
| 10 | GPIO buzzer works and does not crash boot | UNVERIFIED | |
| 11 | Memory is stable over multi-day uptime | UNVERIFIED | |
| 12 | Reboot bounding behaves as designed | UNVERIFIED | |

---

## 1–3. Camera lifecycle

The unit tests in `tests/test_phase1_camera_lifecycle.py` pin the *contract*
(stop → close, idempotent, reference cleared) against a fake. They cannot tell you
whether real Picamera2 tolerates the sequence.

```bash
# 1. does it open and produce a frame at all
python scripts/preflight.py && cat data/preflight.json

# 2/3. release/reopen in a loop -- the thing the recovery ladder depends on.
#      Watch for a growing fd count, which is the leak this was fixed for.
python - <<'PY'
import os, time
from core.camera import Camera
import yaml
cfg = yaml.safe_load(open("config.yaml"))["camera"]
cam = Camera(cfg)
pid = os.getpid()
for i in range(20):
    frame = cam.read()
    assert frame is not None and frame.size > 0, f"no frame on iteration {i}"
    cam.reopen()
    fds = len(os.listdir(f"/proc/{pid}/fd"))
    print(f"iter {i:2d}  frame {frame.shape}  open fds {fds}", flush=True)
    time.sleep(0.2)
cam.release()
print("open fds after release:", len(os.listdir(f"/proc/{pid}/fd")))
PY
```

**Pass:** 20 successful reopens, and the fd count is flat rather than climbing.
**Fail signature (the bug this tests for):** fd count rises ~1–3 per iteration, or
reopen throws "Device or resource busy".

---

## 4–6. Hailo / AI HAT+

`core/accel.py`'s `HailoYOLO` has **never executed on hardware** — the project's own
`docs/ACCELERATOR.md` says so. Treat first contact as a debugging session, not a
verification. Known-suspicious spots found by review, worth watching:

- `predict()` re-enters the `InferVStreams` context and re-activates the network
  group on **every frame**. This may be slow, or may not be valid at all.
- `HailoYOLO.predict()` takes no `device` kwarg, but some callers pass one.
- The class-name table is hardcoded COCO regardless of what the `.hef` contains.

```bash
# 4. is the device there
hailortcli fw-control identify
ls -l /dev/hailo0

# 5. get a prebuilt model and run it (no Dataflow Compiler needed -- see ACCELERATOR.md)
mkdir -p models
curl -fL -o models/yolov11n.hef \
  https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8l/yolov11n.hef
# point config.yaml at it: accelerator.yolo_hef: /home/pi/sensordyme/models/yolov11n.hef
sudo systemctl restart sensordyme
journalctl -u sensordyme -n 200 | grep -i 'accel\|hailo'
curl -s localhost:8899/api/health | python3 -m json.tool | grep -A3 accelerator
```

**Pass:** health reports "Hailo in use for object detection", and the overlay boxes
look right. **Any init failure falls back to CPU** and is reported — so a silent
"still on CPU" is a fail, not a pass.

```bash
# 6. parity -- the number that decides whether the NPU can be trusted.
#    Same frames through both backends, compare detections.
python benchmarks/accuracy_state_farm.py --per-class 100 \
    --baseline benchmarks/results/hailo_on.json
# then set accelerator.yolo_hef: null, restart, and:
python benchmarks/accuracy_state_farm.py --per-class 100 \
    --compare benchmarks/results/hailo_on.json
```

**Pass:** headline metrics within the default 0.03 tolerance. A larger gap means
quantisation or the differing letterbox/NMS path is changing real detections, and
the two backends are not interchangeable.

---

## 7. Sustained FPS

No FPS number for this product exists on real hardware. Latency measured on the dev
Mac (YOLO11n ~19 ms, dmonitoring ~10 ms) is a floor, not a prediction — a Pi 5 CPU
is roughly 3–6× slower, and the MediaPipe figures measured here were taken on blank
frames where the landmark stage short-circuits.

```bash
python benchmarks/run_benchmarks.py --minutes 5            # live camera
python benchmarks/run_benchmarks.py --video clip.mp4 --minutes 5
cat benchmarks/results/summary_*.csv
```

Record per module: mean and p95 latency, sustained FPS, and CPU temperature at the
end of five minutes. **This is the number that decides whether the product's
temporal thresholds mean what they say** — several are expressed in seconds but
gated on frame counts, so they behave differently at 5 fps than at 30.

---

## 8. Thermal / undervoltage

```bash
vcgencmd measure_temp
vcgencmd get_throttled          # 0x0 = clean; bit 0 = under-voltage now, bit 16 = has occurred
cat /sys/class/thermal/thermal_zone0/temp
curl -s localhost:8899/api/health | python3 -m json.tool | grep -A5 thermal
```

Then load the device (run the benchmark above) and re-read. **Pass:** the health
endpoint's reading tracks `vcgencmd`, and a real under-voltage event surfaces rather
than being silently absorbed.

---

## 9. BLE provisioning

Security caveats found by review and **not fixed** (they need hardware to validate a
fix): the credentials characteristic is declared with a bare `["write"]` flag — no
encryption, no bonding — and the provisioning window defaults to always-available
rather than closing after setup. Verify the current behaviour before changing it.

```bash
sudo systemctl status sensordyme
journalctl -u sensordyme | grep -i 'ble\|gatt\|advertis'
sudo btmgmt info
# from a Mac, with the repo's own client:
python scripts/ble_provision_client.py --scan
```

**Also verify the exposure:** from a second machine in BLE range, confirm whether
the credentials characteristic can be written *without* pairing, and whether it is
still writable long after Wi-Fi is connected. Both are believed true and both are
security findings if confirmed.

---

## 10. GPIO buzzer

`config.yaml` enables `gpio_buzzer` by default and `main.py` constructs it outside
any try block, so a busy pin or an lgpio failure crashes the process before the
runtime exists — which, with `Restart=always`, is one of the reboot-loop triggers.

```bash
python -c "from gpiozero import Buzzer; b=Buzzer(17); b.on(); import time; time.sleep(0.5); b.off()"
sudo systemctl restart sensordyme && journalctl -u sensordyme -n 50
```

**Pass:** the buzzer sounds and the service reaches `READY=1`. **Fail:** the service
crash-loops — that is the unguarded-construction defect, and it is still open.

---

## 11. Memory over time

Nothing here has run for more than a few minutes. Storage growth is a *known* issue
(telemetry writes ~87 bytes twice a second with no rotation — roughly 5.5 GB/year),
so watch disk as well as RSS.

```bash
# leave running for 48h+
while true; do
  echo "$(date -Is) rss_kb=$(ps -o rss= -p $(pgrep -f 'python.*main.py')) \
disk=$(df -h / | awk 'NR==2{print $4}') \
data=$(du -sh data 2>/dev/null | cut -f1)"
  sleep 300
done | tee -a data/soak.log
```

**Pass:** RSS plateaus. **Fail:** steady climb, or `data/` growth that would fill the
card inside the product's expected service interval.

---

## 12. Reboot bounding

`services/runtime.py` was changed in Phase 1 so that reboots are bounded in total
(`health.reboot_max_total`, default 5) and so that a failure to persist the reboot
record **blocks** the reboot instead of being swallowed. Unit tests cover the logic;
the on-device behaviour is unverified.

```bash
cat data/.reboot_state.json          # {"count": N, "last_at": ...}
journalctl -u sensordyme | grep -i 'LAST RESORT\|reboot suppressed\|reboot ABORTED'
```

To exercise it deliberately, set `health.reboot_max_total: 1`, induce a critical
component failure (e.g. unplug the camera), and confirm the device reboots **once**
and then stays up degraded with the suppression message in the journal.

**Do not test this on a device you cannot physically reach.**
