# Hailo hardware validation (M5): procedure for a development Pi

> **STATUS: PREPARED, NOT EXECUTED.** Nothing in this document has been run on a
> Raspberry Pi, a Hailo-8L, a Camera Module 3 or a buzzer. The scripts were
> written on 2026-11-11 (branch `feat/m5-hw-validation`) and only their pure
> logic was tested, on a Mac (`tests/test_m5_hw_validation.py`). Every pass/fail
> criterion below describes what to look for. None of them is a result. When a
> step is run, record its actual output under `docs/evidence/` and update the
> rows in `docs/HARDWARE_VERIFICATION.md`.

Hardware: Raspberry Pi 5, AI HAT+ (Hailo-8L), Camera Module 3 Wide NoIR, piezo
buzzer on BCM 17 (`alerts.gpio_buzzer_pin`). Model:
`damoyolo_tinynasL20_T.hef`, Hailo Model Zoo v2.15.0 hailo8l, sha256
`e69136d3ebf3cc8ebaf3da5417a59962b139f3d3d67d715e9a9b1ae10963960e`
(`accelerator.yolo_hef_sha256`).

## M8.3 additions (2026-11-13): one command per stage, known-good inputs, the device path checked on the fleet

Still **not executed on hardware**. What M8.3 added on top of the steps below:

* **`hwv_stage.py`**: one command per stage, each ending in `STAGE <name>: PASS | FAIL | BLOCKED`.
  - Stages: `pins`, `model`, `parse-hef`, `capture`, `parity`, `latency`, `thermal`, `buzzer`, plus `list` and `all`.
  - Exit codes: 0 PASS, 1 FAIL, 2 BLOCKED, 3 SIMULATED.
  - `--fake` runs a stage against stand-ins (`fake_hailo.py`, a fake GPIO, a synthetic soak CSV) and can never exit 0.
  - The stage table, commands and provisional criteria are in `docs/MONDAY_HARDWARE_SESSION.md` §6.2.
  - The stages wrap the scripts below. They do not replace steps 1 to 12.
* **Known-good inputs** (`known_good/`). Six 640×480 frames with no people:
  - Four photos: a cat, a cup, a clock and a rocket. All are scikit-image v0.24.0 sample data, CC0 or public domain, verified against scikit-image's own sha256 registry and pasted unscaled.
  - Two generated frames: flat grey 114 and a test card.
  - Licensing and hashes are in `SOURCES.json`. `make_known_good.py` rebuilds the set.
  - Because the frames are camera-sized, `predict()` letterboxes them at scale 1.0 with no resampling, so the Pi's uint8 input tensor is bit-identical to the float run's. The `capture` stage checks this by sha256.
* **Float expectations** (`known_good/expected_float.json`), made through the device's own `predict()` with the float ONNX standing in for the chip (fleet job `20261113-043857-f182d1`). For each image they hold:
  - the input-tensor sha256;
  - per-head statistics;
  - the legacy detections at 0.05.

  The `parity` stage compares the INT8 capture against them without needing ONNX on the Pi.
* **Clean refusals.** Every script here now refuses a missing input or missing hardware with one line and exit 2, never a traceback (`hwv_common.run_cli`). `stage_timing.py health` refuses when nothing is listening, instead of polling a dead port for the whole run.
* **Confirmed on the fleet** (`docs/research/M83_INFERENCE_PATH.md`):
  - The float ONNX heads are NCHW `(1, 68|81, G, G)`. So the 81st class channel comes from the model, not from compiler padding (max 0.0079 over val), and the `to_nhwc` assumption in `compare_float_vs_int8.py` holds.
  - The device path reproduces the Mission 5 AP.
  - **Decoder fixes** (`core/accel.py`): malformed device results now raise errors. Before, they decoded silently to wrong boxes or to no boxes.

## 0. Questions this answers

| # | Question | Step | Script |
|---|---|---|---|
| Q1 | Do the int8 Hailo outputs match the float model on identical inputs? Is each head what we think it is (box DFL logits, class sigmoid)? | 6, 12 | `capture_hailo_outputs.py`, `compare_float_vs_int8.py` |
| Q2 | How much do detections change between the production decoder ("legacy": anchor offset 0.5, argmax class) and the DAMO-YOLO reference (offset 0, every class above threshold), and between letterbox and stretch preprocessing, on the same device tensors? | 6, 12 | same |
| Q3 | Actual processed FPS on the device, and where the time goes per stage | 3, 7 | `stage_timing.py health` / `stages` |
| Q4 | Camera-to-alert latency | 11 | `ms_clock.html`, phone video, `stage_timing.py latency` |
| Q5 | Thermal behaviour over 30 minutes of real load | 4 | `thermal_soak.sh` |
| Q6 | Wide-lens intrinsics and the right `camera_hfov_deg` | 9, 12 | `capture_calibration_frames.py`, `calibrate_intrinsics.py` |
| Q7 | Buzzer timing, and (by a person) whether it is audible | 8 | `buzzer_check.py`, phone audio |
| Q8 | Is the device running the code, config and model we think it is? | 2, 12 | `integrity_manifest.py` |

All scripts live in `scripts/hw_validation/`. The two shared modules are
`hwv_decode.py` (both decoders, NMS, matching; pure numpy, the same file on the
Pi and the worker) and `hwv_common.py` (safety gates, hashing, preprocessing
identical to production, bundle format, parity statistics).

## 1. Ground rules

1. **Development Pi only.** Never a unit that is protecting a driver or anyone
   else. While the service is stopped (steps 5 to 10) the unit monitors nothing.
2. **The installed software is never modified.** No step writes to
   `/home/pi/sensordyme` (code, `config.yaml`, `models/*.hef`, `.venv`, `data/`)
   or to `/etc/systemd/system/sensordyme.service`. Each script refuses an
   output path inside the installed tree, and the Pi-side scripts disable
   bytecode writing so that importing installed code leaves no `__pycache__`.
   All output goes to `~/hwval/` on the Pi.
3. **The validation code runs from a separate directory**,
   `~/hwval/sensordyme_model`. It is never `make deploy`ed. The candidate
   decoder is compared offline on captured tensors, so the installed decoder is
   never replaced to test it.
4. **The scripts never stop or start the service.** The chip, the camera and
   the buzzer GPIO are single-user, so the scripts that need them refuse while
   `sensordyme.service` is active (or its state cannot be read) and print the
   reversible operator commands instead. The operator runs those commands.
5. **Frames.** The product never writes camera frames to disk. Steps 6 (camera
   variant) and 9 do, so they need `--allow-saving-frames` (step 6) and must be
   pointed at a test scene or a checkerboard, not at people. Delete them once
   the worker has a copy.

### Needs explicit owner approval first

| Action | Why |
|---|---|
| Running any step on a unit that is not a designated development Pi | Stopping the service removes protection. Even read-only steps load the device. |
| Stopping the service at all (step 5) | It is reversible, but nothing is monitored while it is stopped |
| Saving camera frames (step 6 `--camera`, step 9) | Departs from the product's no-frames-on-disk guarantee |
| Setting `HAILO_TEMP_CMD` for the soak | It opens a second handle on a chip the service holds. Untested. |
| Restoring the backup (section 10) | Overwrites the installed tree. Only needed if something unexpectedly changed. |
| **Any** change to installed software or config as a result: decoder (legacy to reference), `camera_hfov_deg`, `.hef` or its pin, preprocessing, buzzer patterns | Product behaviour changes. Each goes through its own reviewed PR, tests and deploy, never through this procedure. Changing `camera_hfov_deg` also invalidates the stored head-pose calibration. |
| Installing packages on the Pi | Not needed: the Pi side uses only the service venv's existing packages (numpy, cv2, yaml, hailo_platform, picamera2, gpiozero). onnx/onnxruntime are needed on the worker only. |

## 2. Preparation (Mac / worker; nothing on the Pi yet)

1. Pick the commit to validate from and note its sha:
   `git rev-parse feat/m5-hw-validation`.
2. **Test images.** Use a fixed, shareable set so the worker can rejoin results
   to labels: the Mission 4/5 Open Images subset (same image ids) plus around 50
   driver-like frames, with no identifiable people unless consent is on record.
   Keep it to about 100 images. Each sample is about 6 MB uncompressed per
   preprocessing mode, and the capture script refuses if the SD card lacks
   1.5x that plus 1 GB.
3. **Float model, on the worker:** the Hailo Model Zoo
   `damoyolo_tinynasL20_T` pretrained zip (2022-12-19). Record its sha256. Look
   in the zip's yaml/alls for the input normalisation. The compare script
   prints any `normalization`/`mean`/`std` lines it finds as
   `normalisation_hints`. The default assumption is mean 0, std 1 on 0-255 RGB
   (DAMO-YOLO's convention); if the zip says otherwise, pass `--mean/--std`.
4. Copy the validation code to the Pi **into its own directory** (the operator
   runs this, not an agent):

   ```bash
   C=$(git rev-parse feat/m5-hw-validation)
   git archive --prefix=sensordyme_model/ "$C" | ssh pi@<dev-pi> 'mkdir -p ~/hwval && tar -x -C ~/hwval'
   ssh pi@<dev-pi> "echo $C > ~/hwval/sensordyme_model/HWVAL_COMMIT"
   scp -r <images> pi@<dev-pi>:~/hwval/images
   ```

   `HWVAL_COMMIT` is how the scripts record which commit they came from
   (`git_commit: "file:<sha>"`; step 12 verifies it by hash).

All Pi commands below use the service's own interpreter, read-only:
`PY=/home/pi/sensordyme/.venv/bin/python` and `V=~/hwval/sensordyme_model/scripts/hw_validation`.

## 3. Steps (all on the development Pi unless marked)

### Step 1: Backup (service running)

```bash
mkdir -p ~/hwval/backup && cd ~/hwval/backup
df -h /home                                             # need ~2x the tree size free
cp -p /home/pi/sensordyme/config.yaml config.yaml.$(date +%s)
systemctl cat sensordyme.service > sensordyme.service.unit
tar --exclude='.venv' -czf sensordyme_tree_$(date +%Y%m%dT%H%M%S).tgz -C /home/pi sensordyme
sha256sum * > SHA256SUMS
```

**Pass:** the archive exists, `tar -tzf` lists `sensordyme/config.yaml` and
`sensordyme/models/damoyolo_tinynasL20_T.hef`, and SHA256SUMS is written. The
backup holds `data/` and `config.yaml` (which contains the dashboard token).
Keep it on the device or encrypted storage, and never commit it.

### Step 2: Integrity (service running; read-only)

```bash
$PY $V/integrity_manifest.py snapshot --expect-role driver_safety \
    --out ~/hwval/integrity_$(date +%Y%m%dT%H%M%S).json
```

The script hashes every deployable file under `/home/pi/sensordyme`, using the
same excludes as `make deploy`. It hashes `config.yaml` and the `.hef`
separately and checks the `.hef` against `accelerator.yolo_hef_sha256`. It runs
the checkout's `scripts/deploy_preflight.py` against the device's config (a
pure check, printed only) and records the unit-file hash, `systemctl show`
state and the venv's `pip freeze`. Drop `--expect-role` if this unit is not a
driver unit. Also run by hand:
`hailortcli parse-hef /home/pi/sensordyme/models/damoyolo_tinynasL20_T.hef`
(expect six outputs and no NMS/score threshold).

**Pass:** `model.pin_ok: true`; the preflight prints `"ok": true`, or only
refusals that are already understood; six outputs. **Fail (stop here):** the
pin does not match (the runtime refuses this model too), or the preflight
refuses for a reason nobody can explain. Step 12 shows which commit the
installed code matches.

### Step 3: Baseline processed FPS (service running; read-only)

Close the app and any browser tab on the dashboard. `/api/health` windows are
drained per caller, so another client steals samples.

```bash
export SENSORDYME_TOKEN="$($PY -c "import yaml;print(yaml.safe_load(open('/home/pi/sensordyme/config.yaml'))['dashboard']['token'] or '')")"
$PY $V/stage_timing.py health --minutes 10 --interval 5 --out ~/hwval/health_$(date +%s).jsonl
```

This uses only metrics the runtime already publishes: capture fps,
`processed_frames` and the processed-interval distribution
(`frame_bus.processed_interval_stats`), Hailo inference count, temperature,
throttling and the monitoring state. It also records Hailo inferences per
second (= processed fps / `yolo_every_n_frames` when healthy).

**Pass (the measurement is valid):** at least 100 windows, no poll errors,
`states_seen == ["ready"]`. **Record:** processed fps p5/p50/p95, processed
interval p95 and max, `stalls_1s_total`. There is no fixed FPS threshold in this
procedure. **Flag** any `stalls_1s_total > 0`: a 1 s stall discards in-progress
event evidence (EventEngine occlusion grace).

### Step 4: Thermal soak (service running; read-only; 30 min)

```bash
bash $V/thermal_soak.sh --minutes 30 --interval 5 --out ~/hwval/soak_$(date +%s).csv
```

It logs `vcgencmd measure_temp / get_throttled / measure_clock arm /
measure_volts core`, fan rpm, load, processed fps, camera fps, processed p95
and the monitoring state. The Hailo die temperature is **not** read unless
`HAILO_TEMP_CMD` is set, and setting it needs owner approval (see the table
above). Do not run it at the same time as step 3; both drain the same health
window.

**Pass:** `get_throttled` stays `0x0` (no bit 0/2 now, no bit 16/18 sticky), the
ARM clock never drops below its starting value, and processed fps at minute 30
is within about 10% of minute 1 (advisory). **Record:** peak SoC temperature,
the time to plateau, and the fan rpm.

### Step 5: Stop the service (operator; owner-approved)

```bash
systemctl is-active sensordyme.service      # write the answer down: the state to restore
sudo systemctl stop sensordyme.service      # explicit stop: Restart=always does not fire
systemctl is-active sensordyme.service      # expect: inactive
ls -l /proc/*/fd 2>/dev/null | grep hailo0  # expect: nothing
```

The scripts check the same things themselves (`systemctl is-active`, and
`/proc/*/fd` for `/dev/hailo0`) and refuse otherwise.

### Step 6: Raw Hailo capture (service stopped)

```bash
cd ~/hwval
$PY $V/capture_hailo_outputs.py --images ~/hwval/images --modes letterbox,stretch \
    --out ~/hwval/capture_$(date +%Y%m%dT%H%M%S)
```

What it does: it checks the service is stopped, nothing holds `/dev/hailo0`,
the output is outside the install, and disk space suffices. It verifies the
`.hef` sha256 against the config pin and refuses on a mismatch or a missing
pin. It records HailoRT version, `hailortcli fw-control identify`, the device
ids, config sha256, the code commit and the `core/accel.py` hash (checkout vs
installed). It then builds `core.accel.HailoYOLO` itself, so loading is the
production path. For each image and mode it saves the exact uint8 input tensor
and all six raw output tensors (`samples/*.npz`) and records names, shapes,
dtypes, HEF quant info and timing (preprocess, infer, production decode, the
offline legacy and reference decodes). It also records the production
decoder's own boxes for letterbox samples. The `.hef` is re-hashed at the end
(`model.sha256_after`).

Variants:

* `--code-root /home/pi/sensordyme` imports the **installed** `core/accel.py`
  (read-only) instead of the checkout's. Run this once if
  `code.same_as_installed` is false, so that "legacy" means what the device
  actually runs.
* `--native-outputs` adds a second pass that saves the quantised integer
  outputs and checks `(q - zp) * scale` against the float outputs
  (`native.dequant_check`). It is experimental and optional.
* `--camera 20 --allow-saving-frames` uses live frames (production `Camera`,
  640x480) instead of images. It needs owner approval and a test scene. It
  also serves the glass-to-capture latency check: put `ms_clock.html` in view,
  then read the clock digits in each saved frame against `capture_wall_ns`
  (step 11).

**Pass:** `status: complete`; `model.sha256 == model.sha256_after == pin`;
six outputs at grids 80/40/20 with 68 and 81 channels; no "class head outside
[0,1]" warning; the timing summary prints. **Record:** p50/p95 of `infer`,
`preprocess` and each decode. **Fail:** any refusal, or a class head outside
[0,1] (that would mean the on-chip sigmoid assumption in `_decode_raw` is
wrong).

### Step 7: Per-stage latency (service stopped)

```bash
$PY $V/stage_timing.py stages --minutes 3 --out ~/hwval/stages_$(date +%s)
```

This builds the configured module the way `benchmarks/run_benchmarks.py` does,
with the installed config read in memory and the `.hef` path made absolute. It
runs with its working directory set to `--out`, so any relative write lands
there. Per frame it times capture, preprocess, `hailo_infer`, decode,
`detector_total`, `facemesh`, `hands`, `module_logic` (`process()` minus the
timed sub-stages: rules, trackers, fusion), `process_total` and `frame_total`.
The timing wraps attributes of already-built objects inside this process only.
Use `--video clip.mp4` for a repeatable input.

How to read it next to step 3: the harness is single-threaded, so
`sequential_fps = 1 / (capture + process)`. Production overlaps capture with
processing, so its processed fps is bounded by
`min(camera fps, processing_bound_fps)`. Step 3's measured processed fps should
come out close to that bound. A large gap means the runtime adds work this
harness does not include (alert dispatch, custom rules, telemetry, stream
encode), and that gap is itself a finding. Detector stages appear only on
`yolo_every_n_frames` ticks; `frames_with_stage` says how many.

**Pass:** `instrumented.hailo: true` (the detector really was the Hailo path),
and the `detector_total` p50 is consistent with step 6's
preprocess + infer + decode.

### Step 8: Buzzer (service stopped; a person present)

```bash
$PY $V/buzzer_check.py run --out ~/hwval/buzzer_$(date +%s).json
```

The script plays every production pattern (chirp, single, double, urgent,
critical, impaired) through the production `GPIOBuzzerSink` and pattern player,
announces each one three seconds ahead, and logs when the GPIO was
**commanded** on and off.

At the same time, a person records audio with a phone 50 cm and 1 m from the
device (in the car if possible, engine off and then on), using an SPL meter app
for a dB(A) reading at 1 m. Afterwards, annotate each audible pulse's onset and
offset from the waveform (e.g. Audacity) into a CSV
`pattern,step,onset_ms,offset_ms` and run:

```bash
python buzzer_check.py analyse --timeline buzzer_….json --annotations audio.csv
```

**What software cannot verify:** software cannot tell whether a pattern was
audible. There is no feedback line from the buzzer (HARDWARE_VERIFICATION
M2-2). The commanded timeline only shows that the pin writes succeeded; a
disconnected buzzer produces an identical log. A logic analyser or scope on
BCM 17 is the only way to measure the electrical timing. **Pass (by a
person):** every pattern is heard and distinguishable from the others, with
audio pulse timing within about ±30 ms of the definition (advisory), and SPL
recorded. **Fail:** any pattern missing from the recording (`not_heard`).

### Step 9: Calibration frames (service stopped; checkerboard only)

```bash
$PY $V/capture_calibration_frames.py --stream --count 30 --interactive --out ~/hwval/calib_640x480
$PY $V/capture_calibration_frames.py --full   --count 30 --interactive --out ~/hwval/calib_full
```

Print a checkerboard (e.g. 10x7 squares = 9x6 inner corners, measure the
square), mount it flat, and move it to cover the corners and edges at several
tilts. `--stream` uses the production `Camera` (640x480 RGB888, `rotate_180`
honoured). `capture_meta.json` records ScalerCrop and the sensor mode. That is
how to tell how much of the 16:9 sensor (and so of the lens's FOV) the 4:3
stream keeps.

### Step 10: Restore the service (operator)

```bash
sudo systemctl start sensordyme.service
systemctl is-active sensordyme.service                    # expect: active
curl -s -H "X-Device-Token: $SENSORDYME_TOKEN" localhost:8899/api/health \
  | $PY -c 'import json,sys; d=json.load(sys.stdin)["device"]; print(d["monitoring_active"], d["monitoring"]["state"])'
$PY $V/integrity_manifest.py snapshot --out ~/hwval/integrity_after.json   # then diff with step 2
```

**Pass:** `True ready` within about 60 s, and the step-2 and after snapshots
show the same file hashes, config sha256 and model sha256. If anything differs,
stop and tell the owner. **Restore (owner approval only):**
`sudo systemctl stop sensordyme.service && tar -xzf ~/hwval/backup/sensordyme_tree_*.tgz -C /home/pi && sudo systemctl start sensordyme.service`.

### Step 11: Camera-to-alert latency (service running; phone video)

Setup: open `scripts/hw_validation/ms_clock.html` full-screen on a laptop or
tablet (it is offline and draws `Date.now()` ms plus a square that flips every
display refresh). Place it in view of both the Pi camera and a phone filming at
240 fps (slo-mo). The phone must also see the buzzer (an LED on the same GPIO
helps, but the audio track is enough) and the stimulus. Record the display
machine's NTP offset and `chronyc tracking` on the Pi, which you need to
compare with device timestamps.

Trials (10 or more per alert type). Use stimuli with a known designed delay:

* **eyes closed (drowsiness):** the subject closes their eyes at a clock
  reading. The designed delay is the module's configured eyes-closed duration.
* **phone use:** the subject raises a phone into view. The designed delay is
  the module's phone-use persistence.

For each trial, annotate from the video: `stimulus_ms` (the first frame where
the eyes are closed or the phone is visible, read off the on-screen clock) and
`alert_onset_ms` (the first video frame with buzzer sound onset or LED on).
Store the configured delay as `designed_delay_ms`, then:

```bash
python stage_timing.py latency --trials trials.csv
```

`minus_designed_delay_ms` is the pipeline's own latency: capture, processing
stride, debouncing granularity and actuation. Resolution is about ±4 ms (240
fps video) plus ±17 ms (60 Hz display). Optionally cross-check the device's
own event timestamp (`data/events.csv`, journal) against the clock using the
NTP offsets, to split stimulus-to-event from event-to-buzzer.

**Glass-to-capture** (service stopped; the step-6 camera variant): for each
saved frame, read the displayed ms from the image and put it in a CSV
`displayed_ms,captured_ms,offset_ms` (`captured_ms = capture_wall_ns / 1e6`;
`offset_ms` = Pi clock minus display clock). Then run
`stage_timing.py latency --glass glass.csv`.

**Pass:** the measurement is complete with 10 or more trials per type.
**Record:** p50/p95 of `minus_designed_delay_ms`. No threshold has been agreed;
propose one from the data.

### Step 12: Worker analysis

Copy `~/hwval/capture_*`, `integrity_*.json`, `calib_*`, `stages_*`,
`health_*`, `soak_*` and `buzzer_*` to the worker, then delete the frame-bearing
directories from the Pi.

```bash
# Q1/Q2: int8 vs float, decoders, preprocessing
python scripts/hw_validation/compare_float_vs_int8.py --bundle capture_… \
    --onnx damoyolo_tinynasL20_T.zip --out report_capture_…
# Q6: intrinsics, once per resolution
python scripts/hw_validation/calibrate_intrinsics.py --images calib_640x480 --pattern 9x6 \
    --square-mm <measured> --out intrinsics_640x480.json
# Q8: which commit does the device hold?
python scripts/hw_validation/integrity_manifest.py compare --snapshot integrity_….json \
    --rev 973aba3 --rev integration/all --rev <last deployed tag>
```

`compare_float_vs_int8.py` runs the ONNX on the **identical saved uint8
tensors**. It cuts the graph at `Mul_224, Sigmoid_244, Mul_264, Sigmoid_284,
Mul_304, Sigmoid_324` and pairs each with a HEF output by grid and role. It
reports:

* the tensor name/shape mapping;
* per-tensor mean and max absolute error, RMSE, cosine, Pearson and
  worst-sample cosine;
* device and float value ranges, with a sigmoid-vs-logit verdict;
* per-channel error for each 81-channel class head, in quantisation steps when
  the HEF quant info is known, with person, bottle, cup and cell phone listed
  individually, and the HEF padding channel range;
* int8-vs-float decoded agreement for all four decoder variants (`legacy`,
  `legacy_offset0`, `legacy_multilabel`, `reference`) per preprocessing mode:
  match rates, matched-IoU distribution and score deltas;
* decoder-vs-decoder agreement on the device tensors (the per-stride centre
  shift isolates the anchor offset);
* production pipeline (letterbox + legacy) vs reference pipeline (stretch +
  reference) in frame pixels;
* whether the production decoder's on-device boxes equal the offline legacy
  decoder's.

Reference NMS IoU and the detection cap default to DAMO-YOLO's (0.7, 100);
`--reference-nms-iou` and `--reference-max-det` match whatever the candidate
decoder chooses.

## 4. How results feed back

| Finding | Meaning | Next action (each needs owner approval) |
|---|---|---|
| `production_decoder_parity.all_match` false | The device is not running the decoder the repo describes | Stop. Integrity issue: compare with step 12 Q8 before anything else. |
| Head cosine ≥ 0.99 and class-head error ≲ 1-2 LSB (advisory) | The int8 path is faithful. Mission 4/5 float-ONNX accuracy findings transfer to the device. | Evaluate decoders on labels with the float model, or on these device tensors directly. |
| Head cosine clearly below that, or concentrated on product classes | Quantisation changes the model. Float results do not transfer. | Re-run the Mission 4/5 label evaluation on the **device tensors** from the bundle (it carries image ids, sha256 and geometry) instead of float outputs. |
| Class head outside [0,1], or box head inside [0,1] | The head-layout assumption in `_decode_raw` is wrong | Stop. Fix the decoder assumption before any accuracy work. |
| `legacy_vs_legacy_offset0` centre shift = +stride/2 (it must be, by construction) | Confirms the bundle and decoders are wired correctly | None |
| Reference decoder beats legacy on labels (worker, using device tensors) | The candidate decoder is better on this device | Separate PR: the candidate decoder's output on these bundles must match `hwv_decode` `reference` (use the bundle as a fixture), plus the existing raw-decode tests updated. Then a deploy to the dev Pi, then step 3 again. |
| `camera_hfov_deg` recommendation (paraxial HFOV at 640x480) | The head-pose intrinsics are currently `fx = w` (~53°) | A config change through review. It invalidates the stored calibration, so the device recalibrates. Use the paraxial value, not the 102° datasheet edge-ray angle (see `calibrate_intrinsics.py`). |
| Processed fps or stalls (steps 3, 7) | Whether time-based thresholds behave as configured | Update `docs/PERFORMANCE_BUDGET.md` with measured values. |
| Throttling in the soak | Cooling is insufficient for sustained load | Hardware decision (cooler or case) |
| A buzzer pattern not heard or not distinguishable | M2-1 fails | Hardware or pattern change. M2-2 stays not detectable in software. |

Update `docs/HARDWARE_VERIFICATION.md` with pasted outputs: items 7 (sustained
FPS), 8 (thermal), 10 and M2-1 (buzzer), 4 to 6 (accelerator; item 6's parity
is now answered by Q1 instead of a CPU baseline). Commit only summaries
(`report.json`, `summary.md`, intrinsics JSON, CSVs) under `docs/evidence/`.
Never commit raw bundles, frames, backups or configs.

## 5. What is not verified, even by this package

* None of the scripts has run on a Pi. HailoRT API details are written
  defensively, but they are untested on HailoRT 4.x/v2.15: `vstream_info.quant_info`
  field names, `Device.scan()`, the `FormatType.AUTO` second pass, and
  `hailortcli` output.
* `compare_float_vs_int8.py` was tested end to end on the Mac with a stand-in
  float model. The ONNX graph surgery test needs the `onnx` package, which the
  Mac venv lacks, so it was **skipped**. The real Model Zoo zip has not been
  opened: the end-node names, the ONNX head layout (NCHW assumed, with flattened
  layouts handled) and the input normalisation are assumptions for the worker
  to confirm.
* The reference decoder here implements DAMO-YOLO's postprocess from its
  published behaviour (offset 0, multilabel `score > thr`, class-wise NMS 0.7,
  top 100). It is a measuring stick, not the separately developed candidate.
  Agreement between the two has to be checked when the candidate exists.
* The per-stage harness instruments private attributes (`_yolo_model`,
  `_face_mesh`, `_hands`, `_ensure_ready`, `_decode_raw`). If those names
  change, `instrumented` in the output says what was found.
* Audibility, loudness and electrical timing of the buzzer: by a person or an
  instrument only.
* Hailo die temperature: not read by default (see step 4).
