# Hardware session (M8.3, "Monday session"): the accelerator, the camera, and one end-to-end alert

> **STATUS: PREPARED, NOT EXECUTED.** Written on 2026-11-13 on a Mac and the compute
> fleet. The device (Raspberry Pi 5 + AI HAT+ Hailo-8L + Camera Module 3 Wide NoIR)
> arrives **Sunday 2026-11-15** (M8.4; first planned for Monday 2026-11-16). "Monday"
> below means that first session. The session order, the one-command preflight
> (`sunday_preflight.py`) and the pilot decision are in **`docs/SUNDAY_DEVICE_ACTIVATION.md`**,
> which cites this page as MHS §x. No line below is a hardware result. Section A lists what
> was checked without the device, and every number there cites a job id or a test.

This page is the session plan for the inference hardware. It adds to the runbook
and does not repeat it:

* **`docs/MONDAY_RUNBOOK.md`** (sections A to R) has every install, backup,
  autonomy (K), pairing (L) and rollback (O) command. Steps are cited as
  **RB-F3**, **RB-K1**, and so on.
* **`docs/HAILO_HARDWARE_VALIDATION.md`** (M5) has the measurement procedures,
  cited as **M5 step N**.
* **`docs/MONDAY_OPERATOR_CHECKLIST.md`** gives the risk and approval class of
  every runbook step.

What is new here:
- the one-command stages `scripts/hw_validation/hwv_stage.py`;
- the known-good image set and its float expectations;
- the end-to-end acceptance;
- the A/B/C split;
- the inference-path findings from `docs/research/M83_INFERENCE_PATH.md`.

---

## 0. Variables

```bash
# Mac
export RC=feat/m8.4-leapfrog   # M8.4 candidate (pending owner approval; was feat/m8.3-hardware-truth, docs/M8_3_RELEASE_DECISION.md); the integrator records `git rev-parse $RC`
export PI=<user>@<pi-host>  REL=~/sensordyme-release  VENV=$HOME/.venvs/sensordyme-m01/bin
# Pi (RB-0 / RB-C1 discover U and D first)
export U=<service user>  D=/home/$U/sensordyme  PY=$D/.venv/bin/python
export V=~/hwval/sensordyme_model/scripts/hw_validation  EV=~/m6-evidence  S=~/hwval/stages
```

The validation package is copied from `$RC` into its own directory (RB-A4). It is never deployed.

## 1. A / B / C: what is known, what Monday measures, what is blocked

### A. Completed remotely or in simulation (evidence exists; none of it is hardware)

| Check | Result | Source |
|---|---|---|
| The device's own `predict()` path (letterbox, BGR→RGB, HEF-shaped head pairing, decode, NMS, class map, frame mapping), with the float ONNX standing in for the chip, on 1,585 Open Images val images | input tensor bit-identical to an independent letterbox in 1585/1585; legacy and candidate decodes equal the independent decoder in 1585/1585 (max \|Δ\| 3e-5 px) | fleet `20261113-043857-f182d1` (commit `989ecfd`) |
| Accuracy through the device path (legacy, the shipping decoder) | person AP50 0.660, phone AP50 0.728: reproduces Mission 5 (0.656 / 0.728) | same job; `docs/research/M83_INFERENCE_PATH.md` |
| Channel order RGB (what `predict()` sends) | swapping R and B lowers AP (person 0.635, phone 0.707); weak but consistent | same job |
| 81st class channel | present in the float ONNX too; max 0.0079; dropped by every decoder | same job |
| Malformed device results (missing head, NCHW, no batch axis, wrong grid, NaN/inf, bad `.hef` layout, non-uint8 frame) | now explicit errors; previously some returned **wrong boxes or no detections, silently** (fixed) | commit `ee9f6e6`, `tests/test_m83_inference_path.py` |
| Model pins: `setup_pi.sh` = `config.yaml` = `core/model_registry.yaml`; decoder `legacy`; EXP-08 inactive | consistent | `hwv_stage.py pins` (runs anywhere: PASS on the Mac) |
| Every hw_validation script without hardware | refuses with one line and exit 2, never a traceback | `tests/test_m83_hw_package.py` |
| Every `hwv_stage.py` stage against stand-ins (`--fake`) | executes end to end; exit 3 (SIMULATED), never 0 | same tests |
| Runbook K and L against the code | 1 wrong step (L2) and 4 wording fixes, corrected | commit `9f9d0c5`; §9 |
| Host-CPU per-stage time of the driver module | Hands ≫ FaceMesh ≫ decode ≈ preprocess ≈ module logic | fleet `20261113-050231-7108be`; Mac run; §8 |

### B. Physical checks on Monday (only the device can answer them)

1. The `.hef` loads through the production path, and its parse-hef layout is what the decoder assumes (stages `model`, `parse-hef`).
2. **INT8 vs float**:
   - the Hailo's head tensors and detections on the known-good images match the float expectations within the provisional tolerances (stages `capture`, `parity`);
   - head-level error on a worker (M5 step 12).
3. Hailo latency, host-side duty cycle, and `predict()` against the production stride budget (stage `latency`).
4. Processed FPS of the running service, and camera-to-alert latency (RB-H2, M5 steps 3, 7 and 11).
5. Thermal behaviour over 30 minutes under the real service (stage `thermal`, RB-J).
6. **Buzzer:**
   - commanded timing (stage `buzzer`);
   - audibility, judged by a person (RB-G2).
7. End to end (§6): app↔device video, status, stop/resume, a drowsiness alert from a seated, stationary subject, and the event in the app timeline.
8. Autonomy and pairing as in RB-K and RB-L (now corrected).

### C. Blocked by missing equipment or access (state it; do not improvise)

| Blocked | Needs | Effect if missing |
|---|---|---|
| Hailo die temperature during the soak | owner approval for `HAILO_TEMP_CMD` (M5 step 4) | SoC temperature only |
| Electrical buzzer timing | a logic analyser or scope on BCM 17 | commanded timing plus a person's ear only |
| On-chip NPU utilisation (as opposed to the host-observed duty cycle) | `hailortcli monitor` with `HAILO_MONITOR=1` working on this HailoRT build | duty cycle only; the stage records the monitor text if it works |
| Camera-to-alert latency | a 240 fps phone and a laptop showing `ms_clock.html` | not measured |
| App notifications and the app timeline | a backend reachable from the Pi, plus the iPhone with the ORYN EDGE app signed in | device-local evidence only (`events.csv`, dashboard) |
| In-vehicle validation (cabin heat, vibration, sunlight, night) | a parked vehicle and the owner's approval | desk results only. **Never on public roads.** |
| Any accuracy claim about drivers | consented in-cabin footage (`docs/research/INCABIN_PILOT_PROTOCOL.md`) | none made Monday |

## 2. Equipment

Everything in **RB-A5**, plus:

- **Mount:** a rigid camera mount at driver-face height, either a desk rig or a parked car's dash. Nothing hand-held.
- **Printed targets:**
  - the 9×6 checkerboard;
  - one sheet showing a phone-sized dark rectangle;
  - optionally printouts of `scripts/hw_validation/known_good/kg_coffee.png` and `kg_chelsea.png` at A4.
- **A consenting adult** for §6 drowsiness, who has read the consent text in `docs/research/INCABIN_PILOT_PROTOCOL.md`. No other people in frame.
- **The iPhone** with the ORYN EDGE app, an account, and the device's claim code (RB-L1).
- **A laptop** for `ms_clock.html` (only if latency is in scope); an SPL meter app.

## 3. Software, models and artifacts

| Item | Value | Verify with |
|---|---|---|
| Release candidate | `$RC` (the integrator records the commit) | RB-A0 `make verify-release REF=$RC` |
| Detector `.hef` | `damoyolo_tinynasL20_T.hef`, Hailo Model Zoo v2.15.0 hailo8l, 18,554,430 bytes, sha256 `e69136d3ebf3cc8ebaf3da5417a59962b139f3d3d67d715e9a9b1ae10963960e` | stage `model`; RB-F2 `sha256sum` |
| Decoder | `accelerator.yolo_decoder: legacy` (the reference decoder is opt-in, research only; do not switch it Monday) | stage `pins` |
| Float source (worker only) | Model Zoo `damoyolo_tinynasL20_T.onnx` (2022-12-19), sha256 `02d7ef7f35da6abfecdf2e04505b560451eb2dd9eff1cf335ab8e5370c126657` | `research/m83/inference_path_val.py` refuses any other |
| Known-good images | `scripts/hw_validation/known_good/*.png`, sha256 per file in `SOURCES.json` | the stages refuse a modified file |
| Float expectations | `known_good/expected_float.json` (job `20261113-043857-f182d1`, commit `989ecfd`): per image, the exact input-tensor sha256, head statistics, and float detections at 0.05 (legacy) | stage `parity` |
| Face, hands, pose | MediaPipe 0.10.14 on the Pi CPU (`requirements-device.txt`) | RB-C3 |
| HailoRT | the 4.x that `hailo-all` installs: record `hailortcli --version` | RB-D1 |
| Not active | EXP-08 3-class detector (experiment only); gaze and hand-held classifiers (experimental) | stage `pins`; preset limitations |

## 4. Network and authentication

These come from RB-L1. Never write a secret into `$EV`, chat or git.

- **Wi-Fi:** one known network for the device and the phone, plus a second AP or hotspot for RB-K3.
- **Dashboard token:** `dashboard.token`, at least 16 characters. Send it as the header `X-Device-Token: $TOKEN` (a query token is accepted only on `/stream`).
- **Backend:** `cloud.enabled: true`, `cloud.device_key: dk_…`, `cloud.base_url`. Use `https://`, or `ca_cert_path` for a self-signed certificate.
- **Command auth:** `cloud.command_auth` stays `permissive` (the default).
- **BLE provisioning:** stays `legacy`. No key rotation.
- **App:** sign in, claim the device with the claim code, and enter the token under Settings → Live feed.

## 5. Safe camera test setup

- The camera is **stationary**: on a desk rig, or in a **parked car with the engine off**. **Never drive on public roads**, and never run a test on a unit that is protecting a real driver (M5 §1).
- **People:** only consenting adults who agreed beforehand to be filmed. Nobody else in frame, no faces of bystanders, no children.
- **Objects:** use printed targets and objects (the phone sheet, a real phone held by the consenting subject, a cup) rather than images of people.
- **Frames:** the product never writes frames to disk. The stages here use the bundled known-good PNGs, so nothing from the camera is saved. A camera capture (`capture_hailo_outputs.py --camera … --allow-saving-frames`, M5 step 6) needs owner approval, a test scene, and deletion once the worker has a copy.
- **Ears:** the buzzer is loud at 50 cm. Warn the people present before each buzzer step.

## 6. Commands: preflight, stages, end to end

### 6.1 Preflight (Mac, then Pi)

```bash
make verify-release REF=$RC                         # RB-A0: expect RELEASE VERIFICATION: PASS
$VENV/python scripts/hw_validation/hwv_stage.py pins --install-dir .   # expect STAGE pins: PASS
$VENV/python scripts/hw_validation/hwv_stage.py all --fake             # dry run: every stage SIMULATED, exit 3
```
Then on the Pi:
- RB-B (power `0x0`, storage, cooling) and RB-C, RB-D (OS, Hailo `HAILO8L`, camera `imx708_wide_noir`, buzzer low);
- RB-E backups;
- RB-F0 to F4 install;
- then `$PY $V/hwv_stage.py pins` on the device (`--install-dir $D` is the default `/home/pi/sensordyme`; pass `--install-dir $D` if `$U` is not `pi`).

### 6.2 Accelerator stages (one command each)

Every stage prints one `[ok ]` / `[FAIL]` line per check and ends with `STAGE <name>: PASS | FAIL | BLOCKED`. Exit codes: 0 PASS, 1 FAIL, 2 BLOCKED, 3 SIMULATED. Each stage writes a JSON report to `$S` (`--out`). Copy `$S` into `$EV` at the end.

**Service state:**
- **stopped** stages need M5 step 5 first: `sudo systemctl stop sensordyme.service`. The scripts never stop or start the service themselves; they print the command and exit BLOCKED.
- **running** means the service is started again.

| # | Stage | Command (Pi) | Service | Hardware only? | Dry run on the Mac (no hardware) | `--fake` dry run | PASS means |
|---|---|---|---|---|---|---|---|
| 1 | pins | `$PY $V/hwv_stage.py pins --install-dir $D` | any | no | PASS | PASS (nothing simulated) | the three pins agree; decoder `legacy`; production `.hef` is DAMO-YOLO-T |
| 2 | model | `$PY $V/hwv_stage.py model --install-dir $D` | stopped | **yes** | BLOCKED (service state unknown / no `/dev/hailo0`), exit 2 | SIMULATED-PASS, exit 3 | `.hef` sha256 = pin; `core.accel` loads it (integrity check); raw outputs, no NMS; grids 80/40/20, strides 8/16/32, 81 class channels; 640×640 input; 0 = person, 67 = cell phone; one inference completes; `.hef` unchanged |
| 3 | parse-hef | `$PY $V/hwv_stage.py parse-hef --install-dir $D` | any | **yes** (HailoRT tools) | BLOCKED (no `hailortcli`), exit 2 | SIMULATED-PASS on documented text, exit 3 | `hailortcli parse-hef`: compiled for HAILO8L, six outputs `80x80x68, 80x80x81, 40x40x68, 40x40x81, 20x20x68, 20x20x81`, no NMS |
| 4 | capture | `$PY $V/hwv_stage.py capture --install-dir $D --bundle ~/hwval/capture_kg` | stopped | **yes** | BLOCKED, exit 2 | SIMULATED-PASS, exit 3 (input-tensor check real: 6/6) | bundle complete; sha256 before = after = pin; six outputs; class heads in [0,1], box heads are logits; **the uint8 input tensors are bit-identical to the float run's (6/6)** |
| 5 | parity | `$PY $V/hwv_stage.py parity --bundle ~/hwval/capture_kg` (Pi or worker) | any | needs the stage 4 bundle | BLOCKED without a bundle, exit 2 | SIMULATED-PASS, exit 3 | the production decoder equals the offline legacy decoder on the device tensors; every float detection ≥ 0.30 found on INT8 with IoU ≥ 0.80 and \|Δscore\| ≤ 0.10; no INT8 detection ≥ 0.30 without a float counterpart ≥ 0.15; nothing ≥ 0.20 on the flat-grey frame |
| 6 | latency | `$PY $V/hwv_stage.py latency --install-dir $D` | stopped | **yes** | BLOCKED, exit 2 | SIMULATED-PASS, exit 3 | the detector is the Hailo path; `predict()` p95 ≤ the production stride budget (camera 30 fps, `yolo_every_n_frames` 3, so 100 ms); the paced run at that rate keeps up (< 1 % late). Records p50/p95 of preprocess, `hailo_infer` and decode, the host-observed duty cycle, and `hailortcli monitor` text when available |
| 7 | thermal | `SENSORDYME_TOKEN=$TOKEN $PY $V/hwv_stage.py thermal --minutes 30` | **running** | **yes** | BLOCKED (service not running), exit 2 | SIMULATED-PASS on a synthetic CSV, exit 3 | `get_throttled` 0x0 throughout; ARM clock never below its start; fps at the end within 10 % of the start; monitoring `ready`. Records peak SoC °C |
| 8 | buzzer | `$PY $V/hwv_stage.py buzzer --install-dir $D` | stopped | **yes** (+ a person) | BLOCKED, exit 2 | SIMULATED-PASS with a fake GPIO, exit 3 | every production pattern commanded with every pulse, timing within 30 ms. **MANUAL:** a person hears every pattern at 50 cm and 1 m (RB-G2) |

The "Dry run on the Mac" and `--fake` columns were run on 2026-11-13 and are pinned by `tests/test_m83_hw_package.py`. The parity, latency, thermal and buzzer criteria are **provisional**: engineering tolerances set before any hardware run (§7), not product thresholds.

After stage 5, also run the head-level comparison on a worker (M5 step 12): `compare_float_vs_int8.py --bundle capture_kg --onnx damoyolo_tinynasL20_T.zip`. It gives cosine and per-channel LSB error for the person and cell-phone channels.

Restore the service when the stopped stages are done: `sudo systemctl start sensordyme.service`, then `systemctl is-active` should print `active` (M5 step 10).

### 6.3 End-to-end verification (service running; app paired per RB-L1)

| Step | Do | Expected output | Evidence |
|---|---|---|---|
| E1 status | `curl -s -H "X-Device-Token: $TOKEN" localhost:8899/api/health \| $PY -c 'import json,sys;d=json.load(sys.stdin)["device"];print(d["monitoring"]["state"], d.get("monitoring_active"));print(json.dumps(d.get("accelerator"))[:300])'` | `ready True`; `accelerator` = `{"state": "ok", "detail": "running on the Hailo-8 accelerator", "metrics": {"hailo_present": true, "runtime_available": true, "inferences": <rising>, "last_inference_s": <10}}` (`core/device_health.py`). FAIL on "accelerator idle", "running on CPU", "object detection unavailable" or `detector_available: false`. Note: RB-R1's filter on a `components` key prints `[]` because the payload has no such key; the components are top-level (`camera`, `inference`, `accelerator`, `network`, `system`) | `$EV/e2e_health.json` |
| E2 video | in the app open the device: live feed | live video within a few seconds; `/stream` without a token answers 401 (RB-L1) | screenshot |
| E3 status in the app | device screen | the same monitoring state as E1; the Driver Safety capabilities show working (RB-M7 notes) | screenshot |
| E4 drowsiness | the seated, consenting adult faces the camera, looks ahead for 30 s (calibration, RB-R2), then closes their eyes for about 4 s (`drowsy_seconds` 2.0) | the buzzer sounds (critical tone); one `drowsiness` row in `$D/data/events.csv` (`timestamp,type,duration_s,confidence,module,contributing_signals`, module `driver_attention`) | `grep drowsiness $D/data/events.csv \| tail -1 > $EV/e2e_event.txt`; a phone video of the subject and the buzzer |
| E5 app timeline | in the app, the device's activity or timeline | the E4 drowsiness event appears once, with the same time (± clock offset); a push notification if notifications are on | screenshot |
| E6 stop / resume | app: **Stop monitoring** → confirm "Stop monitoring?" → wait 10 s → **Resume monitoring** | journal `[PROTECTION] STOPPED by …` then `[PROTECTION] resumed by …`, each once; while stopped, closing eyes produces **no** alert and the app shows the paused state; after resuming, E4 alerts again | `journalctl -u sensordyme --since "-5 min" \| grep PROTECTION > $EV/e2e_protection.txt` |
| E7 phone is not phone use | the subject holds a phone in view while **looking at the road** for 5 s | **no** `phone_use` alert (a visible phone is logged as `phone_detected`, never alerted). Optional, with the owner's go-ahead: the subject looks down at the phone past `phone_use_seconds` (2.0 s), which should give a `phone_use` alert | `grep -E "phone_(detected\|use)" $D/data/events.csv \| tail -3` |

## 7. Pass / fail

The session **passes** when all of these hold:
- stages 1 to 6 and 8 PASS;
- stage 7 PASS, or BLOCKED with a recorded reason;
- E1 to E7 match the expected outputs;
- RB-K and RB-L pass as written (corrected);
- the RB-P acceptance list is ticked with evidence.

**Stop and diagnose (§9) when:**
- the model or parse-hef stage fails;
- capture's input-tensor check fails;
- parity's "production == offline" check fails (an integrity problem: the device does not run the decoder the repo describes);
- any E-step contradicts its expected output.

**About the provisional parity tolerances** (IoU ≥ 0.80, |Δscore| ≤ 0.10, the ghost rule):
- A parity FAIL on these is a **finding, not a verdict on the product**. Run the head-level worker comparison (M5 step 12) before concluding that the INT8 model differs.
- Do not change a threshold or the decoder on Monday.

## 8. Measurements: what exists and what Monday adds

| Quantity | Known now (labelled) | Monday |
|---|---|---|
| Decode (legacy) incl. NMS, conf 0.20 | x86: p50 1.66 / p95 3.19 ms; reference p50 2.73 ms (1,585 images, `20261113-043857-f182d1`) | Pi: stage 6 `decode` |
| `predict()` preprocess | x86 p50 0.37 ms; Mac p50 0.41 ms | Pi: stage 6 |
| FaceMesh / Hands / module logic | x86: 3.0 / 32.4 / 0.19 ms p50 (`20261113-050231-7108be`); Mac: 4.6 / 37.5 / 0.29 ms p50 | Pi: M5 step 7 `stage_timing.py stages` |
| Hailo inference | **unknown**. The x86 ONNX stand-in (65–70 ms) is not a Hailo number. Phase 5 measured 23.6 ms on a device in 2026-08 (`docs/ACCELERATOR.md`); re-measure | stage 6 `hailo_infer` |
| Accelerator utilisation | unknown | stage 6 duty cycle (+ `hailortcli monitor`) |
| Processed FPS, stalls | unknown | RB-H2 |
| Thermal | unknown | stage 7 / RB-J |
| Camera → alert latency | unknown | M5 step 11 |

## 9. Failure diagnosis

| Symptom | Likely cause | Next step |
|---|---|---|
| `model`: sha256 ≠ pin | wrong or corrupted `.hef` | stop; re-provision with `setup_pi.sh` (it refuses a mismatch); never edit the pin |
| `model`: load fails, "Hailo init failed" | HailoRT and firmware mismatch, the HAT not seated, or the device busy | `hailortcli fw-control identify`; `ls -l /proc/*/fd \| grep hailo0`; RB-D1 |
| `model` / `capture`: `Hailo output … has shape …` or `non-finite` | the device returned a layout or values the decoder refuses (an M8.3 explicit error) | record the message and the HailoRT version; run parse-hef; do not work around it |
| `parse-hef`: NMS present, or not six outputs | a different `.hef` (an on-chip-NMS build caps phone recall) | stop: this is not the pinned model |
| `capture`: input tensors not bit-identical | preprocessing differs on the Pi (OpenCV build or version) | record `cv2.__version__`; compare one sample's input against `expected_float.json` |
| `parity`: production ≠ offline | the installed code is not the reviewed code | `integrity_manifest.py snapshot` and `compare` (RB-F3 / M5 step 12) |
| `parity`: missed or moved boxes, or ghosts | INT8 quantisation effect | worker head-level comparison; record and do not tune |
| `latency`: p95 over budget | CPU contention, thermal throttling, or a slow HailoRT path | check `vcgencmd get_throttled`; rerun once; record both runs |
| `thermal`: throttled bits set | cooling or power | RB-B2; fan; PSU; record |
| E4: no buzzer, but a `drowsiness` row exists | sink or wiring | stage 8, RB-G2, RB-B1 wiring |
| E4: no event | face not found or calibration not done | `/api/health` `device.monitoring` capabilities (`face`, `attention` STARTING?), RB-R2; lighting |
| E5: event missing in the app | uplink | RB-K3 journal `cloud_uplink`; `uplink_backlog` in health (RB-R4) |
| A quoted journal line is missing | `print()` buffering under systemd | wait 30 s and re-grep (RB-K note) |

## 10. Evidence and rollback

**Evidence** (format and rules in RB-N):
- `$EV/LOG.md`, one line per step;
- `$S/*.json`, the stage reports;
- the capture bundle, which goes to the worker; it holds no camera frames, only the bundled PNG inputs;
- `$EV/e2e_*`;
- phone videos and screenshots.

Copy everything to `$REL/evidence-<date>/` on the Mac. Never copy `config.yaml`, tokens or keys.

**Rollback:** RB-O1 (previous release), RB-O2 (pre-Monday state), RB-O3 (SD image). Nothing in this session changes installed software, config, the model or thresholds. The only service actions are the stop and start the operator runs.

## 11. What was corrected in the runbook for Monday (M8.3)

Commit `9f9d0c5`. Every command, endpoint, field and quoted log line in RB-K and RB-L was checked against the code. Corrected:
- **RB-L2:** on this `role: driver_safety` unit, Elderly Monitoring is **refused** (409 `role_forbids_preset`), not switched to.
- **RB-L3:**
  - the app screen is Ask Jupiter;
  - the expected journal lines are `rules: enforcing N AI-configured rule(s)` and `[MONITORING] mode=standard_plus_custom …`;
  - `/api/state` `policy.mode` and `/api/health` `device.monitoring.custom_rules` show the rule;
  - the app labels are Stop/Resume monitoring.
- **M6.5 note:** the link is "All examples".
- **RB-K:** a note that `print()`-based journal lines can arrive late.

Found outside K and L and **not edited** here (another owner's section): RB-R1's `components` filter. Use E1 above instead.
