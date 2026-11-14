# Sunday device activation (M8.4): the first hours with the Pi

> **STATUS: PREPARED, NOT EXECUTED.** Written on 2026-11-13 on a Mac. The device
> (Raspberry Pi 5 + AI HAT+ Hailo-8L + Camera Module 3 Wide NoIR) arrives
> **Sunday 2026-11-15**, one day earlier than the "Monday" in the older documents.
> No line below is a hardware result. Every expected value was read from the code
> at `c9029fa` (+ this stream's scripts). The only things that have run are the
> Mac dry runs in §1.3, and they are labelled SIMULATED / BLOCKED.

This is the plan for the first hours with the device: 25 steps, in order. It
**reuses** and does not repeat:

| Short ref | Document | What it holds |
|---|---|---|
| **RB-x** | `docs/MONDAY_RUNBOOK.md` (sections A–R) | every install, backup (E), autonomy (K), pairing (L), rollback (O) command; the acceptance list (P) |
| **OC** | `docs/MONDAY_OPERATOR_CHECKLIST.md` | the risk/approval/disruption class of each RB step |
| **MHS §x** | `docs/MONDAY_HARDWARE_SESSION.md` | the one-command `hwv_stage.py` stages (§6.2), end-to-end E1–E7 (§6.3), failure diagnosis (§9) |
| **M5 step N** | `docs/HAILO_HARDWARE_VALIDATION.md` | measurement procedures (capture, stage timing, thermal, calibration, buzzer, integrity) |
| | `docs/STATE_MODEL.md`, `docs/FAILURE_MATRIX.md`, `docs/EVENT_PIPELINE.md`, `docs/VIDEO_TRANSPORT.md` | state truth, failure rows, event identity and delivery evidence, stream headers |
| | `docs/BLE_PROOF_OF_POSSESSION.md` "On the hardware (Monday)" | the optional authenticated-onboarding session (same steps, now Sunday or later) |

New in this stream:
- `scripts/hw_validation/sunday_preflight.py`: **the one entry point** for the read-only hardware preflight (§1);
- `scripts/hw_validation/check_live_status.py`: live status vs the state-model invariants, and the stream headers (§4);
- the version/compatibility checklist (§2), the 25 steps (§3), the behavioural cases (§5) and the pilot decision (§6).

---

## 0. Conventions

### 0.1 Variables

Use RB-0 unchanged, with one change: the release candidate is now the M8.4 candidate.

```bash
# Mac
export RC=f63ddd9454f27a387623722fb948140dce61912d   # M8.5 final candidate (feat/m8.5-pre-hardware; adds session renewal and the zero-to-ready README); supersedes 378c189 and the M8.4 candidate 06dbca3
export PI=<user>@<pi-host>  REL=~/sensordyme-release  VENV=$HOME/.venvs/sensordyme-m01/bin
# Pi (RB-0 / RB-C1: discover U and D from the unit first)
export U=<service user>  D=/home/$U/sensordyme  PY=$D/.venv/bin/python
export V=~/hwval/sensordyme_model/scripts/hw_validation      # the package copied from $RC (RB-A4)
export EV=~/m6-evidence/sunday && mkdir -p $EV                # evidence on the Pi
export SENSORDYME_TOKEN="$TOKEN"                              # RB-0 reads $TOKEN from config.yaml; never type it on a command line
```

### 0.2 Result vocabulary (every step records exactly one)

| Result | Means | Never |
|---|---|---|
| **PASS** | observed **on the device**, matching the expected observation, with an evidence file | never from a dry run, a simulator, `--fake`, or a Mac |
| **FAIL** | observed on the device, contradicting the expectation | |
| **BLOCKED** | could not be observed: missing equipment, access, approval, or a prerequisite step failed | not a FAIL and not a PASS |
| **SIMULATED** | the logic ran against stand-ins (`--fake`, the virtual device, recorded sim fixtures) | **never counts as physical validation** |
| **NOT RUN** | deliberately not run (out of scope, not approved, optional) | |

The scripts follow the same rule mechanically: exit codes 0 PASS, 1 FAIL, 2 BLOCKED,
3 SIMULATED; a `--fake` run never exits 0; off a Raspberry Pi the preflight cannot
report a hardware check as PASS.

### 0.3 Evidence

One line per step in `$EV/LOG.md` (RB-N format, with the vocabulary above):

```
<UTC time> | SUN-<step> | PASS/FAIL/BLOCKED/SIMULATED/NOT RUN | <observed value or file> | <decision / follow-up>
```

Files go in `$EV/` with the names given per step (`SUN-<nn>_*.{json,txt,md}`), photos and
videos as `SUN-<nn>_*.{jpg,mov}`. At the end, copy to the Mac (step 24). Never copy
`config.yaml`, tokens, the device key, Wi-Fi PSKs or backups into evidence, chat or git.

### 0.4 Safety (non-negotiable)

- **Stationary only.** A desk rig, or a **parked car, engine off, parking brake on**. **Never drive on public roads**, never test on a unit protecting a real driver.
- People: only **consenting adults** who agreed beforehand (consent text in `docs/research/INCABIN_PILOT_PROTOCOL.md`); nobody else in frame; no children.
- Targets: printed sheets (phone-sized rectangle, `known_good/kg_*.png` printouts), real objects held by the consenting subject. No images of other people.
- Frames: the product never writes frames to disk. Only the checkerboard calibration (RB-I) saves frames, deliberately.
- Ears: the buzzer is loud at 50 cm; announce every buzzer step.
- **Never unplug the camera or the HAT with power on.** Physical faults are induced only by covering the lens or by network/config steps the runbook defines.
- Every **(CHANGE)** step needs the owner's go-ahead at that moment (OC "Appr").

### 0.5 Transfer package (M8.5 PACKAGE)

Everything the Sunday session carries to the device now comes from **one verified
directory**: `make transfer-package REF=$RC` → `~/sensordyme-release/transfer-<id>/`
(docs/DEPLOYMENT.md §10). It holds the release bundle + manifest (the RB-A3 tarball),
the hashed aarch64 wheel set for an offline install (`FIND_LINKS`), the production
`.hef` already verified against `e69136d3…960e`, the config template, install scripts
and generated `INSTALL.md`/`MODELS.md`, with a package-wide `SHA256SUMS`
(`shasum -a 256 -c SHA256SUMS` before leaving the Mac). The §2 rows "Release
candidate", "Model artifact" and "Python packages" are satisfied from this package.

---

## 1. The one entry point: `sunday_preflight.py`

### 1.1 What it does

Runs the read-only checks in this order and writes **one** JSON + Markdown summary
(`sunday_preflight_<UTC>.json/.md`, default `~/hwval/sunday/`) with a result per check,
then prints the **next manual step**:

| Check id | Step | Reads (never writes) |
|---|---|---|
| `identity` | 1 | `/proc/device-tree/model`, `/proc/cpuinfo` (Revision; Serial present, never recorded), `core.identity.device_id()` / `display_name()`, the avahi TXT `id=` |
| `power` | 1 | `vcgencmd get_throttled` (expects `throttled=0x0`), `vcgencmd pmic_read_adc EXT5V_V` |
| `os` | 3 | `/etc/os-release` (bookworm), `uname -rm` (aarch64), `$D/.venv/bin/python --version` (3.11) |
| `packages` | 3 | `dpkg-query -W` (hailo-all, hailort, rpicam-apps, python3-picamera2, …), `hailortcli --version`, every device pin (the hashed lock; simpleaudio is not installed on Linux) vs the install venv |
| `software_revision` | 5 | the package's commit (`HWVAL_COMMIT` or git) vs `--expect-commit` |
| `pins` | 5 | `hwv_stage.py pins` (config / `setup_pi.sh` / `model_registry.yaml` agree; decoder `legacy`) |
| `model_artifact` | 5 | sha256 of `$D/models/damoyolo_tinynasL20_T.hef` vs `accelerator.yolo_hef_sha256` (hash only, no load) |
| `hailo` | 7 | `lspci`, `/dev/hailo0`, `hailortcli scan` |
| `parse_hef` | 8 | `hwv_stage.py parse-hef` |
| `camera_listing` | 6 | `rpicam-hello --list-cameras` (`imx708_wide_noir`; BLOCKED, not FAIL, if the running service holds it) |
| `service` | 4 | `systemctl show sensordyme.service` (active, `NRestarts=0`, `WorkingDirectory` = `--install-dir`) |
| `auth` | 4 | `/api/health`: no token 401, header token 200, `?token=` 401 |
| `health_truth` | 15 | `GET /api/health` + `sim/invariants.report_violations`; ready, accelerator `ok` with `hailo_present`, `config.source=config`, no `saved_state` faults, id = identity, not simulated |
| `perception` | 10 | `device.monitoring.perception`: `sd-perception/1`, not simulated, `object_detector` and `face_landmarks` ok, consumers |
| `stream` | 13 | `GET /stream` for `--stream-seconds` (headers only judged), `GET /stream` without token = 401, `/api/stream/stats` |
| `backend_truth` | 12 | with `--backend` and `$SENSORDYME_BEARER`: `GET /v1/devices` row, `status_violations`, user state, agreement with the device |
| `privacy_stopped` | 22 | only when the device reports `protection: stopped`: Decision C stream check |
| `stage_model`, `stage_capture`, `stage_parity`, `stage_latency`, `stage_buzzer` | 8, 9, 10, 19 | **NOT RUN** unless `--with-stopped-stages` (`--with-buzzer`) **and** a typed `STOP` at a terminal |
| `stage_thermal` | 11 | always NOT RUN here (30 min); the command is printed |
| `config_unchanged` | – | sha256 of `$D/config.yaml` before and after: FAIL if anything changed it |

State changes: only with `--with-stopped-stages` / `--with-buzzer`, and only after the
operator types `STOP` at a terminal (no TTY → refused). It then runs
`sudo systemctl stop sensordyme.service`, the stopped stages (`hwv_stage.py model,
capture, parity, latency [, buzzer]`), and **always** `sudo systemctl start
sensordyme.service` (in a `finally`), recording `service_restored`. It never edits
`config.yaml`, never installs anything, never touches the network.

### 1.2 Usage

```bash
# Pi, any time (read-only):
$PY $V/sunday_preflight.py --install-dir $D --expect-commit <rc sha> --out $EV/preflight
#   + the backend view (step 12):
SENSORDYME_BEARER=<owner bearer> $PY $V/sunday_preflight.py --install-dir $D --expect-commit <rc sha> \
    --backend https://<backend> --out $EV/preflight
# Pi, approved stopped window (CHANGE: monitoring off for ~5-10 min; warn people for the buzzer):
$PY $V/sunday_preflight.py --install-dir $D --expect-commit <rc sha> --with-stopped-stages [--with-buzzer] --out $EV/preflight
# Pi, after "Stop monitoring" in the app (step 22):
$PY $V/sunday_preflight.py --install-dir $D --only privacy --out $EV/preflight
# Mac dry run (never evidence):
$VENV/python scripts/hw_validation/sunday_preflight.py --fake [--with-stopped-stages]
```

Bearer token without shell history: `read -s PW; export SENSORDYME_BEARER=$(curl -s -X POST
https://<backend>/v1/login -H 'content-type: application/json' -d "{\"email\":\"<owner>\",\"password\":\"$PW\"}"
| python3 -c 'import json,sys;print(json.load(sys.stdin)["token"])'); unset PW`.

### 1.3 Dry-run output on the Mac (2026-11-13, `c9029fa` + this stream)

`--fake --with-stopped-stages` (exit **3**, SIMULATED): every check ran against stand-in
answers; `PASS 0, FAIL 0, BLOCKED 0, SIMULATED 21, NOT RUN 2` (backend not given, thermal).
`NEXT: this was a dry run: on the Pi run it without --fake`.

No flags, no hardware (exit **2**, BLOCKED): `PASS 1` (`pins`, a repo check that runs
anywhere), `BLOCKED 15` (every hardware check: "not a Raspberry Pi: … needs the device";
`software_revision` asks for `--expect-commit`), `NOT RUN 7`.
`NEXT: step 1 (identity): run this on the Pi`.

Evidence: `~/sensordyme-audit/2026-11-10/m84-evidence/sunday/` (`dryrun_fake*`,
`dryrun_mac_nohw*`, `live_status_fixture_ready*`). Pinned by `tests/test_m84_sunday_preflight.py`.

Note: `--with-buzzer --fake` can report the buzzer stage SIMULATED-FAIL on a heavily
loaded Mac (one pulse onset 37.6 ms late against the 30 ms provisional tolerance,
load average 32). That is wall-clock jitter in the stand-in, not a defect; on the Pi the
real stage decides.

---

## 2. Version / compatibility checklist (before step 6)

Everything below must be recorded; the left four must **match** the approved candidate.
The compatibility matrix across app / backend / device / model versions is owned by
**PRIVSEC**: `docs/SECURITY_REVIEW_M6.md` §7 (compatibility matrix; each row pinned by
`tests/test_m84_privsec_compat.py`), with camera privacy in `docs/PRIVACY.md` §1. On any
mismatch: stop, record it, and resolve against that matrix with PRIVSEC/the integrator;
never "fix" it on the device by hand.

| Item | Must match | How to check (command) | Expected | On mismatch |
|---|---|---|---|---|
| Release candidate (repo) | `$RC` commit recorded by the integrator | Mac: `git rev-parse $RC`; `make verify-release REF=$RC` (RB-A0) | `RELEASE VERIFICATION: PASS` | do not take it to the device |
| Device agent (installed tree) | `$RC` | Pi: `$PY $V/integrity_manifest.py snapshot --root $D --expect-role driver_safety --out $EV/SUN-05_integrity.json`, Mac: `make release-verify VENV=$VENV RELEASE_DIR=$REL SNAPSHOT=…` (RB-C4/F3); runtime version: `/api/health` → `device.monitoring.versions.software` (`"0.2"`, `services/runtime.py` `_VERSION`) | `VERDICT: PASS against <commit>` | not on the RC → RB-E backups, then RB-F install (CHANGE); FAIL after install → RB-O1 |
| Validation package | `$RC` | preflight `software_revision` (`--expect-commit`) | PASS | re-copy (RB-A4) |
| Model artifact | pin `e69136d3ebf3cc8ebaf3da5417a59962b139f3d3d67d715e9a9b1ae10963960e`, 18,554,430 bytes | preflight `model_artifact` + `pins`; `/api/health` → `device.monitoring.model.detector.sha256`, `versions.model` (`damoyolo_tinynasL20_T@e69136d3ebf3`) | equal | stop; `setup_pi.sh` re-provisions; never edit the pin (MHS §9) |
| Decoder | `accelerator.yolo_decoder: legacy` | preflight `pins` | `legacy` | do not switch on Sunday |
| Model capability schema | perception `sd-perception/1`, health stamp `sd-health-v1` | preflight `perception` / `health_truth` (`stamp.schema`) | equal | an app/backend that does not know the schema shows "unknown": record, coordinate with PRIVSEC |
| Device config | schema-valid, `role: driver_safety`, pins the model, not edited by the session | Mac: `make deploy-preflight VENV=$VENV ROLE=driver_safety` or the RB-F0 manual block; Pi: preflight `config_unchanged`; `/api/health` → `device.config.source == "config"` | `OK to deploy`; unchanged | REFUSED for no `role` → owner adds it deliberately (RB-F0); others: stop |
| Python packages | device pins (hashed `requirements-device.lock`) | preflight `packages` | all pins match | RB-F2 `.venv/bin/python -m pip install --require-hashes --only-binary=:all: -r requirements-device.lock` (CHANGE) |
| HailoRT / firmware | the 4.x `hailo-all` installs | preflight `packages` (`hailortcli --version`); `hailortcli fw-control identify` in the stopped window (RB-F1) | recorded; `HAILO8L` | MHS §9 "Hailo init failed" |
| Backend | the backend built from `$RC` (M8.3+ status contract: `status.protection_command`, `health_freshness`) | `curl -s https://<backend>/v1/health` (`{"ok": true, "service": "sensordyme-cloud", "id": …}`); `check_live_status.py --backend …` shows `freshness` and `protection_command` fields; backend before device (RB-R7) | fields present | an older backend lacks stop/resume outcomes: record; RB-R7 order |
| iOS app | built from `$RC` (or the documented fall-back, `docs/M8_3_RELEASE_DECISION.md`) | the builder records the commit and `CURRENT_PROJECT_VERSION` (`ios/project.yml`; e.g. `CURRENT_PROJECT_VERSION=20261115.0900`); on the Mac, Xcode → Window → Devices and Simulators lists the installed version/build. The app does not report its version to the backend. | commit recorded | an app from another commit: record; the status contract is additive (older apps tolerate new fields) |
| Command auth / BLE | `cloud.command_auth: permissive`, `network.provisioning.auth: legacy` | read-only grep of the device config by the owner (do not paste it) | defaults | leave defaults (RB M6.5 notes) |

---

## 3. The 25 steps

### 3.0 Overview

| # | Step | Command / reference | Expected (short) | HW-only? |
|---|---|---|---|---|
| 1 | Device identity & hardware revision | preflight `identity`, `power`; RB-B1, RB-B2, RB-C2 | Pi 5, one `SD-XXXXXXXX` everywhere, `throttled=0x0` | yes |
| 2 | Camera + accelerator inventory | RB-B1 (photos); preflight `hailo`, `camera_listing`; RB-D1–D3 | HAT seated, Hailo on PCIe, `imx708_wide_noir`, pin 17 low | yes |
| 3 | OS / runtime compatibility | preflight `os`, `packages`; RB-C1, RB-C3, RB-B3; RB-E6 | bookworm aarch64, Python 3.11, pins match, ≥4 GB free | yes |
| 4 | Secure device access | SSH as the owner; preflight `service`, `auth`; RB-E3, RB-E4 | 401/200/401; unit active; config 600 | yes |
| 5 | Software + model artifact revisions | §2; preflight `software_revision`, `pins`, `model_artifact`; RB-C4; (RB-E, RB-F if not on the RC) | installed tree = `$RC`, `.hef` = pin | yes (Mac part: RB-A0–A4) |
| 6 | Camera capture | §4.2 stream probe; `/api/health` `camera`; RB-I orientation | frames flow, age ≤ 2 s, driver upright | yes |
| 7 | Accelerator detection | preflight `hailo`; `hailortcli fw-control identify` (stopped window); `/api/health` `accelerator` | `HAILO8L`; `running on the Hailo-8 accelerator`, inferences rising | yes |
| 8 | Load the deployed candidate model | MHS §6.2 stages `model`, `parse-hef`; RB-G1 | sha = pin, 6 raw outputs, no NMS | yes |
| 9 | Preprocessing / output parity | MHS §6.2 stages `capture`, `parity`; M5 step 12 | inputs bit-identical 6/6; production == offline | yes |
| 10 | FPS + inference latency | stage `latency`; RB-H2; RB-G3 `run_ms`; M5 step 11 | p95 ≤ 100 ms budget; fps/stalls recorded | yes |
| 11 | Thermal | stage `thermal` (30 min); RB-J | never throttled; fps within 10 % | yes |
| 12 | Device/backend sync | RB-L1 (CHANGE); `check_live_status.py --device … --backend … --samples 6` | agree; fresh; same id | yes |
| 13 | Authenticated live streaming | `check_live_status.py --device … --stream-seconds 6`; RB-L1 curls | 401 without token; seq rising; `annotated` | yes |
| 14 | Connect the iOS app | RB-L1 step 3; MHS E2, E3 | LIVE, same state and id as the device | yes (+ iPhone) |
| 15 | Monitoring truth | `check_live_status.py` (device + backend); app screen; RB-M7 notes | no violations; app = backend user state | yes |
| 16 | Stop / resume acknowledgements | MHS E6; `check_live_status.py --samples`; journal | `confirmed` only after the echo; each journal line once | yes |
| 17 | Controlled detection scenarios | §5 (stationary; consenting adult; printed targets) | per §5 | yes (+ person) |
| 18 | Event creation + delivery | `events.csv`, `/api/history`, `GET /v1/devices/{id}/events`; MHS E5 | one row, one stored event, `alert.output` reported | yes |
| 19 | Physical alert output | stage `buzzer`; RB-G2 (a person listens) | every pattern heard at 50 cm and 1 m | yes (+ person) |
| 20 | Network-interruption recovery | RB-K3 (+ RB-R4 optional) | local alert during outage; delivered once after | yes |
| 21 | Safe degradation | lens covered; RB-K2 offline; RB-K4a/b (CHANGE); FAILURE_MATRIX rows | the state says why; never "All clear" when blind | yes |
| 22 | Privacy: camera off while stopped | preflight `--only privacy` / `check_live_status.py --stream-seconds 6` while stopped | no camera picture leaves the device (Decision C) unless owner-enabled | yes |
| 23 | Run the relevant runbook sections | RB map in §3.23 | RB-P items ticked with evidence | yes |
| 24 | Record evidence | §0.3; RB-N | `$EV` complete, copied, hashed | no (needs the files) |
| 25 | Supervised-pilot readiness decision | §6 | GO / GO-WITH-LIMITS / NO-GO with reasons | no (decision on evidence) |

Each step below: **Do** (commands or references), **Expect**, **Result**, **Evidence**,
**If it fails** (diagnosis), **Rollback**.

### 3.1 Step 1: device identity and hardware revision
- **Do:** RB-B1 physical inspection and photos; power on with the official 27 W PSU; then `$PY $V/sunday_preflight.py --install-dir $D --only identity power --out $EV/preflight`. Also RB-C2 by hand (the `grep -o 'id=SD-…'` line) if the preflight's avahi value is empty.
- **Expect:** `board` starts with "Raspberry Pi 5"; `revision` recorded (e.g. `d04170`); `serial_present: true`; `device_id` = `SD-XXXXXXXX` and equal to the avahi `id=`; `display_name` `SensorDyme-XXXX`; `throttled=0x0`, EXT5V about 5.0–5.2 V.
- **Result:** `SUN-01 | ____ |` **Evidence:** `$EV/SUN-01_photos/`, the preflight JSON.
- **If it fails:** no Serial → the id comes from the MAC (finding, RB-C2); a throttle bit → fix power before anything else (RB-B2); a 5 V buzzer on GPIO → do not power on (OC §1).
- **Rollback:** none (read-only).

### 3.2 Step 2: camera and accelerator inventory
- **Do:** RB-B1 (HAT seated, ribbon locks, camera port noted, cooler fan, buzzer on BCM 17 / pin 11); preflight `--only hailo camera`; RB-D3 `pinctrl get 17`.
- **Expect:** `lspci` line "Hailo Technologies Ltd. Hailo-8 AI Processor", `/dev/hailo0`, `hailortcli scan` lists one device; `0 : imx708_wide_noir [4608x2592 …]`; pin 17 output, low.
- **Result:** `SUN-02 | ____ |` **Evidence:** preflight JSON, photos.
- **If it fails:** no PCIe device → power off, reseat, `dtparam=pciex1` (RB-D1); no camera while the service runs → BLOCKED, re-check in the stopped window; none with the service stopped → power off, reseat the CSI ribbon (RB-D2).
- **Rollback:** none.

### 3.3 Step 3: OS / runtime compatibility
- **Do:** preflight `--only os packages`; RB-C1 (unit, user, directory), RB-C3, RB-B3 (storage).
- **Expect:** `VERSION_CODENAME=bookworm`, `aarch64`, venv Python 3.11.x, `hailo-all`/`hailort` present, every device pin (the hashed lock; simpleaudio is not installed on Linux) installed, ≥ 4 GB free, no I/O errors.
- **Result:** `SUN-03 | ____ |` **Evidence:** preflight JSON; `$EV/pip_freeze_before.txt` (RB-C3).
- **If it fails:** not Bookworm 64-bit / filesystem errors → RB-E6 clean-install criteria (decision recorded, owner approval); pin mismatches → RB-F2 (CHANGE, after backups).
- **Rollback:** none (read-only); RB-O for anything installed later.

### 3.4 Step 4: secure device access
- **Do:** SSH as the owner (key-based; agents never SSH). RB-E3 `stat -c '%a %U' $D/config.yaml`, RB-E4 `sudo -l -U $U` (findings only). Preflight `--only service auth`. No Tailscale without separate approval (RB-M).
- **Expect:** unit `active`, `NRestarts=0`, `WorkingDirectory` = `$D`; `/api/health`: no token **401**, `X-Device-Token` **200**, `?token=` **401**; `config.yaml` `600 $U` (anything else is a finding, not fixed on Sunday).
- **Result:** `SUN-04 | ____ |` **Evidence:** preflight JSON; `$EV/SUN-04_access.txt` (mode, sudo rights; no secrets).
- **If it fails:** an empty token with a LAN bind gives 401 everywhere by design (RB-C5); the token is set in RB-L1 (CHANGE).
- **Rollback:** none.

### 3.5 Step 5: software and model artifact revisions
- **Do:** §2 checklist. Preflight `--only revision pins model_artifact --expect-commit <rc sha>`; RB-C4 integrity snapshot and compare on the Mac. If the device is **not** on `$RC`: RB-E1/E2/E5 backups first (nothing is installed before E5), RB-E6 decision, then RB-F0–F4 (CHANGE).
- **Expect:** `software_revision` PASS; `pins` PASS; `model_artifact` "model sha256 matches the pin (e69136d3ebf3…)"; RB-F3 `VERDICT: PASS against <commit>`.
- **Result:** `SUN-05 | ____ |` **Evidence:** `$EV/SUN-05_integrity.json`, preflight JSON, the release-verify output.
- **If it fails:** wrong `.hef` → stop (MHS §9); install drift → RB-F3 explains each failing check.
- **Rollback:** RB-O1 (previous tree) / O2 (E1 backup) / O3 (SD image).

### 3.6 Step 6: camera capture
- **Do:** with the service running and the subject seated: `$PY $V/check_live_status.py --device http://localhost:8899 --stream-seconds 6 --out $EV/SUN-06_stream.json`; `/api/health` → `device.camera` (`state ok`, `detail "capturing"`, `metrics.frame_age_s`) and `device.monitoring.last_frame_age_s`. Look at the live feed (dashboard or the app) for orientation (RB-I).
- **Expect:** frames flow, `X-Frame-Seq` strictly rising, `X-Frame-Age-Ms` ≤ 2000, `camera` ok, frame age well under 1 s; the driver upright and fully in frame. NoIR under daylight looks pink/washed: expected.
- **Result:** `SUN-06 | ____ |` **Evidence:** `$EV/SUN-06_stream.json`, one screenshot of the dashboard (no third parties in frame).
- **If it fails:** `camera` impaired "no new frame for …" → FAILURE_MATRIX row 1 (record timing); upside down → `camera.rotate_180` is a deliberate, recorded config change (RB-I), never made by a script.
- **Rollback:** revert the config change (owner) if made.

### 3.7 Step 7: accelerator detection
- **Do:** preflight `--only hailo health`; in the stopped window (RB-F1 / step 8) `hailortcli fw-control identify | tee $EV/SUN-07_fw_identify.txt`.
- **Expect:** `Device Architecture: HAILO8L` + firmware version; `/api/health` `accelerator` = `{"state": "ok", "detail": "running on the Hailo-8 accelerator", "metrics": {"hailo_present": true, "runtime_available": true, "inferences": <rising>, …}}` (`core/device_health.py`).
- **Result:** `SUN-07 | ____ |` **Evidence:** fw identify text, preflight JSON.
- **If it fails:** "accelerator idle", "running on CPU", `hailo_present: false` → RB-D1; MHS §9.
- **Rollback:** none.

### 3.8 Step 8: load the deployed candidate model
- **Do (stopped window, CHANGE: monitoring off):** `$PY $V/sunday_preflight.py --install-dir $D --expect-commit <sha> --with-stopped-stages --out $EV/preflight` (runs `model`, `capture`, `parity`, `latency` and restarts the service), or the stages by hand per MHS §6.2 (1–3, 6). Then RB-G1 with the service running.
- **Expect:** stage `model` PASS (sha = pin; `core.accel` loads it; raw outputs; grids 80/40/20, strides 8/16/32, 81 class channels; 640×640; one inference); `parse-hef` PASS (HAILO8L, six outputs, no NMS); `service_restored` PASS; RB-G1 `ready True`.
- **Result:** `SUN-08 | ____ |` **Evidence:** `$EV/preflight/stages/model_*.json`, `parse-hef_*.json`.
- **If it fails:** MHS §9 rows "model" / "parse-hef" (do not work around a layout error).
- **Rollback:** nothing is changed by the stages; if the service did not come back, `sudo systemctl start sensordyme.service`, then RB-O1 if it will not stay up.

### 3.9 Step 9: preprocessing / output parity
- **Do:** stages `capture` then `parity` (same stopped window; the preflight passes one bundle to both); worker head-level comparison (M5 step 12).
- **Expect:** bundle `complete`; sha before = after = pin; class heads in [0,1], box heads logits; **input tensors bit-identical to the float run, 6/6**; production decoder == offline legacy decoder; provisional INT8-vs-float criteria (MHS §6.2 row 5).
- **Result:** `SUN-09 | ____ |` **Evidence:** stage JSONs, the bundle (no camera frames: known-good PNGs only), the worker report.
- **If it fails:** inputs differ → record `cv2.__version__` (MHS §9); production ≠ offline → integrity problem (RB-F3); provisional tolerance miss → a **finding, not a verdict**; do not change thresholds or the decoder.
- **Rollback:** none.

### 3.10 Step 10: processed FPS and inference latency
- **Do:** stage `latency` (stopped window); service running: RB-H2 `$PY $V/stage_timing.py health --minutes 10 --interval 5 --out $EV/SUN-10_health.jsonl`; RB-G3 `run_ms`; optional camera-to-alert (M5 step 11, needs the 240 fps phone and `ms_clock.html`).
- **Expect:** `predict()` p95 ≤ 100 ms (30 fps / `yolo_every_n_frames` 3); paced run < 1 % late; `hailo_infer` p50 recorded (Phase 5 measured 23.6 ms: re-measure); processed fps p5/p50/p95 and `stalls_1s_total` recorded; `states_seen` `["ready"]`.
- **Result:** `SUN-10 | ____ |` **Evidence:** latency JSON, health JSONL.
- **If it fails:** check `get_throttled`; rerun once; record both (MHS §9).
- **Rollback:** none.

### 3.11 Step 11: thermal
- **Do (service running, 30 min, closed enclosure if available):** `SENSORDYME_TOKEN=$TOKEN $PY $V/hwv_stage.py thermal --minutes 30 --install-dir $D --out $EV/stages` (= RB-J).
- **Expect:** `throttled` 0x0 throughout; ARM clock never below its start; fps at the end within 10 % of the start; monitoring `ready`; peak SoC °C recorded. Sustained ≥ 80 °C would report `degraded` with a heat reason (M8.3 TRUTH): record, it is the designed behaviour.
- **Result:** `SUN-11 | ____ |` **Evidence:** soak CSV + stage JSON.
- **If it fails:** cooling/power (RB-B2); fix before any later measurement.
- **Rollback:** none.

### 3.12 Step 12: device/backend sync
- **Do:** RB-L1 (CHANGE: token, `cloud.*`, claim) if not paired. Then `SENSORDYME_BEARER=… $PY $V/check_live_status.py --device http://localhost:8899 --backend https://<backend> --samples 6 --interval 5 --out $EV/SUN-12_sync.json` (or preflight `--backend`).
- **Expect:** backend row `hardware_id` = the step 1 id; `online: true`, `freshness.state: fresh`, `user_state: ready`; **no** `status_violations`; device and backend agree on `monitoring_state` (a single-sample disagreement is uplink lag; two in a row is a FAIL); journal `cloud_uplink: connected to …` (a `print()` line: wait 30 s before calling it missing, RB-K note).
- **Result:** `SUN-12 | ____ |` **Evidence:** `$EV/SUN-12_sync.json`.
- **If it fails:** `credentials` block in the row says why a key is refused; `http://` is announced as cleartext (development only); RB-K3 journal.
- **Rollback:** RB-L1 changes are config edits: restore `config.yaml` from E1 (owner) if pairing must be undone.

### 3.13 Step 13: authenticated live streaming
- **Do:** `$PY $V/check_live_status.py --device http://localhost:8899 --stream-seconds 6 --out $EV/SUN-13_stream.json`; RB-L1 step 4 curls; from another LAN machine, `curl -s -o /dev/null -w '%{http_code}\n' http://<pi>:8899/stream` → 401.
- **Expect:** `GET /stream` without token **401**; with the header **200**; response headers `X-Stream-Id`, `X-Stream-Max-Frame-Age-Ms: 5000`; every part carries `X-Frame-Seq` (strictly rising), `X-Frame-Age-Ms` (≤ 2000), `X-Frame-Source: annotated` after the first ~2 s, one `X-Stream-Id`; `/api/stream/stats` `viewers` ≥ 1, `stale_frames_withheld`/`client_send_timeouts` recorded (`docs/VIDEO_TRANSPORT.md`).
- **Result:** `SUN-13 | ____ |` **Evidence:** the JSON.
- **If it fails:** `camera` parts while monitoring runs = "Not analysed" (module not drawing: check `/api/health` reasons); ages > 2 s → CPU load (step 10).
- **Rollback:** none.

### 3.14 Step 14: connect the iOS app
- **Do:** iPhone with the app built from `$RC` (§2 records the build). RB-L1 step 3 (sign in, claim, token under Settings → Live feed); MHS E2, E3.
- **Expect:** live video within a few seconds, badge LIVE (the app requires a new seq with age ≤ 2000 ms in the last 2 s); the device screen shows the same state as step 12's `user_state` and the same device id; Driver Safety capability rows match `/api/health` (RB-M7 notes).
- **Result:** `SUN-14 | ____ |` **Evidence:** screenshots `SUN-14_*.png`, app build commit/number.
- **If it fails:** URLSession per-part behaviour on iOS is untested on hardware (`docs/M8_3_RELEASE_DECISION.md` risks): record exactly what the app shows; MJPEG over the LAN needs the phone on the same network.
- **Rollback:** reinstall the previous app build (`docs/M8_3_RELEASE_DECISION.md` §rollback).

### 3.15 Step 15: monitoring truth
- **Do:** with the subject seated: `check_live_status.py --device … --backend … --samples 6`; compare with the app; during calibration (first ≤ 30 s after a start or `--recalibrate`) sample once more.
- **Expect:** no invariant violations (`docs/STATE_MODEL.md` table); during calibration `attention` is `starting` and the state is not "All clear" before it is `ready` (RB-R2); the app never shows "All clear" unless the backend says `ready` + fresh.
- **Result:** `SUN-15 | ____ |` **Evidence:** JSON + screenshots.
- **If it fails:** any violation is a FAIL of M8.3 TRUTH: save the JSON, the `/api/health` (local only: it contains the SSID) and the journal (RB-N).
- **Rollback:** none.

### 3.16 Step 16: stop / resume acknowledgements
- **Do:** MHS E6 from the app (**Stop monitoring** → confirm → wait 10 s → **Resume monitoring**), with `check_live_status.py --device … --backend … --samples 8 --interval 3 --out $EV/SUN-16_ack.json` running alongside; `journalctl -u sensordyme --since "-5 min" | grep PROTECTION > $EV/SUN-16_protection.txt`.
- **Expect:** `protection_command` `pending` → `confirmed` only with `applied_at` set and `reported_state` stopped (and the reverse for resume); user state ready → stopped → ready (or via resuming/activation_pending) with no `state_transition` finding; analysis capabilities `paused` while stopped; journal `[PROTECTION] STOPPED by …` and `[PROTECTION] resumed by …` exactly once each; while stopped, eye closure gives **no** alert. Optional (owner-approved, AP off): a stop that cannot reach the device times out and is withdrawn (FAILURE_MATRIX row 10).
- **Result:** `SUN-16 | ____ |` **Evidence:** the JSON + journal grep.
- **If it fails:** `ack_before_success` or a stopped state without an applied stop is a FAIL (STATE_MODEL "Stop / resume").
- **Rollback:** Resume from the app; a restart also comes back protecting (stops are not persisted).

### 3.17 Step 17: controlled detection scenarios
- **Do:** §5, in order, with the consenting adult seated in front of the mounted camera, stationary. Calibrate first (look ahead 30 s). Log each trial start/end time in `$EV/SUN-17_trials.csv` (`utc_start,utc_end,case,variant,observed_events,notes`).
- **Expect:** §5 per case.
- **Result:** one line per case `SUN-17.<case> | ____ |`. **Evidence:** `grep -E 'drowsiness|distraction|eyes_down|phone_|attention_lost|frequent_glancing|_risk' $D/data/events.csv | tail -50 > $EV/SUN-17_events.txt`; a phone video of the subject and the buzzer (consent covers it).
- **If it fails:** a missed alert or an unexpected loud alert is a finding with its trial row; thresholds are not tuned on Sunday.
- **Rollback:** none.

### 3.18 Step 18: event creation and delivery
- **Do:** after a step 17 drowsiness trial: `tail -3 $D/data/events.csv` (`timestamp,type,duration_s,confidence,module,contributing_signals`); `curl -s -H "X-Device-Token: $TOKEN" 'localhost:8899/api/history?limit=5'`; backend `curl -s -H "Authorization: Bearer $SENSORDYME_BEARER" https://<backend>/v1/devices/<id>/events?limit=5`; app timeline (MHS E5).
- **Expect:** one CSV row and one dashboard history entry; **one** stored backend event with `event_key`, `occurred_at` + `time_basis`, `simulated: false`, and `alert` = the device's delivery report (`decision: alerted`, `episode: "<boot>-<n>"`, `output: sounded | pending | unconfirmed | failed`) (`docs/EVENT_PIPELINE.md` §4); the app shows it once, at the same time (± clock offset).
- **Result:** `SUN-18 | ____ |` **Evidence:** `$EV/SUN-18_events.json` (backend JSON), CSV tail, screenshot.
- **If it fails:** missing in the app → uplink (RB-K3 journal, `uplink_backlog`); twice → episode uniqueness defect (EVENT_PIPELINE §robustness).
- **Rollback:** none.

### 3.19 Step 19: physical alert output
- **Do:** stage `buzzer` (stopped window: `--with-buzzer`, or `hwv_stage.py buzzer`), then RB-G2: a person listens at 50 cm and 1 m; SPL at 1 m; phone audio recording; `buzzer_check.py analyse` with annotations.
- **Expect:** every production pattern commanded with every pulse within 30 ms (provisional), each **heard and distinguishable**; during step 17 the stored event's `alert.output` is `sounded` (it means "the GPIO pattern was driven", never "someone heard it").
- **Result:** `SUN-19 | ____ |` **Evidence:** buzzer JSON, recording, SPL figures.
- **If it fails:** no sound but `sounded` → wiring / buzzer type (RB-B1); **no audible alert → stop: the product's core claim fails** (OC §4).
- **Rollback:** none.

### 3.20 Step 20: network-interruption recovery
- **Do:** RB-K3 (AP off 3 min, one drowsiness trial during the outage, AP back), recording `$EV/SUN-20_uplink.txt`; optional RB-R4 (outage + power cut, owner-approved). Monitor with `check_live_status.py --backend … --samples 20 --interval 10`.
- **Expect:** the local buzzer still sounds during the outage; backend row goes `offline`/`stale` with everything withheld (never the last "All clear"); `uplink_backlog.queued` ≥ 1 then 0; journal `cloud_uplink: connection lost … device continues standalone`, then `connected to …` within ~30–45 s of Wi-Fi; the outage alert stored **exactly once**.
- **Result:** `SUN-20 | ____ |` **Evidence:** journal grep, the JSON.
- **If it fails:** lost → spool durability (RB-R4); duplicated → episode/event_key defect.
- **Rollback:** restore the AP; nothing on the device changed.

### 3.21 Step 21: safe degradation
Only inductions that are safe and reversible:
- **Lens covered** (hand or card, 5 s, service running): frames keep arriving, so `camera` stays ok; the module loses the face and the person → `attention_lost` after `face_lost_seconds` 2.5 s (critical tone). Record whether `/api/health` stays `ready`: covering is a driver-safety event, not a device fault, at `c9029fa`.
- **Offline monitoring** (RB-K2, CHANGE): local alerts still sound with no Wi-Fi.
- **Damaged config** (RB-K4a/K4b, CHANGE, restore after): boots on the last-known-good, `config.source: last_known_good`, one `config_fault` row; without a saved copy exit 78, no reboot loop.
- **Heat:** not induced. Record only what step 11 shows.
- **Not on Sunday:** unplugging the camera or the HAT (hardware damage risk); Hailo loss (FAILURE_MATRIX row 2 stays simulator-proven).
- **Expect:** in every case the state says why (`reasons`), `monitoring_active` false when the role's required capability is gone, never "All clear" while blind.
- **Result:** `SUN-21.<case> | ____ |` **Evidence:** health JSON per case, journal.
- **If it fails:** a contradiction → step 15 evidence set.
- **Rollback:** uncover; restore Wi-Fi; RB-K4 restore line; RB-O1 on a boot loop.

### 3.22 Step 22: privacy, camera off while stopped (Mission 8.4 Decision C)
Decision C (merged in M8.4; `docs/PRIVACY.md` §1, `docs/VIDEO_TRANSPORT.md` §2a): **by default no camera
picture reaches a REMOTE viewer while monitoring is stopped** (`protection.state == "stopped"`). The owner can
enable a separate **camera-only mode** from the app: Settings → device → Live feed → "Camera only while
stopped" (backend `POST /v1/devices/{id}/camera-mode {camera_only, confirm}`, delivered to the device as a
signed `set_camera_mode` command; reported as `monitoring.camera` on the device and `status.camera_mode` in
`GET /v1/devices`). A viewer ON the device (loopback) may still see `camera_local` frames for maintenance
(`dashboard.local_camera_while_stopped`, default true).
- **Do (remote check, from the Mac on the LAN, NOT from the Pi):** stop monitoring from the app (step 16),
  then on the Mac `python scripts/hw_validation/check_live_status.py --device http://<pi-ip>:8899 --stream-seconds 6 --out $REL/SUN-22_privacy.json`
  (token via `$SENSORDYME_TOKEN`). Then the same from the Pi over loopback for the maintenance case.
- **Expect, camera-only OFF (the default):** remote: no JPEG part at all (no `camera`, `camera_only` or `annotated`);
  a client that asks for heartbeats gets `X-Stream-Status: camera_off_while_stopped`; the response carries
  `X-Stream-Camera: off_while_stopped`; `GET /v1/devices` → `status.camera_mode.camera_only == false`; the app shows
  "CAMERA OFF · Monitoring stopped" with the last picture hidden. Loopback on the Pi: `camera_local` frames
  (allowed; the checker says it cannot prove the remote case from loopback).
- **Then (owner-approved, optional):** enable "Camera only while stopped" in the app (confirmation sheet); wait
  until `status.camera_mode.camera_only == true` and `pending == false` (device-reported); repeat the remote
  check with `--camera-while-stopped`: frames flow labelled `X-Frame-Source: camera_only`, never `annotated` or
  `camera`; the app says "LIVE · Camera only — not monitoring". Turn it OFF again and confirm remote frames stop.
- **Restart persistence:** with camera-only OFF, restart the service and repeat the remote check: still no frames.
- **Result:** `SUN-22 | ____ |` **Evidence:** the JSON (headers only: no frames are saved by the probe).
- **If it fails:** any remote JPEG part while stopped with camera-only OFF is a privacy FAIL: stop the session's remote viewing; record.
- **Rollback:** Resume monitoring.

### 3.23 Step 23: run the relevant existing runbook sections
In this order, skipping what steps 1–22 already covered (cite the step):

| RB section | When on Sunday | Covered by |
|---|---|---|
| A0–A4 (Mac) | before driving to the device | §2 |
| B, C, D | first hour | steps 1–4 |
| E1, E2, E5, E6 | **before any CHANGE** | – (mandatory) |
| F0–F4 | only if the device is not on `$RC` | step 5 |
| G1–G3, R1–R3 | after install | steps 7, 8, 10, 15, 17 |
| H1, H2 | stopped window / running | steps 9, 10 |
| I | with the checkerboard | step 6 (orientation), RB-I calibration |
| J | 30 min | step 11 |
| K1–K4 | after the first evidence copy | steps 20, 21 |
| L1–L3 | after K1 | steps 12, 14, 16 (L2 presets incl. the role refusal, L3 custom rule) |
| K5, R4, R5, R6 | optional, separately approved | – |
| M (Tailscale) | not without separate approval | – |
| O | only if needed | per-step rollback |
| P | end of session | tick only with evidence |

### 3.24 Step 24: record evidence
- **Do:** at the end (and at every break): `cp -r ~/hwval/stages ~/hwval/sunday $EV/ 2>/dev/null; cd $EV && find . -type f ! -name SHA256SUMS -exec sha256sum {} + > SHA256SUMS`; on the Mac `rsync -a $PI:m6-evidence/sunday/ ~/sensordyme-audit/2026-11-10/m84-evidence/sunday/hardware-<date>/` (exclude frame-bearing directories; the capture bundle holds only known-good PNG inputs). Verify with `shasum -a 256 -c SHA256SUMS` on the Mac.
- **Expect:** `LOG.md` has one line per step 1–25 with a result; every PASS names a file; no secrets (`grep -rE 'dk_|token|psk' $EV` finds nothing sensitive).
- **Result:** `SUN-24 | ____ |`.
- **Rollback:** n/a.

### 3.25 Step 25: supervised-pilot readiness decision
See §6. Record `SUN-25 | GO / GO-WITH-LIMITS / NO-GO | <criteria table> | <owner>`.

---

## 4. Camera / stream diagnostics and state-truth commands

```bash
# 4.1 Device truth: the health sample the uplink sends, checked against sim/invariants.report_violations
$PY $V/check_live_status.py --device http://localhost:8899 --out $EV/truth_$(date +%s).json
# 4.2 Video evidence: /stream part headers (X-Frame-Seq / X-Frame-Age-Ms / X-Frame-Source / X-Stream-Id),
#     401 without a token, /api/stream/stats. Nothing is saved but headers and sizes.
$PY $V/check_live_status.py --device http://localhost:8899 --stream-seconds 6
# 4.3 What the app is told: GET /v1/devices row, status_violations, state_model.derive, transitions across samples,
#     and device/backend agreement
SENSORDYME_BEARER=... $PY $V/check_live_status.py --device http://localhost:8899 --backend https://<backend> --samples 6
# 4.4 Re-check recorded evidence offline (Mac)
$VENV/python scripts/hw_validation/check_live_status.py --health-file e2e_health.json --devices-file devices.json
# 4.5 Raw headers by hand (no script)
curl -sN -H "X-Device-Token: $TOKEN" --max-time 3 localhost:8899/stream | grep -a -E '^X-(Frame|Stream)-' | head -20
curl -s -H "X-Device-Token: $TOKEN" localhost:8899/api/stream/stats | python3 -m json.tool
```

Reading the results:

| Observation | Meaning | Next |
|---|---|---|
| `X-Frame-Seq` repeats or stops, `X-Frame-Age-Ms` grows | nothing new captured; with heartbeats (`X-Stream-Status: camera_stale`) the camera itself stalled | `/api/health` `camera`; FAILURE_MATRIX row 1 |
| `X-Frame-Source: camera` while monitoring runs | video is live but not analysed (module degraded / not drawing) | `/api/health` reasons; the app shows "Not analysed" |
| `X-Frame-Source: camera` while stopped | the M8.3 camera fallback: violates Decision C unless owner-enabled | step 22 |
| `X-Stream-Id` changes between runs | the dashboard process restarted | journal; `NRestarts` |
| `stale_frames_withheld` rising | frames older than 5 s were not sent (correct) | CPU/thermal (steps 10, 11) |
| a `stale_definition` / `stale_withholds` violation | the backend claims more than its evidence | step 15 evidence set |
| `device_backend_agree` | device and backend disagree in two consecutive samples | uplink (step 12, RB-K3) |
| verdict SIMULATED | the data is from the virtual device (`simulated: true` or an `SD-SIM-` id) | it is not evidence: you are not talking to the Pi |

The checker is GET-only, reads tokens from `$SENSORDYME_TOKEN` / `$SENSORDYME_BEARER`, never
writes them, and keeps selected fields only (no SSID, no full health dump). Tests:
`tests/test_m84_sunday_live_status.py` (fixtures recorded from the virtual device in
`tests/fixtures_m84_sunday/`, and a loopback `DashboardServer`).

---

## 5. Safe behavioural test cases (stationary; step 17)

Setup: camera mounted at driver-face height (desk rig or parked car, engine off); the
consenting adult seated; preset Driver Safety **Standard**; calibrate first (look at the
"road" ~30 s; `attention` `starting` → `ready`). Expectations are those of the **current
software** (`modules/driver_attention.py`, `config.yaml` `driver_attention`, `presets.yaml`
`driver_safety`, `main.py` `ALERT_EVENT_TYPES_PI`), with `behavior_v2: false`. Validated
alert set for this module: drowsiness, distraction, eyes_down, eyes_up, phone_use,
attention_lost (loud/critical), fatigue_warning, frequent_glancing (soft); `elevated_risk`,
`high_risk`, `erratic_hand_movement` sound today but are **experimental** (presets
limitation). `phone_detected` and `eating_drinking` are log-only.

Expectations are for the M8.4 candidate with ALPHA merged (Decision A: experimental alert families are
logged observations, never audible; Decision B: a visible phone contributes no risk). The shadow behaviour
engine (`docs/BEHAVIOR_ENGINE.md`) runs alongside and never alerts; record its abstentions from
`/api/health` telemetry (`engine_*`) as findings.

| # | Case (how) | Expected (M8.4 candidate) | Must NOT happen | Notes |
|---|---|---|---|---|
| B1 | **Eyes closed 3 s**, head level, face visible | `drowsiness` after ~2.0 s of continuous closure (`drowsy_seconds` 2.0, + ~0.3 s median lag), critical tone, one CSV row; `elevated_risk` may be LOGGED as an experimental observation (`experimental: true`, severity `log`): never a sound or push (M8.4 Decision A) | blinks or a 1 s closure alerting; any audible `elevated_risk`/`high_risk` | Decision A merged (`alerts.experimental_alerts_audible: false`) |
| B2 | **Sustained look-away**: head turned ~40° to the side (passenger), held 4 s | `distraction` at ~3.0 s (`distraction_seconds`; yaw > 35° calibrated), loud | an alert under 3 s | a turn beyond 55° gets `rearview_tolerance_seconds` 5.0 s; face lost mid-turn with the body visible coasts the turn, `attention_lost` waits 8 s |
| B3 | **Phone mounted, unused**: phone in a dash/vent mount, screen on, eyes on the road, both hands visible on the wheel, 60 s; then (B3b) a 4 s look down at the mounted phone, once right after placing it and once with hands out of view | at most one `phone_detected` (log-only, rising edge); **no `phone_use`**; no loud alert while the eyes are on the road. B3b: **`eyes_down` at ~2.5 s** (or `distraction` at ~3.0 s for a sideways look), loud; **no `phone_use`** | `phone_use` at any point (M8.5); silence through a ≥ 3 s look | M8.5 (preset v5): a mounted/unheld phone is not handled, so a look at it is judged by eyes_down/distraction only, whether or not hand tracking has latched it "set down" and whether or not hands are seen. If the device runs with `--debug`, record the `phone_handling` line (level, `handling_cue`) of the once-a-second `DRIVER_ATTENTION STATE` block during B3b. Finding, not defect: `phone_use` here means the mount's box moved ≥ 10 % of its diagonal (vibration, detector jitter) or the mount sits directly below the face in the image (read as the lap zone). A larger passenger's phone is not the driver's |
| B4 | **Phone held + eyes off road**: subject holds the phone in the lap and looks down at it 4 s; repeat with the hand hidden from the camera | `phone_use` at ~2.0 s (`phone_use_seconds`), loud, detail "hand overlapping the phone (experimental hand tracking)", or with no hand seen "phone visible; no hand confirmed on it; possible handling: phone in lap zone / phone moving (unvalidated phone-handling evidence)"; `eyes_down` at 2.5 s (pitch < −20°); `elevated_risk`/`high_risk` only as logged experimental observations, never audible (Decision A) | no alert at all through the 4 s look | owner go-ahead (MHS E7). With no hand seen and the phone held still away from below the face, M8.5 gives `eyes_down` only (stated limitation): record which |
| B5 | **Mirror glance**: a 1 s look at the rear-view or side mirror, back to the road; repeat 3× spaced out | nothing | `distraction` | 5 glances past 35° within 45 s → `frequent_glancing` (soft) is designed behaviour: record if it fires |
| B6 | **Sunglasses** (IR-pass and IR-block if available), eyes open 30 s, then closed 3 s | **unvalidated** (presets: "sunglasses and night driving are untested"). Record: face found? false `drowsiness` with eyes open? closure detected? If the face is lost and no person is detected → `attention_lost` after 2.5 s | a false `drowsiness` with eyes open is a finding | the iris veto only suppresses false closures; `behavior_v2` occlusion handling is off |
| B7 | **Hands out of view** (hands in the lap, below the frame), eyes on the road, 30 s | no alert; hand evidence "unknown" | any alert | M8.5: with a phone visible and eyes off the road, `phone_use` fires only on a handling cue (hand seen, phone moving, at the face, or in the lap with the gaze down); an unhandled phone gives `eyes_down`/`distraction` by their own rules |
| B8 | **Camera blocked**: card over the lens 5 s (do not unplug) | `attention_lost` after ~2.5 s (face and person gone), critical tone; `camera` stays ok (frames still arrive) | "All clear" with a black picture **and no alert** | record `/api/health` state during the block (step 21) |

Each case: record start/end UTC, events, tone heard (yes/no), and the `/api/health` state.
A deviation from this table is a finding, not a reason to change thresholds on Sunday.

---

## 6. Supervised-pilot readiness decision (step 25)

"Supervised pilot" here means the **stationary in-cabin pilot** in
`docs/research/INCABIN_PILOT_PROTOCOL.md` (parked vehicle, consenting adults, an operator
present). It is **never** on-road use. Decision options:

| Decision | All of these hold |
|---|---|
| **GO** | (1) steps 1–5 PASS (identity, inventory, OS, access, revisions = `$RC`, `.hef` = pin); (2) steps 7–9 PASS (model loads; inputs bit-identical; production == offline); (3) step 10 PASS (latency within budget) and step 11 PASS (no throttling); (4) steps 12–16 PASS with **zero** invariant violations; (5) step 17: B1 drowsiness, B2 distraction, B4 phone use and B8 attention_lost each fire as expected in ≥ 3 of 3 trials, and **no loud false alert** in B3, B5, B7; (6) step 18 PASS (one stored event per alert) and step 19 PASS (every pattern heard); (7) step 20 PASS (local alert during the outage, delivered once); (8) step 21: no case shows "All clear" while blind; (9) step 22 PASS (remote check from the Mac); (10) the pilot protocol, consent form and data path are approved in writing (protocol §0) |
| **GO-WITH-LIMITS** | (1)–(4), (6), (8) and (10) hold; a non-safety item is BLOCKED or FAIL with a written limit (e.g. step 14 app feed unverified → the operator uses the device dashboard; B6 sunglasses unvalidated → excluded from the pilot script; parity tolerance miss explained by the worker report) |
| **NO-GO** | any of: no audible alert (step 19); drowsiness not detected (B1); a state contradiction (steps 15/16); the model not the pin or not loading (steps 5, 8); production ≠ offline decoder (step 9); throttling under load (step 11); an alert lost or duplicated (steps 18, 20); camera video leaving the device while stopped with Decision C merged and the setting off (step 22); the protocol not approved |

A dry run, the virtual device or `--fake` output never satisfies any criterion. The owner
signs the decision line in `$EV/LOG.md`.

---

## 7. TODO-INTEGRATOR

1. ~~Record the candidate commit~~ (M8.5 final: `f63ddd9`; Python 3,956 passed / 0 failed; `make acceptance-full`: every software suite and all 30 owner journeys PASS (simulated), XCUITest 26/28 in the full run with the 2 overload casualties (test21, test24) passing on a quiet rerun; verify-release PASS; use `--expect-commit f63ddd9454f27a387623722fb948140dce61912d`; transfer package `~/sensordyme-release/transfer-f63ddd9454f2`).
2. ~~Step 22: PRIVSEC's names~~ (done: `status.camera_mode`, `POST /v1/devices/{id}/camera-mode`, `camera_only` / `camera_local` labels; `check_live_status.py` judges them).
3. ~~§2: link PRIVSEC's matrix~~ (done: `docs/SECURITY_REVIEW_M6.md` §7).
4. ~~§5 rows B1, B3, B4~~ (done: re-read against ALPHA's merge; Decisions A and B; the mounted-phone `phone_use` look is an open owner decision).
5. ~~BETA model change~~ (none: EXP-09 rejected; production `.hef` and `legacy` decoder unchanged).
6. ~~Monday dates~~ (done: FAILURE_MATRIX now points at the Sunday session).
