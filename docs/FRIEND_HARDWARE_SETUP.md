# SensorDyme physical setup: the technician guide

**For:** a technically capable person who has the Raspberry Pi and has **not** followed
SensorDyme's development. **From:** the owner. **Status:** written 2026-11-14 on a Mac.
**Nothing in this guide has run on this hardware yet.** Every "Expect" value comes from the
code and the earlier runbooks. Your results are the first real evidence.

This guide is the one you follow, in order. It points to the detailed runbooks only where
they hold the full procedure:
- `docs/SUNDAY_DEVICE_ACTIVATION.md` (the 25-step session);
- `docs/MONDAY_RUNBOOK.md`, cited as **RB-x** (install, backup, rollback);
- the generated `INSTALL.md` inside the transfer package (the exact commands for this release).

Every command here exists in the repository, or is a standard Raspberry Pi OS command.

---

## 0. Before you start

### 0.1 Who does what

| Person | Has | Does |
|---|---|---|
| **You (technician)** | the Pi, the AI HAT+, the camera, the cooler, the PSU, an SD card, the buzzer | the physical work and the commands below; you share sanitized evidence |
| **Owner** | the MacBook, the repository, the backend, the iPhone with the app | approves every **CHANGE** step, builds and sends the release package, does the backend and iPhone steps (22–25) |

### 0.2 Rules (please read them all)

1. **Stationary only.** Desk rig, or a parked car with the engine off and the parking brake on. Never test while driving, never on a public road.
2. **Power off before touching any ribbon, the HAT, the SD card or the buzzer wiring.** Never rewire powered hardware.
3. **Read-only first.** Steps marked **Modifies the Pi: no** are always safe. Steps marked **yes** need the owner's go-ahead **at that moment**: a message that says "go, step N".
4. **Never send in chat, email or a ticket:**
   - passwords;
   - the dashboard token;
   - the device key (`dk_…`);
   - the claim code or the setup code;
   - Wi-Fi names or passwords;
   - `config.yaml`, the backup archives, or camera photos of people.

   Send the evidence bundle from §0.5 and the photos of the **hardware**.
5. **Never on your own initiative:**
   - flash or reformat the SD card;
   - overwrite or rename the `.hef` model;
   - change `accelerator.yolo_decoder` or any threshold;
   - rotate keys;
   - edit network settings;
   - install Tailscale;
   - run `scripts/demo_cycle.sh` (it refuses on this device by design).
6. **People on camera:** only consenting adults who agreed beforehand. No one else in frame. The product never stores video. Tests in this guide record no video unless a step says so.
7. **If anything fails and you do not know why, stop.** Record what you saw (§0.5) and message the owner.

### 0.3 The result vocabulary (record exactly one per step)

| Result | Means |
|---|---|
| **PASS** | observed on the device, matching **Expect**, with the evidence captured |
| **FAIL** | observed on the device, contradicting **Expect** |
| **BLOCKED** | could not be observed (missing part, access, approval, or an earlier step failed) |
| **NOT RUN** | deliberately skipped |

The tools follow the same rule: exit code 0 PASS, 1 FAIL, 2 BLOCKED, 3 SIMULATED. A
"SIMULATED" result is never hardware evidence.

### 0.4 Shell variables (set once per terminal on the Pi)

Do not guess the user name. Step 6 finds it.

```bash
export U=<service user from step 6>  D=/home/$U/sensordyme  PY=$D/.venv/bin/python
export V=~/hwval/sensordyme_model/scripts/hw_validation     # created in step 9
export EV=~/hwval/evidence && mkdir -p $EV
# the dashboard token, read from the device's own config into the environment (never typed, never printed):
export SENSORDYME_TOKEN="$(sudo -u $U $PY -c "import yaml;print((yaml.safe_load(open('$D/config.yaml')).get('dashboard') or {}).get('token') or '')")"
```

### 0.5 Sharing evidence: one command

Run this any time the service is running, and at the end of every session:

```bash
$PY $V/collect_evidence.py --install-dir $D --expect-commit <RC from the owner> --minutes 2
```

**What it does:**
- It is read-only and changes nothing.
- It runs the preflight checks.
- It samples for 2 minutes: fps, per-stage latency, CPU, memory, temperature, throttling and the monitoring state.
- It collects every earlier stage result under `~/hwval`.

**What it writes:** a report built from allow-listed fields only. It scans for and redacts:
- tokens and keys;
- Wi-Fi details, e-mail, IP and MAC addresses;
- codes and the board serial.

It refuses to write anything if a secret survives. No frames, video or audio are included.

**Output:** `~/hwval/evidence/evidence_<device-id>_<UTC>.tgz`, with its sha256 printed.
- Send the `.tgz` through the private channel the owner names.
- Send **only the sha256** in chat.
- Then add one line per step to `$EV/LOG.md`, in this format:

```
<UTC time> | step <N> | PASS/FAIL/BLOCKED/NOT RUN | <observed value or file name> | <note>
```

### 0.6 What needs you at the Pi, and what can be remote

| Task | Where |
|---|---|
| Photos, ribbons, HAT, SD card, power supply, buzzer wiring, camera aim | **You, at the Pi** (power off for every physical change) |
| Hearing the buzzer; controlled event tests with a person | **You, at the Pi** (plus the participant) |
| Power-cycle / unplug tests; SD imaging | **You, at the Pi** |
| BLE Wi-Fi setup from the iPhone | **The iPhone must be next to the Pi** (Bluetooth range) |
| Live video in the app | the iPhone on the **same Wi-Fi** as the Pi (LAN only, by design) |
| Preflight, evidence bundle, timing, thermal, parity | anyone with a shell on the Pi. Remote shell access is only over owner-approved SSH (or Tailscale, approved separately; never needed by the product) |
| Pairing with the owner's account | the Pi must reach the owner's backend. Today the backend runs on the owner's MacBook (home network). **No hosted backend is deployed.** So steps 22–25 and 27 need the Pi on the owner's network, or an owner decision (see step 22) |

---

## Part A — Hardware (power OFF throughout until step 5)

### Step 1. Hardware inventory
| | |
|---|---|
| **Do** | Lay out and photograph each part. **Expected parts:** (1) Raspberry Pi 5; (2) Raspberry Pi AI HAT+ with Hailo-8L (13 TOPS); (3) Camera Module 3 **Wide NoIR** plus the **Pi 5 camera cable (22-pin to 15-pin)**, because the camera's own 15-pin cable does not fit a Pi 5; (4) the official Active Cooler; (5) the official 27 W USB-C PSU (5.1 V / 5 A) for bench work; (6) a microSD (high-endurance) or the existing card/SSD; (7) the buzzer and two female-female jumper wires, if present. |
| **Access** | physical |
| **Expect** | every part present; the HAT kit includes its GPIO stacking header and standoffs |
| **Capture** | `step01_parts.jpg`; close-ups of the AI HAT+ (chip marking), the camera board label, the PSU rating label and the buzzer's markings |
| **Pass / fail** | PASS: all present. BLOCKED: a part is missing (say which) |
| **Common failure** | the 15-pin camera cable only; a 15 W phone charger instead of the 27 W PSU |
| **Recovery** | stop; tell the owner what is missing. Do not substitute a lower-rated supply |
| **Modifies the Pi** | no |
| **Owner approval** | no |

### Step 2. Electrical safety checks
| | |
|---|---|
| **Do** | (a) Read the PSU label: it must say **5V⎓5A**, or it is the official 27 W PSU. (b) Read the buzzer: is it an **active** buzzer (it beeps on DC) and **rated for 3.3 V logic**? Photograph its markings and datasheet if you have one. (c) Check the GPIO pins for bends. |
| **Access** | physical, power OFF |
| **Expect** | PSU 5 V / 5 A; buzzer active and 3.3 V-rated (README Phase 1 item 7) |
| **Capture** | `step02_psu_label.jpg`, `step02_buzzer_markings.jpg` |
| **Pass / fail** | PASS: both as expected. **STOP (do not power on with the buzzer connected):** the buzzer is 5 V-only, passive, a speaker or relay, or unmarked |
| **Common failure** | an unmarked 5 V buzzer from a kit |
| **Recovery** | leave the buzzer **disconnected** and continue without it. A 5 V or higher-current part needs a transistor driver off the 5 V rail. That is the owner's design decision; do not improvise one |
| **Modifies the Pi** | no |
| **Owner approval** | yes, to connect any buzzer whose rating is not clearly 3.3 V-compatible |

### Step 3. Camera ribbon installation
| | |
|---|---|
| **Do** | (1) Put the 22-pin end of the Pi 5 camera cable into one of the Pi 5's two camera connectors; either works, and the software finds it. Note which (CAM/DISP 0 or 1). (2) Put the 15-pin end into the camera. (3) At each end: lift the latch, seat the ribbon fully and squarely, close the latch. **Orientation:** follow Raspberry Pi's official Camera Module 3 / Pi 5 cable guide, or the diagram shipped with the cable. This repository does not pin the per-connector orientation (README Phase 1 step 3). |
| **Access** | physical, power OFF |
| **Expect** | both latches closed; ribbon straight and fully inserted |
| **Capture** | `step03_ribbon_pi.jpg`, `step03_ribbon_camera.jpg` (both ends, latches visible) |
| **Pass / fail** | PASS: seated per the official diagram. The camera is proven later, in step 14 |
| **Common failure** | ribbon reversed or half-seated, which shows as "no cameras" in step 14 |
| **Recovery** | power off, reseat, re-photograph |
| **Modifies the Pi** | no (physical assembly) |
| **Owner approval** | no |

### Step 4. Hailo (AI HAT+) board installation
| | |
|---|---|
| **Do** | (1) Fit the Active Cooler first, with its fan lead in the Pi's FAN connector. (2) Mount the AI HAT+ on the GPIO stacking header with its standoffs. (3) Connect the HAT's **PCIe FFC ribbon** to the Pi's PCIe connector, **locked at both ends**. (4) Confirm the HAT sits flat and that physical pins 9 and 11 are still reachable for the buzzer (README Phase 1 step 2). |
| **Access** | physical, power OFF |
| **Expect** | HAT flat on its standoffs; PCIe ribbon locked at both ends; no bent pins |
| **Capture** | `step04_hat_top.jpg`, `step04_pcie_ribbon.jpg`, `step04_cooler_fan_lead.jpg` |
| **Pass / fail** | PASS: as expected. The accelerator is proven later, in step 15 |
| **Common failure** | the standoffs too short above the cooler; the PCIe ribbon not locked |
| **Recovery** | if the stacking does not fit above the cooler, stop and tell the owner (README Phase 1 step 2 warning) |
| **Modifies the Pi** | no (physical assembly) |
| **Owner approval** | no |

**Buzzer wiring (only if step 2 passed):** positive lead to **GPIO 17 = physical pin 11**,
negative lead to **ground, physical pin 9**. This matches `gpio_buzzer_pin: 17` in
`config.yaml`. Photograph it as `step04_buzzer_wiring.jpg`.

### Step 5. Power supply and cooling (first power-on)
| | |
|---|---|
| **Do** | Insert the SD card. Connect the 27 W PSU. After boot, in a terminal (keyboard and monitor, or SSH if the card already has it): `vcgencmd get_throttled` · `vcgencmd pmic_read_adc EXT5V_V` · `vcgencmd measure_temp` · `cat /sys/devices/platform/cooling_fan/hwmon/*/fan1_input 2>/dev/null` (RB-B2/B3) |
| **Access** | physical, then terminal |
| **Expect** | `throttled=0x0`; EXT5V about 5.0–5.2 V; idle temperature under ~60 °C; a fan reading of 0 rpm at idle is normal |
| **Capture** | the four outputs in `$EV/LOG.md` (they contain no secrets) |
| **Pass / fail** | FAIL: any throttle bit set (`0x1` under-voltage now, `0x10000` since boot) |
| **Common failure** | a weak supply or cable |
| **Recovery** | power off; use the official 27 W PSU. Fix power before any later measurement |
| **Modifies the Pi** | no |
| **Owner approval** | no |

## Part B — What is on the Pi, and backups

### Step 6. Existing OS inspection (do NOT reflash a card you have not looked at)
| | |
|---|---|
| **Do** | Read-only: `uname -m` · `cat /etc/os-release` · `free -h` · `df -h` · `ls ~/sensordyme 2>/dev/null && echo "existing install present"` · `ls ~/sensordyme.releases 2>/dev/null` · `systemctl show sensordyme.service -p User -p WorkingDirectory -p ActiveState -p NRestarts` · `cat /proc/device-tree/model; echo`. Set `U` and `D` (§0.4) **from `User=` and `WorkingDirectory=`**. If there is no service, `U=$(id -un)` and `D=~/sensordyme` (the fresh-install layout). |
| **Access** | terminal |
| **Expect** | `aarch64`; `VERSION_CODENAME=bookworm` (Debian 12); "Raspberry Pi 5 Model B…"; at least 4 GB free |
| **Capture** | the outputs in `LOG.md`, **minus the hostname** if it is personal |
| **Pass / fail** | PASS: recorded. Note "existing install: yes/no" |
| **Common failure** | a 32-bit or non-Bookworm OS (this release needs Bookworm 64-bit, Python 3.11 and HailoRT 4.x) |
| **Recovery** | do not change anything: step 8 decides |
| **Modifies the Pi** | no |
| **Owner approval** | no |

### Step 7. Data and device-identity backup (before ANY change)
| | |
|---|---|
| **Do** | **Only if step 6 found an existing install.** (a) Record the identity, read-only (RB-C2): `cd $D && sudo -u $U $PY -c "from core.identity import device_id, display_name; print(device_id(), display_name())"`. (b) Back up the tree, config, system glue and model hashes exactly as **RB-E1** (the block starting `B=~/m6-backup-$(date +%Y%m%dT%H%M%S)`), then copy `$B` to a USB stick and verify `sha256sum -c SHA256SUMS` there. (c) Optional full SD image, **RB-E5**: power-off plus imaging on a computer, owner-approved. |
| **Access** | terminal; USB stick |
| **Expect** | the archive lists `config.yaml`, the `.hef` and `data/`; checksums verify on the stick |
| **Capture** | the device id (`SD-XXXXXXXX`, not secret) and `model_hashes.txt` in `$EV`. **Never send the backup itself: it contains the device key, the token and Wi-Fi passwords.** |
| **Pass / fail** | PASS: backup verified off the Pi |
| **Common failure** | not enough space (RB-B3: at least 4 GB free) |
| **Recovery** | free space by moving files off the Pi (ask the owner first), never by deleting `data/` |
| **Modifies the Pi** | writes only the backup folder (E1); E5 powers off |
| **Owner approval** | E1: no (it only adds a folder). E5: yes |

### Step 8. Fresh-install vs upgrade decision
| | |
|---|---|
| **Do** | Apply the **RB-E6** criteria and write the decision with reasons in `LOG.md`. **Clean install only if one of these holds:** filesystem/I/O errors; the OS is not Bookworm 64-bit or `hailo-all` cannot be installed; an unidentifiable tree with a complete backup; unexplained users, keys or services; a service user of root or a broken venv. **Otherwise: upgrade in place. A blank card means a fresh install.** |
| **Access** | — |
| **Expect** | one of: FRESH (blank card) · UPGRADE (existing, healthy) · CLEAN-REINSTALL (criterion met) |
| **Capture** | the decision line |
| **Pass / fail** | the owner confirms the decision |
| **Common failure** | reflashing "to be safe", which loses the device's credentials and history |
| **Recovery** | — |
| **Modifies the Pi** | no |
| **Owner approval** | **yes** (always) |

**Fresh install (blank card), on a computer:**
1. In **Raspberry Pi Imager**, choose **Raspberry Pi OS (64-bit), Bookworm**. If the Imager's default is a newer release, pick the Bookworm 64-bit image instead. If Bookworm is not offered, stop and tell the owner: the wheels target Bookworm, Python 3.11.
2. In the Imager settings, set:
   - a hostname you recognise;
   - **SSH with a public key** (not a password);
   - your Wi-Fi and locale (README Phase 2).

## Part C — Install the approved release

### Step 9. Verified release-package transfer
| | |
|---|---|
| **Do** | **Owner (Mac):** `make verify-release REF=$RC` (must print `RELEASE VERIFICATION: PASS`), then `make transfer-package REF=$RC`, which creates `~/sensordyme-release/transfer-<id>/` where `<id>` is the first 12 characters of the commit. The owner sends you the directory (USB drive or a private share; the build refuses to package secrets) and, **separately in chat**, the commit `$RC` and the sha256 of the package's `SHA256SUMS` file. **You (Pi), in the copied directory `~/transfer`:** `sha256sum SHA256SUMS` (must equal the owner's value) · `sha256sum -c SHA256SUMS` (every line `OK`). Then the validation tools, in their own directory (RB-A4 equivalent): `mkdir -p ~/hwval/sensordyme_model && tar -xzf ~/transfer/bundle/sensordyme-<id>.tar.gz -C ~/hwval/sensordyme_model && echo <RC> > ~/hwval/sensordyme_model/HWVAL_COMMIT` |
| **Access** | file copy; terminal |
| **Expect** | both checks pass; `~/hwval/sensordyme_model/scripts/hw_validation/` exists |
| **Capture** | the two check outputs (no secrets) |
| **Pass / fail** | FAIL: any hash differs. Do not install |
| **Common failure** | a partial copy, a different release directory |
| **Recovery** | re-copy the directory; never "fix" a hash |
| **Modifies the Pi** | writes `~/transfer` and `~/hwval` only |
| **Owner approval** | no (nothing is installed yet) |

### Step 10. Checksums and model hashes
| | |
|---|---|
| **Do** | `sha256sum ~/transfer/models/*.hef` · read `~/transfer/INSTALL.md` "Contents" (the manifest sha256) and `~/transfer/MODELS.md`. On an existing install also: `sha256sum $D/models/*.hef` |
| **Access** | terminal |
| **Expect** | `damoyolo_tinynasL20_T.hef` = `e69136d3ebf3cc8ebaf3da5417a59962b139f3d3d67d715e9a9b1ae10963960e` (18,554,430 bytes). An installed `.hef` with `4c71eab8…add2` is the old unverified file, which the runtime refuses |
| **Capture** | the hash lines |
| **Pass / fail** | FAIL: the package's `.hef` is not the pin |
| **Common failure** | — |
| **Recovery** | stop. **Never edit a pin to match a file** |
| **Modifies the Pi** | no |
| **Owner approval** | no |

### Step 11. Required dependencies (the compatibility floor)
| | |
|---|---|
| **Do** | `grep VERSION_CODENAME /etc/os-release` · `uname -m` · `python3 --version` · `dpkg -l \| grep -E 'hailo\|hailort\|rpicam\|libcamera0\|python3-picamera2\|bluez\|network-manager' \| awk '{print $2, $3}'` · `hailortcli --version 2>/dev/null` |
| **Access** | terminal |
| **Expect** | `bookworm`, `aarch64`, Python `3.11.x`. On a fresh card, `hailo-all` is absent until step 12 (`setup_pi.sh` installs it with apt) |
| **Capture** | the outputs |
| **Pass / fail** | FAIL: not Bookworm, not aarch64, or not Python 3.11 |
| **Common failure** | a newer OS image |
| **Recovery** | back to step 8 (owner decision) |
| **Modifies the Pi** | no |
| **Owner approval** | no |

### Step 12. Safe installation
| | |
|---|---|
| **Do** | Follow the package's `~/transfer/INSTALL.md`, which pins every command for this exact release. **FRESH:** INSTALL.md §A; the entrypoint is `cd ~/sensordyme && bash setup_pi.sh`. It runs apt (including `hailo-all`), installs the hashed lock (wheels only), verifies the model pin, and installs and starts the service. **UPGRADE, existing adopted device** (`~/sensordyme.releases` exists): INSTALL.md §B (`release_guard.py stage` → `activate` → `status --wait`); the release rolls itself back if unhealthy. **UPGRADE, legacy tree** (no `sensordyme.releases`): RB-F0–F4 (stage beside, swap, keep the previous tree for rollback). |
| **Access** | terminal with sudo; internet for apt (and PyPI unless the package's `wheels/` are used, as INSTALL.md shows) |
| **Expect** | §A: the script ends with the service started; §B: `status --wait` exits 0 = CONFIRMED |
| **Capture** | the last 30 lines of the install output: `… 2>&1 \| tee $EV/step12_install.txt` (no secrets are printed) |
| **Pass / fail** | FAIL: any error, a model-pin refusal, or "hailo-all not installed" |
| **Common failure** | no internet for apt; a pip refusal (the hash lock); not enough disk |
| **Recovery** | stop and send the output. Upgrades: step 28. Never install packages by hand to "fix" it |
| **Modifies the Pi** | **yes** |
| **Owner approval** | **yes**, including which path (§A / §B / RB-F) |

### Step 13. Systemd service installation
| | |
|---|---|
| **Do** | `systemctl show sensordyme.service -p ActiveState -p NRestarts -p User -p WorkingDirectory -p OnFailure` · `systemctl cat sensordyme.service \| grep -E 'StartLimitAction\|OnFailure'` · `journalctl -u sensordyme -b --no-pager \| grep -E 'runtime up\|DEGRADED\|ERROR\|accelerator' \| tail -20`. After a **fresh** install, adopt the staged layout once (README Phase 3A): `sudo systemctl stop sensordyme.service` → `python3 ~/transfer/install/release_guard.py adopt --root ~/sensordyme` → `sudo systemctl start sensordyme.service` |
| **Access** | terminal |
| **Expect** | `ActiveState=active`, `NRestarts=0`, `OnFailure=sensordyme-recover.service`; journal `runtime up: profile=pi module=driver_attention … accelerator=…Hailo…`, no DEGRADED |
| **Capture** | the outputs (the journal lines contain no secrets; if one shows a URL with a host you do not want shared, omit it) |
| **Pass / fail** | FAIL: inactive, restarting, or DEGRADED with a reason |
| **Common failure** | the camera ribbon (the service reports why); no Hailo |
| **Recovery** | steps 14–15 diagnose; `sudo journalctl -u sensordyme -n 50 --no-pager` explains |
| **Modifies the Pi** | adopt: **yes** |
| **Owner approval** | adopt: **yes** |

Then set §0.4's `SENSORDYME_TOKEN` (it may be empty until step 22) and run the read-only
preflight: `$PY $V/sunday_preflight.py --install-dir $D --expect-commit <RC> --out $EV/preflight`.

## Part D — Camera, accelerator and model on the real hardware

### Step 14. Camera hardware validation
| | |
|---|---|
| **Do** | The running service holds the camera, so list it in a short stopped window: `sudo systemctl stop sensordyme.service && rpicam-hello --list-cameras; sudo systemctl start sensordyme.service`. With a monitor attached you may also run `rpicam-hello -t 5000` (a 5 s preview) inside the same stopped window. With the service running: `$PY $V/check_live_status.py --device http://localhost:8899 --stream-seconds 6 --out $EV/step14_stream.json` |
| **Access** | terminal |
| **Expect** | `0 : imx708_wide_noir [4608x2592 …]`; stream parts with a rising `X-Frame-Seq` and `X-Frame-Age-Ms` ≤ 2000 (a 401 means the token is not yet set: continue after step 22). NoIR in daylight looks pink: expected |
| **Capture** | the listing line; `step14_stream.json` (headers only, no frames) |
| **Pass / fail** | FAIL: "No cameras available" with the service stopped |
| **Common failure** | ribbon reversed or loose |
| **Recovery** | **power off**, reseat both ends (step 3), retry |
| **Modifies the Pi** | stops monitoring for a few seconds |
| **Owner approval** | yes for the stopped window (monitoring off briefly) |

### Step 15. Hailo recognition
| | |
|---|---|
| **Do** | `lspci \| grep -i hailo` · `ls -l /dev/hailo0` · `hailortcli scan` (RB-D1). In a stopped window: `hailortcli fw-control identify \| tee $EV/step15_fw_identify.txt` |
| **Access** | terminal |
| **Expect** | "Co-processor: Hailo Technologies Ltd. Hailo-8 AI Processor"; `/dev/hailo0`; one device; `Device Architecture: HAILO8L` plus the firmware version |
| **Capture** | the outputs |
| **Pass / fail** | FAIL: no PCIe device |
| **Common failure** | the PCIe ribbon not locked |
| **Recovery** | **power off**, reseat (step 4). RB-D1 also names `dtparam=pciex1` and the HAT EEPROM; that is a boot-config change, **owner approval** only |
| **Modifies the Pi** | identify needs a stopped window |
| **Owner approval** | yes for the stopped window |

### Step 16. Production `.hef` validation
| | |
|---|---|
| **Do** | `$PY $V/hwv_stage.py pins --install-dir $D` (any state) · `$PY $V/hwv_stage.py parse-hef --install-dir $D` · in a stopped window: `$PY $V/hwv_stage.py model --install-dir $D` |
| **Access** | terminal |
| **Expect** | `STAGE pins: PASS` (config / setup / registry agree; decoder `legacy`); `parse-hef` PASS (HAILO8L, 6 outputs 80/40/20 × 68/81, no NMS); `model` PASS (sha = pin, loads, 640×640) |
| **Capture** | the reports in `~/hwval/stages/` (collected by §0.5) |
| **Pass / fail** | the STAGE line of each |
| **Common failure** | a wrong `.hef` (refused by design) |
| **Recovery** | stop (`docs/MONDAY_HARDWARE_SESSION.md` §9); never swap the file by hand |
| **Modifies the Pi** | `model` needs a stopped window; nothing is written to the install |
| **Owner approval** | yes for the stopped window |

### Steps 17–18. Legacy decoder baseline, then the reference-decoder INT8 parity experiment
One stopped window covers 16–19 (≈ 10 minutes of monitoring OFF). Run the commands in this order (the M10.1 hardware gate H1):

```bash
sudo systemctl stop sensordyme.service                       # monitoring OFF from here
$PY $V/hwv_stage.py model --install-dir $D
$PY $V/hwv_stage.py capture --install-dir $D --bundle ~/hwval/capture_kg
$PY $V/hwv_stage.py parity --bundle ~/hwval/capture_kg --decoder both
$PY $V/hwv_stage.py latency --install-dir $D
sudo systemctl start sensordyme.service && systemctl is-active sensordyme.service   # must print: active
```

| | Step 17 (legacy baseline) | Step 18 (reference INT8 parity) |
|---|---|---|
| **Do** | the `capture` stage, then the `[legacy]` arm of `parity` | the `[reference]` arm of the same `parity` run |
| **Access** | terminal | terminal |
| **Expect** | bundle `complete`; inputs **bit-identical** to the float run 6/6; production decoder == offline legacy decoder; INT8-vs-float agreement recorded | INT8-vs-float agreement recorded for the reference decoder, which decodes **offline** from the same INT8 tensors |
| **Capture** | `capture_*.json`, `parity_*.json` (known-good PNG inputs only; no camera frames) | same files |
| **Pass / fail** | the STAGE line; a provisional-tolerance miss is a **finding, not a verdict** | numbers only. **No switch is made here** |
| **Common failure** | input mismatch: record `cv2.__version__` | — |
| **Recovery** | `docs/MONDAY_HARDWARE_SESSION.md` §9 | — |
| **Modifies the Pi** | stopped window only; **production decoder and model never change** | same |
| **Owner approval** | yes (stopped window) | yes (stopped window); switching decoders is a separate owner decision under the promotion rule |

### Step 19. Processed FPS and per-stage latency
| | |
|---|---|
| **Do** | Stopped window: the `latency` stage above. Service running (close app and browser tabs): `$PY $V/hwv_stage.py observe --minutes 10 --interval 10 --install-dir $D` and `$PY $V/stage_timing.py health --minutes 10 --interval 5 --out $EV/step19_health.jsonl` |
| **Access** | terminal |
| **Expect** | `predict()` p95 ≤ 100 ms (the stride budget at 30 fps, detector every 3rd frame), < 1 % late frames; `observe` reports `is_pi_measurement: true` with Hailo inference, decoder CPU, face landmarks, frame-to-decision p95 and processed fps; states `["ready"]` |
| **Capture** | `latency_*.json`, `observe_*/record.summary.json`, the JSONL |
| **Pass / fail** | latency per the stage; `observe`: numbers recorded (these are the first Pi figures) |
| **Common failure** | throttling (step 5); a browser tab streaming video adds load |
| **Recovery** | fix power/cooling, rerun once, record both |
| **Modifies the Pi** | `latency`: stopped window; `observe`/`health`: no |
| **Owner approval** | stopped window only |

### Step 20. CPU, memory and temperature
| | |
|---|---|
| **Do** | `$PY $V/collect_evidence.py --install-dir $D --expect-commit <RC> --minutes 5`, then the 30-minute soak with the service running (closed case if you have one): `SENSORDYME_TOKEN=$SENSORDYME_TOKEN $PY $V/hwv_stage.py thermal --minutes 30 --install-dir $D` |
| **Access** | terminal |
| **Expect** | CPU %, service RSS, memory, SoC °C and ARM clock in the report; soak: `throttled` stays `0x0`, ARM clock never below its start, fps at minute 30 within 10 % of minute 1 |
| **Capture** | the evidence `.tgz` sha256; the thermal report |
| **Pass / fail** | FAIL: any throttling, or an fps drop over 10 % |
| **Common failure** | cooler fan not connected |
| **Recovery** | power off, connect the fan lead (step 4), rerun |
| **Modifies the Pi** | no |
| **Owner approval** | no |

### Step 21. Buzzer: electrical, then functional
| | |
|---|---|
| **Do** | **Only if step 2 passed and it is wired per step 4.** Read-only first: `pinctrl get 17` (expect output, level low). Then, in a stopped window, **warning everyone nearby (it is loud at 50 cm)**: `$PY $V/hwv_stage.py buzzer --install-dir $D`, or `$PY $V/buzzer_check.py run --out $EV/step21_buzzer.json` (RB-G2). A person listens at **50 cm and 1 m** |
| **Access** | terminal + a person listening |
| **Expect** | every production pattern commanded on time, and each one **heard and distinguishable** |
| **Capture** | the stage JSON; "heard: yes/no" per pattern; an SPL-app reading at 1 m if available |
| **Pass / fail** | **No audible alert is a hard stop**: the product's core claim fails. Note: `output: sounded` in events means the pin was driven, never that someone heard it |
| **Common failure** | a passive buzzer (silent on DC); wrong pins |
| **Recovery** | **power off** before any rewiring; tell the owner |
| **Modifies the Pi** | stopped window |
| **Owner approval** | yes |

## Part E — Backend, iPhone and the full product (owner plus technician)

### Step 22. Backend configuration (owner-led)
| | |
|---|---|
| **Do** | **Owner, on the backend host (the MacBook):** `.venv/bin/python cloud/manage.py create-device "Car unit"` prints a device key (`dk_…`) and a claim code. **These two values are secrets: never in chat.** Then the device's `config.yaml` gets `cloud.enabled: true`, `cloud.base_url` (the backend's stable name, e.g. `http://sensordyme-backend.local:8787`, never a bare IP), `cloud.device_key`, and a `dashboard.token` from `openssl rand -hex 16`, followed by `chmod 600 $D/config.yaml && sudo systemctl restart sensordyme` (README Phase 6; RB-L1). **Preferred:** the owner makes this edit over owner-approved SSH, or in person, so the key never passes through you. |
| **Access** | owner on the backend host; config edit on the Pi |
| **Expect** | within ~30 s, `sudo journalctl -u sensordyme -n 50 \| grep cloud_uplink` shows `connected to …`; the auth check gives no token **401**, the header token **200**, `?token=` **401** (RB-L1 step 4) |
| **Capture** | the 401/200/401 codes; "uplink connected: yes/no" (do not paste the URL if it names a private host) |
| **Pass / fail** | FAIL: the uplink never connects |
| **Common failure** | the Pi is not on the backend's network: the backend runs on the owner's MacBook at home, and **no hosted backend is deployed** |
| **Recovery** | bring the Pi to the owner's network. A hosted backend (`fly.toml`, `cloud/README.md`) is an owner decision that is not made yet. The device keeps monitoring and buzzing without any backend |
| **Modifies the Pi** | **yes** (config) |
| **Owner approval** | **yes** |

### Step 23. iPhone pairing (owner's iPhone, next to the Pi)
| | |
|---|---|
| **Do** | The **owner** builds and installs the app from Xcode (README Phase 5; `docs/IOS_DEVICE_BUILD.md`). Pairing (README Phase 6): sign in → antenna button "Connect over Bluetooth" → `SensorDyme-XXXX` → choose Wi-Fi → enter the claim code → wait for "online". If the Pi is already on Wi-Fi (Imager), use **＋ Add device** (claim-only) instead of Bluetooth |
| **Access** | the iPhone within Bluetooth range; the same network as the backend |
| **Expect** | the device appears online with the **same `SD-XXXXXXXX`** as step 7/13 |
| **Capture** | a screenshot of the device card (no account e-mail visible) |
| **Pass / fail** | FAIL: not found / not online after 120 s |
| **Common failure** | the BLE window is closed because the device is online: power-cycle for a 5-minute setup window. "hasn't reported to the server": step 22 is incomplete |
| **Recovery** | README Phase 6 "When pairing fails" table |
| **Modifies the Pi** | BLE Wi-Fi setup saves a Wi-Fi profile on the Pi: **yes** |
| **Owner approval** | **yes** (the owner does it) |

### Step 24. Live video
| | |
|---|---|
| **Do** | In the app, on the device screen: Settings → Live feed → enter the dashboard token (the owner types it; it is never sent in chat). Phone on the **same Wi-Fi** as the Pi |
| **Access** | iPhone on the LAN |
| **Expect** | video within a few seconds; badge LIVE (a new frame ≤ 2 s old within the last 2 s); the same state as `/api/health` |
| **Capture** | a screenshot of the hardware only, with nobody else in frame |
| **Pass / fail** | FAIL: no video on the same LAN with the token entered |
| **Common failure** | guest Wi-Fi with client isolation; token typo |
| **Recovery** | a phone hotspot is the reliable fallback (README Phase 9) |
| **Modifies the Pi** | no |
| **Owner approval** | no |

### Step 25. Distracted Driving activation
| | |
|---|---|
| **Do** | In the app: select **Driver Safety** (the distracted-driving mode); optionally a variant (e.g. Strict). Then on the Pi: `$PY $V/check_live_status.py --device http://localhost:8899 --out $EV/step25_state.json`. Also try selecting **Elderly Monitoring**: it must be **refused** with the role reason (RB-L2), which is the safety floor working, not a failure |
| **Access** | iPhone + terminal |
| **Expect** | confirmed only when the device reports running it; `ready`; zero invariant violations; the refusal shows its reason |
| **Capture** | the JSON; screenshots |
| **Pass / fail** | FAIL: "confirmed" without the device reporting it, or any invariant violation |
| **Common failure** | the first ~30 s after a start are calibration (`attention` STARTING): wait |
| **Recovery** | RB-L2/R2 |
| **Modifies the Pi** | preset state (expected product use) |
| **Owner approval** | no (normal app use) |

### Step 26. Controlled event testing (stationary, consenting adult)
| | |
|---|---|
| **Do** | Camera at driver-face height; participant seated; look ahead for 30 s to calibrate. Run the cases in `docs/SUNDAY_DEVICE_ACTIVATION.md` §5, **B1–B8**: eyes closed 3 s; a 4 s look-away; a mounted phone left unused, then looked at; a handheld phone; mirror glances; sunglasses; hands out of view; lens covered with a card. Log each trial's start and end time. **For measured results** use `cabin_session.py` with an **independent observer** (§26a) |
| **Access** | at the Pi, with the participant |
| **Expect** | per §5 (e.g. B1 `drowsiness` at ~2.0 s + ~0.3 s, critical tone; B2 `distraction` at ~3.0 s; B3 **no** `phone_use`; B5 nothing; B8 `attention_lost` at ~2.5 s) |
| **Capture** | the trials CSV; `$PY $V/collect_evidence.py …` afterwards; tone heard yes/no |
| **Pass / fail** | a deviation is a **finding**: record it; thresholds are never changed during the session |
| **Common failure** | no calibration; a passenger in view during calibration |
| **Recovery** | recalibrate per README Phase 4, with only the driver in view |
| **Modifies the Pi** | no |
| **Owner approval** | the participant's consent; the owner's go-ahead for the session |

**26a. Measured sessions: a cue is not proof.** `cabin_session.py run` reads cues out loud. Only an **independent observer** (not the person reading the cues), in a **second SSH session on the Pi** (same clock), records what actually happened:

```bash
python3 $V/cabin_session.py plan --preset core --out ~/hwval/cabin/S01
python3 $V/cabin_session.py run --plan ~/hwval/cabin/S01 --subject S01 --session S01-<date>-a --vehicle V01 \
    --seat driver --lighting day --eyewear none --occupancy driver_only --mount dash_centre
# observer, second ssh session, at the same time:
python3 $V/cabin_session.py observe --session ~/hwval/cabin/S01 --annotator O1 --clock same_host \
    --consent-ref <consent record id> --independent
python3 $V/cabin_session.py score --session ~/hwval/cabin/S01 --labels ~/hwval/cabin/S01/labels_O1.json
```

Without `--labels`, the score is labelled `diagnostic_cue_schedule` and withholds every
accuracy figure. Even with labels, **one session is diagnostics, never an accuracy study**.
`docs/REAL_DATA_EVALUATION_PROTOCOL.md` has the full rules: subject pseudonyms, consent
reference ids, no names.

### Step 27. Automatic reboot and reconnection
| | |
|---|---|
| **Do** | (a) `sudo reboot` (RB-K1). After 60–90 s: `systemctl is-active sensordyme.service` · `systemctl show sensordyme.service -p NRestarts` · the device-id one-liner from step 7. (b) The README Phase 7 **power-cycle loop**: unplug mid-monitoring → plug back in → touch nothing → within ~2 min the app shows the same device, the same preset, protecting; a test look-away alerts. (c) Wi-Fi outage (RB-K3): turn off the access point for 3 minutes, trigger one closure, turn it back on |
| **Access** | at the Pi; iPhone |
| **Expect** | `active`, `ready`, the **same** device id; the app online within ~30 s of Wi-Fi; during the outage the **buzzer still sounds**; afterwards the alert appears in the app **exactly once** |
| **Capture** | outputs; `journalctl -u sensordyme --since "-10 min" \| grep cloud_uplink > $EV/step27_uplink.txt` |
| **Pass / fail** | FAIL: a different id, no local alert during the outage, a lost or duplicated event |
| **Common failure** | the uplink lines are `print()` output and can reach the journal late: wait 30 s and re-grep |
| **Recovery** | RB-K notes |
| **Modifies the Pi** | reboot / power cut |
| **Owner approval** | **yes** |

### Step 28. Safe rollback
| | |
|---|---|
| **Do** | **Staged layout** (after step 13's adopt): `python3 ~/sensordyme/scripts/release/release_guard.py switch --root ~/sensordyme --to previous --user $(id -un)` (README Phase 3), or from the owner's Mac `make deploy-rollback DEVICE=<user>@<host> DIR=sensordyme`. **Legacy tree swap (RB-F):** RB-O1. **Exact pre-session state:** RB-O2 (the E1 backup). **Total:** RB-O3 (write the E5 image back; check the disk number twice) |
| **Access** | terminal |
| **Expect** | the service `active` on the previous release; `config.yaml`, calibration and events kept (device state lives outside the releases) |
| **Capture** | `release_guard.py status` output; `systemctl is-active` |
| **Pass / fail** | PASS when the previous release is active and healthy |
| **Common failure** | a deploy interrupted by a power cut leaves `pending.json`; resolve it with `release_guard.py confirm-or-rollback` (`docs/DEPLOYMENT.md` §4) |
| **Recovery** | O2, then O3 |
| **Modifies the Pi** | **yes** |
| **Owner approval** | **yes** (unless the device is already failing and the owner asked for it) |

---

## Your first three actions

1. **Steps 1–2, power OFF:** photograph every part, the PSU label and the buzzer markings. Send the photos and say whether the buzzer is clearly 3.3 V-rated. If it is not, leave it disconnected.
2. **Steps 3–5:** assemble (cooler, HAT with the PCIe ribbon locked, camera ribbon per the official diagram), power on with the 27 W PSU, and send the four step 5 outputs (`get_throttled`, EXT5V, temperature, fan).
3. **Step 6, read-only:** send `uname -m`, `/etc/os-release`, `free -h`, `df -h`, and whether `~/sensordyme` / `sensordyme.service` exist (`User=` and `WorkingDirectory=`). **Change nothing** until the owner has decided step 8.
