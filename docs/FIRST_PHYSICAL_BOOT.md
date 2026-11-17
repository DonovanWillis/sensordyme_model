# First physical boot (M14 "Zero Day"): the operator sequence

**Status:** DRAFT, written on a Mac on 2026-11-16, before the hardware arrived. **Nothing here has run on
this Pi yet.** Every "Expect" comes from the code, the earlier runbooks and the Phase 5 hardware record
(Pi 5 + Hailo-8L, HailoRT 4.20.0). Your results are the first evidence.

This page is the order of operations and the decision points. The full procedure for each step is in
`docs/FRIEND_HARDWARE_SETUP.md` (cited as **FHS step N**). Commands not shown here are there.

- The Pi stages are 1, 2, 5–13 and 21–24.
- The iPhone stages (3, 4, 14–20) are placeholders, filled by the iPhone stream.
- Placeholders: `<user>@<host>` is whatever the owner set in Raspberry Pi Imager. Never guess it.
  `<M14 release>` is **TO BE FILLED BY THE INTEGRATOR** with the release id (the first 12 characters of
  the commit) and the full SHA.

**Shell variables on the Pi:** use FHS §0.4 (`U`, `D`, `PY`, `V`, `EV`), and set `U` and `D` from step 6.
The hardware tools default to `/home/pi/sensordyme`, so always pass `--install-dir $D`.

**Results:** PASS / FAIL / BLOCKED / NOT RUN. Tool exit codes are 0 PASS, 1 FAIL, 2 BLOCKED,
3 SIMULATED. A SIMULATED result is never hardware evidence.

---

## Pi stages

### 1. Power, with the proper supply
- **Do:** follow FHS steps 1–5. Power OFF for every ribbon, HAT and buzzer change. Use the official 27 W
  USB-C supply (5.1 V / 5 A). After boot, run `vcgencmd get_throttled` and
  `vcgencmd pmic_read_adc EXT5V_V`.
- **Expect:** `throttled=0x0` and EXT5V about 5.0–5.2 V.
- **If it fails:** a set bit means a weak supply or cable.
  - Recovery: power off and use the 27 W PSU. Fix power before you measure anything else.
- **Approval:** none to read. The owner approves connecting a buzzer that is not clearly 3.3 V-rated
  (FHS step 2).

### 2. Network reachable from the MacBook
- **Do (Mac):** `ping -c 3 <host>.local`. If mDNS does not resolve, use the router's DHCP list.
- **Expect:** replies.
- **If it fails:** check these in order:
  - the Pi is not on the same LAN (guest Wi-Fi with client isolation);
  - the Imager Wi-Fi settings are wrong;
  - mDNS is blocked.
  - Recovery: an Ethernet cable, or keyboard and monitor. Do not edit network settings on the Pi without
    the owner.
- **Approval:** none.

### 3. (iPhone stage) — placeholder
*Filled by the iPhone stream.*

### 4. (iPhone stage) — placeholder
*Filled by the iPhone stream.*

### 5. Establish authorized SSH
- **Do (Mac):** **the user runs this themselves:** `ssh <user>@<host>.local`, with the key set in the
  Imager. Nobody else opens this connection for them.
- **Expect:** a shell prompt.
- **If it fails:**
  - "Permission denied (publickey)": the wrong key or user.
  - A host-key warning: the card was reflashed, or this is a different Pi. Stop and ask the owner;
    never delete `known_hosts` lines on your own.
  - Recovery: keyboard and monitor on the Pi.
- **Approval:** the owner authorizes access. Password SSH and Tailscale are owner decisions.

### 6. Inspect the existing installation (read-only)
- **Do:** run the FHS step 6 commands. Add `ls -la ~/sensordyme ~/sensordyme.releases ~/sensordyme.state
  2>/dev/null`.
- **Expect:** `aarch64`; `VERSION_CODENAME=bookworm`; at least 4 GB free. One of three layouts:
  - **fresh:** no `~/sensordyme`;
  - **legacy:** `~/sensordyme` is a directory and `~/sensordyme.releases` does not exist;
  - **adopted:** `~/sensordyme` is a symlink into `~/sensordyme.releases/`.
- **If it fails:**
  - `VERSION_CODENAME=trixie`, or Python 3.13: **hardware gate HG-OS** (below). Stop. Do not reflash.
  - armhf / 32-bit: also HG-OS.
- **Approval:** none. Read-only.

### 7. Preserve identity and files
- **Do:**
  - **Fresh card:** nothing to preserve.
  - **Existing install:** record the device id (FHS step 7a). Then take the RB-E1 backup in
    `docs/MONDAY_RUNBOOK.md`.
  - **Since M14 the E1 `tar` uses `-h`.** On an adopted device `~/sensordyme` is a symlink. Without
    `-h` the archive held only that symlink, with no config, no `data/local_identity.pem` and no `.hef`.
  - Copy `$B` off the Pi and run `sha256sum -c SHA256SUMS` there.
- **Expect:** `tar -tzf $B/sensordyme_tree.tgz | grep -E 'config.yaml$|\.hef$|local_identity.pem$'` lists
  all three. On an adopted device they appear under `sensordyme/` because `-h` follows the links.
- **If it fails:**
  - Not enough space: move files off the Pi, with the owner's approval. Never delete `data/`.
- **What preserves identity:** none of the install paths creates, rotates or reads out a credential.
  - `adopt` moves `data/`, `models/`, `logs/`, `.secrets` and `.env` into `~/sensordyme.state/`.
  - Every release links back to that state, and `config.yaml` is carried into each release.
- **Approval:** none for E1, which only adds a folder. The owner approves an SD image (RB-E5, power-off).

### 8. Verify the package
- **Do (Mac):** in `~/sensordyme-release/transfer-<M14 release>/`, run `shasum -a 256 -c SHA256SUMS`.
  Copy the directory to the Pi as `~/transfer` (README §3.0).
- **Do (Pi):** in `~/transfer`, run:
  - `sha256sum SHA256SUMS`, which must equal the value the owner sent separately;
  - `sha256sum -c SHA256SUMS`;
  - `sha256sum models/*.hef`;
  - `grep -A6 'Compatibility floor' INSTALL.md`.
- **Expect:**
  - every line `OK`;
  - `damoyolo_tinynasL20_T.hef` = `e69136d3ebf3cc8ebaf3da5417a59962b139f3d3d67d715e9a9b1ae10963960e`.
- **If it fails:** re-copy the package. Never edit a hash or a pin to match a file.
- **Approval:** none. This writes only `~/transfer`.

### 9. Hardware preflight
- **Before installing (any card, read-only):**
  - `grep VERSION_CODENAME /etc/os-release; uname -m; python3 -c 'import struct,sys; print(sys.version.split()[0], 8*struct.calcsize("P"))'`
  - `lspci | grep -i hailo`
  - `rpicam-hello --list-cameras` (only while `sensordyme.service` is not running)
  - `vcgencmd get_throttled`
- **Expect:**
  - `bookworm`, `aarch64`, `3.11.x 64`;
  - a Hailo PCIe line;
  - `imx708_wide_noir`;
  - `0x0`.
  - On a fresh card, `/dev/hailo0` does not exist yet. `hailo-all` installs it in stage 11.
- **After an install (any time, read-only):** `$PY $V/sunday_preflight.py --install-dir $D --expect-commit
  <M14 release> --out $EV/preflight`. The validation tools are unpacked as in FHS step 9.
- **If it fails:**
  - No Hailo on PCIe or no camera: power off and reseat (FHS steps 3–4, 14–15).
  - OS/Python wrong: HG-OS.
- **Approval:** none. `rpicam-hello` on a running device needs the owner's stopped window (FHS step 14).

### 10. Installation approval
- **Do:** write the decision line (FHS step 8). The three paths:
  - **FRESH:** a blank card. Use INSTALL.md §A.
  - **UPGRADE-ADOPTED:** `~/sensordyme.releases` exists. Use INSTALL.md §B.
  - **UPGRADE-LEGACY:** back up (stage 7), adopt once, then use §B.
- **Expect:** the owner replies "go, stage 11" and names the path.
- **Approval:** **owner, always.** No install without it.

### 11. Install and start
- **FRESH** (from `~/transfer`): the single chained command in `INSTALL.md` §A. It does all of this:
  - `mkdir ~/sensordyme` (it stops at once if the directory exists);
  - extracts the bundle;
  - copies `config/config.yaml.example` to `config.yaml` with mode 0600;
  - copies the `.hef` and the wheels to `~/sensordyme.wheels`;
  - runs `bash setup_pi.sh`.
- **What `setup_pi.sh` does, in order:**
  1. A read-only **platform preflight**. Bookworm, aarch64, 64-bit CPython 3.11 and `config.yaml` must all
     be present. Otherwise it exits 1, prints "Nothing was installed or changed", and names HG-OS.
  2. apt, including `hailo-all`. It prints the HailoRT version it got.
  3. The hashed lock from `~/sensordyme.wheels` (`--no-index`).
  4. The model pin.
  5. Groups `netdev bluetooth video gpio audio`.
  6. Units, then a service start.
- If it ends with **"/dev/hailo0 is not present yet"**:
  - Run `sudo reboot` once, with the owner's approval. The Hailo driver loads at boot.
  - Until then the service honestly reports no accelerator (`monitoring_active: false`) and never
    simulates detections.
- Then adopt once: `sudo systemctl stop sensordyme.service && python3 ~/transfer/install/release_guard.py
  adopt --root ~/sensordyme && sudo systemctl start sensordyme.service`.
- **UPGRADE-ADOPTED / -LEGACY:** INSTALL.md §B (`release_guard.py stage` → `activate --detach` →
  `status --wait`).
  - With the package wheels, add `--find-links ~/sensordyme.wheels`.
  - A release that does not stay healthy is switched back automatically.
- **Expect:**
  - §A: "Setup complete."
  - §B: `status --wait` exits 0 (CONFIRMED).
- **If it fails:**
  - A pip refusal: the hash lock (HG-PIP).
  - Exit 78 in the journal: `config.yaml` is invalid. Read the "config ERROR" lines.
  - A model-pin refusal: stop. Never swap the file.
  - Recovery: stage 24. Send `… 2>&1 | tee $EV/stage11_install.txt`.
- **Approval:** **owner** (the install, the reboot, the adopt).

### 12. Verify camera and Hailo
- **Do:**
  - `systemctl show sensordyme.service -p ActiveState -p NRestarts -p WorkingDirectory`
  - `journalctl -u sensordyme -b --no-pager | grep -E 'runtime up|config ERROR|config FAULT|could not start the Pi camera|Hailo|integrity|REFUSED' | tail -20`
  - `ls -l /dev/hailo0 && hailortcli scan`
  - `$PY $V/hwv_stage.py pins --install-dir $D`
  - `$PY $V/hwv_stage.py parse-hef --install-dir $D`
  - `$PY $V/check_live_status.py --device http://localhost:8899 --stream-seconds 6 --out $EV/stage12_stream.json`
  - Stopped window (FHS steps 14–16): `rpicam-hello --list-cameras`, `hailortcli fw-control identify`,
    `$PY $V/hwv_stage.py model --install-dir $D`
- **Expect:**
  - `active`, `NRestarts=0`;
  - `runtime up: profile=pi … accelerator=…Hailo…`;
  - `STAGE pins: PASS` with decoder `legacy`;
  - parse-hef: HAILO8L, 6 outputs, no NMS;
  - stream frames with a rising `X-Frame-Seq` and `X-Frame-Age-Ms` ≤ 2000;
  - `model` PASS.
- **If it fails:**
  - "could not start the Pi camera: …": the ribbon, or a broken picamera2 install. Since M14 the reason
    is the real error, never a webcam fallback.
  - No `/dev/hailo0` after a reboot: the PCIe ribbon (FHS step 15).
  - parse-hef or model FAIL on a HailoRT other than 4.20.0: **HG-HAILORT**.
- **Expected bounded recovery, not a fault:** a camera that never opens leads to one process restart by
  the watchdog, then at most one reboot, which is rate-limited (`health.reboot_min_interval_s: 1800`). Power
  off before you reseat anything.
- **Approval:** the owner approves each stopped window (monitoring off briefly).

### 13. Initial latency and FPS (production stays on the legacy decoder)
- **Do:**
  - Stopped window: `$PY $V/hwv_stage.py latency --install-dir $D`
  - Service running: `$PY $V/hwv_stage.py observe --minutes 10 --interval 10 --install-dir $D`
- **Expect:**
  - `predict()` p95 ≤ 100 ms and under 1 % late frames;
  - `observe` reports `is_pi_measurement: true`, per-stage timings and processed fps.
  - These are the first Pi figures, so record them.
- **If it fails:**
  - Throttling: go back to stage 1.
  - A browser tab or app streaming video adds load: close it and rerun once.
- **What does not change:** the decoder. `hwv_stage.py parity --decoder both` is an offline experiment
  (FHS steps 17–18). A switch is a separate owner decision.
- **Approval:** the owner approves the stopped window for `latency`.

### 14.–20. (iPhone stages) — placeholders
*Filled by the iPhone stream (pairing, live video, protection state, alerts in the app).*

### 21. Restart the Pi
- **Do:** `sudo reboot`. Wait 60–90 s, then SSH in again (stage 5).
- **Expect:** the Pi is reachable again.
- **If it fails:**
  - Not back after 3 minutes: check power (stage 1) at the Pi.
  - If the device keeps rebooting: the recovery unit bounds it to 3 start-limit reboots in 6 h. Read
    `journalctl -b -1 -u sensordyme` and `-u sensordyme-recover`.
- **Approval:** **owner.**

### 22. Verify automatic recovery
- **Do:**
  - `systemctl is-active sensordyme.service; systemctl show sensordyme.service -p NRestarts`
  - `cd $D && $PY -c "from core.identity import device_id; print(device_id())"`
  - `python3 $D/scripts/release/release_guard.py status --root ~/sensordyme`
  - `$PY $V/check_live_status.py --device http://localhost:8899 --out $EV/stage22_state.json`
- **Expect:**
  - `active`, `NRestarts=0`;
  - the **same** `SD-XXXXXXXX` as stage 7;
  - the same release;
  - state `ready` (or `ready` after the ~30 s attention calibration).
- **If it fails:** a different device id means the identity was lost. Stop and restore from the stage 7
  backup.
- **Approval:** none.

### 23. Capture evidence
- **Do:** `$PY $V/collect_evidence.py --install-dir $D --expect-commit <M14 release> --minutes 5`.
  Read-only. It runs the preflight, samples the running service, redacts secrets and writes
  `~/hwval/evidence/evidence_<device-id>_<UTC>.tgz`.
- **Expect:** a `.tgz` and its sha256 printed.
- **Sharing:** send the file privately. Only the sha256 goes in chat (FHS §0.5).
- **If it fails:** "refused: a secret survived": do not share. Send the owner the console output only.
- **Approval:** none.

### 24. Rollback available
- **Do:** `python3 ~/sensordyme/scripts/release/release_guard.py status --root ~/sensordyme`.
- **Expect:** the staged layout, the running release, and `previous` in `confirmed.json`.
- **Rollback, when needed:** `python3 ~/sensordyme/scripts/release/release_guard.py switch --root
  ~/sensordyme --to previous --user "$(id -un)"`, or from the Mac
  `make deploy-rollback DEVICE=<user>@<host> DIR=sensordyme`.
  - This needs no pip and no network.
  - The previous release keeps its own venv and its pre-migration `config.yaml`.
- **Fresh install:** the only release is `legacy-<time>` (the adopted first install). Rollback means:
  - the stage 7 backup (RB-O2), or the owner-approved SD image (RB-O3);
  - or, after the next staged deploy, `switch --to previous`.
- **If it fails:**
  - "a deploy is pending": run `release_guard.py confirm-or-rollback` (`docs/DEPLOYMENT.md` §4).
- **Approval:** **owner** (unless the device is already failing and the owner asked for it).

---

## Hardware gates (only the Pi can answer these)

| Gate | Question | Where it is answered | If it fails |
|---|---|---|---|
| **HG-OS** | Raspberry Pi OS **Bookworm** (Debian 12), aarch64, 64-bit userland, CPython 3.11? | Stages 6 and 9. `setup_pi.sh` refuses anything else before any change. | **Trixie** (Debian 13, Python 3.13) or 32-bit: **stop**. This release has no cp313 wheel set. Reflashing to Bookworm 64-bit erases the card (identity, config, events), so it needs the owner's approval and the stage 7 backup first. |
| HG-POWER | `throttled=0x0` idle and under load? | Stage 1; the 30-min soak (FHS step 20) | the 27 W PSU and the Active Cooler fan lead |
| HG-PIP | Does the hashed cp311 lock install on Bookworm's pip, offline from the package wheels? | Stage 11 | stop and send the output; never install by hand |
| HG-HAILO | PCIe, `/dev/hailo0` and `hailortcli scan` after the post-install reboot? | Stages 9, 11, 12 | power off and reseat the PCIe ribbon. `dtparam=pciex1` or the HAT EEPROM is a boot-config change: owner only |
| **HG-HAILORT** | Does the pinned `.hef` (Model Zoo v2.15.0) parse and load on the HailoRT that `hailo-all` installed? It was measured on 4.20.0 only. | `setup_pi.sh` prints the version. Then `hwv_stage.py parse-hef` and `model` (stage 12) | the runtime refuses a model it cannot load and reports DEGRADED (no CPU fallback). Record `hailortcli --version` and the error. The owner decides; never swap the `.hef` or edit a pin |
| HG-CAMERA | `imx708_wide_noir` listed; fresh frames in the stream? | Stages 9 and 12 | the ribbon orientation or seating (FHS step 3) |
| HG-SYSTEMD | Does the unit follow the `~/sensordyme` symlink, reach READY within `TimeoutStartSec=180`, and survive a reboot? | Stages 12 and 21–22 | `journalctl -u sensordyme -b` (output is unbuffered since M14) |
| HG-AUDIO | What the `sound` sink does on a Pi 5 (no analog audio out) | the `audible_alert` capability in the monitoring state, after the first alert | since M14 a failed `aplay` is reported as a failed playback, not as "played". The GPIO buzzer is the audible output |
| HG-BUZZER | Each alert pattern is heard at 50 cm and 1 m | FHS step 21 (stopped window, a person listening) | no audible alert is a hard stop for driver safety |
| HG-LATENCY | `predict()` p95 and processed fps on the Pi | Stage 13 | a finding to record, not a reason to change thresholds or the decoder |
| HG-RECOVERY | The same device id, release and preset after a reboot and a power cut | Stages 21–22; FHS step 27 | stop; restore from the stage 7 backup |
| HG-ROLLBACK | `switch --to previous` on real systemd | Stage 24, after the first staged deploy | RB-O2 / RB-O3 |
