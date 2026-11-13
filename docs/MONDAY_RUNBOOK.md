# Monday hardware runbook — Raspberry Pi 5 + Hailo-8L + Camera Module 3 Wide NoIR

> **STATUS: NOT EXECUTED — requires the physical device.** Every command, expected
> output and pass criterion below was written on a Mac on 2026-11-11 from the code
> and the M5 package. None has been run against the Pi. Where an expected output
> is a best expectation rather than measured, it says so. Record what you
> actually see in the evidence log (§N) even when it matches.
>
> **M8 update (2026-11-12):** brought up to the M8 release candidate. What changed for
> Monday is collected in **§R** (new checks, the rewritten K4, the optional
> proof-of-possession onboarding session, and a map of every Monday task A–Q to its step).
> Still NOT EXECUTED: nothing in M8 ran on the Pi, the Hailo, the camera or a radio.

Who runs this: the **owner/operator**, at the device. Agents prepared it and are
not authorized to SSH to the Pi, change its networking, install Tailscale,
rotate identities or deploy.

Reuses, does not duplicate:
* `docs/HAILO_HARDWARE_VALIDATION.md` (M5: capture / compare / stage timing / thermal / calibration / buzzer / integrity), referred to as **M5 step N**.
* `docs/RELEASE_MANIFEST.md` (manifest build/verify).
* `docs/HARDWARE_DEPENDENCY_MATRIX.md` (what each step proves).
* `docs/SECURITY_REVIEW_M6.md`.

---

## 0. Conventions and ground rules

```bash
# On the Mac (repo checkout at the release candidate):
export PI=<user>@<pi-host-or-ip>      # e.g. pi@sensordyme.local. The user may NOT be `pi` (Bookworm)
export RC=feat/m8-product-intelligence  # the approved release ref (or its merged successor); record `git rev-parse $RC`
export REL=~/sensordyme-release        # manifest + tarball + evidence on the Mac
export VENV=$HOME/.venvs/sensordyme-m01/bin  # M8.1: the Makefile defaults to .venv/bin, which this checkout does not have

# On the Pi, discover the real service user and install directory first (§C1), then:
export U=<service user>  D=/home/$U/sensordyme
export PY=$D/.venv/bin/python
export V=~/hwval/sensordyme_model/scripts/hw_validation   # M5 package, own directory (§A4)
export EV=~/m6-evidence && mkdir -p $EV                    # evidence on the Pi
export TOKEN="$(sudo -u $U $PY -c "import yaml;print((yaml.safe_load(open('$D/config.yaml')).get('dashboard') or {}).get('token') or '')")"
```

Rules:
1. Steps marked **(read-only)** change nothing. Steps marked **(CHANGE)** need the owner's go-ahead at that moment and a recorded result.
2. Never paste `config.yaml`, the dashboard token, the device key, Wi-Fi passwords or backup archives into chat, tickets or git. Hash them or reference them by path.
3. Stop at the first unexplained **FAIL**, record it (§N), and decide before continuing. Rollback is §O.
4. People on camera: only consenting adults, following the M5 consent workflow. No frames leave the Pi except the M5 bundles, by the documented transfer.

---

## A. Before Monday (Mac; no device needed)

**A0. One-command offline release verification** (read-only; M6.5; ~15 s on the Mac):
```bash
make verify-release REF=$RC        # = bash scripts/release/verify.sh $RC
```
Expect `RELEASE VERIFICATION: PASS`. It checks the device dependencies (no test/dev packages,
every device import pinned), builds the manifest from git objects and runs the release gates
(model pins consistent, decoder `legacy`, every `/v1` route the device and app call exists in the
backend, `requirements-device.lock` agrees with the pins, `config.yaml` passes the schema, the
status tiers in `scripts/release/release_status.yaml` are well formed), validates the systemd
units where `systemd-analyze` exists (SKIP on macOS; run it on a worker), and runs the focused
release tests. **FAIL:** do not take this ref to the device. The status-tier table in
`manifest.md` (A2) is what this release may and may not claim; nothing in it is hardware-validated.

**A1. Tests on the release candidate** (read-only):
```bash
git switch --detach $RC && git status --short            # expect: nothing
PYTHONDONTWRITEBYTECODE=1 ~/.venvs/sensordyme-m01/bin/python -m pytest -p no:cacheprovider -q \
  tests/test_m6_*.py tests/test_m02_*.py tests/test_m01_*.py tests/test_m5_hw_validation.py \
  tests/test_m7_*.py tests/test_m8_*.py
```
Expect: `… passed` with 0 failed. Record the line; it goes into the manifest. (The full suite,
`make test`, is ~3,200 tests and ~7 minutes on the Mac; run it once on the exact RC commit.)

**A2. Build the manifest** (read-only on the repo; writes `$REL`):
```bash
make release-manifest VENV=$VENV REF=$RC RELEASE_DIR=$REL TESTS="<the A1 summary line>, <date>, <machine>"
```
Expect: `<commit>  <N> files  aggregate <hash>…  model pins consistent: True`, exit 0.
**FAIL** if the pins are not consistent: do not take this ref to the device.

**A3. Release tarball** from git objects (never the working tree):
```bash
C=$(git rev-parse "$RC^{commit}"); S=${C:0:12}
git archive --format=tar --prefix=sensordyme/ "$C" | gzip -n > $REL/sensordyme-$S.tgz
shasum -a 256 $REL/sensordyme-$S.tgz | tee $REL/sensordyme-$S.tgz.sha256
```

**A4. M5 validation package** in its own directory (as M5 §2 says):
```bash
git archive --prefix=sensordyme_model/ "$C" | ssh $PI 'mkdir -p ~/hwval && tar -x -C ~/hwval'
ssh $PI "echo $C > ~/hwval/sensordyme_model/HWVAL_COMMIT"
```
(A4 writes only to `~/hwval`, which is not the install. Do it on Monday if SSH is not yet available.)

**A5. Have ready:** the official 27 W USB-C PSU (5.1 V/5 A); a spare SD card at least as large as the device's, plus a card reader; a USB stick for backups; a printed 9×6 inner-corner checkerboard; a phone that can film at 240 fps; an SPL meter app; a laptop for `ms_clock.html`; the iPhone with the ORYN EDGE app; a known Wi-Fi network; and a second AP or phone hotspot for the network-loss tests.

---

## B. Inspect hardware and power (physical, then read-only commands)

**B1. Physical** (record photos in §N):
- AI HAT+ (Hailo-8L) seated on the PCIe FFC with the ribbon locked at both ends; standoffs fitted.
- Camera Module 3 Wide NoIR ribbon on the correct CSI port (CAM/DISP 0 or 1; note which), contacts the right way round.
- Active cooler fitted and its fan connected to the FAN header.
- Buzzer: signal to **BCM 17 = physical pin 11**, ground to a GND pin (e.g. pin 9). Note whether it is an *active* buzzer (beeps on DC) and its rated voltage (3.3 V logic). A 5 V buzzer driven straight from GPIO is a wiring fault: record it and do not power on until resolved.
- PSU rating label: official 27 W expected. A 15 W supply limits USB current and triggers under-voltage under Hailo load.

**B2. Power** (read-only, after boot):
```bash
vcgencmd get_throttled            # expect: throttled=0x0
vcgencmd pmic_read_adc EXT5V_V    # expect: ~5.0–5.2 V (Pi 5)
dmesg | grep -iE 'under-?voltage|voltage' | tail
```
**FAIL:** any bit set (0x1 = under-voltage now, 0x10000 = has occurred since boot). Fix power before measuring anything.

**B3. Storage and cooling** (read-only):
```bash
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT,MODEL; df -h / /home
sudo dmesg | grep -iE 'mmc|nvme|I/O error|EXT4-fs error' | tail
vcgencmd measure_temp; cat /sys/devices/platform/cooling_fan/hwmon/*/fan1_input 2>/dev/null
```
Expect: root filesystem under 80 % used and **at least 4 GB free** (the backups need about 2× the install, and M5 capture needs 1.5× its bundle plus 1 GB); no I/O errors; idle temperature under ~60 °C. A fan reading of 0 rpm at idle is normal. **FAIL:** filesystem errors. Image the card (§E5) before anything else; a dying card is the clean-install criterion in §E6.

---

## C. Identify OS and installed software (read-only, service running)

**C1. OS, firmware, service user and directory:**
```bash
cat /proc/device-tree/model; echo
cat /etc/os-release | head -4; uname -a
vcgencmd version; sudo rpi-eeprom-update | head -5
systemctl show sensordyme.service -p User -p WorkingDirectory -p ExecStart -p ActiveState -p SubState -p NRestarts
systemctl cat sensordyme.service > $EV/unit_before.txt; sha256sum /etc/systemd/system/sensordyme.service
systemctl list-unit-files | grep -iE 'sensordyme|hailo|avahi|bluetooth|NetworkManager'
```
Expect: "Raspberry Pi 5 Model B…"; Debian 12 (bookworm); `ActiveState=active`. **Set `U` and `D` from `User=`/`WorkingDirectory=`** (§0). Expect the unit to show `StartLimitAction=reboot` (pre-M6). If it already shows `OnFailure=sensordyme-recover.service`, someone installed M6 units already: record it.

**C2. Identity** (what the app and backend know this device as):
```bash
grep -i '^Serial' /proc/cpuinfo                    # expect a non-zero serial on a Pi 5
cat /sys/firmware/devicetree/base/serial-number; echo
cd $D && sudo -u $U $PY -c "from core.identity import device_id, display_name; print(device_id(), display_name())"
grep -o 'id=SD-[0-9A-F]*' /etc/avahi/services/sensordyme.service
```
Expect: one `SD-XXXXXXXX` id, the same in all places, and `SensorDyme-XXXX` as the display name. **Record the id**: it must be identical after every reboot and reinstall (§K). If `/proc/cpuinfo` has **no** Serial line, the id is derived from the wlan0/eth0 MAC instead. That is stable but interface-dependent; record it as a finding.

**C3. Installed packages and code** (read-only):
```bash
python3 --version; $PY --version
dpkg -l | grep -E 'hailo|hailort|rpicam|libcamera0|python3-picamera2|bluez|network-manager' | awk '{print $2, $3}'
hailortcli --version
$PY -m pip freeze > $EV/pip_freeze_before.txt
sha256sum $D/models/*.hef
cd $D && git rev-parse HEAD 2>/dev/null || echo "not a git checkout (expected: make deploy rsyncs files)"
```
Expect: hailo-all / HailoRT 4.x installed; the `.hef` sha256 is `e69136d3…960e` (pinned) or `4c71eab8…add2` ("evaluated": unverified provenance, which the runtime refuses).

**C4. What code is installed** (M5 step 2; read-only):
```bash
$PY $V/integrity_manifest.py snapshot --root $D --expect-role driver_safety --out $EV/integrity_before.json
```
On the Mac: `scp $PI:m6-evidence/integrity_before.json $REL/` then
`python scripts/hw_validation/integrity_manifest.py compare --snapshot $REL/integrity_before.json --rev 973aba3 --rev integration/all --rev c01e43b`.
Record the best-matching revision and every drift file. This is the "previous release" baseline.

**C5. Health now** (read-only):
```bash
curl -s -H "X-Device-Token: $TOKEN" localhost:8899/api/health \
 | $PY -c 'import json,sys;d=json.load(sys.stdin);dv=d["device"];print(d.get("state"), dv.get("monitoring_active"), dv["monitoring"]["state"])'
journalctl -u sensordyme -b --no-pager | tail -50 > $EV/journal_before.txt
```
Expect: `ok True ready` (or a reason). An empty `$TOKEN` with a LAN bind gives **401** by design (fail-closed); use `localhost` with the token, or record that the token is unset.

---

## D. Hailo, camera, buzzer wiring (read-only)

**D1. Hailo:**
```bash
lspci | grep -i hailo                       # expect: "Co-processor: Hailo Technologies Ltd. Hailo-8 AI Processor"
hailortcli scan                             # expect: "Hailo Devices:" then "[-] Device: 0000:01:00.0"
ls -l /dev/hailo0
hailortcli parse-hef $D/models/damoyolo_tinynasL20_T.hef | tee $EV/parse_hef.txt   # expect 6 outputs, no NMS
```
`hailortcli fw-control identify` needs the device free. Run it in the stopped window (§F1, or M5 step 6, which records it). Expect `Device Architecture: HAILO8L` and the firmware version. **FAIL:** no PCIe device. Check the ribbon, then `dtparam=pciex1` and the HAT EEPROM.

**D2. Camera:**
```bash
rpicam-hello --list-cameras
```
Expect a line like `0 : imx708_wide_noir [4608x2592 10-bit RGGB] (/base/axi/pcie@120000/rp1/i2c@…/imx708@1a)` (exact path not verified). If it lists no cameras while the service runs, re-run in the stopped window. **FAIL:** "No cameras available" with the service stopped. Reseat the ribbon (power off first).

**D3. Buzzer GPIO state** (no output driven):
```bash
pinctrl get 17            # expect: output, level lo (when no alert is active)
```

---

## E. Back up everything that makes this device *this* device (CHANGE: writes only to the backup location)

What must survive a reinstall, and why:

| Item | Path | Why |
|---|---|---|
| Device config | `$D/config.yaml` | `cloud.device_key` (backend pairing), `dashboard.token` (app pairing), `role`, camera orientation, model pin. **Secret.** |
| Device state | `$D/data/` (`device_state.json` saved preset and zones, `monitoring.json` custom rules, `calibration.json`, `gaze_calibration.json`, `.reboot_state.json` reboot budget, `events.csv`, `telemetry.csv`, `preflight.json`) | Restores the mode, rules and calibration after reinstall. Contains personal data |
| Model | `$D/models/*.hef` | The pinned binary. Re-downloadable, but keep it to avoid a network dependency |
| Wi-Fi networks | `/etc/NetworkManager/system-connections/*.nmconnection` | Includes the PSKs. **Secret** |
| Bluetooth bonds | `/var/lib/bluetooth/` | Pairing keys, if any were bonded ("Just Works" usually leaves none) |
| System glue | `/etc/systemd/system/sensordyme*.service`, `/etc/sudoers.d/sensordyme-reboot`, `/etc/polkit-1/rules.d/10-sensordyme-network.rules`, `/etc/avahi/services/sensordyme.service` | Needed to restore the previous release exactly |
| Identity | (derived from the SoC serial; nothing to back up) | Record the C2 id |

**E1. Install tree and config** (M5 step 1, extended):
```bash
B=~/m6-backup-$(date +%Y%m%dT%H%M%S); mkdir -p $B && chmod 700 $B && cd $B
cp -p $D/config.yaml config.yaml
tar --exclude='.venv' -czf sensordyme_tree.tgz -C $(dirname $D) $(basename $D)
sudo tar -czf system_glue.tgz /etc/systemd/system/sensordyme*.service /etc/sudoers.d/sensordyme-reboot \
  /etc/polkit-1/rules.d/10-sensordyme-network.rules /etc/avahi/services/sensordyme.service \
  /etc/NetworkManager/system-connections /var/lib/bluetooth 2>/dev/null
sudo chown $USER system_glue.tgz && chmod 600 *
sha256sum $D/models/*.hef > model_hashes.txt
sha256sum * > SHA256SUMS; cat SHA256SUMS
tar -tzf sensordyme_tree.tgz | grep -E 'config.yaml$|\.hef$|data/device_state.json$' 
```
Expect: the archive lists `config.yaml`, the `.hef` and `data/`. Copy `$B` to the USB stick (or encrypted storage on the Mac) and verify `sha256sum -c SHA256SUMS` there. **Treat as secret.**

**E2. Record model hashes:** done in `model_hashes.txt`. Also copy it into `$EV`.

**E3. Config file mode** (finding only; do not change in this step): `stat -c '%a %U' $D/config.yaml`. Anything other than `600 $U` is noted for the security backlog (SECURITY_REVIEW_M6 §3).

**E4. Service user's sudo rights** (finding only): `sudo -l -U $U`.

**E5. Recovery path: full SD image (CHANGE: power-off).** This is the only rollback that survives a bad card or a botched OS.
```bash
sudo systemctl stop sensordyme.service && sudo poweroff
# Mac: insert the card
diskutil list                                   # identify /dev/diskN by SIZE. Get this wrong and you overwrite the wrong disk
diskutil unmountDisk /dev/diskN
sudo dd if=/dev/rdiskN of=$REL/pi-sd-$(date +%Y%m%d).img bs=4m status=progress
shasum -a 256 $REL/pi-sd-*.img > $REL/pi-sd.img.sha256
diskutil eject /dev/diskN
```
Expect: the image size equals the card size. Keep it until the acceptance checklist passes. (No Mac disk space? Write the image to an external drive. The repo Mac is nearly full.)

**E6. Is a clean install necessary?** Decide here and write the decision and reasons in §N.

Clean install (fresh Raspberry Pi OS Bookworm 64-bit + `setup_pi.sh`) **only if one of these holds**:
- filesystem or I/O errors (B3), or the card fails `fsck`;
- OS is not Bookworm 64-bit, or `hailo-all` cannot be installed or upgraded on it;
- the install tree cannot be identified (C4 matches no revision, with large unexplained drift) **and** the backup (E1) is complete;
- unknown users, keys, services or cron jobs that cannot be explained (possible compromise);
- the service user is root, or the venv is broken beyond `pip install -r requirements-device.txt`.

Otherwise **upgrade in place** (§F). It keeps NetworkManager, BlueZ, avahi and the identity untouched. A clean install loses NetworkManager profiles unless restored, and needs the backend device key and the dashboard token re-entered from the E1 backup.

---

## F. Install ONLY the approved release candidate (CHANGE)

**F0. Precheck the device config against the release** (Mac, read-only on the device):
```bash
make deploy-preflight VENV=$VENV ROLE=driver_safety     # copies only config.yaml to a temp dir; prints REFUSED/OK
```
`make deploy-preflight` is hard-wired to `pi@sensordyme.local:~/sensordyme`. If §C1 found a different user, host or directory, run the same check by hand (M8.1):
```bash
DPI=<install dir from §C1, e.g. /home/<user>/sensordyme>     # on the Mac; $D exists only on the Pi
T=$(mktemp -d) && scp -q $PI:$DPI/config.yaml $T/config.yaml && \
$VENV/python scripts/deploy_preflight.py --config $T/config.yaml --expect-role driver_safety \
  --hef-sha256 "$(ssh $PI "sha256sum $DPI/models/damoyolo_tinynasL20_T.hef | cut -d' ' -f1")"; rm -rf $T
```
**Expected first REFUSED on a pre-M0.1 install:** a config with no `role:` runs as `general` and is refused for a `driver_safety` deploy (M8.1 dry run on the `50e9441` config). Add `role: driver_safety` to the device's config by hand (a deliberate, recorded change: deploys never change a role), then re-run.
**FAIL:** any REFUSED. Fix the device config deliberately first (for example `profile: pi`, the model pin). Read every warning (token length, `http://`, frame capture) and record them.

**F1. Stop, then stage the new tree beside the old one:**
```bash
# Mac → Pi
scp $REL/sensordyme-$S.tgz $REL/sensordyme-$S.tgz.sha256 $PI:
# Pi
cd ~ && sha256sum -c <(sed "s#  .*#  sensordyme-$S.tgz#" sensordyme-$S.tgz.sha256)   # expect: OK
systemctl is-active sensordyme.service > $EV/state_before_stop.txt
sudo systemctl stop sensordyme.service && systemctl is-active sensordyme.service   # expect: inactive
hailortcli fw-control identify | tee $EV/fw_identify.txt                           # now the device is free
STAGE=$(dirname $D)/sensordyme.rc-$S && mkdir -p $STAGE && tar -xzf ~/sensordyme-$S.tgz -C $STAGE --strip-components=1
# carry over device state: never overwritten by a release
cp -a $D/config.yaml $STAGE/ && cp -a $D/data $STAGE/ && mkdir -p $STAGE/models && cp -a $D/models/*.hef $STAGE/models/
cp -a $D/.venv $STAGE/.venv              # same final path, so the venv's absolute paths stay valid
```

**F2. Swap, keeping the previous release for rollback:**
```bash
PREV=$(dirname $D)/sensordyme.prev-$(date +%Y%m%dT%H%M%S)
mv $D $PREV && mv $STAGE $D && echo "$PREV" > ~/m6-previous-release-path
cd $D && .venv/bin/pip install -q -r requirements-device.txt && .venv/bin/pip install -q gpiozero==2.0.1 lgpio==0.2.2.0
python3 scripts/migrate_config.py config.yaml             # prints added keys; never removes any
sudo bash scripts/release/install_units.sh --user $U      # installs sensordyme.service + the recovery unit + guard
systemctl cat sensordyme.service | grep -E 'StartLimitAction|OnFailure'   # expect: none / sensordyme-recover.service
sha256sum $D/models/*.hef                                 # expect: e69136d3…960e
```
**FAIL:** a pip error (no network? see §K3, or restore the venv from `$PREV/.venv`); a model hash that is not the pin.

> M6.5 notes for F2. `requirements-device.txt` no longer contains `pytest` or `anthropic`; pip
> does not uninstall them, so an existing venv keeps them (harmless; `release-verify` compares
> only the pins). Do not switch to `requirements-device.lock` (`--require-hashes`) or to the
> least-privilege unit (`deploy/systemd/sensordyme.service.hardened`) on Monday: both are
> hardware-pending and are a separate, later session (`docs/LEAST_PRIVILEGE.md`). F2 installs
> the **stock** unit, as before. No identity changes: device id, device key and claim code are
> carried over by F1 unchanged, and no key is rotated.

**F3. Verify the installed tree against the manifest before starting** (read-only):
```bash
$PY $V/integrity_manifest.py snapshot --root $D --expect-role driver_safety --out $EV/integrity_rc.json
# Mac:
scp $PI:m6-evidence/integrity_rc.json $REL/ && make release-verify VENV=$VENV RELEASE_DIR=$REL SNAPSHOT=$REL/integrity_rc.json
```
Expect: `VERDICT: PASS against <commit>`: deployed files exact, model hash = pin, device config pins it, requirements installed, unit matches, preflight ok. **FAIL:** do not start. Explain each failing check, or roll back (§O).

**F4. Start:**
```bash
sudo systemctl start sensordyme.service; sleep 60
systemctl is-active sensordyme.service; systemctl show sensordyme.service -p NRestarts
journalctl -u sensordyme -b --no-pager | grep -E 'runtime up|DEGRADED|ERROR|accelerator' | tail -20
```
Expect: `active`, `NRestarts=0`, `runtime up: profile=pi module=driver_attention … accelerator=…Hailo…`, no DEGRADED.

---

## G. Model loading and real alert output

**G1. Model verified and running:**
```bash
curl -s -H "X-Device-Token: $TOKEN" localhost:8899/api/health | $PY -c '
import json,sys;d=json.load(sys.stdin);dv=d["device"]
print(dv["monitoring"]["state"], dv.get("monitoring_active")); print(json.dumps(dv.get("accelerator") or dv.get("model") or {}, indent=1)[:800])'
```
Expect: `ready True`; the accelerator reports Hailo live with an increasing inference count; the model identity carries sha256 `e69136d3…`. **FAIL:** "accelerator idle" or "model sha256 does not match" (the runtime refused the `.hef`).

**G2. Real alert output** (a person present; M5 step 8 for the timed measurement):
```bash
sudo systemctl stop sensordyme.service
$PY $V/buzzer_check.py run --out $EV/buzzer_$(date +%s).json     # every pattern, announced 3 s ahead
sudo systemctl start sensordyme.service
```
Pass (by a person): every pattern is audible and distinguishable, recorded at 50 cm and 1 m with dB(A) at 1 m. Then the end-to-end check: with the service running, a consenting driver closes their eyes past the drowsiness threshold in a stationary vehicle. Expect an audible alert, a timeline entry, and (if the cloud is configured) an app notification. Software cannot tell an audible buzzer from a disconnected one; a person must hear it.

---

**G3. Shared perception report (M7)** (read-only):
```bash
curl -s -H "X-Device-Token: $TOKEN" localhost:8899/api/health | $PY -c '
import json,sys;d=json.load(sys.stdin);m=d["device"]["monitoring"];p=m.get("perception") or {}
print(p.get("schema"), p.get("simulated"))
for n,c in (p.get("components") or {}).items(): print(n, c.get("state"), c.get("runs"), c.get("run_ms"), c.get("consumers"), c.get("reason"))
for n in p.get("nodes") or []: print(n["id"], n.get("state"), n.get("validation"), n.get("missing_capabilities"))'
```
Expect:
- `sd-perception/1 False`.
- `object_detector ok` with `consumers` including `driver_attention`, a rising `runs`, and a `run_ms` you record. That is the first real Hailo per-frame cost.
- `face_landmarks ok`.
- Node `driver_safety active`, validation `synthetic_tests`.

**FAIL:**
- `simulated: True`;
- any component `ok` while the matching capability is DEGRADED, or the reverse;
- an `object_detector` with more runs than frames analysed.

Record `runs`/frames for 60 s, with and without a custom person rule. The rule must not raise the detector's run count above one per analysed frame.

## H. Decoding parity and performance

**H1. Legacy vs reference decoding, INT8 parity** (service stopped; M5 steps 5–6 and 12):
```bash
sudo systemctl stop sensordyme.service
cd ~/hwval && $PY $V/capture_hailo_outputs.py --images ~/hwval/images --modes letterbox,stretch \
  --out ~/hwval/capture_$(date +%Y%m%dT%H%M%S)
sudo systemctl start sensordyme.service
# worker: compare_float_vs_int8.py --bundle capture_… --onnx damoyolo_tinynasL20_T.zip --out report_…
```
Pass: `status: complete`; model sha256 before = after = pin; six outputs (80/40/20 grids, 68 and 81 channels); no "class head outside [0,1]". The worker report gives INT8-vs-float error and agreement for all four decoder variants. **Decision rule:** `accelerator.yolo_decoder` stays `legacy` on this device unless the report shows the reference decoder's INT8 agreement at least matches legacy's, and the owner approves the switch (docs/DECODER_CANDIDATE.md). Switching is a config change on the device. Record it in §N, then repeat G1.

**H2. Processed FPS and latency** (service running; M5 steps 3, 7, 11). Close the app and dashboard tabs first:
```bash
$PY $V/stage_timing.py health --minutes 10 --interval 5 --out $EV/health_$(date +%s).jsonl
```
Record: processed fps p5/p50/p95, processed interval p95/max, `stalls_1s_total` (flag any value above 0), and `states_seen` (expect `["ready"]`). For per-stage timing (service stopped) and camera-to-alert latency (`ms_clock.html` plus 240 fps video), follow M5 steps 7 and 11. No threshold has been agreed: propose one from the data.

---

## I. Camera calibration (service stopped; checkerboard only; M5 step 9)

```bash
sudo systemctl stop sensordyme.service
$PY $V/capture_calibration_frames.py --stream --count 30 --interactive --out ~/hwval/calib_640x480
sudo systemctl start sensordyme.service
# worker: calibrate_intrinsics.py --images calib_640x480 --pattern 9x6 --square-mm <measured> --out intrinsics_640x480.json
```
Also check orientation: in the live feed (§L1) the driver is upright. If not, set `camera.rotate_180` in the device config (a deliberate config change; record it). Run the driver-attention EAR calibration once the driver is seated: `sudo systemctl stop sensordyme && cd $D && sudo -u $U .venv/bin/python main.py --no-display --recalibrate`, then Ctrl-C after it reports calibration saved, and `sudo systemctl start sensordyme`.

---

## J. Thermal soak (service running; 30 min; M5 step 4)

```bash
bash $V/thermal_soak.sh --minutes 30 --interval 5 --out $EV/soak_$(date +%s).csv
```
Pass: `get_throttled` stays `0x0` throughout (no sticky bits 16/18); ARM clock never below its start value; processed fps at minute 30 within ~10 % of minute 1. Record: peak SoC temperature, time to plateau, fan rpm. In-vehicle: repeat with the cabin warm and the enclosure closed, as that is the real envelope.

---

## K. Autonomy: boot, Wi-Fi loss, reconnection, reboot, restart policy

> **Journal timing (M8.3 check against the code).** Four quoted lines in K and L are written with `print()` to stdout, not the logger: `restored the last selected mode` (main.py), `switched to preset` and `rules: enforcing` (services/runtime.py), and `cloud_uplink: connection lost` / `connected to` (core/uplink.py). The unit runs Python without `-u`, so under systemd these can reach `journalctl` late or out of order with the `[PROTECTION]`/`[MONITORING]` logger lines. If a `grep` for one of them finds nothing, wait 30 s and re-run it before recording a FAIL.

**K1. Reboot and reconnection** (CHANGE: reboot):
```bash
sudo reboot
# after ~60–90 s
systemctl is-active sensordyme.service; systemctl show sensordyme.service -p NRestarts
curl -s -H "X-Device-Token: $TOKEN" localhost:8899/api/health | $PY -c 'import json,sys;d=json.load(sys.stdin)["device"];print(d["monitoring"]["state"])'
cd $D && sudo -u $U $PY -c "from core.identity import device_id; print(device_id())"   # must equal the C2 id
```
Expect: `active`, `ready`, the same device id, the app shows the device online within ~30 s of Wi-Fi coming up (uplink backoff cap), and the preset and custom rules restored (journal `restored the last selected mode`).

**K2. First boot without Wi-Fi: offline monitoring and BLE provisioning** (CHANGE: removes the known network temporarily; owner-approved; needs the E1 backup):
```bash
nmcli -t -f NAME,TYPE connection show | grep wireless > $EV/nm_profiles.txt
sudo nmcli connection down "<current SSID profile>"   # or power off the AP
```
Expect within ~10 s (`network.provisioning.poll_s`):
- monitoring continues: `/api/health` on localhost is still `ready`, and an alert still sounds (repeat the G2 end-to-end check);
- BLE advertises `SensorDyme-XXXX`;
- the app's setup flow finds it, lists networks, and joins.

The app should report success, then `nmcli -t -f ACTIVE,SSID dev wifi | grep ^yes` shows the network. Also check that a refused credential reaches the app as an error. With `scripts/ble_provision_client.py provision --ssid "$(printf 'x%.0s' {1..33})" --password x` from a Mac, expect status `error` with "longer than 32 bytes" and no nmcli attempt in the journal. (Do not put a real password on a shell command line; it lands in shell history.)

**K3. Network loss and uplink reconnection** (cloud configured):
1. With the device online and paired, power off the AP (or move the device out of range) for 3 minutes, and trigger one alert during the outage.
2. Journal: `cloud_uplink: connection lost … device continues standalone`. Retries back off up to 30 s with jitter, rather than every 2 s.
3. Restore the AP. Expect `cloud_uplink: connected to …` within about 30–45 s of Wi-Fi association, and the alert from step 1 appearing in the app exactly once.

`journalctl -u sensordyme --since "-10 min" | grep cloud_uplink > $EV/uplink_outage.txt`.

**K4. Damaged config: boots on the last-known-good, never loops** (CHANGE; restore after). M8 changed this step: a damaged `config.yaml` no longer leaves the device dark when a known-good copy exists.

The runtime records `data/config.last_good.yaml` (0600) once it has started successfully on a config, so after F4 the copy should exist.

*K4a, with a saved copy:*
```bash
sudo systemctl stop sensordyme && cp -p $D/config.yaml ~/config.good
ls -l $D/data/config.last_good.yaml          # expect: -rw------- $U ... (recorded after F4). Absent? -> K4b only, record why
echo "module: [unclosed" | sudo -u $U tee -a $D/config.yaml >/dev/null
sudo systemctl start sensordyme; sleep 60
systemctl show sensordyme -p ActiveState -p NRestarts -p ExecMainStatus
curl -s -H "X-Device-Token: $TOKEN" localhost:8899/api/health | $PY -c '
import json,sys;d=json.load(sys.stdin)["device"];print(d["config"]);print(d["monitoring"]["state"])'
grep config_fault $D/data/events.csv | tail -1
```
Expect:
- `ActiveState=active`, `NRestarts=0`, no reboot;
- `config` = `{"source": "last_known_good", "fault": "<parse error …>", "running_from": "…/data/config.last_good.yaml", …}`;
- monitoring `ready` (or the same state as before the damage);
- one `config_fault` row;
- `config.yaml` itself is still the damaged file, because the device never rewrites it.

*K4b, with no saved copy* (the pre-M8 behaviour, still the fallback):
```bash
sudo systemctl stop sensordyme && sudo -u $U mv $D/data/config.last_good.yaml ~/last_good.bak
sudo systemctl start sensordyme; sleep 15
systemctl show sensordyme -p Result -p ExecMainStatus -p NRestarts -p ActiveState
```
Expect: `ExecMainStatus=78`, `ActiveState=failed`, `NRestarts=0`, no reboot, and `journalctl -u sensordyme-recover` reporting "unit result is 'exit-code' … no reboot".

Restore both: `sudo -u $U cp -p ~/config.good $D/config.yaml && sudo -u $U mv ~/last_good.bak $D/data/config.last_good.yaml && sudo systemctl reset-failed sensordyme && sudo systemctl start sensordyme`.

**K5. Crash loop is bounded across boots** (CHANGE; optional, owner-approved; leaves a reboot budget record):
Simulate a startup crash, e.g. `sudo systemctl edit sensordyme` with `[Service]` / `ExecStart=` / `ExecStart=/bin/false`. Expect: 5 failures within ~20 s, then the unit enters `failed (start-limit-hit)`, `sensordyme-recover` logs "start limit hit: rebooting (1 of 3 …)" and the Pi reboots. After the third reboot inside 6 h it logs "leaving the service failed and the device up", and SSH stays available. Clean up with `sudo systemctl revert sensordyme && sudo systemctl reset-failed sensordyme && sudo rm /var/lib/sensordyme-recover/reboots.json && sudo systemctl start sensordyme`.

---

## L. App pairing, presets, custom rules

**L1. Secure iPhone pairing** (CHANGE: config):
1. Dashboard token: if unset or under 16 characters, generate one on the Pi with `openssl rand -hex 16`, set `dashboard.token` in `$D/config.yaml` (edit in place; never copy the repo's config over it), `chmod 600 $D/config.yaml`, and restart.
2. Backend: `python cloud/manage.py create-device "Car unit"` on the backend host. Set `cloud.enabled: true`, `cloud.device_key: dk_…` and `cloud.base_url` in the device config. Prefer `https://` with a real certificate, or `ca_cert_path` for a self-signed one. An `http://` base URL works but is announced as cleartext in the journal and the preflight (development only).
3. In the app: sign in, add the device with the claim code, then enter the dashboard token under Settings → Live feed.
4. Verify:
   ```bash
   curl -s -o /dev/null -w '%{http_code}\n' localhost:8899/api/health                             # 401
   curl -s -o /dev/null -w '%{http_code}\n' "localhost:8899/api/health?token=$TOKEN"             # 401 (M6: query token only on /stream)
   curl -s -o /dev/null -w '%{http_code}\n' -H "X-Device-Token: $TOKEN" localhost:8899/api/health  # 200
   ```
   The app shows the live feed, the device's real monitoring state, and the device id equal to the C2 id.

**L2. Presets:** this unit is `role: driver_safety` (config.yaml; `core/roles.py` allows only the `driver_safety` preset on that role). From the app, select Driver Safety (and a variant, e.g. Strict): expect it confirmed within ~2 poll intervals and the journal `switched to preset …`. Then select Elderly Monitoring: on this unit it must be **refused** with the reason `this device's role is 'driver_safety': preset 'elderly_monitoring' would stop driver_attention; allowed: driver_safety` (409 `role_forbids_preset` from the backend / 409 locally / the refusal shown in the app), and Driver Safety keeps running. That refusal is the M0.1 safety floor, not a failure. (Corrected in M8.3: this step used to expect Elderly Monitoring to switch, which the code refuses on this role.)

**L3. Custom rules:** in the app, open **Ask Jupiter** (formerly "Configure with AI"), e.g. "alert me if a phone is visible for 5 seconds"; review, then **Activate**. Expect:
- journal `rules: enforcing 1 AI-configured rule(s)` and `[MONITORING] mode=standard_plus_custom module=driver_attention base_monitoring=enabled …` (the rules version goes upstream as `rules_version`; it is not logged);
- `/api/state` → `policy.mode` is `standard_plus_custom` (custom rules run beside Driver Safety on this role; `policy` holds `role`, `protection`, `mode`, `last_rejection`, `features`, not the rules), and `/api/health` → `device.monitoring.custom_rules` lists the rule;
- holding a phone in view for 5 s produces an alert in the app. This is a custom "phone visible" rule, not the product's `phone_use` alert: a visible phone alone is not phone use.

Then select Driver Safety again and confirm the rules stop (journal `[MONITORING] custom monitoring exited: explicit mode switch`). Stop/resume: from the app, **Stop monitoring** (confirm "Stop monitoring?"), then **Resume monitoring**. Expect `[PROTECTION] STOPPED by …` / `[PROTECTION] resumed by …`, each logged exactly once.

> **M6.5 notes for K and L.**
>
> - **Device auth.** Signed commands (`docs/DEVICE_AUTH_CONTRACT.md`) default to `cloud.command_auth: permissive` on the device and the backend's default mode. On Monday leave both at their defaults: existing pairing keeps working, and a command with a bad signature is never applied.
>   - Switching to `required` is a later, deliberate step: the backend first, then each device.
>   - Do not run `scripts/migrate_device_key.py` (dry-run by default) and do not rotate keys on Monday.
> - **BLE provisioning.** It stays `network.provisioning.auth: legacy`, so K2 uses the existing app flow. The proof-of-possession flow (`docs/BLE_PROOF_OF_POSSESSION.md`) needs a setup-code label, an app build with its setup screen, and its own hardware session. Do not set `deployment: production` on Monday: with `legacy` that is a schema error by design.
> - **Person rules.** Custom rules on `person` now receive the driver's person box (fixed in M6.5; before, "nobody in the seat" fired with the driver seated). If L3 includes a person rule, record whether it fires correctly.
> - **Ask Jupiter.** In the app use the full Ask Jupiter screen (the "All examples" link or the custom-rules card). The inline Ask Jupiter bar on the device screen failed UI automation: its field did not take keyboard focus. If it works by hand on the phone, record that in `$EV/LOG.md`.
> - **`calibration.json`.** A `calibration.json` that is not a JSON object is now ignored and the driver recalibrates. Before, the device booted FAILED every time.


> **M7 notes for G, K and L.** Shared perception engine and monitoring nodes (`docs/M7_IMPLEMENTATION_MAP.md`).
>
> - **The Driver Safety card:** in the app, tapping the Driver Safety row now opens its details sheet.
>   - The sheet shows the device-reported capabilities (working / not working with a reason / unknown) and the validation tier.
>   - Activate counts as confirmed only when the device runs the mode AND monitoring is not failed.
>   - On Monday, check that the capability rows match G3 and that a refused change shows the device's reason.
> - **Companion node:** "Also count people" turns on the `people_counter` node. It reads the same detector run as driver monitoring, with no second model. Check in G3 that `people_counter` appears in `consumers` and that the detector's runs do not rise. Node alerts never sound the buzzer.
> - **New modes:** Occupancy and Workplace Safety exist but are not allowed on a `driver_safety`-role device, which refuses them by design. Do not change the role on Monday.
> - **Detector truth:** if the Hailo detector fails, the device must report `objects` DEGRADED, with face-based drowsiness still alerting. That is the M7 failure-isolation claim, and only the simulator has shown it so far. Record what happens.
> - **No model changes on Monday:**
>   - the production `.hef` and the decoder default (`legacy`) are unchanged;
>   - the EXP-08 3-class model stays experiment-only;
>   - EXP-09 (`docs/research/M7_DETECTOR_ANALYSIS.md`) is proposed and not run.

---

## M. Optional: developer remote access (Tailscale)

Developer-only, for remote debugging during validation. **Not** a consumer requirement and not part of the product image. It does **not** do Wi-Fi provisioning (BLE does). It changes the device's networking, so it needs explicit owner approval at the time; agents may not install it. If approved, install from Tailscale's signed apt repository for Debian bookworm (not a `curl | sh`), run `sudo tailscale up`, record the tailnet IP in §N, and remove it after validation (`sudo tailscale down && sudo apt purge tailscale`).

---

## N. Recording results and failures

Keep `$EV` on the Pi and copy it to `$REL/evidence-<date>/` on the Mac at the end (excluding frame-bearing directories, which follow the M5 transfer procedure). For every step write one line in `$EV/LOG.md`:

```
<UTC time> | <step id> | PASS/FAIL/SKIP | <observed value or file> | <decision / follow-up>
```

On any FAIL also save `journalctl -u sensordyme -b --no-pager > $EV/journal_<step>.txt`,
`journalctl -u sensordyme-recover --no-pager`, `vcgencmd get_throttled`, and the
`/api/health` JSON (it contains the SSID; keep it local).

---

## O. Rollback

**O1. To the previous release** (minutes; the tree was kept in §F2):
```bash
sudo systemctl stop sensordyme.service
PREV=$(cat ~/m6-previous-release-path)
mv $D $(dirname $D)/sensordyme.rc-failed-$(date +%s) && mv $PREV $D
cd ~/m6-backup-*/ && sudo tar -xzf system_glue.tgz -C / etc/systemd/system/sensordyme.service
sudo systemctl daemon-reload && sudo systemctl start sensordyme.service
systemctl is-active sensordyme.service && sha256sum /etc/systemd/system/sensordyme.service   # equals the C1 hash
```
The previous unit has no `OnFailure=`, so the recovery unit stays installed and inert. `config.yaml` and `data/` changed after the swap (for example a token set in §L1) live in the failed tree: copy them back deliberately if wanted.

**O2. To the exact pre-Monday state:** restore E1. Stop the service, then `tar -xzf sensordyme_tree.tgz -C $(dirname $D)` (the venv is not in the archive, so keep the existing `.venv` or recreate it with setup_pi.sh), then `sudo tar -xzf system_glue.tgz -C /`, `sudo systemctl daemon-reload`, and start the service.

**O3. Total:** write the E5 image back to the card. On the Mac: `diskutil unmountDisk /dev/diskN && sudo dd if=$REL/pi-sd-<date>.img of=/dev/rdiskN bs=4m status=progress`. Check the disk number twice.

**O4. Least-privilege unit** (only if a later session adopted it; not part of Monday): run the
ROLLBACK block printed by `bash scripts/release/least_privilege_plan.sh --from-dir $D`. It
reinstalls the stock unit with `install_units.sh`, removes the drop-in, and copies
`/var/lib/sensordyme` back into `$D/data`.

---

## P. Acceptance checklist

All items are **NOT EXECUTED — requires the physical device**. Tick only with evidence in `$EV/LOG.md`.

- [ ] A0 `make verify-release REF=$RC` PASS (Mac), and the systemd step PASS on a worker
- [ ] B2 power: `throttled=0x0` at idle and after the J soak; 27 W PSU
- [ ] B3 storage: no I/O errors; ≥4 GB free; cooling present
- [ ] C1–C5 baseline recorded: OS, firmware, packages, service user/dir, device id, installed-code revision, health
- [ ] D1 Hailo-8L enumerated; `parse-hef` shows 6 outputs; `fw-control identify` recorded
- [ ] D2 camera `imx708_wide_noir` listed
- [ ] E1/E2 backup complete, checksums verified off-device; E5 SD image taken and hashed
- [ ] E6 clean-install decision recorded, with reasons
- [ ] F0 device-config preflight OK, warnings recorded
- [ ] F3 `release-verify` **PASS** against the approved manifest commit
- [ ] F4 service active, `NRestarts=0`, no DEGRADED
- [ ] G1 model hash = pin; accelerator live; monitoring `ready`
- [ ] G2 every buzzer pattern heard and recorded; end-to-end drowsiness alert heard
- [ ] G3 perception report real (`simulated: False`); detector runs ≤ analysed frames with a custom rule active; `run_ms` recorded
- [ ] H1 INT8 capture complete; worker parity report produced; decoder decision recorded (default `legacy`)
- [ ] H2 processed FPS/latency recorded; `stalls_1s_total` recorded and explained if > 0
- [ ] I calibration frames captured; intrinsics computed; orientation correct
- [ ] J 30-min soak: no throttling; fps stable
- [ ] K1 reboot: same device id, `ready`, app online, preset and rules restored
- [ ] K2 offline monitoring still alerts; BLE provisioning joins Wi-Fi from the app; a refused credential reaches the app
- [ ] K3 outage: backoff observed; reconnect ≤ ~45 s; queued alert delivered exactly once
- [ ] K4a damaged config → boots on the last-known-good, `config.source=last_known_good`, `config_fault` row, no reboot; K4b with no saved copy → exit 78, unit stopped, no reboot
- [ ] K5 (optional) crash loop → at most 3 reboots in 6 h, then the device stays up
- [ ] L1 pairing: header token 200, missing or query token 401; app shows the live feed and the real state
- [ ] L2 preset switches confirmed; a role-violating preset refused with a reason
- [ ] L3 custom rule fires in the app; stop/resume each applied once
- [ ] N evidence copied off the Pi; frame-bearing data handled per M5
- [ ] O1 rollback path confirmed available (`~/m6-previous-release-path` exists, E1 backup verified)
- [ ] R1 M8 health fields as expected (`config.source=config`, no `saved_state` faults, `uplink_backlog.persistent`, accelerator `detector_available`)
- [ ] R2 `attention` capability STARTING during calibration, then READY; drowsiness alerts throughout
- [ ] R3 G3 with tracking: `tracking.updates` = detector runs; occupancy-style counts stable with one seated person
- [ ] R4 outage + power cut: queued alert delivered exactly once after power returns
- [ ] R5 (optional, separate session) proof-of-possession onboarding, BLE doc steps 1–10, all recorded
- [ ] R6 (optional) rollback guard: one deliberately broken `make deploy` rolls back by itself

---

## R. M8 additions (2026-11-12): what changed for Monday, and the extra checks

M8 (`feat/m8-product-intelligence`) changes device behaviour in ways the steps above must expect. Every M8 claim is **simulator- or test-verified only**. The steps below are where the Pi proves or disproves them.

### R0. Monday task map (task → step). Hardware-only gates are marked **HW**; everything else was already proven in software on the Mac.

| Task | Where | Notes |
|---|---|---|
| A. Initial physical inspection | B1–B3 | **HW** |
| B. Safe backup of the installed system | E1–E5 | **HW**; nothing is flashed before E5 |
| C. Clean install vs upgrade | E6 | default: upgrade in place; never factory-reset by default |
| D. Version-pinned deployment | A0–A3, F0–F4 | `make verify-release REF=$RC` must PASS; tarball from git objects |
| E. Camera and Hailo verification | D1, D2, G1 | **HW** |
| F. Legacy vs corrected decoder | H1 | **HW**; stays `legacy` unless the INT8 report justifies a switch and the owner approves |
| G. Model artifact validation | C3, F2 (`sha256sum`), F3, G1 | pin `e69136d3…960e`; the EXP-08 3-class model stays experiment-only |
| H. Camera geometry calibration | I | **HW**; the wide-lens `camera_hfov_deg` decision waits for this (R2) |
| I. Processed FPS and latency | G3, H2 (+ R3 tracking cost) | **HW** |
| J. Buzzer and alert output | G2 | **HW**; a person must hear it |
| K. Wi-Fi provisioning | K2 (legacy BLE) | **HW** |
| L. Authenticated BLE pairing | R5 (optional, separate session) | **HW** + an M8 app build on a physical iPhone |
| M. iPhone association and reconnection | L1, K1, R5 steps 8–9 | **HW** |
| N. Preset activation | L2 (+ R2) | |
| O. Custom policy activation | L3 (+ R3) | |
| P. Restart and recovery | K1, K3, K4, K5, R4 | **HW** |
| Q. Rollback | O1–O3 (+ R6) | |

### R1. New health fields (read-only, after F4)

```bash
curl -s -H "X-Device-Token: $TOKEN" localhost:8899/api/health | $PY -c '
import json,sys;d=json.load(sys.stdin)["device"]
print("config", d.get("config")); print("saved_state", d.get("saved_state")); print("uplink_backlog", d.get("uplink_backlog"))
acc=[c for c in (d.get("components") or []) if "accel" in json.dumps(c).lower()]; print("accelerator", json.dumps(acc)[:400])'
```

Expect:
- `config.source: "config"` with `fault: null`;
- `saved_state.faults: []`;
- `uplink_backlog.persistent: true` when the cloud is enabled;
- an accelerator component that does not say "detector unavailable".

**FAIL:**
- `config.source: last_known_good` on a config nobody touched;
- any `saved_state` fault: a damaged `data/device_state.json` or `monitoring.json` was kept aside as `<name>.corrupt`, so record it;
- an accelerator with `detector_available: false` while D1 enumerated the Hailo.

Field names were read from the M8 code; the exact `components` layout of `/api/health` was not checked on hardware, so record what you see.

### R2. Driver Safety changes (with G2/G3, L2)

- **Calibration.** A new capability, `attention` (the driver's calibrated forward gaze), is STARTING until calibration completes, at most ~30 s, then READY.
  - During it: drowsiness and attention-lost still alert; distraction, eyes-down, glancing and phone use do not yet.
  - The device no longer reports READY while it cannot judge gaze.
  - Check this after `--recalibrate` (§I) and record how long it took.
- **Phone use.** A phone counts only inside the driver's person box.
  - A phone that is held (a hand overlapping it) alerts after `phone_use_seconds`.
  - A still phone in a mount with both hands on the wheel does not alert.
  - With no hand evidence, it alerts as before and says "hold unknown".
  - **HW check:** a phone in a dash mount, with the driver glancing at it while both hands are visible on the wheel, must not raise phone use. Record it.
- **Looking down vs closing the eyes.** A look down (lowered lids) is no longer drowsiness; a real closure still is.
  - **HW check:** read something on the lap for 5 s and expect eyes_down, not drowsiness. Closing the eyes for 3 s with the head down must still give drowsiness.
- **Camera stalls** no longer count as evidence. Record `stalls_1s_total` in H2; a stall-heavy camera now delays alerts slightly instead of firing them early.
- **Wide lens.** `camera_hfov_deg` stays `null` (fx = frame width, about 53°). Set it only after §I measures the intrinsics and the owner approves; changing it invalidates the stored calibration.

### R3. Shared tracking and nodes (with G3, L3)

Add to the G3 printout:
```bash
curl -s -H "X-Device-Token: $TOKEN" localhost:8899/api/health | $PY -c '
import json,sys;p=json.load(sys.stdin)["device"]["monitoring"].get("perception") or {}
print(p.get("tracking")); [print(n["id"], n.get("state"), n.get("metrics"), n.get("experimental_features")) for n in p.get("nodes") or []]'
```

Expect:
- `tracking.updates` equals the detector's `runs`. One shared tracker, never one per node.
- Record `tracking.update_ms`, the first Pi measurement of tracker cost; the Mac measured 0.03–0.07 ms for a 5-person room.
- With one seated person, `people_counter` (if enabled) holds count 1 for 60 s with no entry/exit flapping.

Two people crossing can swap IDs; that is a known limit, not re-identification.

**Child Monitoring default changed.** Its experimental pose alerts (flailing, choking-hazard) are now **off by default** for new configs. Existing devices keep their own `enable_pose` value, because `migrate_config.py` only adds keys. The app shows them as experimental, with an opt-in setting. Child Monitoring is refused on a `driver_safety`-role device anyway; do not change the role on Monday.

### R4. Outage plus power cut: the queued alert survives (CHANGE: power pull, owner-approved; after K3)

1. With the cloud configured, take the AP down (as in K3).
2. Trigger one drowsiness alert. Confirm `uplink_backlog.queued` ≥ 1, and that `$D/data/uplink_spool.jsonl` exists, owned by `$U`.
3. Pull the power. Restore it with the AP still down. After boot, `uplink_backlog.restored` ≥ 1.
4. Bring the AP back. Expect the alert in the app **exactly once**; the backend dedupes by `event_key`.
5. Record the time from AP up to the event appearing.

**FAIL:** the alert is missing (spool lost: check fsync on the card), or it appears twice.

### R5. Proof-of-possession onboarding (OPTIONAL, separate owner-approved session; never before E5 and L1 pass)

This is the authenticated BLE pairing. It needs:
- a printed or noted setup code, generated **on the Pi**;
- `network.provisioning.auth: pop` on the device;
- an M8 build of the app on a **physical** iPhone. Build it from `$RC` with the owner's Apple signing team, launched **without** `-orynSimulatedBLE`.

Follow `docs/BLE_PROOF_OF_POSSESSION.md` → "On the hardware (Monday)", steps 1–10. They cover: the wrong code, the lockout countdown, another device's QR refused, a wrong Wi-Fi password, the plain-write refusal on 5d1e0003, the claim, re-provisioning without a second claim, and the MTU and timing measurements.

Rollback: set `auth: legacy` and restart. No data migration. **Do not** set `deployment: production`.

Known limits (documented, not defects):
- the lockout counter resets on a power cycle;
- offline guessing from a recorded session is possible; the 60-bit code and PBKDF2 are the defence.

### R6. Rollback guard (OPTIONAL; last; owner-approved; only if `make deploy` targets this device)

`make deploy` now:
1. snapshots the running release (`data/release/previous/`, with config.yaml at 0600);
2. deploys and restarts;
3. rolls back by itself if the unit is not stably `active` within the settle time, or if any step fails (`scripts/release/release_guard.py`).

It is hard-wired to `pi@sensordyme.local:~/sensordyme`. Skip R6 if this device's user or directory differs (§C1).

To test it: deploy a throwaway commit whose `main.py` exits 1 at import. Expect:
- `make deploy` exits 1;
- the journal shows the rollback;
- the service is `active` on the previous release (F3's `release-verify` against the previous manifest passes again);
- `release_guard.py status` shows no pending deploy.

Things only the Pi can show:
- whether `Type=notify` and `NRestarts` behave as the guard assumes under systemd 252;
- whether the rollback's `pip install` of the old pins works offline.

Record both.

### R7. Security notes (M8)

Leave every default. Do not run the new `cloud/manage.py` commands (`revoke-device`, `require-signed`, `allow-bearer`) on Monday, and rotate nothing.

**Backend before device.** The M8 backend adds tables on start (`auth_nonces`, `revoked_device_keys`, `devices.auth_mode`) and signs clock answers. Update the backend before the device:
- an M8 device set to `command_auth: required` cannot recover from a clock more than 300 s off behind a pre-M8 backend;
- at the default `permissive`, every old/new combination works.

The backend's request-nonce memory now survives restarts, so a captured request cannot be replayed after a backend restart.

### R8. Model research for the hardware session (no model change on Monday)

`docs/research/M8_CV_RESEARCH.md` recommends a single 3-class (person/phone/hand) detector as the next architecture. It is **not** for Monday.

What Monday can collect, without changing the model:
- the H1 INT8 capture;
- the per-frame Hailo cost (G3 `run_ms`; desk baseline 22.07 ms);
- Pi CPU cost for Face Mesh and Hands. Record the per-stage times from M5 step 7 if run.

A future compile of the 3-class model needs at least 1,024 NoIR calibration frames, collected only with consent under the M5 workflow.

All image-level accuracy figures in the research doc are from Open Images, not driving events.
