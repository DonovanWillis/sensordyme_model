# First physical boot (M14 "Zero Day"): the operator sequence

**Status:** written on a Mac on 2026-11-16, before the hardware arrived. **Nothing here has run on
this Pi yet.** Every "Expect" comes from the code, the earlier runbooks and the Phase 5 hardware record
(Pi 5 + Hailo-8L, HailoRT 4.20.0). Your results are the first evidence.

This page is the order of operations and the decision points. The full procedure for each step is in
`docs/FRIEND_HARDWARE_SETUP.md` (cited as **FHS step N**). Commands not shown here are there.

- Pi stages: 1, 2, 5–13 and 21–24. iPhone stages: 3, 4, 14–20 and the app half of 22.
- iPhone conventions: `<UDID>` is the identifier from stage 4 (never paste it into Git); `$REPO` is the
  repository checkout on the Mac. Stage 19 is **stationary only**: never test an alert in a moving
  vehicle or on a public road.
- Placeholders: `<user>@<host>` is whatever the owner set in Raspberry Pi Imager. Never guess it.
- **The M14 release** (`docs/RELEASE_M14.md`): release id `f3442f642a73`, tested commit
  `f3442f642a73bcb6290e656b5d46ddc5f0589e87`, package `~/sensordyme-release/transfer-f3442f642a73/`, whose
  `SHA256SUMS` has sha256 `01ecbbf20ed7667a3e4257ea9248e7c20cbad5542ceed4f21bda186fa31dc8b6`. The rollback
  candidate is M13.1, `transfer-0a597349e83c/`.

**Shell variables on the Pi:** use FHS §0.4 (`U`, `D`, `PY`, `V`, `EV`), and set `U` and `D` from step 6.
The hardware tools default to `/home/pi/sensordyme`, so always pass `--install-dir $D`.

**Results:** PASS / FAIL / BLOCKED / NOT RUN. Tool exit codes are 0 PASS, 1 FAIL, 2 BLOCKED,
3 SIMULATED. A SIMULATED result is never hardware evidence.

---

## Stages (Pi and iPhone, in order)

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

### 3. Connect the iPhone to the Mac by data cable (iPhone)

| | |
|---|---|
| **Action** | Plug the iPhone into the Mac with a USB-C **data** cable (a charge-only cable shows nothing in stage 4). Unlock the phone. When iOS asks **Trust This Computer?**, tap **Trust** and enter the passcode. |
| **Expected** | The phone shows it is charging. Finder's sidebar lists the iPhone. |
| **If it fails** | No Trust prompt: unlock the phone and re-plug. Prompt dismissed by mistake: Settings → General → Transfer or Reset iPhone → Reset → **Reset Location & Privacy**, then re-plug (this resets every app's location and privacy permissions on the phone; ask the owner first). Not in Finder: try another cable or port; a hub can drop data. |
| **Recovery** | Unplug, unlock, re-plug. |
| **Approval** | Owner (it is their phone and their passcode). |

### 4. Confirm Xcode sees the iPhone (iPhone)

| | |
|---|---|
| **Command** | `xcrun devicectl list devices` then `xcrun xctrace list devices` |
| **Expected** | `devicectl` lists the iPhone with **State `available (paired)`** and its identifier (the `<UDID>`). `xctrace` lists it under *Devices*, not *Devices Offline*. |
| **If it fails** | `unavailable`: the phone is locked, untrusted (stage 3), or on a cable without data. `connected (no DDI)` or "preparing": wait for Xcode to copy the developer disk image (Xcode → Window → Devices and Simulators shows progress; minutes on the first connection). "Developer Mode disabled": stage 14, step 3. |
| **Recovery** | Unlock, re-plug, open Xcode → Window → Devices and Simulators and wait for it to finish preparing the phone. |
| **Approval** | None (read-only). |

**M14 check (Xcode vs iOS version).** On 2026-11-16 this Mac had **Xcode 26.1.1** (iOS SDK 26.1), and
the iPhone was last seen on **iOS 26.5.2**, with Developer Mode on and paired. When stage 4 shows the
phone, run `xcrun devicectl device info ddiServices --device <UDID>`.
- **Expect:** it mounts the developer disk image and succeeds.
- **If Xcode says this iOS version is not supported, or the DDI will not mount:** the fix is a newer
  Xcode. That is a large download, needs tens of GB free, and is the owner's decision. Do the Pi
  stages meanwhile.
- The first connection also copies the phone's debug symbols (often several GB): keep at least
  10 GB free (`df -h /System/Volumes/Data`).

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
- **Do (Mac):** in `~/sensordyme-release/transfer-f3442f642a73/`, run `shasum -a 256 -c SHA256SUMS`.
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
  f3442f642a73bcb6290e656b5d46ddc5f0589e87 --out $EV/preflight`. The validation tools are unpacked as in FHS step 9.
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
- **The decoder experiment (optional, owner-approved stopped window, ~20–30 min, never timed):**
  `$PY $V/hwv_stage.py decoder-exp --install-dir $D --package ~/transfer` with the service stopped
  (`sudo systemctl stop sensordyme.service` first, `start` after; the tool never stops it itself and
  prints BLOCKED instead). It writes one directory, `~/hwval/stages/decoder_exp_<time>/`:
  - legacy and reference decoding of the **same** captured INT8 tensors;
  - parity, latency and end-to-end fps for each decoder;
  - `thermal.csv`, provenance and `SHA256SUMS`;
  - `DECISION.md`, which reads **KEEP LEGACY** unless every criterion C0–C12 passes
    (`docs/HAILO_HARDWARE_VALIDATION.md`, M14 section).
  Even then it reads only "ELIGIBLE FOR OWNER REVIEW", and it never edits config.
- **What does not change:** the decoder. `hwv_stage.py parity --decoder both` is an offline experiment
  (FHS steps 17–18). A switch is a separate owner decision.
- **Approval:** the owner approves the stopped window for `latency`.

### 14. Build and install the app (iPhone)

**Prerequisites. These are the owner's actions; no script can do them.**

1. **Apple ID in Xcode.** Xcode → Settings… → Accounts → `+` → Apple Account,
   signed in as the Apple ID that owns development team `D6ZR59ZR7Q`. This is
   the team already set in `ios/project.yml`. On 2026-11-13 the Mac had that
   team's signing certificate, but no signed-in account and no provisioning
   profile (`docs/IOS_DEVICE_BUILD.md` §a).
2. **Team.** Nothing to change: `ios/project.yml` sets
   `DEVELOPMENT_TEAM: D6ZR59ZR7Q` with `CODE_SIGN_STYLE: Automatic`. Pick
   another team only if the owner signs in with a different Apple ID. Then
   change it in `project.yml`; do not edit the generated project.
3. **Developer Mode** on the phone: Settings → Privacy & Security → Developer
   Mode → On → Restart → Turn On. The switch appears only after the phone has
   been paired with Xcode (stage 4), or after a first install attempt fails.
4. **Disk.** `df -h /System/Volumes/Data` must show at least 3 GB free.

**Command** (from `docs/IOS_DEVICE_BUILD.md` §c):

```sh
cd $REPO/ios && xcodegen generate && cd $REPO
xcodebuild -project ios/SensorDyme.xcodeproj -scheme SensorDyme \
  -configuration Debug -destination 'id=<UDID>' \
  -derivedDataPath /tmp/sd-device-dd -allowProvisioningUpdates build
xcrun devicectl device install app --device <UDID> \
  /tmp/sd-device-dd/Build/Products/Debug-iphoneos/SensorDyme.app
xcrun devicectl device info apps --device <UDID> | grep -i sensordyme
rm -rf /tmp/sd-device-dd
```

Or in Xcode: open `ios/SensorDyme.xcodeproj`, choose the iPhone as the run
destination, and press Run (⌘R).

The first launch needs one more step on the phone: Settings → General → VPN &
Device Management → the developer app entry for the Apple ID → **Trust**.

| | |
|---|---|
| **Expected** | `** BUILD SUCCEEDED **`; `devicectl` reports the app installed; `device info apps` shows `com.sensordyme.app` with the build's version. The ORYN EDGE icon appears and the app opens on the sign-in screen. |
| **If it fails** | `No Account for Team "D6ZR59ZR7Q"`: do prerequisite 1. `No profiles for 'com.sensordyme.app'`: the phone is not registered yet. Keep it connected and unlocked and rerun with `-allowProvisioningUpdates`. "Developer Mode disabled": prerequisite 3. "Untrusted Developer" at launch: the Trust step above. "No space left on device": free disk space and delete `/tmp/sd-device-dd`. |
| **Recovery** | Fix the named cause and run the same command again. A failed install changes nothing on the phone. |
| **Limits** | A free personal team's profile lasts **7 days** (`docs/IOS_DEVICE_BUILD.md` §d): after that the app will not open until it is reinstalled. Distribution to other phones (TestFlight or the App Store) needs the paid programme. It is not part of this boot. |
| **Approval** | Owner: signs in to Xcode, enables Developer Mode and taps Trust. |

### 15. Pair the camera over Bluetooth (iPhone)

On a real phone this is the **real CoreBluetooth** flow against the Pi's BLE
provisioning service (`services/ble_pop.py`, `docs/BLE_PROOF_OF_POSSESSION.md`).
It has only been run over the simulated transport.

| | |
|---|---|
| **Action** | 1. The Pi's Bluetooth window must be open: it opens when the Pi has no network, for `reconfigure_window_s` after boot, or when `always_available` is set. 2. On the iPhone, sign in (or skip sign-in), then on *Your Devices* tap **Connect over Bluetooth** (the radio-waves button). 3. Allow **Bluetooth** when iOS asks (`NSBluetoothAlwaysUsageDescription`). Allow **Local Network** when asked (`NSLocalNetworkUsageDescription`). 4. If the device uses `auth: pop`, enter the setup code from the label (`XXXX-XXXX-XXXX`). 5. Pick the Wi-Fi network and enter its password, or keep the Pi's current Wi-Fi. 6. Signed in: enter the claim code from the card. |
| **Expected** | The app finds exactly the device whose id matches the label. The device proves the code back before the password is sent. The phone enrols its own key over the proven session (`docs/LOCAL_ACCESS.md` §4.1). On the Pi, `scripts/local_clients.py list` shows this iPhone with `source: ble`. The app lists the camera as **Connected directly on this network** or, when signed in, as the account device. |
| **If it fails** | "Waiting for Bluetooth to be turned on": turn Bluetooth on (Control Center is not enough if it is off in Settings). "Still nothing after N s": the Pi's window is closed, or the phone is more than a few metres away. Check `journalctl -u sensordyme` on the Pi for the BLE service state. Wrong code: each wrong code spends a challenge. 5 failures lock the device out for 30 s, doubling. Wait, do not retry rapidly. Join failed: wrong Wi-Fi password, or a 5 GHz-only network the Pi cannot see. |
| **Recovery** | Close and reopen the setup sheet. To start clean: *Forget this camera on this iPhone* on the camera's screen, and `scripts/local_clients.py revoke <client_id>` on the Pi. |
| **Approval** | Owner (the setup code, the claim code and the Wi-Fi password are theirs). |

### 16. Open the real live feed (iPhone)

| | |
|---|---|
| **Action** | Open the camera from *Your Devices*. |
| **Expected** | The picture shows the Pi's camera and **Live video** is in the corner. **No "Simulated" badge**: the badge appears only when the overlay says `sim: true`, so on real hardware it must be absent. The feed runs over this iPhone's own sd-local-v1 session: no token is typed anywhere. |
| **If it fails** | Placeholder "Live picture on your Wi-Fi": the phone and the Pi are not on the same network, or Local Network permission was declined (Settings → Privacy & Security → Local Network → ORYN EDGE). "This iPhone needs to be paired again": the Pi answered 403 `unknown_client`; repeat stage 15. **A "Simulated" badge on a real feed is a release blocker: stop and report it.** |
| **Recovery** | Same Wi-Fi on both, permission on, reopen the screen. |
| **Approval** | None. |

### 17. AI view and Present (iPhone)

| | |
|---|---|
| **Action** | On the live camera tap **AI view** (bottom left of the picture), then **Present**. |
| **Expected** | AI view off by default. When on, it shows the device-drawn analysis state and the device's measured fps. It **never shows percentages and never uses the word "gaze"** (`docs/LIVE_OVERLAY.md`). Present opens full screen with the AI view on, the distracted-driving panel, and the mode chips. When the account server is off, the presentation state says **Direct connection**. |
| **If it fails** | "Not analysed": the device sends no overlay for this frame (camera fallback). Check the Pi's perception health. No fps: the overlay header is missing. Check the device runs a build with sd-overlay/1. |
| **Recovery** | Close Present (top-left ✕) and reopen it. |
| **Approval** | None. |

### 18. Activate Distracted Driving (iPhone)

| | |
|---|---|
| **Action** | In Present (or on the camera's *Mode* list), choose Driver Safety, the Distracted Driving product, in its Standard or Strict variant. |
| **Expected** | The chip shows a spinner and becomes selected **only when the Pi reports the new mode**. The app never selects a mode on its own. The Pi's journal (`journalctl -u sensordyme`) records the preset change, sent by `local_app` over the signed local path. A mode the device's role does not allow is refused with the device's own reason (409). |
| **If it fails** | The spinner ends with the old chip still selected: the device refused or did not confirm. Read the message under the chips. |
| **Recovery** | Choose the mode again. Nothing changes on the device unless it confirms. |
| **Approval** | Owner (it changes what the device alerts on). |

### 19. Safe stationary event test (iPhone)

**Never on a public road, never in a moving vehicle.** The vehicle is parked
with the engine off, or the Pi is on a desk. The person playing the driver
consents, and the buzzer is announced before every trigger.

| | |
|---|---|
| **Action** | With Driver Safety active and the iPhone app **open in the foreground**, the seated person looks clearly away from the road (head turned) for longer than the mode's threshold, then looks back. Use the stationary cases in `docs/SUNDAY_DEVICE_ACTIVATION.md` §5. |
| **Expected** | The Present panel shows "Head turned away" developing, then confirmed. The Pi records one `distraction` event (`data/events.csv`, `/api/events`). The why line under the picture names that event. |
| **If it fails** | No event: check the calibration and the driver seat region, and the Pi's perception health (FaceMesh). An event the person did not cause: record it as a false positive with the time; do not tune thresholds during the boot. |
| **Recovery** | Wait for the next cycle; the event history keeps everything. |
| **Approval** | Owner and the person in the seat (consent to being filmed and to the buzzer). |

### 20. Device buzzer and iPhone foreground alert (iPhone)

| | |
|---|---|
| **Action** | Repeat stage 19 once with the phone's ringer **on** and the app in the foreground. A person listens at the Pi. |
| **Expected** | **The Pi's buzzer sounds**: this is the safety alert, and it never depends on the phone. Within about 5 s (the local poll), the iPhone shows **one** banner "Driver alert" for the event. Because this is a real, non-simulated event under 60 s old, **the iPhone plays the standard notification sound exactly once**, if *Alert sound on this iPhone* is on (Settings → Alerts). The same event reached again through the account later shows **no second banner and no second sound** (`AlertSound.decide`, `docs/LOCAL_ACCESS.md` §7). |
| **Not expected** | A sound for a simulated, experimental or log-only event, or for one older than 60 s. Any sound while the app is in the background: local notifications are posted only by the app in the foreground. There is no push (APNs). |
| **If it fails** | No banner: notification permission was declined (Settings → Notifications → ORYN EDGE). No sound but a banner: Silent Mode, a Focus, or *Alert sound on this iPhone* is off. iOS rules, and the app does not override them. **No buzzer: stop. The product's core claim fails** (Sunday OC §4). |
| **Recovery** | Fix the setting and repeat stage 19. |
| **Approval** | Owner, and everyone within earshot is warned about the buzzer. |

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

### 22b. The iPhone reconnects by itself after a Pi restart (iPhone)

| | |
|---|---|
| **Action** | With the app open on the camera, restart the Pi (`sudo reboot`, with the owner's approval) or restart its service (`sudo systemctl restart sensordyme`). Do not touch the phone. |
| **Expected** | Within seconds the screen says the camera **isn't answering on this network**. It never keeps showing "protecting" or a frozen picture as live. When the Pi is back (about 1–2 min for a reboot), the app opens a new device-signed session by itself, **Live video** returns, and monitoring reads fresh from the Pi. No re-pairing. Events recorded before the restart are not announced again; after a reboot the history marks them with the camera clock note. |
| **If it fails** | Still "isn't answering" after 3 min: the Pi changed IP and Bonjour is not reaching the phone (check `_sensordyme._tcp` with `dns-sd -B _sensordyme._tcp` on the Mac), or the service did not start (`systemctl status sensordyme`). "Pair again": the Pi's `data/local_identity.pem` or `local_clients.json` changed. That is a fault on the Pi: do not delete them. |
| **Recovery** | Leave and reopen the camera screen. If the Pi's IP changed, keep the phone on the same Wi-Fi; discovery finds the new address. |
| **Approval** | Owner (a reboot stops monitoring for its duration). |

**What the Simulator already proves for the iPhone stages (and what it does not)**

* `test32_BackendDownCustomerJourney` (ios/UITests) runs stages 16–18, the app's
  half of stages 19–20 and stage 22, with the account server stopped. It runs
  against the real device protocol on the **simulated** device: scripted camera
  and models, an `SD-SIM-` identity, no Bluetooth, no Hailo, no buzzer. Its
  events are flagged simulated, so the journey checks that the phone decides
  **once** and stays **silent**. The once-only sound for a real event is
  checked in `ios/Tests/LocalEventsTests.swift` and `AlertSoundTests.swift`.
* Not proven anywhere yet:
  * installing on the phone;
  * real CoreBluetooth pairing;
  * the Secure Enclave key (`docs/LOCAL_ACCESS.md` §10 gate 6);
  * an audible buzzer or phone sound;
  * the Local Network permission prompt on a phone;
  * reconnecting across a real IP change.

### 23. Capture evidence
- **Do:** `$PY $V/collect_evidence.py --install-dir $D --expect-commit f3442f642a73bcb6290e656b5d46ddc5f0589e87 --minutes 5`.
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
