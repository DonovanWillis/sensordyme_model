# SensorDyme

A modular edge computer-vision platform built around a driver-safety camera. One
codebase, every part of the product:

- **the device** — Raspberry Pi 5 + AI HAT+ (Hailo-8L) + Camera Module 3 Wide
  NoIR, running headless as a systemd service (`main.py`, `core/`, `modules/`,
  `services/`);
- **the backend** (`cloud/`) — accounts, device pairing, alert relay and remote
  commands; runs on a Mac as a LaunchAgent or hosted (Fly.io);
- **the iPhone app** (`ios/`) — pairing, live view, presets, alert history;
- **a Mac dev profile** that runs the same pipeline on a webcam.

> **Detector: DAMO-YOLO on the Hailo-8L. No copyleft in the device image.**
> The device runs `damoyolo_tinynasL20_T` (Apache-2.0) as a checksum-pinned Hailo
> `.hef` — Model Zoo v2.15.0, sha256 `e69136d3…`, verified at install **and at
> load** (a mismatch is refused, fail closed). `requirements-device.lock` ships
> neither `ultralytics` nor `torch`; every install path is hash-pinned and
> wheels-only, and a model can only arrive through pinned provisioning.
> `docs/LICENSING.md` carries the full analysis. This is a factual account of
> dependency licences, not legal advice — get counsel before a commercial release.

---

# Quick Start — SensorDyme + iPhone

For a device that has already been through first-time setup (below):

1. **Power on SensorDyme** (plug it in). It boots on its own: the service
   starts, saved Wi-Fi reconnects, the camera and the pinned detector load, and
   the preset, zones and rules you last chose are restored. Nothing to press.
2. **Open the SensorDyme app** on your iPhone. You stay signed in; the app
   finds its server again by itself and your device appears in *Your Devices*
   with its real, current state within roughly 30 seconds of the Pi booting.
3. **Confirm the status is genuine.** The device card shows the true monitoring
   state ("All clear" appears only when the device itself reports ready and
   fresh — never from stale data). A device that is still booting shows as
   starting; a problem names its reason.
4. **Review or change the mode** from the device screen (Driver Safety and its
   variants; refusals are explained). A deliberate *Stop monitoring* is never
   persisted — after any reboot the device comes back protecting.
5. **Watch the live feed** (phone on the same Wi-Fi as the device) and open the
   activity timeline for alert history.

No terminal, no SSH, no reconfiguration, no re-pairing, no re-entering Wi-Fi.
If any of that seems needed, something is wrong — see
[Phase 9 — Troubleshooting](#phase-9--troubleshooting).

**First time?** The one-time setup below takes an afternoon: assemble the
hardware, flash the Pi, install the verified release, install the app on your
iPhone, pair, verify. After that, the loop above is the whole experience.

---

# Complete Setup Guide — From Unboxing to iPhone Monitoring

This is the authoritative zero-to-ready path. Follow the phases in order. Every
command here is either a standard OS procedure or a script that exists in this
repository; commands that can only run on the physical Pi are marked **(Pi)**.
Nothing in this guide has been run on the real hardware yet — hardware-dependent
results are marked **NOT YET VERIFIED** until the device arrives and
`docs/SUNDAY_DEVICE_ACTIVATION.md` is executed. The software behind every step
is verified at release `f63ddd9`: the full Python suite (3,956 passed, 0
failed), every software acceptance suite and the 30-journey matrix
(simulated), and the 28 iPhone UI journeys (26 in the full run; the 2 that
failed while the Mac was overloaded passed on a quiet rerun).

### How the pieces talk (read this once)

```
 iPhone app ──(HTTPS/HTTP, accounts+status+events+commands)──▶  backend  ◀──(outbound only)── Pi device
 iPhone app ──(MJPEG over your Wi-Fi, direct, token-gated)──────────────────────────────────▶ Pi device
 iPhone app ──(Bluetooth LE, first-time Wi-Fi setup only)───────────────────────────────────▶ Pi device
```

- **The backend is required** for pairing, status, history and remote commands.
  Two supported homes:
  - **A Mac on your home network** (the verified path):
    `./scripts/install_backend_mac.sh` installs it as a LaunchAgent that
    survives reboots and advertises itself over mDNS. The Mac then acts as the
    home server appliance — it is not needed *near* the device, just powered on
    and on the same network. Note it starts at login, so the Mac should
    auto-login or stay logged in.
  - **Hosted (Fly.io)** for status and alerts away from home: `fly.toml` +
    `cloud/README.md` give real HTTPS with no app changes. The device reaches
    out only (no inbound ports, NAT-friendly).
- **Live video never goes through the backend.** The app streams directly from
  the device over your Wi-Fi, so the live feed needs the phone and the device
  on the same network. Away from home you still get status, alerts and history
  (with a hosted backend); you do not get live video. Frames never leave the
  device otherwise — see `docs/PRIVACY.md`.
- **The device needs no development machine.** Once installed it talks to its
  configured backend by name (`cloud.base_url`) and runs fully standalone when
  offline: capture, inference, local buzzer alerts and the on-device dashboard
  all continue, and events queue on disk until the uplink returns.

### Development machines (not part of the product)

These exist for building and validating SensorDyme. **None of them is required
for ordinary operation** — the finished camera needs only power, Wi-Fi, a
running backend, and your iPhone.

| Machine | Role |
|---|---|
| **MacBook Pro** | development controller: Xcode/iPhone installs, release builds, SSH orchestration during install sessions, and (optionally) the home of the backend LaunchAgent |
| **oryn** (Windows desktop, RTX 4060) | primary CUDA compute worker for model training/evaluation, via `~/sensordyme-fleet/sdrun` |
| **quantcompute** (ASUS laptop, RX 7700S) | secondary evaluation worker (DirectML) |

---

## Phase 1 — Hardware assembly

### Parts

Street-price estimates, not quotes. The AI accelerator was added to the build
after the original sub-$200 list; the realistic total is now **~$250**.

| # | Part | Est. | Why this one |
|---|---|---|---|
| 1 | **Raspberry Pi 5 (4 GB)** | $60 | The compute. 4 GB, not 8/16: this workload is not RAM-bound. |
| 2 | **Raspberry Pi AI HAT+ 13 TOPS (Hailo-8L)** | $70 | Runs the pinned DAMO-YOLO detector. There is deliberately no CPU fallback on the device (`docs/ACCELERATOR.md`) — without the HAT the device boots degraded and says so. |
| 3 | **Camera Module 3 NoIR (Wide) + IR illuminator + Pi 5 camera cable (22-pin → 15-pin)** | $47 | Night vision is when drowsiness detection matters most; Wide FoV because the camera won't be dead-centre. **The adapter cable is not optional**: the Pi 5 has 22-pin mini camera connectors and the camera ships with a 15-pin cable that won't fit. IR: 850 nm is brighter with a faint red glow, 940 nm is invisible but dimmer; prefer a USB/5 V illuminator. |
| 4 | **Active Cooler (official)** | $5 | Sustained CV inference throttles an uncooled Pi 5, especially in a car. |
| 5 | **High-endurance microSD, 64 GB** | $13 | Abrupt power loss is the #1 field failure; dashcam-rated cards are built for it. 64 GB is plenty — frames are never written to disk. |
| 6 | **USB-C PD car charger (27 W+, must state 5V⎓5A) + e-marked cable** | $26 | Chargers that top out at 5 V/3 A cause throttling and brownouts under CV load. Verify later with `vcgencmd get_throttled` (`0x0` = healthy). |
| 7 | **Active piezo buzzer, 2-wire, rated for 3.3 V logic, + 2 female-female jumper wires** | $3 | The audible alert (`core/alerts.py` GPIOBuzzerSink). **Must be 3.3 V-compatible** — see the wiring rule below. |
| 8 | **Dashcam-style case + heavy-duty suction mount (¼-20)** | $29 | A cheap mount vibrates loose and un-aims the camera. 3D-printing: PETG/ABS/ASA only (never PLA — dashboards exceed 70 °C), open holes in front of camera and IR LEDs, real airflow for the cooler, and rigid — a flexing mount invalidates the head-pose calibration. |

### Assembly order

No soldering. Do all of it **before first power-on**.

1. **Clip the Active Cooler onto the Pi 5** and plug its fan lead into the fan
   connector.
2. **Mount the AI HAT+** on the GPIO stacking header and standoffs it ships
   with, and connect its **PCIe FFC ribbon to the Pi's PCIe connector, locked
   at both ends**. The stacking header passes the 40-pin GPIO through the HAT
   so the buzzer pins stay reachable. ⚠ **Confirm at assembly:** that your
   HAT+ kit includes the taller standoffs/header for use above the Active
   Cooler, and that pins 9/11 are accessible once seated — if not, stop and
   resolve the stacking before continuing.
3. **Connect the camera**: 22-pin end into one of the Pi 5's two camera
   connectors (either works; the software finds it), 15-pin end into the
   camera. **Ribbon orientation:** the exposed metal contacts must face the
   connector's own contacts at each end. ⚠ **Confirm at assembly** against
   Raspberry Pi's official Camera Module 3 / Pi 5 camera-cable guide (or the
   diagram shipped with the adapter cable) — this repository does not pin the
   per-connector orientation, and it has not been checked on this hardware
   yet. The latch lifts, the ribbon seats fully and squarely, the latch
   closes. A reversed ribbon shows no camera; re-seat with the power off only.
4. **Wire the buzzer**: positive lead → **GPIO 17 (BCM 17 = physical pin 11)**,
   negative lead → any ground pin (physical **pin 9** is adjacent). This
   matches `gpio_buzzer_pin: 17` in `config.yaml`.
   **The electrical rule (non-negotiable):** Pi GPIO is **3.3 V**, a few mA.
   - An **active piezo buzzer rated for 3 V/3.3 V logic at low current** may be
     driven directly from the pin — this is the intended part.
   - **Never connect a 5 V-only buzzer, or anything inductive or
     higher-current (speakers, relays, motors), directly to a GPIO pin.** A
     part like that needs a transistor driver off the 5 V rail.
   - ⚠ **Confirm at assembly:** if your buzzer's logic rating or current draw
     is unmarked or unknown, treat the wiring as an open decision — the
     hardware checklist (`docs/MONDAY_OPERATOR_CHECKLIST.md` §1) makes "5 V
     buzzer driven straight from GPIO" an explicit do-not-power-on stop.
5. **Insert the flashed microSD** (Phase 2), close the case, fit the mount.
6. **Aim** the camera at where the driver's face will be. You will calibrate
   for the exact mounting angle later — off-centre is fine, loose is not.

### Verify before supplying power

- HAT seated flat, PCIe ribbon locked at both ends, no bent GPIO pins.
- Camera ribbon fully seated at both ends, latches closed, contacts oriented
  as above.
- Buzzer on pins 11 (GPIO 17) and 9 (GND) — and rated for 3.3 V (the rule above).
- Cooler fan lead connected; nothing blocking airflow.
- Power supply label says **5V⎓5A** (or it's the official 27 W PSU for bench
  setup).

Then power on. First-boot electrical health check **(Pi)**:
`vcgencmd get_throttled` → `throttled=0x0`.

---

## Phase 2 — Prepare the Raspberry Pi

### Operating system

- **Raspberry Pi OS Bookworm, 64-bit (aarch64)** — this is the release's
  compatibility floor (Python 3.11, cp311 wheels, HailoRT 4.x from apt).
  32-bit or older OS versions will not run this release.
- Storage: the high-endurance microSD from the parts list.

### If the device may already contain an installation — inspect first

**Do not reflash a card you haven't looked at.** Boot it (or mount the card)
and inspect read-only **(Pi)**:

```bash
cat /etc/os-release | head -2            # which OS is on it
ls ~/sensordyme 2>/dev/null && echo "existing install present"
ls ~/sensordyme.releases 2>/dev/null     # staged-release layout already adopted?
systemctl status sensordyme 2>/dev/null | head -5
cat ~/sensordyme/config.yaml >/dev/null 2>&1 && echo "config present (do NOT copy it anywhere public: it can hold the device key)"
```

- **Existing install found** → take backups before anything else, then use the
  **upgrade-in-place** path (Phase 3B). Backup procedure: runbook sections E1
  (tree + config archive, hash-verified off-device) and E5 (full SD image) in
  `docs/MONDAY_RUNBOOK.md`; the clean-install-vs-upgrade decision criteria are
  E6. Nothing below E-backups should modify a device with history on it.
- **Blank or expendable card** → clean install (next).

### Clean install

1. **Raspberry Pi Imager** (on the Mac) → *Raspberry Pi OS (64-bit)* (Bookworm)
   → your SD card.
2. In the Imager's settings gear before writing — this is the secure way to
   enable access, instead of enabling anything after boot:
   - hostname: something you'll recognise (e.g. `sensordyme-pi`);
   - **enable SSH with a public key** (not password auth);
   - your Wi-Fi SSID/password (so the first session needs no cable) and locale.
3. Boot the Pi on it.

### Find it on the network and confirm the OS

From the Mac:

```bash
ping sensordyme-pi.local         # mDNS: Raspberry Pi OS advertises <hostname>.local
ssh <user>@sensordyme-pi.local   # your Imager-configured user and key
```

On the Pi, confirm the floor before installing anything:

```bash
grep VERSION_CODENAME /etc/os-release   # bookworm
uname -m                                # aarch64
python3 --version                       # 3.11.x
```

---

## Phase 3 — Install SensorDyme software

Everything a device needs — including one with no internet — comes from **one
verified directory**: the transfer package.

### The five entrypoints (there are no others)

| Step | One command |
|---|---|
| Build the verified package (Mac) | `make transfer-package REF=<release commit>` |
| Preflight (read-only, Pi) | `scripts/hw_validation/sunday_preflight.py` |
| Install / update (Pi) | `setup_pi.sh` (first time) · `release_guard.py stage/activate` or `make deploy` (updates) |
| Post-install verification | `release_guard.py status --wait` + the preflight again |
| Rollback | `release_guard.py switch --to previous` (or `make deploy-rollback`) |

### 3.0 The package

The current tested release is **`f63ddd9454f2`** (commit
`f63ddd9454f27a387623722fb948140dce61912d`). Its package is already built at
`~/sensordyme-release/transfer-f63ddd9454f2/` on the MacBook; rebuild it for
any release with `make transfer-package REF=<commit>` (deterministic: the same
commit yields byte-identical content). It contains the release bundle +
per-file sha256 manifest, all 40 device wheels (hash-locked, for offline
install), the production `.hef` (verified against its pin before packing), a
config template, the install scripts, and a generated **`INSTALL.md` that pins
the exact commands and hashes for that release** — when in doubt, that file
wins over this section.

Verify the package, then copy it to the Pi:

```bash
# Mac
cd ~/sensordyme-release/transfer-f63ddd9454f2 && shasum -a 256 -c SHA256SUMS   # every line: OK
scp -r ~/sensordyme-release/transfer-f63ddd9454f2 <user>@sensordyme-pi.local:transfer
```

### 3A. Fresh device (no SensorDyme install yet)

**(Pi)** — from the copied package directory:

```bash
cd ~/transfer
sha256sum -c SHA256SUMS                                  # verify again, on the device
mkdir -p ~/sensordyme && tar -xzf bundle/sensordyme-f63ddd9454f2.tar.gz -C ~/sensordyme
mkdir -p ~/sensordyme/models && cp models/*.hef ~/sensordyme/models/
cp -r wheels ~/sensordyme.wheels                         # enables the offline install
cd ~/sensordyme && bash setup_pi.sh
```

What `setup_pi.sh` does (and what you should see):

- installs apt packages: picamera2/libcamera, BLE, mDNS, alsa-utils, and
  **`hailo-all`** (HailoRT + drivers). If `hailo-all` cannot install it says so
  loudly — the device would boot degraded with no detector;
- creates the venv and installs **only the hashed lock**
  (`pip install --require-hashes --only-binary=:all: -r
  requirements-device.lock`) — no package code runs at install time;
- **verifies the model** against
  `e69136d3ebf3cc8ebaf3da5417a59962b139f3d3d67d715e9a9b1ae10963960e` and
  refuses a mismatch; MediaPipe's bundled `.tflite` files are verified too
  (and re-verified at every boot, fail closed);
- installs and starts `sensordyme.service` (systemd: starts on every boot,
  `Restart=always`, watchdog, bounded reboot budget).

Then adopt the staged-release layout **once** (this is what makes every future
update atomic and self-rolling-back):

```bash
sudo systemctl stop sensordyme.service
python3 ~/transfer/install/release_guard.py adopt --root ~/sensordyme
sudo systemctl start sensordyme.service
```

Check it's alive:

```bash
sudo systemctl status sensordyme     # active (running)
journalctl -u sensordyme -f          # live logs
```

### 3B. Update an existing (adopted) device — the normal update

From the Mac with this repo: one command, which stages beside the live release,
verifies every file against the manifest, switches with a single atomic
`rename(2)`, and **rolls itself back** (no network, no pip) if the new release
doesn't stay healthy:

```bash
make verify-release REF=<commit>       # must print: RELEASE VERIFICATION: PASS
make deploy DEVICE=<user>@sensordyme-pi.local DIR=sensordyme REF=<commit> ROLE=driver_safety
```

By hand with only the package (e.g. no internet on the Pi): the exact staged
commands, with this release's hashes filled in, are in the package's
`INSTALL.md` §B. A deploy interrupted by a power cut leaves a `pending.json`
that the next deploy refuses until resolved — `docs/DEPLOYMENT.md` §4 covers
both resolution commands.

**Rollback (any time):**

```bash
# on the device, no Mac needed:
python3 ~/sensordyme/scripts/release/release_guard.py switch --root ~/sensordyme --to previous --user $(id -un)
# or from the Mac:
make deploy-rollback DEVICE=<user>@sensordyme-pi.local DIR=sensordyme
```

Device state (`config.yaml`, calibration, events, the model) lives outside the
releases and survives every switch in both directions.

---

## Phase 4 — Configure and verify the AI camera

### One read-only command checks everything

**(Pi)** — from the installed tree (the preflight ships with the release):

```bash
~/sensordyme/.venv/bin/python ~/sensordyme/scripts/hw_validation/sunday_preflight.py \
    --install-dir ~/sensordyme --expect-commit f63ddd9454f27a387623722fb948140dce61912d
```

It writes one JSON+Markdown summary and prints the next manual step. The checks
and their expected answers (each is **NOT YET VERIFIED** on hardware until it
actually passes on your Pi):

| Check | What "good" looks like |
|---|---|
| Camera | `rpicam-hello --list-cameras` lists `imx708_wide_noir` |
| Hailo-8L | `lspci` shows "Hailo Technologies"; `/dev/hailo0` exists; `hailortcli scan` finds one device |
| Production model | sha256 of `models/damoyolo_tinynasL20_T.hef` equals the pin `e69136d3…` |
| Decoder config | `pins` check: config / setup_pi.sh / model registry agree, decoder **`legacy`** (do not switch it — the alternative decoder is a measured candidate awaiting on-device INT8 parity, `docs/DECODER_CANDIDATE.md`) |
| Facial landmarks | MediaPipe `.tflite` files match their registry pins (also enforced at every boot) |
| Service | unit active, zero restarts, correct working directory |
| Health truth | `/api/health` reports ready; accelerator `running on the Hailo-8 accelerator` with rising inference count; no invariant violations |

### Role and monitoring

The device ships with `role: driver_safety` and `module: driver_attention` in
`config.yaml` — the production configuration; nothing to select. The role is a
safety floor: presets that would stop driver monitoring are refused by design
(you'll see the refusal, with its reason, if you try from the app). Confirm
monitoring is genuinely operational **(Pi)**:

```bash
TOKEN=<dashboard token>   # set in Phase 6
curl -s -H "X-Device-Token: $TOKEN" localhost:8899/api/health | python3 -m json.tool | head -40
# monitoring state "ready", protection "protecting", camera "capturing", accelerator "ok"
```

The first ~10 seconds after a start calibrate the driver's eye baseline and
neutral head pose; the state is `starting` until then, and the app never says
"All clear" before the device itself reports ready. After mounting the camera
in its final position, recalibrate once while looking at the road normally:

```bash
sudo systemctl stop sensordyme
cd ~/sensordyme && sudo -u $(id -un) .venv/bin/python main.py --no-display --recalibrate   # Ctrl-C after "calibration saved"
sudo systemctl start sensordyme
```

### Buzzer test, safely

The buzzer is loud at 50 cm — warn anyone nearby, then run the dedicated stage
in a stopped window (it stops monitoring, runs every production alert pattern,
and always restarts the service):

```bash
~/sensordyme/.venv/bin/python ~/sensordyme/scripts/hw_validation/sunday_preflight.py \
    --install-dir ~/sensordyme --expect-commit <commit> --with-stopped-stages --with-buzzer
```

A person must actually hear each pattern — the stored event field `output:
sounded` means "the GPIO pattern was driven", never "someone heard it". **No
audible alert is a hard stop** (the product's core claim), not a tuning issue.

**Model policy:** the production `.hef` + `legacy` decoder is the only shipped
configuration. Experimental artifacts (the EXP-08 3-class model) are registry-
marked experiment-only and are not promoted by any install path. No INT8
accuracy or real-camera performance number is claimed anywhere until the
on-device stages above produce one.

---

## Phase 5 — Install the iPhone app

Personal Apple ID, free personal team, no App Store or paid membership needed.
Full detail and current blockers: `docs/IOS_DEVICE_BUILD.md`. One-time, ~5
minutes of clicking plus the build:

1. **Sign into Xcode**: Xcode → Settings… → Accounts → `+` → your Apple ID
   (the one that owns the personal team in `ios/project.yml`). This is the one
   step no script can do.
2. **Connect the iPhone by USB cable** (first time; Wi-Fi works after). Unlock
   it and tap **Trust This Computer**.
3. **Developer Mode** on the phone: Settings → Privacy & Security → Developer
   Mode → on → the phone restarts → confirm. (If the switch isn't visible yet,
   run step 4 once — the first install attempt makes it appear.)
4. **Build and install** — either open `ios/SensorDyme.xcodeproj` in Xcode,
   pick your iPhone in the run-destination bar, press **Run (⌘R)** — or from
   the terminal:

   ```bash
   cd ios && xcodegen generate && cd ..
   xcodebuild -project ios/SensorDyme.xcodeproj -scheme SensorDyme \
     -configuration Debug -destination 'id=<your iPhone's UDID>' \
     -derivedDataPath /tmp/sd-device-dd -allowProvisioningUpdates build
   xcrun devicectl device install app --device <UDID> \
     /tmp/sd-device-dd/Build/Products/Debug-iphoneos/SensorDyme.app
   ```

   Signing is already configured in `ios/project.yml` (automatic signing); on a
   first build Xcode mints the provisioning profile itself once you're signed
   in with the phone attached.
5. **Trust the developer** on the phone at first launch: Settings → General →
   VPN & Device Management → trust your certificate.
6. **Permissions the app asks for, when it first needs them:** Local Network
   (to find your server and camera — declining makes discovery impossible, and
   the app says so), Bluetooth (only when you open device setup), and
   Notifications (when you first open a device's screen; alerts are local
   notifications — no push service to configure).

**Know it's the real device build:** it's running on your physical iPhone (the
Simulator can't do Bluetooth), and the release-configuration audit
(`ios/Tests/release_config_audit.sh`) proves a Release product carries no debug
affordances or simulated transports.

**Free-team expiry:** a personal team's app signature lasts **7 days**. After
that the icon stays but won't launch — plug in (or share Wi-Fi with a paired
Xcode) and press Run again. Your session, paired camera and settings all
survive reinstalls; only deleting the app loses them.

---

## Phase 6 — Pair SensorDyme with your iPhone

### Two different codes — don't mix them up

| Code | Looks like | Comes from | Used for |
|---|---|---|---|
| **Claim code** | `1A2B-3C4D-5E6F` (hex) | printed once by the backend's `create-device` — "the card in the box" | adding the device to **your account** |
| **Setup code** | `XXXX-XXXX-XXXX` (letters+digits) + QR | generated **on the device**, printed on its label | proving to the device over **Bluetooth** that you're its owner (only when `auth: pop` is enabled; the default `legacy` build skips this screen) |

### One-time operator step (before the app can pair)

The device cannot register itself with the backend — a device row and its
credentials are created deliberately, once, where the backend runs:

```bash
# on the backend host (Mac):
.venv/bin/python cloud/manage.py create-device "Car unit"
#   -> prints device_key (dk_…)  and  claim_code (1A2B-3C4D-5E6F)
```

Then put the credentials on the Pi **(Pi)** — edit `~/sensordyme/config.yaml`
in place (never copy the repo's config over it):

```yaml
cloud:
  enabled: true
  base_url: http://sensordyme-backend.local:8787   # the backend's own stable mDNS name — never a bare IP (docs/NETWORKING.md)
dashboard:
  token: <output of: openssl rand -hex 16>          # gates the live feed
```

Put the `dk_…` key in `cloud.device_key` (or better, a 0600 file via
`cloud.device_key_file` — `scripts/migrate_device_key.py`), then
`chmod 600 ~/sensordyme/config.yaml && sudo systemctl restart sensordyme`.
Keep the claim code and the dashboard token for the app steps below. For a
hosted backend use its `https://…` URL as `base_url`.

### The pairing journey in the app

1. **Power on SensorDyme.** Whenever it has no Wi-Fi it advertises for setup;
   after each boot it also accepts setup for 5 minutes even when online —
   so to re-run setup on a working device, just power-cycle it.
2. **Open the app and sign in** (create the account on first use). The app
   finds the backend by itself on your Wi-Fi (allow Local Network when asked).
3. Tap the **antenna button — "Connect over Bluetooth"** on the home screen.
   Allow Bluetooth when asked.
4. **It finds "SensorDyme-XXXX"** and shows it; tap **Connect**. (It won't
   appear in iOS Bluetooth Settings — that's normal; find it in the app.)
5. **Prove it's yours** *(only on a `pop`-configured device)*: type the setup
   code from the device's label, or paste the QR text. Wrong code: the device
   refuses, and after 5 tries locks setup for a while — the app counts the
   lockout down. The default development build (`auth: legacy`) skips this
   screen entirely.
6. **Choose its Wi-Fi**: the list is what the *device* can see. Pick the
   network, enter the password (8+ characters). The device joins and — by
   design — drops Bluetooth when it succeeds. The joined network is **saved
   with autoconnect** on the device, permanently.
7. **Add it to your account**: enter the **claim code** (the app reminds you:
   "Not the setup code — it's on the card in the box").
8. **Wait for it to report in**: the app polls until the device shows online
   (typically well under a minute).
9. **Choose what it watches for** (Driver Safety) and **Start watching** — or
   "Not now".
10. **Live feed token**: on the device's screen, Settings → Live feed → enter
    the dashboard token from the operator step. Status and alerts work without
    it; the video needs it.

**Already-on-network device** (configured over SSH, no Bluetooth needed): the
**＋ Add device** button keeps a claim-only path — tap through to the pair step
and enter the claim code.

### When pairing fails

The app explains each failure in place; the facts behind the messages:

| Symptom | Cause / fix |
|---|---|
| Device never found over Bluetooth | It's online (window closed): unplug/replug — setup is open for 5 min after boot. Or Bluetooth permission denied: the app shows an Open Settings card. |
| "The setup code was wrong" | Check the label; each challenge allows one guess; 5 misses = device-held lockout (30 s doubling to 1 h, survives reconnects; a power cycle clears it). |
| Wi-Fi join fails | Usually the password; also captive-portal networks and 5 GHz-only networks. A phone hotspot is the most reliable fallback. The device stays on its previous network if it had one (a failed join rolls back). |
| "Already on another account" (409) | By design there is no self-service release. Sign in with the first account — its Wi-Fi setup did still apply. Account deletion releases its devices. |
| "That claim code isn't one the server knows" (404) | Typo, or the backend you reached isn't yours (the app pins your server's identity — Settings → Manual server address → Forget remembered server if you rebuilt the backend). |
| Joined Wi-Fi but "hasn't reported to the server yet" (120 s) | On a real device this almost always means the operator step is incomplete: `cloud.enabled` false, wrong `device_key`, or a `base_url` that doesn't resolve. Check **(Pi)** `sudo journalctl -u sensordyme -n 50 \| grep cloud_uplink` — it prints exactly what it can't reach and why. |
| Too many claim attempts (429) | Rate-limited (10/IP/5 min). Wait a few minutes. |

---

## Phase 7 — Automatic reconnection after first setup

This is the designed normal experience, and each piece is implemented and
covered by named tests (simulated end-to-end; the physical run is the Phase 8
checklist):

**Power on → monitoring, untouched:**

| What happens by itself | How |
|---|---|
| Service starts on boot, restarts on failure | systemd `WantedBy=multi-user.target`, `Restart=always`, watchdog; bounded reboot budget as last resort |
| Wi-Fi rejoins | NetworkManager autoconnect on the saved profile (BLE provisioning saves it `autoconnect yes` and prefers the newest network) |
| Identity restored | `SD-XXXXXXXX` is derived from the SoC serial — stable across reboots, reflashes of the same board, and network changes |
| Credentials restored | `cloud.device_key`/`base_url` + dashboard token live in the device's config (0600); a damaged config boots on the last-known-good copy and reports the fault |
| Monitoring configuration restored | preset, zones, companion nodes (`data/device_state.json`) and custom rules (`data/monitoring.json`) are re-applied at boot, re-validated against the role floor |
| A deliberate Stop is **not** restored | by design: a reboot always comes back protecting (`stop` lives only in memory and the backend withdraws an undelivered stop after 30 s) |
| Backend uplink reconnects | 2 s cadence, backoff capped at 30 s, instant recovery on first success; events queued on disk through outages and power cuts, delivered exactly once |
| Models verified, honestly | the `.hef` and `.tflite` pins are re-checked at every boot; a mismatch runs degraded **and says so** — the device never reports protection while a required component is missing |

**Open the app → your device, signed in, true state:**

| What happens by itself | How |
|---|---|
| Still signed in | the session token lives in the Keychain; **the backend renews it once per app launch** (`/v1/session/refresh`), so a session used at least once a month never ages out onto the sign-in screen |
| Server found again | known addresses are health-checked first, then Bonjour; a discovered server must prove the pinned identity; 20 s heartbeat + revalidate-on-foreground afterwards |
| Device list + real state | fetched on open and every 10 s; "All clear" is shown only from a fresh, device-reported ready |
| Live feed reconnects | direct LAN stream re-uses the cached address, re-discovers when stale, and if the Pi finishes booting *after* you opened the app, the device list takes a throttled re-look at the LAN once the backend sees it online |
| Paired camera remembered | the linked camera id + address and the device token survive relaunches, re-installs, and even session expiry |
| Revoked/expired credentials handled truthfully | an expired or signed-out session lands on sign-in with an explanation; an operator-revoked device key is named as the first reason on the device card (never a bare, indefinite "status unknown") |

**What iOS actually permits in the background (no overpromising):** reconnection
above happens **when you open the app** (or return to it). In the background
the app does not keep connections alive; alert delivery while closed goes
through the notification path, and opening the app is what refreshes state.
There is no claim of continuous background streaming or polling — Apple does
not permit it for an app like this.

**What the app needs where:** on your home Wi-Fi — everything. Elsewhere —
status, alerts, history and commands via a hosted (HTTPS) backend; live video
is LAN-only by design. With the Mac backend and the phone away from home, the
app says truthfully that it can't reach the server.

**Power-cycle test plan** (run after first setup, and after any update): unplug
the device mid-monitoring → plug it back in → do not touch anything → within
~2 minutes the app's device card must return to the same preset, protecting,
with the same device id, and a test look-away must alert. That exact loop is
Phase 8's items 12–13 and `docs/SUNDAY_DEVICE_ACTIVATION.md` steps 16/20/25
(runbook K1/K3).

---

## Phase 8 — Verify the complete product

Every row has a concrete method; rows marked **HW** are **NOT YET VERIFIED**
until run on the physical device (they are the Sunday session's steps —
`docs/SUNDAY_DEVICE_ACTIVATION.md` has each one's full procedure, expected
values and rollback).

| # | Check | How to verify | Status |
|---|---|---|---|
| 1 | Pi boots | power on → `systemctl status sensordyme` active | HW |
| 2 | Camera works | preflight `camera` → `imx708_wide_noir`; live feed shows the seat | HW |
| 3 | Hailo recognised | preflight `hailo` → `/dev/hailo0`, `hailortcli scan` | HW |
| 4 | Models load, pins hold | preflight `model_artifact` + `/api/health` accelerator `ok`, inferences rising | HW |
| 5 | Monitoring health correct | `check_live_status.py --device …` → zero invariant violations | HW |
| 6 | Paired to the account | app *Your Devices* row online; backend `hardware_id` = device id | HW |
| 7 | iPhone sees it | device card live state = device's own `/api/health` state | HW |
| 8 | Distracted Driving activates | select in app → confirmed only when the device reports running it | HW |
| 9 | Live preview works | LIVE badge; stream headers fresh (`check_live_status.py --stream-seconds 6`) | HW |
| 10 | Controlled events appear | scripted scenarios (consenting adult, stationary; Sunday §5) → alert + one history row each | HW |
| 11 | Buzzer verified safely | buzzer stage + a human listener at 50 cm/1 m | HW |
| 12 | Reconnects after reboot | the power-cycle loop above; same id, same preset, app online ≤ ~30 s | HW |
| 13 | Survives network interruption | AP off 3 min: local alert still sounds; on reconnect the queued alert is delivered exactly once | HW |
| 14 | Stopped-device camera privacy | while stopped, a remote stream request gets `camera_off_while_stopped`, no frames (Sunday step 22) | HW |
| 15 | Settings restored | preset, zones, rules back after reboot (also pinned by `tests/test_m8_lifecycle_e2e.py`, `tests/test_m85_journeys_app.py`) | HW (sim ✓) |
| 16 | No false protection claims | cover the lens: alert within ~2.5 s, never a silent "All clear" (also `tests/test_m84_reliability_recovery.py`) | HW (sim ✓) |

Software-side, the same claims are already held by `make acceptance-full`: all
30 owner journeys, 893 checks, with the real backend, real device runtime and
real app code — camera, models and radios simulated, and every report labelled
`simulated: true`.

---

## Phase 9 — Troubleshooting

First reflex for anything device-side: **(Pi)** `sudo journalctl -u sensordyme
-n 50 --no-pager` — the device explains itself. For app-vs-server confusion:
**(Mac)** `.venv/bin/python scripts/diagnose_backend_mac.py <pi-hostname>` —
one report that says which side is actually broken.

| Problem | Diagnose → fix |
|---|---|
| Pi won't boot | Red PWR LED only / no green activity: re-seat the SD card, re-flash if unreadable elsewhere. Check PSU is 5V/5A (`vcgencmd get_throttled` once up). |
| Camera not detected | `rpicam-hello --list-cameras` (service stopped — it holds the camera while running: a busy camera is BLOCKED, not broken). Nothing listed → power off, re-seat both ribbon ends, check orientation (Phase 1.3). |
| Hailo missing | `lspci \| grep -i hailo`; nothing → power off, re-seat the PCIe FFC; `dtparam=pciex1` and the HAT EEPROM per `docs/MONDAY_RUNBOOK.md` D1. `hailortcli scan` needs the service stopped. |
| Model file invalid | Health says `model integrity: … refused` — the designed fail-closed state. Re-stage the release (Phase 3B) or `release_guard.py switch` to a good one. **Never edit a pin to match a file.** |
| Decoder mismatch | preflight `pins` FAIL → the config was hand-edited; restore `accelerator.yolo_decoder: legacy`. |
| Not discoverable over Bluetooth | It's online (window closed) → power-cycle for a 5-min window. `auth: pop` with a broken code file keeps Bluetooth shut *by design* — health says "setup code unusable"; fix the file (0600, service user). |
| Wrong setup code / lockout | See Phase 6 failure table. |
| Wi-Fi provisioning fails | Phase 6 failure table; the device rolls back to its previous network, so nothing is lost. |
| "Already claimed" | Phase 6 failure table (409). |
| iPhone can't connect / app offline | `diagnose_backend_mac.py` (above): is the backend process actually running? Is its mDNS record current? Then: phone on the same Wi-Fi? Local Network permission granted? A guest/hotel network with client isolation blocks discovery entirely — phone hotspot is the reliable fallback (`docs/NETWORKING.md`). |
| Device shows offline but works locally | Classic cause: `cloud.base_url` holds a bare IP or a dead hostname. The journal names the exact address it can't reach. Fix: `http://sensordyme-backend.local:8787` (the backend's own stable name), restart the service. |
| No video feed | Same-LAN only. Then: dashboard token entered in the app (Settings → Live feed) and equal to `dashboard.token`? The device rejects a token with a stray trailing space (the app trims; a hand-typed curl might not). While *stopped*, no remote video is the privacy design, not a fault. |
| Degraded monitoring reported | Read the reason in the app or `/api/health` — it names the component. The device deliberately never hides this. Lens covered/dirty → `attention_lost` alerts; that's detection, not a fault. |
| Buzzer silent | Event shows `output: sounded` but no sound → wiring (pin 11/GND), buzzer type (3.3 V-rated? active, not passive?), or a driver stage needed (Phase 1.4). This is a hard stop until fixed. |
| Device restarts repeatedly | `systemctl status` + journal. A config error exits 78 and *stops* (no loop) with the reason; a damaged config auto-falls back to last-known-good and reports `config_fault`. Thermal: `vcgencmd measure_temp`, check the cooler. |
| Loses network | `nmcli connection show` — profile present with autoconnect? Signal? The device keeps monitoring and alerting locally throughout; events queue and deliver once on return. |
| Update fails | It rolls itself back automatically and records why (`data/release/rolled_back.json`). A deploy interrupted mid-flight: `release_guard.py confirm-or-rollback` resolves it either way (`docs/DEPLOYMENT.md` §4). |
| Need to go back | Phase 3's rollback one-liner. Releases keep their venv and config, so rollback needs no network and loses no state. |

**Never** reset/reflash or delete user data as a first step: the config holds
the device's credentials, `data/` holds its calibration and history, and nearly
everything above is diagnosable read-only. Backups (runbook E1/E5) come before
any destructive recovery.

---

# Development

Everything below is for working on SensorDyme itself.

## Source of truth

This repository (`sensordyme_model`) is the one SensorDyme codebase: the
on-device detector (`main.py`, `core/`, `modules/`, `services/`), the backend
(`cloud/`), the VLM rule interpreter (`jupiter/`), and the iOS app (`ios/`).

What is deliberately **not** in git, and must be carried by hand when moving a
running installation between checkouts (all gitignored):

| Path | What it is | If it is missing |
|---|---|---|
| `cloud/sensordyme.db` | accounts, devices, events | backend starts empty |
| `cloud/.jwt_secret` | session signing key (mode 600) | a new one is minted; every signed-in phone must sign in again |
| `cloud/.backend_id` | stable backend identity (mDNS TXT + `/v1/health`) | a new one is minted; phones that pinned the old one refuse it until Settings → Manual server address → Forget remembered server |
| `data/` | device calibration, `monitoring.json`, preflight results, datasets | recalibration on first run |
| `ios/SensorDyme.xcodeproj`, `ios/Info.plist` | generated by `cd ios && xcodegen generate` from `ios/project.yml` | regenerate; never hand-edit |

Stop the backend before copying the database, or copy it with
`sqlite3 <old>/cloud/sensordyme.db ".backup cloud/sensordyme.db"`.

## Quickstart: Mac (30 seconds)

```bash
./setup_mac.sh
source .venv/bin/activate
python main.py
```

A window opens on your webcam with the live face mesh, EAR/yaw readouts and the
full alert pipeline; the first 10 seconds calibrate your EAR baseline and
neutral head pose. `make dev-debug` adds every raw signal to the overlay.
Events land in `data/events.csv`.

## The virtual device and acceptance

`make sim` runs a complete virtual device against the real backend;
`make acceptance` runs every software suite plus the owner's 30-journey matrix
(~25 min, no Simulator), and `make acceptance-full` adds all 28 XCUITest
journeys (`docs/ACCEPTANCE.md`, `docs/VIRTUAL_DEVICE.md`, `docs/DIGITAL_TWIN.md`).

## mac_max profile: what it's for (and what it isn't)

**The pi profile is the shipping product.** `mac_max` is an additive, Mac-only
profile for live demos, ground-truth comparisons and a GPU-backed accuracy
ceiling. It requires Apple Silicon + torch MPS and never affects `profile: pi`
defaults.

```bash
make dev-max      # = python main.py --profile mac_max --debug
```

What it layers on (in order of signal impact): iris-refined gaze fusion, tiered
drowsiness (microsleep → drowsiness) with a personal blink-rate baseline,
confidence-stacked phone/object fusion (phone alone is log-only; phone + hand +
eyes-off-road is the loud tier), full-frame YOLO11-x every frame on MPS, and
~5-frame rolling-median smoothing of every raw signal. All of it funnels
through the same `core/alerts.py` AlertManager and event log as the pi profile.
Threshold citations (NHTSA glance guidance, PERCLOS cutoffs, blink norms) are
comments in `config.yaml` under `mac_max:`, next to the numbers they justify.

### The frontier VLM judge (mac_max only)

A slow, precise second tier: every `vlm.interval_s` a burst of frames goes to a
vision model and returns a strict-JSON verdict; an alert fires only after
`confirm_judgments` consecutive confident verdicts, so one flaky classification
never makes a sound. Backend-pluggable via `vlm.provider`:

| `provider` | Runs | Frames leave the Mac? |
|---|---|---|
| `anthropic` (default) | Claude vision via the Anthropic API | Yes (API) |
| `ollama` | a local model (e.g. `qwen2.5vl:7b`) | **No — fully on-device** |
| `openai_compatible` | NVIDIA NIM, LM Studio, vLLM, Gemini's OpenAI endpoint | depends on `base_url` |

Graceful without credentials: no key → one clear line, the judge disables
itself, the local pipeline continues. **Privacy tradeoff, stated plainly:**
with `vlm.enabled: true` frames are transmitted (in memory, never to disk) to
the configured API. Mac-only, announced at startup, one config line to turn
off; the pi profile never constructs the judge.

### Measuring accuracy

`make accuracy` walks a timed scenario protocol and grades `data/events.csv`
against it (per-phase recall, false positives, a mounted-phone control);
`benchmarks/run_benchmarks.py --profile pi|mac_max --video clip.mp4` is the
same-video FPS/latency comparison. See `docs/EVALUATION.md` for the real-video
evaluation pipeline and `docs/DETECTION_THRESHOLDS.md` for the detector
confidence pins. For behaviour tuning, every `driver_attention` knob is
documented next to its default in `config.yaml`.

## Architecture

```
sensordyme/
├── main.py                  # entrypoint: config, camera, module, runtime
├── config.yaml              # module: driver_attention  <- one line switches the product
├── setup_pi.sh              # one-shot Pi install (hashed lock, pinned model, systemd)
├── core/                    # camera, events, alerts, detectors/accel, state model,
│                            #   identity, config LKG, uplink, secret store, releases
├── modules/                 # driver_attention (product), occupancy, fall_detection,
│                            #   baby_monitoring, comparison modules
├── services/                # runtime coordinator, dashboard, BLE provisioning (+PoP)
├── cloud/                   # FastAPI backend: accounts, claim, events, commands
├── ios/                     # the SensorDyme (ORYN EDGE) iPhone app + logic tests
├── sim/                     # the virtual device / digital twin
├── evaluation/, research/   # eval harness; CV research (fleet-run)
├── scripts/                 # hw_validation/, release/, sim/, provisioning tools
├── deploy/                  # systemd, launchd, avahi, polkit, sudoers, udev
├── tests/                   # ~3,950 tests; no camera or hardware required
└── benchmarks/              # latency/FPS/thermal + accuracy protocol
```

Key design docs: `ARCHITECTURE.md`, `docs/STATE_MODEL.md` (monitoring truth),
`docs/EVENT_PIPELINE.md` (event identity and delivery evidence),
`docs/VIDEO_TRANSPORT.md` (stream liveness), `docs/BEHAVIOR_ENGINE.md`,
`docs/DEVICE_AUTH_CONTRACT.md` (signed commands), `docs/CREDENTIAL_STORAGE.md`,
`docs/LEAST_PRIVILEGE.md`, `docs/PRIVACY.md`.

## Driver attention module (how detection works)

Calibrated, per-user, and blink/glance-proof by construction: a personal EAR
baseline + neutral head pose learned in the first 10 s (`--recalibrate` after
re-mounting; `camera_hfov_deg: 102` for the Wide lens); sustained-condition
tracking in monotonic time so blinks and mirror checks reset cleanly; PERCLOS
for the heavy blinker; separate, more sensitive eyes-down detection; full-frame
DAMO-YOLO on the Hailo plus MediaPipe Hands for phone/food context — **a
visible phone alone is never an alert** (mounted GPS is legitimate; M8.5:
`phone_use` additionally requires a handling cue — a mounted, unheld phone you
glance at is judged by eyes-down/distraction rules only); face-lost as its own
risk signal; and RiskFusion combining everything into one smoothed score with
corroboration for compound danger. Experimental alert families
(`elevated_risk`/`high_risk`/`erratic_hand_movement`) are logged observations,
never audible, by default. Full rationale in the module docstrings and
`docs/BEHAVIOUR_REASONING.md`.

## Privacy design statement

**SensorDyme never writes a camera frame or video clip to disk, ever.** Only
derived signals are persisted (events, numeric telemetry). Frames exist as
in-memory arrays for one loop iteration. The single frame-producing point is
marked with a grep-able comment:

```bash
grep -rn "PRIVACY: frames are never persisted" .
```

The one deliberate, opt-in exception is the Mac-only VLM judge above. On the
device: remote live view is token-gated; while monitoring is stopped, no camera
picture reaches a remote viewer unless the owner explicitly enables camera-only
mode (`docs/PRIVACY.md`).

## Testing

```bash
make test            # the Python suite (~3,950 tests; no camera needed)
make test-ios        # the Swift logic tests (~1,700 checks)
make acceptance      # every suite + the 30-journey matrix (evidence in data/acceptance/)
make verify-release REF=<commit>   # the release gates
```

## Makefile targets

| Target | What it does |
|---|---|
| `make dev` / `dev-debug` / `dev-max` | run on the Mac webcam (normal / tuning overlay / mac_max) |
| `make test` / `test-ios` / `check` | Python suite / Swift logic tests / every CI gate |
| `make sim` / `sim-e2e` / `acceptance` / `acceptance-full` | virtual device / e2e / full software acceptance (+iOS) |
| `make transfer-package REF=` | the verified install package for a device |
| `make deploy DEVICE= DIR= REF=` | staged, verified, atomic deploy with auto-rollback |
| `make deploy-adopt` / `deploy-status` / `deploy-rollback` | one-time layout adoption / outcome / switch back |
| `make bench` / `report` | benchmarks / weekly PDF report |
