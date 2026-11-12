# Monday operator checklist (M8.1): what each runbook step needs, risks and proves

This is a one-page companion to `docs/MONDAY_RUNBOOK.md`. The runbook has the commands; this page classifies every step so the operator knows before starting:
- whether it needs hands on the device;
- whether it is read-only;
- whether the backup must exist first;
- whether the owner must approve it at that moment;
- whether it interrupts monitoring;
- what counts as success;
- what triggers a rollback.

**Not executed. No step has run on the Pi.** Prepared on 2026-11-12 from the code at the M8.1 release candidate.

Legend:

| Column | Meaning |
|---|---|
| Phys | needs physical access (hands, cables, power, ears) |
| RO | read-only: changes nothing on the device |
| Bkp | E1 and E5 must be complete and verified first |
| Appr | the owner approves at that moment |
| Mon | interrupts monitoring (service stopped, reboot, network or config change) |

## 0. Before touching the device (Mac only)

| Step | Phys | RO | Bkp | Appr | Mon | Success | Stop / rollback trigger |
|---|---|---|---|---|---|---|---|
| A0 `make verify-release REF=$RC` | – | yes | – | – | – | `RELEASE VERIFICATION: PASS` | any FAIL → do not take this ref to the device |
| A1 targeted tests, A2 manifest (`VENV=$VENV`), A3 tarball | – | yes | – | – | – | 0 failed; `model pins consistent: True`; tarball sha recorded | pins inconsistent → stop |
| A5 kit ready (27 W PSU, spare SD card ≥ device's, reader, USB stick, checkerboard, 240 fps phone, SPL app, iPhone with the app, a second AP or hotspot) | yes | – | – | – | – | everything present | no spare card → do not start E5 or F |

## 1. Inspect (read-only, service running)

| Step | Phys | RO | Bkp | Appr | Mon | Success | Stop / rollback trigger |
|---|---|---|---|---|---|---|---|
| B1 physical: HAT seated, camera ribbon, fan, buzzer wiring (BCM17 / pin 11, 3.3 V), PSU label | yes | yes | – | – | – | photos recorded; buzzer rated for 3.3 V logic | **5 V buzzer driven straight from GPIO → do not power on** |
| B2 power `get_throttled` | – | yes | – | – | – | `0x0`, EXT5V about 5.0–5.2 V | any bit set → fix power before measuring anything |
| B3 storage and cooling | – | yes | – | – | – | no I/O errors; ≥ 4 GB free; idle under ~60 °C | filesystem errors → E5 image first; clean-install criterion |
| C1–C5 OS, identity, packages, installed code, health | – | yes | – | – | – | Bookworm 64-bit; one `SD-XXXXXXXX` id everywhere; `.hef` sha recorded; installed revision identified | unexplained users, keys or services → stop (possible compromise) |
| D1–D3 Hailo, camera, buzzer GPIO | – | yes | – | – | – | `HAILO8L`; `imx708_wide_noir`; pin 17 output low | no PCIe device or no camera with the service stopped → hardware fix, power off first |

## 2. Back up (writes only to the backup location)

| Step | Phys | RO | Bkp | Appr | Mon | Success | Stop / rollback trigger |
|---|---|---|---|---|---|---|---|
| E1/E2 tree, config, system glue, model hashes | – | device unchanged | – | – | – | `sha256sum -c SHA256SUMS` OK **off-device** | archive missing config.yaml, the `.hef` or `data/` → redo |
| E5 full SD image (power-off) | yes | device unchanged | – | **yes** | **yes** (powered off) | image size = card size; sha recorded | wrong `/dev/diskN` → **stop**; check it twice |
| E6 clean-install vs upgrade decision | – | – | yes | **yes** | – | decision and reasons in `$EV/LOG.md`; default **upgrade in place** | – |

Nothing below this line starts until E1 and E5 are verified.

## 3. Install the release candidate

| Step | Phys | RO | Bkp | Appr | Mon | Success | Stop / rollback trigger |
|---|---|---|---|---|---|---|---|
| F0 device-config preflight (`VENV=$VENV`, or the manual block for a non-`pi` user) | – | yes | – | – | – | `OK to deploy`; warnings recorded | **REFUSED for no `role`** (expected on a pre-M0.1 install) → owner adds `role: driver_safety` by hand, then re-run. Any other REFUSED → stop |
| F1 stop the service, stage beside the old tree | – | – | yes | **yes** | **yes** | tarball sha OK; old tree untouched | sha mismatch → stop |
| F2 swap (previous kept), pip install, `migrate_config.py`, install units | – | – | yes | **yes** | **yes** | `.hef` sha = `e69136d3…960e`; `StartLimitAction=none` / `OnFailure=sensordyme-recover.service` | pip error or wrong model hash → **O1** |
| F3 `release-verify` against the manifest | – | yes | – | – | (stopped) | `VERDICT: PASS against <commit>` | any FAIL → do not start; **O1** |
| F4 start | – | – | yes | **yes** | resumes | `active`, `NRestarts=0`, no DEGRADED, `runtime up … Hailo` | not active within 60 s, crash loop, or DEGRADED with no explanation → **O1** |

## 4. Verify the product (service running; read-only unless stated)

| Step | Phys | RO | Bkp | Appr | Mon | Success | Stop / rollback trigger |
|---|---|---|---|---|---|---|---|
| G1 model loaded | – | yes | – | – | – | monitoring `ready`; Hailo live; sha `e69136d3…` | "accelerator idle" or a sha mismatch → **O1** |
| R1 M8 health fields | – | yes | – | – | – | `config.source=config`, no `saved_state` faults, `uplink_backlog.persistent`, detector available | `last_known_good` on an untouched config → investigate before continuing |
| G2 buzzer patterns + end-to-end drowsiness | yes (ears) | – | yes | **yes** | **yes** (service stopped for `buzzer_check`) | every pattern heard at 50 cm and 1 m; a consenting driver's eye closure is heard | **no audible alert → stop: the product's core claim fails** |
| G3 + R3 shared perception and tracking | – | yes | – | – | – | `simulated: False`; detector runs ≤ frames; `tracking.updates` = runs; `run_ms` recorded | `simulated: True`, or more runs than frames → stop |
| R2 Driver Safety behaviours (mount, lap, calibration) | yes | – | – | – | – | mounted phone: no phone-use alert; reading on the lap: eyes-down, not drowsiness; `attention` STARTING → READY | – (record; thresholds are untuned) |
| H1 INT8 capture (legacy vs reference) | – | yes (model unchanged) | yes | **yes** | **yes** (stopped) | capture complete; worker parity report | decoder stays `legacy` unless the report and the owner both say otherwise |
| H2 FPS/latency, I calibration, J 30-min soak | yes (checkerboard) | yes | – | – | I: **yes** (stopped) | figures recorded; no throttling in the soak | throttling → fix cooling or power before any later step |

## 5. Autonomy and recovery (each one changes state)

| Step | Phys | RO | Bkp | Appr | Mon | Success | Stop / rollback trigger |
|---|---|---|---|---|---|---|---|
| K1 reboot | – | – | yes | **yes** | **yes** | same device id; `ready`; preset and rules restored; app online in about 30 s | id changed → stop (identity defect) |
| K2 no Wi-Fi: offline alerts + legacy BLE provisioning | yes | – | yes | **yes** | network only | local alert still sounds; the app joins Wi-Fi over BLE; a refused credential reaches the app | no local alert while offline → stop |
| K3 outage, backoff, single delivery | yes (AP) | – | yes | **yes** | network only | reconnect ≤ ~45 s; queued alert delivered once | duplicate or lost alert → record; R4 decides |
| R4 outage + power cut | yes | – | yes | **yes** | **yes** | alert delivered once after power returns | lost → spool durability defect |
| K4a/K4b damaged config | – | – | yes | **yes** | **yes** | K4a boots on the last-known-good; K4b exit 78, no reboot | reboot loop → **O1** |
| K5 crash loop (optional) | – | – | yes | **yes** | **yes** | ≤ 3 reboots in 6 h, then the device stays up with SSH | SSH unreachable → O3 |

## 6. App, presets, rules

| Step | Phys | RO | Bkp | Appr | Mon | Success | Stop / rollback trigger |
|---|---|---|---|---|---|---|---|
| L1 secure pairing (token, backend key, claim) | iPhone | – | yes | **yes** | restart | header token 200, query token 401; live feed; real state | – |
| L2 presets; L3 custom rule; stop/resume | iPhone | – | – | **yes** | brief | switches confirmed; a role-violating preset refused with a reason; rule fires; stop/resume applied once | a driver_safety device accepting a non-driver preset → **stop (safety floor broken)** |
| R5 proof-of-possession onboarding (optional, separate session) | yes + physical iPhone with an M8 build | – | yes | **yes** | BLE/network | BLE doc steps 1–10 recorded | rollback: `auth: legacy` + restart |
| R6 rollback-guard test (optional, last) | – | – | yes | **yes** | **yes** | broken deploy rolls back by itself | guard fails → O1 by hand |

## Rollback ladder (runbook §O)

| Rung | When | Cost |
|---|---|---|
| O1 previous release tree | F2–F4 or G1 fails; reboot/crash loop after install | minutes |
| O2 E1 backup | O1 tree missing or damaged | about 10 minutes |
| O3 E5 SD image | OS or card damage; SSH lost | about 30 minutes plus re-imaging |

**Never on Monday:**
- factory reset;
- flashing before E5;
- changing the role (except adding the missing `role: driver_safety` at F0, deliberately);
- rotating keys or `migrate_device_key.py --apply`;
- `command_auth: required`;
- `deployment: production`;
- any model, `.hef` or decoder change;
- installing Tailscale without separate approval;
- enabling Child Monitoring's experimental pose alerts.
