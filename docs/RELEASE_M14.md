# Release record: M14 "Operation Zero Day"

**Status:** a software-verified release, packaged for transfer. Written on 2026-11-17, before the
hardware arrived.

* **No hardware step of M14 has run.** The Raspberry Pi 5 + Hailo-8L and the iPhone were not
  available during the mission.
* Nothing here is a Pi, Hailo or physical-iPhone result. Every measurement is a desktop, Simulator
  or SIMULATED figure, and is labelled as one.
* The operator sequence for the first boot is `docs/FIRST_PHYSICAL_BOOT.md`.

| Item | Value |
|---|---|
| Branch | `feat/m14-zero-day` (based on the M13.1 record `8bbc758`; candidate `0a59734`) |
| Tested commit | `f3442f642a73bcb6290e656b5d46ddc5f0589e87` |
| Production model | `.hef` sha256 `e69136d3ebf3cc8ebaf3da5417a59962b139f3d3d67d715e9a9b1ae10963960e`, decoder `legacy`. **Unchanged.** `config.yaml`, `config/` and `models/` have no diff from M13.1 (`0a59734`) |
| Transfer package | `~/sensordyme-release/transfer-f3442f642a73/` (59 files + `SHA256SUMS`, 354.8 MB). **Not in Git** |
| Package `SHA256SUMS` sha256 | `01ecbbf20ed7667a3e4257ea9248e7c20cbad5542ceed4f21bda186fa31dc8b6` |
| Bundle tarball sha256 | `871f94d621e17e169c736eab7cb8545ccba55686a77e010285cd2544f3f9988b` (826 files) |
| Bundle manifest sha256 | `a441df2e0f4313cb3454564ce0cc6f3f5d85b0ce3028afc3cdb5a7713b2bcb74` |
| Rollback candidate | M13.1, `~/sensordyme-release/transfer-0a597349e83c/`. Re-verified on 2026-11-17: 59/59 OK, `SHA256SUMS` sha256 `652d050f…7a7d`, as recorded in `docs/RELEASE_M131.md` |

## What M14 adds

**Device install and runtime (no hardware needed to verify these).**

* `setup_pi.sh` begins with a read-only platform preflight (`c337379`). Before any `sudo` it
  refuses:
  * Debian 13 "Trixie" (hardware gate HG-OS);
  * a 32-bit OS;
  * a Python that is not 3.11;
  * a missing `config.yaml`.

  On refusal it prints "Nothing was installed or changed". It installs from the package's hashed
  wheels with `--no-index`, prints the HailoRT version, and says when a reboot is needed for
  `/dev/hailo0`.
* The generated `INSTALL.md` (`e9fd22f`):
  * a fresh install copies `config/config.yaml.example` to `config.yaml` (mode 600);
  * an existing tree is refused;
  * the upgrade path names the backup, the legacy adopt and identity preservation.
* A failed `aplay` is a failed playback. It used to be reported as played, which kept
  `audible_alert` ready (`da1d9d8`).
* On a Raspberry Pi board, a broken `picamera2` becomes the camera's stated reason. It is never a
  webcam fallback or a crash (`e609647`).
* The systemd units set `PYTHONUNBUFFERED=1` (`f362e70`).
* The service user joins the `video`, `gpio` and `audio` groups (`402ffa6`).
* Runbook E1 backs up an adopted device's state, not just its symlink (`d26f3c6`).

**Hailo decoder experiment (`8977546`).** `hwv_stage.py decoder-exp` is one command, run on the Pi
with the service stopped.

* It records, in ABBA order:
  * stage outputs;
  * a capture;
  * both decoders;
  * calibration;
  * pipeline timing;
  * thermal readings.
* Everything goes into one hashed evidence directory, with a `DECISION.md`.
* Gates C0–C12 read **KEEP LEGACY** unless every gate passes. Even then the result is only
  "ELIGIBLE FOR OWNER REVIEW".
* The tool never edits config.
* C11 (cabin labels) is NOT EVALUATED unless `--cabin-scores` is given.
* The Mac dry run exits 3 (SIMULATED) with KEEP LEGACY.

**Cabin data collection.** `cabin_session` gains condition presets: `low_light`, `night_ir`,
eyewear variants and `occlusion`.

* It refuses a mismatched condition before it records anything.
* It requires `--consent-ref` and `--passenger-consent-ref`.

**Jupiter beside Driver Safety (`6723eeb`, `2061794`, `7697c70`).**

* The demo rule is: "Alert me when a cell phone stays in this area for 10 seconds", with one area,
  `console`.
* It runs as `standard_plus_custom` on the real `role: driver_safety` unit. The role is not changed
  and no mandatory monitoring is relaxed.
* On the twin the alert fires at about +10.25 s.
* On the one buzzer, a rule's pattern now ranks below every mandatory pattern (`2061794`).
  * It used to share a rank with a driver "warning", and could displace a waiting driver pattern.
  * A category object without the new flag ranks as mandatory (`f3442f6`). This was found by this
    release's own gate; see "Verification".
* The overlay's `last_event` is set only for loud/soft alerts (`a01e600`).
* The 90-second and 5-minute flows are in `docs/INVESTOR_DEMO.md` §2, §3 and §3.1.

**iPhone app (`273815e`).**

* The alert banner says "Driver alert" only for driver alerts, "Rule alert" for a configured rule,
  and otherwise "Alert". With the backend down, a rule alert reads "Alert".
* New XCUITest journey **32**, `test32_BackendDownCustomerJourney`, is the backend-down customer
  journey. It is mapped to journeys 21 and 25.
* DEBUG builds only: an alert-sound decision record (`GET /app/alert-decisions`).
* The unsigned Release build for a generic iOS device succeeds, and the release-configuration audit
  passes on it (`docs/IOS_DEVICE_BUILD.md`).

**The test06 lost tap.** XCUITest journey 06 (Resume) failed once in M13.1, and once before that in
M8.1.

* Reproduced 1 time in 8 runs.
* XCTest's synthesized tap had touch-down and touch-up 164 ms apart with no touch updates.
* A `systemGestureStateChange` gate event fell between them, about 105 ms after touch-down. The
  M13.1 failure had the same shape.
* Journey 06 now taps with a 0.06 s press (`fingerTap`). New `test06b_ResumeHeldPress` checks that a
  held press always fires.
* **Not proven:** that a real finger can never produce such a touch. A user whose tap is lost sees
  the button unchanged and taps again. Nothing is shown as resumed until the device confirms it.
* Evidence: `~/sensordyme-audit/2026-11-10/m14-evidence/streamC/`.

**`docs/FIRST_PHYSICAL_BOOT.md`.** Stages 1–24, plus 22b, as one Pi + iPhone sequence. Each stage
gives the command, the expected result, the diagnosis, the recovery, the approval and its named
hardware gate.

**Unchanged:**

* detection thresholds and every mandatory alert decision;
* driver association;
* relapse escalation (OFF);
* security and privacy controls;
* the production model and decoder.

No experimental model was promoted. `dmonitoring_model.onnx` (the optional openpilot module, not
the production path) has been tracked since `0bc25bf` and shipped in every earlier package.

## Verification

Evidence: `~/sensordyme-audit/2026-11-10/m14-evidence/` (`gates/`, `gates-f3442f6/`, `package/`).

`f3442f6` changes only `core/alerts.py` and one Python test (17 insertions, 1 deletion). The Swift
results at `42e8801` therefore cover the release's app code.

| Gate | Commit | Result |
|---|---|---|
| ruff | f3442f6 | clean |
| pyright | f3442f6 | 0 errors |
| pytest (full) | 42e8801 | **1 failed**, 4,954 passed: `test_m10_observe::test_the_gpio_buzzer_records_command_to_pin_driven`. AttributeError. A regression from `2061794`, fixed in `f3442f6` |
| pytest (full) | **f3442f6** | **4,956 passed**, 6 skipped, 1 strict xfail (escalation, OFF), 0 failed |
| Swift logic tests (`make test-ios`) | 42e8801 | all passed: 2,161 checks |
| `make verify-release` | **f3442f6** | **PASS** (the systemd step is SKIP on macOS) |
| `make acceptance-full` | 42e8801 | **PASS: 901 / 901**, including **33 / 33 XCUITest journeys** in one invocation |
| `make acceptance` (all Python suites) | **f3442f6** | **PASS: 868 / 868** |
| Release gates on the package manifest | **f3442f6** | **PASS**. Checks: pins consistent; decoder legacy; protocol; hashed lock (40 packages); config schema; no hardware claim |
| `shasum -a 256 -c SHA256SUMS` | **f3442f6** | 59 / 59 OK. The bundle's own `SHA256SUMS` is 2 / 2 OK |
| Secrets scan of the package | **f3442f6** | No key, credential, `.env`, identity or database file was found. The builder's secrets-refusal check passed |
| Rebuild, same commit | **f3442f6** | The payload is identical: tarball, wheels, `.hef`, config template and install scripts. Not identical: the manifest's `generated_utc`, the output path recorded in `PACKAGE.json`, and the hashes that include them (`INSTALL.md`, both `SHA256SUMS`). The package is therefore **not byte-reproducible**; its payload is |

### After the package: a test-isolation fix (not in the package)

* While checking this record, a subset of 51 test files failed
  `test_m83_inference_path::test_get_yolo_model_returns_the_verified_hailo_model_and_counts_inferences`.
  The model fell back to ultralytics.
* **Cause:** the `without_ultralytics` fixture in `tests/test_phase4_no_agpl_runtime.py` restored
  `sys.modules` but not the `core` package's `accel` and `detectors` attributes. Any test run after it
  that used `from core import accel` therefore got a fresh, unconfigured copy.
* The full suite never ran them in that order, which is why every full run passed.
* Neither file, nor the code they exercise, changed in M14.
* The fix restores the package attributes too, and changes no assertion. It lands as a test-only
  commit after the release.
* **The device payload is unaffected, and the release stays `f3442f6`.**
* After the fix, the same 51-file subset passes (1,045 passed, 0 failed). The full-suite result is
  in the commit message.
* Evidence: `m14-evidence/post-release/`.

## Hardware gates still open, in order

The first-boot stages are in `docs/FIRST_PHYSICAL_BOOT.md`:

* HG-POWER, HG-OS
* HG-PIP, HG-HAILO, HG-HAILORT
* HG-CAMERA, HG-SYSTEMD
* HG-AUDIO, HG-BUZZER
* HG-LATENCY, HG-RECOVERY, HG-ROLLBACK
* then the iPhone stages:
  * install (signing team, provisioning, Xcode 26.1.1 vs iOS 26.5.2 device support);
  * BLE pairing;
  * live view;
  * alert sound;
  * reconnect.

## What this release does not claim

* Pi fps or latency, Hailo inference, FaceMesh on the Pi, thermal behaviour, either decoder's INT8
  parity, or an audible buzzer.
* That the iPhone app installs or works on a physical phone: signing, BLE pairing, live video, the
  AI view, notification sound, or reconnecting.
* That a real finger can never lose a tap the way XCTest's synthesized tap did.
* That the Jupiter rule detects a propped phone steadily on the real camera.
* Any real-world accuracy figure, or any customer pilot.
