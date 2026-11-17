# Release record: M13.1 "Operation Live Launch" candidate

**Status:** a software-verified candidate, packaged for transfer.

* **No hardware step of M13.1 has run.** The Raspberry Pi 5 + Hailo-8L and the
  iPhone were unreachable for the whole mission:
  * The Pi was not on the tailnet, did not answer mDNS, had no host on the
    dashboard port, and had no Raspberry Pi MAC address on this LAN.
  * The iPhone was paired with Xcode but reported "unavailable".
* Nothing here is a Pi, Hailo or physical-iPhone result. Every measurement is a
  desktop or SIMULATED figure, and is labelled as one.

| Item | Value |
|---|---|
| Branch | `feat/m13-showstopper` (M13 + M13.1; based on the M12 release `eda0952`) |
| Tested commit | `0a597349e83c1be23495f58446a3b6588d4170d4` |
| Production model | `.hef` sha256 `e69136d3ebf3cc8ebaf3da5417a59962b139f3d3d67d715e9a9b1ae10963960e`, decoder `legacy`. **Unchanged.** `config.yaml` and `models/` are unchanged since M12 |
| Transfer package | `~/sensordyme-release/transfer-0a597349e83c/` (59 files, 354.7 MB). **Not in Git** |
| Package `SHA256SUMS` sha256 | `652d050ff03daf972ac94fc4fa5eaf0e0aa199661b416f23bbef205778a27a7d` |
| Bundle tarball sha256 | `3ed24eb611e5b4f89c701d605c666f058500106ee23039d8b14aeeab54857009` (817 files) |
| Bundle manifest sha256 | `37aa7518ba5a697d575455969b84cf48c99c727386bde29fcd6c31386f7d6974` |

## What M13.1 adds

* **AI view and Present on the paired-camera screen.**
  * This is the screen the app opens from the phone's local pairing
    (sd-local-v1).
  * Both work with the backend stopped: the live picture, the sd-overlay/1
    layers, the distracted-driving panel, and a mode switch that only the device
    confirms.
  * Presentation mode is fed by one `PresentationSource`, from either screen.
  * The local client now reads the device's role and mode from its own
    `/api/state`.
  * New XCUITest journey **31** (`test31_LocalPresentation`) runs with the
    backend process stopped.
* **Face mesh drawn on the device in one `cv2.polylines` call**
  (`modules/driver_attention.draw_face_tessellation`).
  * Pixel-identical to mediapipe's drawing, tested across frame sizes,
    off-frame points, low visibility and NaN.
  * 2.31 → 0.90 ms p50 per streamed frame (Mac M3).
* **The accessibility audit's re-check** (`docs/ACCEPTANCE.md`, "The Dynamic
  Type re-check"). **Owner sign-off requested.**
  * It covers Dynamic Type findings, and contrast findings on located text in
    the clear band.
  * Such a finding is audited once more on the still screen. A finding that
    reproduces fails the test. One that does not is written down as
    `not-reproduced`.
  * Without it, the full suite failed about 1 run in 4 **on unchanged M13
    code**. Every text on the Jupiter sheet was flagged at once, and device-row
    texts were flagged that measure 17.4:1 and 10.1:1.

**Unchanged:**

* detection thresholds and every alert decision;
* driver association;
* relapse escalation (OFF);
* security and privacy controls;
* the production model and decoder.

## Verification

Evidence: `~/sensordyme-audit/2026-11-10/m131-evidence/` (`gates/`, `gates2/`,
`gates3/`, `gates4/`, `gates5/`, `gates6/`, `package/`).

The Python and app source last changed at `20bf68a` and `96afa1d`. The commits
after them (`bd9ff99`, `0a59734`) change only
`ios/UITests/AccessibilityAuditUITests.swift` and docs. The gates marked with
those commits therefore cover the candidate's code.

| Gate | Commit | Result |
|---|---|---|
| ruff | 20bf68a | clean |
| pyright | 20bf68a | 0 errors |
| pytest (full) | 20bf68a | **4,886 passed**, 6 skipped, 1 strict xfail (escalation, OFF), 0 failed |
| Swift logic tests (`make test-ios`) | 96afa1d | all passed: 2,155 checks |
| iOS release-configuration audit | 96afa1d | PASS |
| `make verify-release` | **0a59734** | **PASS** (the systemd step is SKIP on macOS) |
| Release gates on the package manifest | **0a59734** | **PASS** (pins consistent, decoder legacy, protocol, hashed lock, config schema, no hardware claim) |
| `shasum -a 256 -c SHA256SUMS` | **0a59734** | 59 / 59 OK |
| `make acceptance-full`, run 1 | **0a59734** | 898 / 899: 30 / 31 UI journeys. `test06_Resume` failed (below) |
| `make acceptance-full`, run 2 | **0a59734** | 898 passed, 0 failed. `test31` MISSING: the Mac's disk filled while xcodebuild wrote its result bundle ("mkstemp: No space left on device"). Not a test result |
| `make acceptance-full`, run 3 | **0a59734** | **PASS: 899 / 899**, including **31 / 31 XCUITest journeys in one invocation**, no retries, and no audit re-check fired |
| test21 alone, `-test-iterations 4` | 0a59734's audit code | 4 / 4 passed. The re-check fired 4 times and nothing reproduced |

**Run 1's failure: `test06_Resume`.**

* What happened, from the app's own os_log and the backend, device and proxy
  logs:
  * The Resume button was enabled and had been still for over 1 s.
  * The tap reached the app window (touch-down 16:33:40.117, touch-up .267).
  * Its action never ran: no request was sent and the button never showed
    "Resuming…". The device stayed stopped.
* **The cause is not identified.** The same failure happened once before, in
  M8.1 (`f22564e`), before any M12/M13 code existed. 30 other full runs passed
  this journey.
* Evidence: `gates4/test06-lost-tap/`.
* A real user whose tap is lost sees the button unchanged and taps again.
  Nothing is ever shown as resumed until the device confirms it.

## Hardware: what is still required, in order

1. **Pi.** Power it on its own USB-C supply, on Ethernet or this Wi-Fi. Tell me
   its hostname or IP and the SSH user, and run the first `ssh <user>@<host>`
   yourself so the host key is yours to accept. Then:
   1. inspect;
   2. check health;
   3. preserve config and identity;
   4. back up;
   5. verify this package (`shasum -c`, `release_guard.py stage`);
   6. **your approval**;
   7. `make deploy` / the staged install;
   8. confirm the service;
   9. verify the hashes;
   10. confirm real frames and Hailo inference;
   11. confirm rollback.
2. **iPhone.** Connect it by USB and unlock it. Tap Trust. Enable Developer Mode
   if iOS asks. Pick the signing team in Xcode if it asks. Then install, pair
   over BLE, and run the live checks.
3. **Benchmarks.** Run `hwv_stage.py latency --latency-decoder both` on the Pi
   and record it against the promotion criteria. **Production stays on the
   legacy decoder.**

## What this release does not claim

* Pi fps or latency, Hailo inference, FaceMesh on the Pi, thermal behaviour,
  INT8 parity of either decoder, or an audible buzzer.
* That the iPhone app works on a physical phone: installation, BLE pairing,
  live video, the AI view, Present, notification sound, or reconnecting after
  a reboot.
* Any real-world accuracy figure.
