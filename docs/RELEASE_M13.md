# Release record: M13 "Showstopper" candidate

**Status:** a software-verified candidate.

* Not pushed, not merged, not packaged for transfer.
* **No hardware gate has passed.** The Raspberry Pi 5 + Hailo-8L was unreachable
  for the whole mission: it was not on the tailnet, had no SSH alias and did not
  appear on this LAN. The iPhone was paired with Xcode but reported
  "unavailable".
* Every measurement is a desktop or SIMULATED figure, and is labelled as one.

| Item | Value |
|---|---|
| Branch | `feat/m13-showstopper` (local; based on the M12 release `0ff8f45`) |
| Tested commit | `faae3dec10aeebd62014cb2648518fcdefac82cd` |
| Production model | `.hef` sha256 `e69136d3…960e`, decoder `legacy`. **Unchanged.** `config.yaml` and `models/` are unchanged since M12. The only new config is the isolated dev overlay `deploy/dev/reference_decoder.overlay.yaml`, which is not used by the product |
| Fallback recording | `~/sensordyme-audit/2026-11-10/m13-evidence/fallback-recording-SIMULATED.mp4`, sha256 `2e93257e3ddea415ec8dec9488e6302cb0f9850ba31f2539339360099e6a63dc` |

## What M13 adds

* **sd-overlay/1** (`docs/LIVE_OVERLAY.md`, `core/overlay.py`).
  * The device sends the clean frame, plus a description of what the AI saw on
    that frame. It goes only to viewers that ask for it.
  * No confidence numbers. Head pose is labelled as head pose, never gaze.
    `det: null` means "could not look". A simulated source is flagged on every
    frame.
  * Performance is measured on the device. `last_event.event_key` ties the live
    "why" line to its history row.
* **The iPhone app:**
  * the AI view;
  * presentation mode;
  * the distracted-driving panel;
  * history rows that say what each event means;
  * a faster Jupiter follow-up;
  * new XCUITest journey `test30_PresentationMode`.
* **Ten flagship scenarios and the Jupiter dwell demo** (`docs/DEMO_SCENARIOS.md`).
  * Result: SIMULATED, 10/10 PASS.
  * Three defects the scenarios found are fixed:
    * G1: a passenger's phone has role `other`;
    * G2: every alert carries a measured reason;
    * J1: a dwell alert's duration is the time in the area.
* **Performance** (`docs/research/M13_PERFORMANCE.md`, desktop):
  * `AdaptiveBaseline` keeps a sorted window;
  * decoder speedups that are bit-identical;
  * the uplink wakes on a new alert (**owner sign-off requested**: this changes
    transport timing, not decisions);
  * benchmarks;
  * a reference-decoder dev config and `hwv_stage.py latency --latency-decoder both`.
* **Investor package** (`docs/INVESTOR_DEMO.md`):
  * the 90-second and 5-minute scripts;
  * preflight, which now includes the live AI view check;
  * the fallback recording;
  * labelled performance evidence;
  * what the product does and does not do;
  * edge vs cloud.

**Unchanged:**

* detection thresholds and every alert decision threshold;
* driver association;
* relapse escalation (OFF);
* security and privacy controls (sd-local-v1, camera privacy while stopped).

## Verification at `faae3de`

Evidence: `~/sensordyme-audit/2026-11-10/m13-evidence/gates/`.

| Gate | Result |
|---|---|
| ruff | clean |
| pyright | 0 errors |
| pytest (full) | **4,871 passed**, 6 skipped, 1 strict xfail (the documented escalation decision, OFF), 0 failed |
| Swift logic tests (`make test-ios`) | all passed: 2,082 checks |
| iOS release-configuration audit | PASS |
| `make verify-release REF=HEAD` | **PASS** (the systemd step is SKIP on macOS) |
| `make acceptance-full` | **PASS: 898 / 898**, including **30 / 30 XCUITest journeys in one run** |

## What this release does not claim

* Pi inference latency, fps, thermal behaviour, INT8 parity of either decoder,
  and an audible buzzer (gates H1–H3 and step 21).
* Any real-world accuracy figure. Simulated scenarios are not accuracy evidence.
* The AI view or presentation mode on a physical iPhone.
