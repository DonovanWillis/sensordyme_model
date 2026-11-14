# Software acceptance (M8.5 "Launch Ready")

One command proves the owner's 30 required journeys on software alone — the
real backend, the real device runtime and the real app code, with the camera,
models and radios simulated — and says exactly what that does NOT prove. None
of its output is ever a hardware result: every report carries `simulated: true`.

## The commands

```bash
make acceptance        # every software suite + the 30-journey matrix (~20-25 min, no Simulator)
make acceptance-full   # the same + every XCUITest journey (REQUIRES the iOS Simulator; ~25 min extra)

# equivalently
python scripts/sim/acceptance.py --out data/acceptance [--with-ios] [--quick] [--only NAME ...] [--list]
```

Evidence lands in `$(ACCEPTANCE_OUT)` (default `data/acceptance/`):
`acceptance.json` and `acceptance.md` — per-suite pass/fail/skip counts and
durations, the commit (and whether the tree was dirty), the twin's fidelity map
(what was real, what was a stand-in), the journey matrix, and whether the
XCUITest journeys ran. Exit status is nonzero if any suite failed; skips are
reported, never hidden. Each suite runs in its own process (the driver suites
load real MediaPipe graphs).

`make acceptance-full` must not run while anything else owns the iOS Simulator
(ports 8787/8790/8797 must be free); `scripts/sim/ios_uitest.sh` builds the real
app, runs all 28 journeys + 2 accessibility audits in one xcodebuild invocation
with no retries, and the acceptance report folds its per-test results in. Plain
`make acceptance` states explicitly that the iOS half did not run.

## The journey matrix

`acceptance.md` ends with one row per required journey (1–30):

* **result** — the worst result of the acceptance suites that prove that
  journey *in this run*: `passed`, `passed (some skips)`, `skipped`, `failed`,
  or `incomplete (suite not run)` when `--only`/`--quick` left a referenced
  suite out.
* **iOS (XCUITest)** — the app-presentation half: the XCUITest journeys that
  cover it and their results, or `not run (--with-ios)` under plain
  `make acceptance`, or `-` where no UI journey applies.
* **suites / tests** — which suites decide the result, and the load-bearing
  `file::test` references inside them (the full audit is the M8.5 mission
  record; `docs/DIGITAL_TWIN.md` §2 maps the M8.4 subset).
* **coverage notes** — what the simulation deliberately does not prove. Every
  note here is a hardware/Sunday gate or a stated design decision, not a TODO.

The same matrix is in `acceptance.json` under `"journeys"` for tooling.

## What "simulated" excludes (the hardware gates)

The twin runs the real runtime, uplink, backend, BLE provisioning service and
app logic, but these are stand-ins, so passing acceptance says nothing about:

| Excluded | Why / where it is proven instead |
|---|---|
| The camera sensor and the real open path | the virtual device injects `SimCamera` past `_open_camera` (the retry path is unit-tested on real code: `test_camera_open_retries.py`, `test_capture_service.py`) — Sunday steps 2, 6 |
| The accelerator (Hailo) and real inference FPS/latency/thermals | scripted detector/FaceMesh/Hands — Sunday steps 7–11 |
| An audible, physical alert | the simulated buzzer records instead of sounding — Sunday step 19 |
| The BLE and Wi-Fi radios | PoP + provisioning run over a simulated transport against the real services (Swift app code where `swiftc` exists) — Sunday steps 4, 14 |
| mDNS discovery on a real LAN | advertiser logic is tested with a fake zeroconf; live only (`make sim-backend --mdns`) |
| Push delivery INTO the iOS app | the device→backend→`send_push` chain is tested with the provider's HTTP stand-in; the app itself polls and posts local notifications (no APNs) |
| A real staged deploy on the Pi | `deploy.sh` runs end to end against recorder fakes for ssh/systemd/pip — Sunday step 5 and the staged-release runbook (R6) |
| The device's real SoC identity | the virtual device uses an `SD-SIM-` identity; `core/identity` is unit-tested — Sunday step 1 |

## How Sunday's physical checks map onto the same journeys

`docs/SUNDAY_DEVICE_ACTIVATION.md` runs 25 steps on the real Pi; each closes
the hardware half of journeys this command proves in software:

| Sunday step(s) | Journey # closed on hardware |
|---|---|
| 1 identity / 5 artifact revisions | 7 (real SoC identity), 29–30 (deployed artifacts verified) |
| 2 camera inventory, 6 camera capture | 8 (real sensor through the real open path) |
| 7–11 accelerator, model, parity, FPS, thermal | 11 (real inference behind the distraction events) |
| 4 secure device access | 27–28 (real key storage and command auth on the unit) |
| 12 device/backend sync | 3, 6, 7, 24 (real unit on the real backend; health freshness live) |
| 13 authenticated live streaming | 18 (stream gate on the real dashboard) |
| 14 connect the iOS app | 1–5 (real app, real radios, real Wi-Fi join) |
| 15 monitoring truth | 9, 24 |
| 16 stop / resume acknowledgements | 17, 19 |
| 17 controlled detection scenarios | 10, 11, 15, 16 |
| 18 event creation and delivery | 12, 13, 14, 22, 23 |
| 19 physical alert output | 12 (audible) |
| 20 network-interruption recovery | 20–23 |
| 21 safe degradation | 20 |
| 22 privacy: camera off while stopped | 18 |
| 23 runbook sections (reboot, staged release R6) | 25, 26, 29, 30 |

The point of the split: anything that fails on Sunday is a hardware/integration
fault by construction, because the identical journey already passed here on the
identical code.

## For the integrator

```bash
make acceptance                                   # at every candidate commit; must PASS
make acceptance-full                              # at the final RC, with the Simulator free
ACCEPTANCE_OUT=... make acceptance                # to steer the evidence directory
python scripts/sim/acceptance.py --list           # the suite catalogue
python scripts/sim/acceptance.py --only deploy m85_journeys --out /tmp/a   # a targeted re-run
```
