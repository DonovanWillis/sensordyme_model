# Phase 5 — Demo readiness

_Generated 2026-08-29. Every number here was measured on the physical device
unless explicitly marked otherwise. Measurements taken against a **different**
detector binary are not reused; where a claim could not be established it says
so rather than estimating._

## Executive verdict

**READY WITH CONDITIONS.**

The demo path is validated end to end on real hardware: all four modes start,
run on the NPU, and report truthfully. Three defects that would have shown up
during a demo were found and fixed in this phase, one of which was actively
rebooting the device. The conditions below are limitations to be *aware of*,
not unfinished work — none of them blocks demonstrating the product.

Phase 5 found the demo device in a state that did not match its own
documentation in three separate ways: it was running an unreproducible model
binary, it had the AGPL stack installed that the licensing analysis said was
removed, and a stale AI rule from thirteen days earlier was silently overriding
the configured module. All three are now closed.

---

## 1. Hardware

| | |
|---|---|
| board | Raspberry Pi 5 Model B Rev 1.1 |
| accelerator | Hailo-8L (AI HAT+), `/dev/hailo0` |
| runtime | HailoRT 4.20.0 (`hailo-all 4.20.0+1`, `hailo-dkms 4.20.0-1`) |
| camera | IMX708 wide NoIR (Camera Module 3 Wide NoIR), 640x480 capture |
| OS / kernel | Debian, 6.12.93+rpt-rpi-2712 |
| power | USB-PD supply advertising 5V/3A, 9V/3A, 12V/2.5A, 15V/2A, 20V/1.5A (30W); firmware negotiated **3000 mA** |
| mounting | `camera.rotate_180: true` (device-specific, preserved across deploys) |

---

## 2. Detector — the authoritative answer

**What exact detector binary is SensorDyme running?**

| | |
|---|---|
| model | `damoyolo_tinynasL20_T` (DAMO-YOLO tinynasL20_T, COCO-80) |
| build | Hailo Model Zoo **v2.15.0**, `hailo8l` |
| **sha256** | **`e69136d3ebf3cc8ebaf3da5417a59962b139f3d3d67d715e9a9b1ae10963960e`** |
| size | 18,554,430 bytes |
| provenance | `https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/damoyolo_tinynasL20_T.hef` |
| structure | 6 raw output tensors, **no on-chip NMS** — confidence thresholds are fully ours |
| licence | DAMO-YOLO Apache-2.0; Model Zoo build carries no copyleft |
| backend | Hailo-8L NPU (`core/accel.py` `HailoYOLO`, off-chip DFL decode at strides 8/16/32) |
| pinned in | `setup_pi.sh` (checksum-verified at install, **install fails on mismatch**), `config.yaml` `accelerator.yolo_hef_sha256` |
| verified on device | config hash == file hash == pinned hash |

**The superseded binary.** Phases 3 and 4 measured a different file (sha256
`4c71eab8…`, 20,088,746 bytes) whose provenance **could not be established** — it
matched no Hailo Model Zoo release (v2.11 through v2.15 all differ in size) and
the script that fetched it did not survive. It has been retired to
`models/_archive_unreproducible/` with a README stating it must never be
deployed. It is kept only so the historic numbers remain traceable to a real
artifact. Eleven other `.hef` files from the Phase 3 detector comparison were
moved to `models/_archive_phase3_comparison/`; `models/` now contains exactly
one `.hef`.

---

## 3. Accuracy — measured on the demo binary

Method: `benchmarks/accuracy_state_farm.py`, State Farm val split
(subject-disjoint: p002, p012, p014, p015, p016, p021), seed 0, 60 images/class
= 600 images, `--backend hailo`. Baseline at
`benchmarks/results/phase5_demo_binary.json`.

**Phone (`cell phone`)**

| conf | recall | FP rate | precision | F1 |
|---|---|---|---|---|
| 0.10 | 99.6% | 86.7% | 69.7% | 0.820 |
| 0.15 | 93.8% | 70.8% | 72.6% | 0.818 |
| **0.20 (shipped)** | **88.3%** | **52.5%** | **77.1%** | **0.823** |
| 0.25 | 82.5% | 30.0% | 84.6% | 0.835 (best F1) |
| 0.35 | 62.5% | 19.2% | 86.7% | 0.726 |
| 0.50 | 23.8% | 2.5% | 95.0% | 0.380 |

0.0% of phone-present frames yield no box at any confidence — the residue is a
model ceiling, not a threshold choice.

**Person (COCO class 0), 600 single-occupant images**

| conf | availability | spurious >1 box |
|---|---|---|
| 0.25 | 100.0% | 60.0% |
| 0.40 | 100.0% | 17.8% |
| **0.50 (shipped)** | **100.0%** | **4.0%** |
| 0.60 | 99.8% | 0.8% |
| 0.70 | 94.7% | 0.5% |

**Face (MediaPipe, detector-independent):** 39.0% availability — unchanged
across binaries, which is the sanity check that the harness is comparing what it
claims to.

**Regression gate vs the Phase 3 baseline** (`--compare`, tolerance 0.03,
exit 0): `face_availability` 0.3900→0.3900 · `person_availability`
1.0000→1.0000 · `phone_best_f1` 0.8475→**0.8354** (−0.0121).

> The harness also prints `phone_recall_at_shipped 0.6625 → 0.8833 improved`.
> **That is not a model improvement** and must not be quoted as one: the Phase 3
> baseline recorded `shipped_conf: 0.35` and the config now ships 0.20. At
> *matched* thresholds the new binary is marginally **worse** (0.35:
> 66.2%→62.5%) and marginally worse on spurious person boxes at low confidence.
> It is also faster. Everything is inside the gate; nothing changed a threshold.

---

## 4. Performance

| metric | value | how |
|---|---|---|
| inference latency | mean **22.07 ms**, p50 21.80, p95 22.75, p99 22.85, max 26.47 | 300 samples, warm-up excluded, demo binary |
| detector ceiling | ~45.3 fps | derived from the mean above |
| pipeline frame rate | **29.5–29.9 fps** (baby_monitoring 25.7–25.9, pose runs every frame) | live `/api/health` across 8 restarts |
| frame interval p95 | 33.5–37.8 ms (baby_monitoring 48.0–52.3) | FrameBus interval telemetry |
| **evidence-losing stalls (>1.0 s)** | **0** in every mode, every cycle | FrameBus interval telemetry |
| stalls >250 ms / >500 ms | 0 / 0 | same |
| RSS | driver_attention 414–415 MB · baby_monitoring 394–396 · fall_detection 330–331 · occupancy 309–310 | stable across two cycles |
| temperature | 58.2–62.6 °C under load | `vcgencmd measure_temp` |
| boot to healthy | 6.1–9.1 s | measured per restart |
| throttling | `throttled=0x0`, **no live bits, no sticky bits** | see §7 |
| power (PMIC rails) | 3.27 W under load — **lower bound**, excludes the HAT, camera, USB, HDMI | `vcgencmd pmic_read_adc` |

---

## 5. Safety

| area | status | evidence |
|---|---|---|
| driver distraction / phone | **VALIDATED (pipeline)** | mode starts, NPU active, 88.3% phone recall at the shipped 0.20 on domain-shifted data |
| person threshold split | **VALIDATED** | phone 0.20 / person 0.50 measured on the demo binary; `tests/test_phase4_detection_thresholds.py` |
| baby monitoring | **VALIDATED (logic)** | 101 tests pass; all 7 identification-safety properties verified in code (§8) |
| fall detection | **VALIDATED (replay)** | 86 tests pass; demo config preserves the measured 30 fps / 1.0 s grace assumptions |
| occupancy / security | **VALIDATED (pipeline)** | starts, NPU 421 inferences, zero stalls |
| alerting / events | **VALIDATED** | events written to `data/events.csv` during the reliability run |
| accelerator failure | **VALIDATED ON HARDWARE** | degrades, does not crash-loop (§6) |
| **staged real fall** | **UNVERIFIED ON HARDWARE** | cannot be safely staged remotely |
| **real infant** | **UNVERIFIED ON HARDWARE** | no infant footage exists |

**Baby-monitoring identification safety** — the "smallest person box = baby"
failure mode is gone and stays gone. Verified present in code: evidence-based
`InfantIdentifier`; crib-zone containment gate
(`INFANT_ZONE_MIN_CONTAINMENT` 0.85); a margin the best candidate must beat
(`INFANT_ID_MIN_MARGIN` 0.12); a minimum observation period
(`INFANT_NO_ZONE_MIN_OBSERVATION_S` 3.0); identity that survives a gap then
expires (`INFANT_REACQUIRE_S` 30.0); an explicit `infant_unidentified` decline;
and that decline being logged rather than silent.

---

## 6. Accelerator failure policy

**Policy: DEGRADE LOUDLY. Never fabricate a fallback, never claim to be
watching.**

Validated on hardware by pointing `yolo_hef` at a missing file with no CPU
fallback installed:

| behaviour | observed |
|---|---|
| service state | `active` — **not** crash-looping (`NRestarts: 0`) |
| preflight | `FAIL detection_backend: NONE — … Object detection will not run. Re-run setup_pi.sh` |
| log | `ERROR DEGRADED: object detection unavailable for driver_attention — monitoring is NOT active` |
| boot banner | `runtime up: … *** DEGRADED: monitoring NOT active ***` |
| `monitoring_active` | **`false`** |
| `inference` | `failed — monitoring NOT active — <reason>` |
| `accelerator` | `warning` + the specific reason |
| dashboard | still served, **HTTP 200** |
| recovery | restoring the config returns `monitoring_active: true` |

The dashboard staying up is the point: a device whose HAT comes loose must still
be able to *tell someone*. Regression tests:
`tests/test_phase4_detection_backend.py`.

---

## 7. Throttle status — historical vs current

The device reported `system: warning — throttled (0x50000)` continuously for
hours while every live bit was clear. `0x50000` is bits **16 and 18**:
"under-voltage **has** occurred" and "throttling **has** occurred" — sticky flags
that persist until the Pi is power-cycled.

`core/device_health.py` now decodes the two halves separately: only bits 0–3
(happening **now**) can make the device unhealthy; bits 16–19 are reported as
history in `throttle_since_boot` and in the detail string, without colouring the
device faulty. A warning that cannot turn off is not a warning.

**Current status: `throttled=0x0` — no live bits, no sticky bits.** The sticky
flags cleared on the reboots during this phase and have not returned under load.

---

## 8. Licensing — what is actually on the device

Phase 4 removed `ultralytics` from the repo's device requirements. Phase 5 found
it **still installed on the physical demo device**:

| package | version | licence | action |
|---|---|---|---|
| `ultralytics` | 8.3.28 | **AGPL-3.0** | **REMOVED** |
| `torch` | 2.13.0+cpu | Apache-2.0 / BSD / MIT | REMOVED (only present as an ultralytics dep, 639 MB) |
| `torchvision` | 0.28.0+cpu | BSD | REMOVED |
| `yolo11n.pt` | — | **AGPL-3.0 weights** | **DELETED from the device** |

Verified after removal: `ultralytics`, `torch`, `torchvision` all report ABSENT;
zero `.pt` files outside `site-packages`; `core.detectors`, all 8 modules,
`services.runtime`, `core.dashboard` and `main` all still import; venv 2.5 G →
1.9 G. The device now has **no CPU fallback at all**, which is why §6 matters.

The two remaining source references are correct by design: `core/detectors.py`
imports `ultralytics` inside a `try` (the optional developer-machine fallback),
and `modules/driver_attention_max.py` imports `torch` inside `require_mps()` on
the Mac-only profile.

---

## 9. Reliability — Phase 5 Demo Reliability Validation

**This is not, and is not called, a 48-hour soak.** It is a demo-shaped test:
repeated startup, mode switching, camera re-acquisition, inference, event
generation and recovery.

**Duration/repetitions: 8 service restarts — 4 modes × 2 cycles — plus 4 further
restarts during degraded-mode testing.**

| cycle | module | boot s | monitoring_active | NPU | fps | p95 ms | stalls >1s | RSS MB | °C |
|---|---|---|---|---|---|---|---|---|---|
| 1 | driver_attention | 9.1 | true | 135 | 29.5 | 37.8 | 0 | 415 | 59.8 |
| 1 | baby_monitoring | 7.0 | true | 167 | 25.9 | 51.2 | 0 | 394 | 60.9 |
| 1 | fall_detection | 6.1 | true | 0 | 29.9 | 33.7 | 0 | 330 | 61.5 |
| 1 | occupancy | 6.1 | true | 421 | 29.9 | 34.2 | 0 | 309 | 58.2 |
| 2 | driver_attention | 6.2 | true | 133 | 29.8 | 36.5 | 0 | 414 | 59.8 |
| 2 | baby_monitoring | 6.3 | true | 170 | 25.7 | 52.3 | 0 | 396 | 60.4 |
| 2 | fall_detection | 6.3 | true | 0 | 29.9 | 34.1 | 0 | 331 | 60.9 |
| 2 | occupancy | 6.2 | true | 421 | 29.9 | 33.5 | 0 | 310 | 58.7 |

Every mode: `camera ok`, `accelerator ok`, `inference ok`, `throttle_now []`.
RSS is stable cycle-to-cycle in every mode. **No reboot occurred** (boot
timestamp identical at start and end). `fall_detection` shows NPU 0 correctly —
it is MediaPipe-Pose-only and consumes no YOLO by design.

---

## 10. Defects found and fixed in Phase 5

1. **Occupancy rebooted the device.** `modules/occupancy.py` built its detector
   in `__init__`, which runs *before* `DeviceRuntime` calls `accel.configure()`,
   so `yolo_hef` was still unset, it fell through to the CPU path, and with
   `ultralytics` gone it raised. `Restart=always` retried, and
   `StartLimitAction=reboot` power-cycled the Pi — repeatedly, because the module
   was still occupancy on the next boot. Fixed two ways: `main.py` now configures
   the accelerator before constructing any module, and occupancy loads lazily
   like every other module. Tests:
   `test_accelerator_is_configured_before_any_module_is_constructed`,
   `test_no_module_loads_a_detector_in_its_constructor`.
2. **`StartLimitIntervalSec` 600 → 120.** With `RestartSec=3`, a genuine crash
   loop burns 5 attempts in ~15 s; ten minutes only served to accumulate
   *unrelated, successful, manually spaced* restarts until the fifth rebooted the
   Pi. 120 s catches every real crash loop with 8x margin.
3. **AGPL stack on the device** (§8).
4. **Sticky throttle bits reported as a live fault** (§7).
5. **A stale AI-configured rule silently overrode the module.**
   `data/monitoring.json`, saved 13 days earlier —
   _"Watching for a bottle present in the scene."_ — had `overrides_mode: true`,
   which put the device in AI-Custom mode running `occupancy` while
   `config.yaml` said `driver_attention`. Nothing in the logs said so. The boot
   log now states plainly that an AI rule set is overriding the configured
   module, quotes the instruction, and gives its age. The rule was preserved
   (`data/monitoring.json.phase5_backup`), not deleted.

---

## 11. Known limitations — explicitly not proven

- **No staged real fall.** All fall evidence is replay of real pose sequences.
  A real body hitting a real floor has not been tested. **UNVERIFIED.**
- **No real infant.** Baby-monitoring identification logic is verified by tests
  and code inspection only. Infant availability at 0.40/0.50/0.60 on crib
  footage is **unmeasured** — which is why the threshold was held at 0.40.
- **Accuracy figures are domain-shifted lower bounds.** State Farm's camera is
  passenger-side across the cabin; SensorDyme mounts dash-centre driver-facing.
  Use them for regression detection, not as SensorDyme's operating accuracy.
- **Phone precision at 0.20 (~77%)** on driver-facing geometry is derived from
  F1 and recall, not measured directly.
- **Per-person recall at security-camera range** unmeasured — why occupancy and
  the security presets were left unchanged.
- **Wall power draw unmeasured.** 3.27 W is PMIC-visible rails only and excludes
  the HAT; no inline meter was available.
- **The supply negotiated 5V/3A (15 W)**, not the 5 A the Pi 5 wants for full
  peripheral current. No live throttling has been observed, but this is a
  plausible cause of the historic sticky bits.
- **No clean-device end-to-end install.** The provisioning *logic* is tested
  (all four cases, §12) but `setup_pi.sh` has not been run start-to-finish on a
  fresh Pi OS image. **UNVERIFIED.**
- **Kernel warnings from the Hailo driver.** `hailo_vdma_buffer_map` calls
  `find_vma()` without holding `mmap_lock` (43 occurrences, bounded — unchanged
  across 500 inferences, 3 device cycles and a service start). Upstream
  out-of-tree driver defect; kernel tainted `W|O`.
- **BLE advertisement registration failed** once during rapid restart testing
  (`org.bluez.Error.Failed`). Not investigated; provisioning was not part of the
  demo path under test.

---

## 12. Deployment reproducibility

`setup_pi.sh` model provisioning, all four cases exercised on the device:

| case | behaviour | exit |
|---|---|---|
| corrupt file present | refuses, prints found vs expected hashes, **fails the install** | **1** |
| retired (evaluated) binary present | recognised by hash, kept, explained | 0 |
| correct pinned binary present | recognised, no re-download | 0 |
| nothing present | downloads from the pinned URL, verifies sha256, installs | 0 |

`make deploy` excludes `*.pt` (and deliberately not `*.onnx` — openpilot's
dmonitoring model is a different licence and the device runs it). `setup_pi.sh`
and `make deploy` both install `requirements-device.txt`, which contains no
`ultralytics` and no `torch`. `tests/test_phase5_demo_readiness.py` asserts
`config.yaml` and `setup_pi.sh` cannot drift apart about the model hash.

**Not verified:** a complete clean install on a fresh Pi OS image.

---

## 13. Regression suite

| | result |
|---|---|
| pytest | **1341 passed, 1 skipped, 0 failed** |
| new failures | **0** |
| skipped | 1 — the local model-hash check, correctly skipped where the gitignored `.hef` is absent |
| ruff (tree-wide) | 336 errors — **all pre-existing**; CI lint was already red before Phase 4 |
| ruff (files Phase 5 touched) | 33, **identical to HEAD** — zero introduced |
| ruff (files Phase 5 created) | **0** |
| pyright | 8 errors, 539 warnings, 198 files — **0 errors in any file Phase 5 touched** |

---

## 14. Deferred production validation

Not required for the demo. **None of these has passed; all are deferred.**

- **48-hour production soak — DEFERRED, NOT REQUIRED FOR CURRENT DEMO.**
  Partial evidence preserved at `docs/evidence/phase4_soak48_partial.csv`:
  26 samples, 1387 s (0.39 h) of 48 h, 40,437 inferences, 29.8–30.3 fps, zero
  stalls at 250/500/1000 ms, RSS 321→323 MB, 0 restarts. **This is 0.8% of the
  required duration and does not constitute a passed soak.**
- Real staged fall with a person.
- Real infant / crib footage, and infant availability at 0.40/0.50/0.60.
- A driver-facing, real-geometry dataset to replace domain-shifted State Farm.
- Clean-device end-to-end provisioning from a fresh Pi OS image.
- Wall power measurement with an inline meter, and a 5 A supply.
- Per-person recall at security-camera range.
- Long-run memory-leak and thermal-drift characterisation.
