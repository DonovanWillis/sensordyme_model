# Phase 6 — Demo readiness completion & software hardening

_2026-08-30. Every hardware claim was executed on the physical Raspberry Pi 5 +
Hailo-8L. Nothing here is simulated and presented as measured. Items that need
real-world conditions are listed as UNVERIFIED or DEFERRED and are not counted
as evidence of anything._

## Verdict

**DEMO READY WITH CONDITIONS** — conditions listed in the final section.

Phase 6 found and fixed **eight defects**, two of which would have ruined a demo
in ways nothing would have explained:

* a restored AI rule set made the device **discard every event its module
  produced while reporting itself healthy** — no alert could ever fire; and
* any failure inside `_build()` (an unplugged camera ribbon; a port still held
  by a stale manual run) leaked the camera and **power-cycled the Pi in a loop**
  instead of booting degraded.

Both are the same shape as every serious finding in Phases 1–5: the reasoning
was sound and nothing checked it against reality.

---

## FIXED — defect found, corrected, regression evidence

| # | Defect | Evidence it was real | Fix | Guard |
|---|---|---|---|---|
| 1 | **A restored AI rule set silenced the device.** `_apply_monitoring_mode()` sat inside the `cloud.enabled` block and `_apply_requested_labels()` was only ever called from inside the inference loop when a *new* rule set arrived. A device that rebooted with rules already saved came up in AI_CUSTOM, threw away every module event, never swapped to the module the rules needed, never told any module which labels to request — and reported `monitoring_active: true`. | Reproduced: 7 frames processed, 1 rule active, **0 events**, module never swapped | Both calls now run unconditionally in `_build()` before the loop starts | Verified on hardware: boot log now prints `[MONITORING] mode=ai_custom module=occupancy base_monitoring=DISABLED` and the running module receives `{'cell phone'}` |
| 2 | **`overrides_mode` defaulted to `True`.** A `monitoring.json` from an older format, hand-edited, or written by anything omitting the key activated AI-custom monitoring and silenced the configured module — the most disruptive possible reading of an absent field. | `core/monitoring_config.py:88` | Defaults `False`; overriding the operator's configured module is now an explicit act | 1362-test suite |
| 3 | **A `_build()` failure power-cycled the Pi.** `self._build()` was called outside `run()`'s `try/finally`, so a camera that would not open, or a dashboard port still held, escaped before any cleanup existed. Camera left open → next start fails too → `Restart=always` every 3s → `StartLimitAction=reboot`. | Reproduced: `CameraError` after 10.0s with `dashboard built? None`, `camera released? False` | `_build()` moved inside the try; new `_shutdown_partial()` releases whatever was constructed, best-effort, without masking the original error | Re-run of the same repro: `camera released? True` |
| 4 | **The dashboard reported 0.0 fps on a healthy 30 fps camera.** `_measure_fps` kept one module-level baseline shared by two callers — the inference loop (~5s cadence) and *every* `/api/health` request on the dashboard thread. Each consumed the other's window. | Measured: loop 30.0, dashboard **0.0**, loop 29.9, same bus, same second | Baseline keyed per caller | On hardware: dashboard now reports 30.2 / 30.1 / 29.7 |
| 5 | **`DashboardServer.is_alive()` could never return False.** It returned `self._httpd is not None`, and `close()` never cleared it — so the registered health check was a constant `True` and its restart path unreachable. If the serve thread died the live feed went dark while health said the dashboard was fine. | `is_alive after close: True` | Tests the serve thread; `close()` calls `server_close()` and clears `_httpd` | `is_alive after close: False` |
| 6 | **`make deploy` shipped developer data to customer devices.** `--exclude='data'` covers only the `data/` directory, so the rsync carried `cloud/sensordyme.db` (the backend's own database, 639 KB) plus `e.csv`, `events.csv`, `t.csv`, `telemetry.csv` from the repo root. | `rsync -avn` transfer list | Excludes `*.db`, `*.sqlite3`, `*.csv`, `.pytest_cache`, `.ruff_cache`, `.claude*` | Dry run: **0** data-bearing files; `test_deploy_ships_no_data_bearing_files` |
| 8 | **A module swap never re-verified the detection backend.** `_verify_detection_backend()` ran once at boot. Neither `_apply_monitoring_mode()` nor `_switch_preset()` re-checked, so a device booting on Elderly Monitoring (Pose-only, nothing degraded) and then switched in the app to Security or Baby would load YOLO lazily inside `process()`, raise every frame, and keep reporting `monitoring_active: true`. | Both swap sites construct a module; neither called the check (AST-verified) | Both re-verify; the check is now idempotent so switching *out* of a degraded state clears the reason | On hardware: a `security_monitoring` preset switch logged `object detection backend for occupancy: hailo` — verification delta +1. Guards: `test_every_module_swap_re_verifies_the_detection_backend`, `test_the_backend_check_clears_a_stale_degraded_reason` |
| 7 | **`setup_pi.sh` and `config.yaml` promised a CPU fallback that no longer exists.** A failed `hailo-all` install was described as harmless; on the current device it means no detector at all. | Both files read | Corrected, with the historical claim preserved as history | `test_no_shipped_comment_promises_a_cpu_fallback` |

---

## VALIDATED — directly tested and passed

**On hardware.** Demo preflight: **23/23 PASS → DEMO READY**. Mode rehearsal:
all four modes start, reach the NPU and report truthfully across 8 restarts with
no reboot (`driver_attention` 6.0s/80 NPU/29.7 fps · `baby_monitoring`
6.3s/122/27.4 · `fall_detection` 6.2s/**0 NPU — correct, Pose-only**/29.8 ·
`occupancy` 6.2s/362/29.8). Model integrity: config pin == file hash ==
`e69136d3…`, exactly one `.hef` in `models/`. Forbidden packages: `ultralytics`,
`torch`, `torchvision` all ABSENT; zero stray `.pt`. Live state: 29.6 fps, p99
36.5 ms, **0 evidence-losing stalls**, 63.9 °C, 853/4049 MB,
`throttle_now=[] since_boot=[]`.

**Negative testing — the preflight has teeth.** Missing model → `DEMO NOT READY`
(3 checks). Stale AI override → `DEMO NOT READY` naming the instruction and its
age. Wrong model hash → `DEMO NOT READY`. Each returns exit 1.

**Failure/recovery matrix** (`tests/test_phase6_failure_matrix.py`, 16 tests):
three accelerator failure modes all degrade rather than lie; a detector dying
mid-run flips `monitoring_active`; recovery un-degrades; missing camera, camera
that never produced a frame, stale frames while capture claims to run, and a
stopped capture service are each reported rather than assumed healthy; a stale
override is detectable without a running device, and a corrupt `monitoring.json`
is reported rather than read as "no override".

**Regression.** pytest **1364 passed, 1 skipped, 0 failed**. Ruff **336
tree-wide — identical to the Phase 6 starting count; zero introduced**; new
Phase 6 files clean. Pyright **8 errors — 0 in any file Phase 6 touched**.

---

## PARTIAL — evidence exists, the real-world condition does not

* **Fall detection.** 86 tests pass; the demo config preserves the measured
  30 fps / 1.0 s grace assumptions; zero stalls exceed the grace. All fall
  evidence remains *replay of real pose sequences*. No staged fall.
* **Baby monitoring.** 101 tests pass; all seven identification-safety
  properties verified in code. No real infant.
* **Deployment reproducibility.** All four provisioning cases pass, including
  a corrupt model **failing the install** (exit 1). No clean-image install.
* **`infant_unidentified`.** Behaviour confirmed unchanged and truthful: the
  module declines to score when it cannot identify the infant, and the decline
  is logged. **Deliberately NOT promoted to a loud notification** — alert
  fatigue on a baby monitor is itself a hazard, and the firing rate is
  unmeasured. Phase 5's deferral stands.

---

## UNVERIFIED — REQUIRES PHYSICAL/REAL-WORLD VALIDATION

Staged real fall · real infant/crib footage · driver-facing real-geometry
dataset · per-person recall at security-camera range · wall power measurement
(the 3.27 W figure is PMIC rails only and excludes the HAT) · clean-device
end-to-end install · long-run memory and thermal behaviour.

Also unverified, from the shipping-image audit: whether shipping a **prepared
SD-card image** triggers LGPL relinking/source-offer terms for `bluez`,
`python3-gi`/GLib and `python3-libcamera` (compounded by the venv being built
`--system-site-packages`); whether `simpleaudio`'s dynamic ALSA linkage matters;
and whether openpilot's MIT licence covers the released `dmonitoring_model.onnx`
artifact as well as its source. **These need legal review, not more testing.**

---

## DEFERRED — postponed until after the competition

* **48-hour production soak.** Partial evidence preserved at
  `docs/evidence/phase4_soak48_partial.csv` (0.39 h of 48 = **0.8%**). Not a pass.
* **Hailo driver warnings — BOUNDED, DEFERRED.** `hailo_vdma_buffer_map` calls
  `find_vma()` without holding `mmap_lock`. 43 per boot, **delta 0** across live
  inference. Kernel tainted `W|O`. **There is nothing to upgrade to:** `apt-cache
  policy hailo-all` reports Installed 4.20.0+1 = Candidate 4.20.0+1, the only
  version in the Raspberry Pi repo. Changing it would mean going off-repo on a
  working demo device. Not justified.
* **Lock the transitive dependency closure.** Only 11 direct pins are exact; 42
  transitive packages resolve fresh at install. `chardet` (transitive via
  `reportlab`) is LGPL-2.1 at ≤5.2.0 and 0BSD only from 7.6.0 — an offline
  wheelhouse or a pinned-back `reportlab` could put LGPL code in the image with
  nothing noticing. Fix is mechanical (`uv pip compile --generate-hashes`) but
  changes what installs on the device; not before a demo.
* **Apache-2.0 §4(d) attribution.** `docs/THIRD_PARTY_NOTICES.md` now exists and
  carries the DAMO-YOLO NOTICE verbatim plus a 53-package table, **and states in
  its own header that it is incomplete**. Per-package licence texts must be
  collected from a provisioned Pi.
* **`opencv-python` / `opencv-contrib-python` collision.** Both provide `cv2`;
  mediapipe pulls the contrib build unpinned, so the pin does not bind and the
  effective version depends on install order. Currently working; changing an
  OpenCV build before a demo is not.
* **`dmonitoring_model.onnx` is unpinned** — no URL, no checksum, unlike the
  `.hef`. Should be provisioned the same way.
* **BLE.** `org.bluez.Error.Failed` was transient; the code falls back to kernel
  mgmt and currently advertises as `SensorDyme-3D24`. **Off the demo path**:
  provisioning only opens when Wi-Fi is down, the device is on Wi-Fi, and
  provisioning is not a health component — a BLE failure cannot mark the device
  unhealthy or affect `monitoring_active` (verified: all components `ok`).

---

## Outstanding items, Phases 1–6

| Item | Current State | Demo Impact | Can Claude Complete Now? | Evidence | Next Action |
|---|---|---|---|---|---|
| 48-hour soak | DEFERRED | None | **No** — needs 48 h | `phase4_soak48_partial.csv`, 0.39 h | Run post-competition |
| Staged real fall | UNVERIFIED | Fall demo is replay-backed only | **No** — needs a person + safe rig | 3/3 confirms on real pose replay | Staged fall with mats |
| Real infant/crib footage | UNVERIFIED | Baby demo is logic-backed only | **No** — needs an infant | 101 tests, 7/7 properties | Consented footage |
| Driver-facing dataset | UNVERIFIED | Accuracy quoted is a lower bound | **No** — needs collection | State Farm is passenger-side | Collect in-car data |
| Security-range recall | UNVERIFIED | Occupancy thresholds unproven at range | **No** — needs footage | Close-range only | Collect at range |
| Clean-image install | PARTIAL | Low — device already provisioned | **No** — needs a spare SD card | 4/4 provisioning cases pass | Flash a spare card |
| Wall-power measurement | UNVERIFIED | Low | **No** — needs an inline meter | PMIC 3.27 W lower bound | USB-PD meter + 5 A PSU |
| Hailo driver warnings | BOUNDED, DEFERRED | None | **Done** — investigated | 43/boot, delta 0; no newer package exists | Report upstream |
| BLE failure | DEFERRED | None — off demo path | **Done** — investigated | Advertising via kernel mgmt | None before demo |
| Repo lint | KNOWN DEBT | None | Partially | 336 tree-wide, **0 introduced** | Sweep post-competition |
| Pyright | VALIDATED | None | **Done** | 8 errors, 0 in touched files | Fix the 8 later |
| Long-run memory | DEFERRED | Low | **No** — needs duration | Stable across 8 restarts | Part of the soak |
| Thermal drift | DEFERRED | Low | **No** — needs duration | 58–64 °C under load | Part of the soak |
| Demo rehearsal | **VALIDATED** | — | **Done** | `make demo-cycle`, ALL MODES OK | Re-run before going on stage |
| Dashboard | **FIXED** | Was showing 0.0 fps; health could not fail | **Done** | 30.2/30.1/29.7 fps; `is_alive` now falsifiable | — |
| Failure/recovery | **VALIDATED** | — | **Done** | 16 automated tests + 3 hardware negatives | — |
| Model provenance | **VALIDATED** | — | **Done** | Hash matches pin on device; corrupt model fails install | — |
| Licensing (Python closure) | **VALIDATED** | — | **Done** | No GPL/AGPL/LGPL in 53-package closure | Lock it post-demo |
| Licensing (attribution) | PARTIAL | None for a demo | Partially | `THIRD_PARTY_NOTICES.md`, self-declared incomplete | Collect texts on the Pi |
| Licensing (system/apt) | UNVERIFIED | None unless shipping an image | **No** — legal question | bluez GPL-2.0, libcamera LGPL-2.1 | Legal review |
| Accelerator failure behaviour | **VALIDATED** | — | **Done** | Degrades loudly; dashboard stays up; recovers | — |
| Startup ordering | **FIXED** | Was rebooting the Pi | **Done** | Repro before/after; 2 regression tests | — |
| Threshold consistency | **VALIDATED** | — | **Done** | Startup banner prints effective values, matches config | — |
| All four demo modes | **VALIDATED** | — | **Done** | `demo_cycle.sh` ALL MODES OK, twice | — |

---

## GO / NO-GO

# DEMO READY WITH CONDITIONS

**Conditions:**

1. **Run `make demo-check` immediately before demonstrating.** It is the whole
   point of this phase — 23 checks, and it refuses when the device is degraded,
   running the wrong mode, or running an unpinned model.
2. **Do not activate an AI-custom rule during the demo unless you intend to
   demonstrate AI-custom mode.** It legitimately overrides the configured module.
   Both the boot log and the preflight now say so loudly; before Phase 6 nothing did.
3. **Do not restart the service more than five times in two minutes.**
   `StartLimitAction=reboot` still power-cycles the Pi by design.
   `scripts/demo_cycle.sh` handles this correctly; ad-hoc `systemctl restart`
   loops do not.
4. **Quote accuracy figures as domain-shifted lower bounds.** State Farm's camera
   is passenger-side; SensorDyme mounts dash-centre.
5. **Do not claim fall detection or baby monitoring are validated against real
   subjects.** They are validated as logic and as replay.
6. **The repo has 38+ uncommitted files and no clean rollback point.** Commit
   before the competition so a bad change can be reverted under pressure.
