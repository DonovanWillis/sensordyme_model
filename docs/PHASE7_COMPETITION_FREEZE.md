# Phase 7 — Competition freeze, final validation & full system audit

_2026-08-30. Every hardware fact was executed on the physical device during this
phase. Phase 6's report was treated as a claim to re-verify, not as evidence._

## Executive verdict

**COMPETITION READY WITH CONDITIONS.**

Five defects were found in Phase 7, four of them by tests Phase 6 did not run.
All five are fixed, regression-tested and verified on hardware. The conditions
are operational, not engineering work.

The most important finding is that **Phase 6 only ever tested fresh boots.**
`demo_cycle.sh` restarts the service between modes, which is not what the demo
does — the demo switches modes *live*. Testing live switching immediately
exposed a health field lying, and it was a field this engagement had already
"fixed" twice.

---

## 1. Starting state — verified, not assumed

Repo matched Phase 6 exactly: 1364 passed / 1 skipped, ruff 336, pyright 8,
branch `integration/all`, HEAD `1b90ec0`. Device matched exactly: model hash ==
config pin == loaded binary, exactly one `.hef`, `ultralytics`/`torch`/
`torchvision` all absent, 0 stray `.pt`, `monitoring_active` true, 29.8 fps, 0
stalls, `throttle_now=[]`. **No regression from Phase 6.**

---

## 2. Defects found in Phase 7

| # | Severity | Defect | Discovery | Fix | Verified |
|---|---|---|---|---|---|
| 1 | **HIGH** | **Live mode switching reported `monitoring_active: false` on a working device.** The NPU counter is process-global and does not reset on a module swap, so after switching into Elderly Monitoring (`fall_detection` is MediaPipe-Pose-only and never touches the NPU) the *previous* module's inferences aged out and looked exactly like a dying detector. | Phase 7's live-switch test (Phase 6 only tested fresh boots, where the counter is 0 and the bug cannot appear) | `accelerator_died` gated on whether the **active** module needs a detector | Hardware: all 5 live switches now `monitoring_active: true`, same PID, 0 restarts |
| 2 | **HIGH** | **App-drawn zones were destroyed by the mode switch their own rules trigger.** The app sends zones and rules in one poll; zones apply first, the rules trigger the swap, and the swap replaced the drawn polygon with the new module's `config.yaml` zones (`{}`). The uplink had already acked `zones_version`, so nothing re-sent. | freeze-hunter, reproduced | Zone reset skipped when the mode is AI_CUSTOM; restored on explicit mode pick | Repro: 400 frames of a person inside the drawn polygon produced **0 alerts** before, zones now survive the swap |
| 3 | **MEDIUM** | **Top-level health read `degraded` for the whole Elderly segment.** `accel.live_status()` was registered as a supervised check without the same mode-awareness as #1, so it reported "pipeline stalled or fell back to CPU" on a perfectly working fall detector. | freeze-hunter, reproduced | Check scoped through `_accelerator_health()` | Hardware: `accelerator: ok — not used by this mode (fall_detection)`; log shows "accelerator recovered" |
| 4 | **MEDIUM** | **Frame-stall counters were a destructive read shared by two consumers.** The inference loop (feeding the uplink) and every `/api/health` request drained the same buffer; whoever read first got the data, the other got `{}` — no keys, so nothing downstream could tell "no stalls" from "not measured". A real 1.2 s stall reached exactly one consumer. | freeze-hunter, reproduced | `interval_stats(caller)` keyed per caller, same pattern already applied to fps | Repro: both consumers now receive the 1.2 s stall |
| 5 | **MEDIUM** | **A dead camera left `monitoring_active: true`.** The inference loop correctly stamps its heartbeat on every no-frame timeout, so `inference` reported "analysing" with no frame for 63 s, and the field documented as authoritative said yes. | freeze-hunter, reproduced | `monitoring_active` also false when the camera is FAILED on the **stale-frame** branch (not the "no frames yet" branch, which is ordinary at boot) | Three cases verified: dead camera False, booting True, healthy True |

### Self-inflicted incident, recorded honestly

Running `pre-commit run --all-files` during Step 14 executed ruff's **safe
autofix** across the tree. It rewrote ~85 files and broke two things:

* it deleted three re-exports from `main.py` (`load_preset_aliases`,
  `resolve_preset`, `same_preset`) as "unused", breaking test collection; and
* it rewrote `interval_stats=lambda: {}` into `interval_stats=dict`, which then
  received the new caller argument and raised `ValueError` inside health.

Both were found and fixed; `main.py` now declares `__all__` and the test stub is
a named function, so neither can be autofixed away again. Every shipping module
imports, `cloud/`, `benchmarks/` and `jupiter/` parse, and the suite is green, so
the changes were **kept** rather than reverted — reverting would have risked more
than it saved. But ~85 files carry mechanical changes that were not individually
reviewed, and tree-wide ruff moved 336 → 221 as a side effect. **Do not run
`pre-commit --all-files` again before the competition.**

---

## 3. Final hardware state

| | |
|---|---|
| board / kernel | Raspberry Pi 5 Model B Rev 1.1 · 6.12.93+rpt-rpi-2712 |
| accelerator | Hailo-8L · HailoRT 4.20.0 · `hailo-all` 4.20.0+1 |
| detector | DAMO-YOLO `damoyolo_tinynasL20_T`, Model Zoo v2.15.0 hailo8l, Apache-2.0, raw-tensor |
| **model SHA-256** | `e69136d3ebf3cc8ebaf3da5417a59962b139f3d3d67d715e9a9b1ae10963960e` (config pin == file == loaded) |
| frame rate | 29.8 fps · p95 36.6 ms · p99 38.0 ms |
| inference latency | 22.07 ms mean (Phase 5, 300 samples, this binary) |
| evidence-losing stalls | **0** |
| temperature | 64.5 °C under load |
| memory | 799 / 4049 MB |
| throttle | `throttle_now=[]`, `throttle_since_boot=[]` |
| restarts | 0 |
| cold boot → DEMO READY | **~21 s** |
| forbidden packages | `ultralytics`, `torch`, `torchvision` all ABSENT · 0 stray `.pt` |

---

## 4. Final regression

| | Phase 6 | Phase 7 |
|---|---|---|
| pytest | 1364 passed, 1 skipped, 0 failed | **1369 passed, 1 skipped, 0 failed** |
| ruff (tree-wide) | 336 | **221** (autofix side effect, see above) |
| pyright | 8 errors | **8 errors**, 0 in touched files |
| pre-commit | — | all hooks pass except the ruff autofix hook, which by design rewrites files |
| `make demo-check` | 23/23 | **25/25 PASS → DEMO READY** (1 warning) |
| `make demo-cycle` | 4/4 | **ALL MODES OK** |

---

## 5. Deployment reproducibility

Real `rsync -avn` dry run: 442 files, **zero** `.pt`/`.hef`/`.db`/`.sqlite3`/
`.csv`/`.log`, no `config.yaml`, no `.secrets`.

**Would a second Pi provisioned from this repo become the same demo system?**
Exactly **four** config keys would differ, all legitimately per-device:

| Key | Repo | This device | Why |
|---|---|---|---|
| `camera.rotate_180` | false | **true** | physical mounting — a reflashed Pi shows an upside-down image |
| `cloud.enabled` | false | true | provisioning state |
| `cloud.base_url` | localhost | the dev Mac | provisioning state |
| `cloud.device_key` | *(empty)* | `dk_…` | secret, correctly absent from the repo |

Everything else — code, model, thresholds, systemd unit, dependency set — is
reproduced by the documented path.

---

## 6. Final demo procedure

```
1.  Power on the Pi.                                    (~21 s to ready)
2.  make demo-check                                     -> must print DEMO READY
3.  Open the dashboard:  http://sensordyme.local:8899
4.  Demonstrate Driver Attention        (the boot default)
5.  Switch preset -> Baby Monitoring    (live, no restart)
6.  Switch preset -> Elderly Monitoring (fall detection; NPU count stays flat — correct)
7.  Switch preset -> Security           (occupancy)
8.  Switch preset -> Driver Safety      (return to primary)
9.  make demo-check                                     -> confirm still DEMO READY
```

`make demo-cycle` rehearses steps 4–8 unattended and prints ALL MODES OK.

---

## 7. Competition-day conditions

1. **Run `make demo-check` immediately before going on.** 25 checks; it refuses
   when the device is degraded, running the wrong mode, or running an unpinned
   model. It has been proven to refuse — see §8.
2. **Expect one WARNING: `cloud_uplink offline`.** That is correct when the
   backend is not running and does not affect a local demo. If you want it clean,
   set `cloud.enabled: false` on the device.
3. **Switch modes with the preset control, not by restarting the service.** More
   than five service starts in two minutes triggers `StartLimitAction=reboot`
   *by design*. Live preset switching has no such limit and is fully rehearsed.
4. **Do not add occupancy/baby/fall zones to `config.yaml` before the demo.**
   `occupancy.set_requested_labels()` overwrites the zone-category classes built
   in `__init__`; harmless as shipped (`zones: {}`) and live the moment a zone is
   added. Documented, deliberately not fixed in a freeze.
5. **Do not run `pre-commit run --all-files`.** Its ruff hook autofixes and
   rewrites the tree — see §2.
6. **Quote accuracy as domain-shifted lower bounds.** State Farm's camera is
   passenger-side; SensorDyme mounts dash-centre.
7. **Do not claim fall detection or baby monitoring are validated against real
   subjects.** They are validated as logic and as replay.

---

## 8. The gate has teeth — proven, not asserted

| Induced failure | Result |
|---|---|
| Model file missing | DEMO NOT READY, 3 checks, exit 1 |
| Stale AI override active | DEMO NOT READY, names the instruction and its age |
| Model hash ≠ pin | DEMO NOT READY, prints found vs expected |
| Corrupt model at install | install **fails**, exit 1 |
| Supervised component unhealthy | surfaced as a WARNING naming the component |
| All restored | DEMO READY, 25/25 |

---

## 9. Known limitations, by category — never blurred

**VALIDATED (hardware evidence):** model provenance and uniqueness · NPU in use ·
no CPU fallback possible · all four modes, fresh boot AND live switch · cold boot
· clean stop/start · degraded-mode behaviour and recovery · throttle live-vs-
sticky · deployment exclusions · preflight refusal behaviour.

**FIXED this phase:** the five defects in §2, each with a regression test.

**PARTIAL:** fall detection (replay of real pose sequences; 3/3 confirms) · baby
monitoring (101 tests, 7/7 safety properties in code) · deployment
reproducibility (4 known per-device deltas) · attribution
(`docs/THIRD_PARTY_NOTICES.md` exists and declares itself incomplete).

**UNVERIFIED — REQUIRES PHYSICAL/REAL-WORLD VALIDATION:** staged real fall · real
infant/crib footage · driver-facing real-geometry dataset · security-camera-range
recall · wall-power measurement · clean-image install · whether shipping a
prepared SD-card image triggers LGPL obligations for `bluez`/`libcamera` (legal,
not technical).

**DEFERRED:** 48-hour soak (0.39 h of 48 preserved — 0.8%, **not a pass**) ·
dependency lock (42 transitive packages unpinned; `chardet` is LGPL ≤5.2.0) ·
Hailo kernel warnings (43/boot, delta 0, and **no newer package exists** —
Installed == Candidate == 4.20.0+1) · `opencv-python`/`opencv-contrib-python`
collision · unpinned `dmonitoring_model.onnx`.

**OUTSIDE DEMO SCOPE, documented not fixed:** a failed AI_CUSTOM module swap
leaves events discarded with health green (trigger not reachable on the shipped
config; signature is `could not switch to '<module>' for ai_custom` in the log) ·
`occupancy.set_requested_labels()` zone-class overwrite (see condition 4) · the
live MJPEG stream freezes on the last frame rather than going dark when degraded,
so **the video is not proof of life — the health payload is**.
