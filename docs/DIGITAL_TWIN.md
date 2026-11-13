# The digital twin (M8.4)

The virtual device (`docs/VIRTUAL_DEVICE.md`) grown into a full-product twin:
for every component, what runs is either the **real production code** or a
named **stand-in**, and every result says which. This page is the fidelity
map, the user journeys, the failure-injection and recovery coverage, the
storage-integrity results, the soaks, and the one command that runs it all.

Nothing here is a hardware result. The physical device (Sunday 2026-11-15)
proves the rows marked *hardware only*.

---

## 1. Fidelity map

Machine-readable in `sim/fidelity.py` (`FIDELITY`); the device's health report
lists its stand-ins (`health.simulation.stand_ins`), and `acceptance.json`
carries the whole map. `tests/test_m84_twin_fidelity.py` keeps this table, the
module and the health report in step.

| Component | Code in the twin | What the stand-in fakes | Hardware only |
|---|---|---|---|
| camera | **stand-in** `sim/camera.py` SimCamera | the image (drawn scene, SIMULATED banner); faults frozen / dropping / stalled | Camera Module 3 exposure, IR, blur, Picamera2 reopen |
| capture | **real + seam**: live = `services/capture.CaptureService` pumping SimCamera; stepped = `advance()` is the pump | stepped mode has no pump thread | frame timing under CPU load |
| FrameBus | **real** `core/frame_bus.py` | — | — |
| perception engine | **real** `core/perception.py` (one detector run per frame, shared) | — | — |
| FaceMesh | **stand-in** ScriptedFaceMesh (`--video`: real MediaPipe) | 478 geometrically real landmarks for the scene's pose and eye state | accuracy on real drivers, glasses, IR |
| Hands | **stand-in** ScriptedHands (`--video`: real) | one hand on a held phone | real hands |
| object detector | **stand-in** ScriptedDetector | person/phone boxes from the scene; raises on `detector_failing`; identity says `simulated` | Hailo-8L INT8 accuracy, latency, NPU faults |
| Pose | **stand-in** ScriptedPose | never finds a pose (pose signals stay unclaimed) | MediaPipe Pose |
| Hailo decoder | **real + seam** `core/accel.HailoYOLO` on `scripts/hw_validation/fake_hailo.py`; M8.4 runs it inside the runtime | the chip returns synthetic head tensors (no quantisation, timing, PCIe) | the real `.hef` outputs, firmware |
| driver_attention logic | **real** (calibration, EAR/iris, pose, gaze, trackers, risk) | — | — |
| monitoring state | **real** (`core/monitoring_state`, runtime, role floors) | — | — |
| alert policy | **real** `core/alerts.py` (episodes, cooldowns, delivery evidence) | — | — |
| buzzer / actuators | **stand-in** SimBuzzerSink for `gpio_buzzer` | actuation recorded, reported `simulated` (never `sounded`); can fail or report late | the buzzer sounding |
| thermal | **real + seam**: `thermal_condition`, degradation, hysteresis | the reading (`Faults.soc_temp_c`, `throttle`) instead of sysfs / vcgencmd | real heat, throttling, fan |
| storage | **real** filesystem (events.csv, spool, saved state, LKG); M8.4 fills a real small volume | — | SD-card wear, power-cut durability of a real card |
| uplink | **real + seam** `core/uplink.CloudUplink` | the transport (`TestClientTransport`, `NetworkGate`, `sim.faults.LinkFaults`) | real Wi-Fi loss patterns |
| network | **stand-in** NetworkGate | the radio (no NetworkManager) | Wi-Fi drop, handover |
| backend | **real** `cloud/app.py` (TestClient on the SimClock, or uvicorn) | AI provider `unavailable`: deterministic fallback, no model calls | — |
| dashboard / `/stream` | **real** `core/dashboard.py` (live mode) | frames are SimCamera's | MJPEG encode cost on the Pi |
| BLE provisioning | **real + seam** `services/provisioning` + `ble_pop` in `sim/ble_peripheral.py`; the app's Swift code via `sim/ble_harness.py` | the radio: in-memory GATT over localhost | BlueZ, CoreBluetooth, the Wi-Fi join |
| identity | **stand-in** `sim/identity.py` (`SD-SIM-…`) | the hardware serial | factory provisioning |
| config / boot | **real** `main.build_runtime`, validation, role floors, LKG | — | — |
| health ladder | **real + seam**: live = HealthMonitor + Supervisor; stepped = monitoring check on the production cadence, optional `supervise_inference` | reboot refused and logged; stepped mode has no restart backoff or budget | systemd watchdog, `Restart=`, start-limit, reboot |
| clock | **stand-in** `sim/clock.SimClock` (deterministic suites only) | `time.monotonic`, and the backend's `time.time` | RTC-less boot, NTP |

On quantcompute (Windows) the twin does not run: its job interpreter has
numpy only (no fastapi, uvicorn, mediapipe, cv2, yaml, psutil; probe job
`20261113-084005-ca4086`), and installing packages there is outside the frozen
fleet. oryn (WSL) runs it headless on CPU with `--no-gpu`.

---

## 2. User journeys (backend / device level)

The iOS stream covers the screens. Here every journey is asserted on real API
responses and on what the device did, with `sim.invariants.InvariantMonitor`
attached (no uplink cycle may break a truth invariant or a state-model edge).
**J** = `tests/test_m84_twin_journeys.py::test_the_whole_product_journey_from_first_boot_to_sign_out`
(one ordered run through every step).

| Journey | Test | Asserted |
|---|---|---|
| Onboarding: account | J; `tests/test_m8_onboarding_e2e.py` | signup/login tokens; device invisible until claimed |
| Onboarding: BLE proof of possession | `tests/test_m8_onboarding_peripheral.py`, `tests/test_m8_onboarding_crosslang.py`, `tests/test_m8_onboarding_e2e.py` (Swift app code vs simulated peripheral; swiftc only), `python -m sim.ble_harness --all` | proof before secrets, one join, lockout, timeouts |
| Onboarding: claim | J (wrong code 404, then claim); `tests/test_m8_security_backend.py` claim limits | row appears with `SD-SIM-` id, `simulated: true` |
| Discovery | J (unclaimed = on no account; catalogue); `tests/test_sim_units.py::test_the_device_local_api_says_simulated` (local API); mDNS live only (`--mdns`) | |
| Camera setup | J: area drawn via `POST /zones`, applied on the device (`_effective_zones`) | |
| Mode activation | J: `driver_safety:strict` active only once the DEVICE reports it | |
| Monitoring confirmation | J: ready / protecting / active, `state_model.derive(...).protecting` | |
| Video | `tests/test_m83_stream_runtime.py`, `tests/test_m83_stream_device.py`, `tests/test_stream_viewer_lifecycle.py`; live soak (§6) reads `/stream` for hours | |
| Activity history | J (paging by `before_id`, newest first, no overlap); `tests/test_m8_driver_activation_sim.py::test_event_history_is_ordered_complete_limited_and_survives_a_restart` | |
| Event detail / explain | J (`/explain` 200; unknown id 404); `tests/test_m83_events_backend.py::test_explain_uses_the_backend_clock_…` | |
| Alert output evidence | J (`alert.output == "simulated"`, `sim_buzzer` played); `tests/test_m83_events_sim.py` | |
| Ask Jupiter review + activation (confirmed only by device) | J: draft → activate → `rules.active` empty until the device runs it → active → fires; `tests/test_sim_e2e.py::test_custom_rule_draft_activate_…` | |
| Stop | J (confirmation required; outcome confirmed only by the echo); `tests/test_m83_truth_protection_ack.py` | |
| Resume | J; ACK suite | |
| Disconnect | J (offline, `health` withheld, nothing claimed) | |
| Reconnect | J; `tests/test_m84_twin_journeys.py::test_an_alert_raised_while_disconnected_reaches_history_once_with_its_true_time` | |
| Reboot | J (same id, mode, rule); `tests/test_sim_e2e.py::test_restart_keeps_identity_pairing_mode_and_rules` | |
| Sign-out | J (`/v1/logout`, token then 401; device still protecting) | |
| Ownership (another account refused) | `test_another_account_can_neither_see_claim_nor_command_this_device` (claim, events, stop, preset, zones, rename, AI configure all refused; nothing changed) | |
| Invalid / stale device state | `test_invalid_requests_and_a_frozen_device_are_told_apart_from_a_healthy_one` (400/409/422/404 codes; a frozen health sample is stale, never ready, then recovers) | |

---

## 3. Failure injection (G3)

Deterministic (SimClock, seeded faults), invariants attached; **F** =
`tests/test_m84_twin_faults.py`, **R** = `tests/test_m84_reliability_recovery.py`,
**S** = `tests/test_m84_reliability_storage.py`.

| Fault | Test | Expected / asserted |
|---|---|---|
| Camera freeze | `test_m83_truth_failure_matrix` F1; `test_sim_e2e` camera freeze; R storms | failed, nothing detecting, then ready |
| Stream latency | `test_m83_stream_device` (slow client latest-frame-only, stale frames withheld, send timeout); soak frame ages | |
| Partial network loss (lossy / slow) | F `test_a_partially_working_link_…[lossy, reordering, mixed]` | every event once (device ledger = backend rows), every command once, ready at the end |
| Duplicate messages | F `[duplicating]` (every request delivered twice); M81 health `repeat` | no duplicate row, no second effect |
| Reordered events | F `[reordering]`, F `test_late_deliveries_never_reorder_history_…`; M81S reordered health | history keeps device time order; old health never overwrites |
| Invalid accelerator tensors | F `test_malformed_accelerator_tensors_…[nan_scores, inf_box, wrong_grid, missing_head]` (real HailoYOLO validation inside the runtime) | degraded, objects impaired, face + alerting kept, recovers; never a crash |
| Corrupt cached configuration | `test_m8_lifecycle_boot` (damaged config → LKG; broken LKG not used) | |
| Stale capability reporting | M83 F14; `test_m82_paused_capabilities`; J invalid/stale | |
| Model load failure | `test_sim_e2e` model failure; `test_m8_lifecycle_boot` rebuild/backoff | |
| Storage exhaustion | S (real full volume) | no crash, health `storage: failed` "full", bounded local loss, 0 events lost to the app |
| Backend restart | F `test_a_backend_restart_during_an_outage_with_a_pending_stop_converges_once`; M81S; events backend | |
| Device process restart | M83 F3/F4; F ack loss (power cut); soak SIGKILLs | |
| Alert-acknowledgement loss | F `test_an_alert_whose_acknowledgement_is_lost_repeatedly_and_across_a_power_cut_is_stored_once` | one row, one notification |
| Clock skew | `test_m81_health_freshness_sim` wall-clock jumps; `test_m83_events_sim` device clock 3 days wrong; `test_command_auth_backend::test_clock_skew_…` | |
| Permission changes | F read-only data dir (live and at reboot); `test_m8_onboarding_setup_code`, `test_ble_pop` (setup-code file mode/owner) | monitoring, alerts, stop all work; health says storage failed |
| BLE setup interruption | `test_m8_onboarding_peripheral` (refused passphrase, timeouts, lockout across reconnects) | |

---

## 4. Fault recovery (D2)

| Requirement | Test |
|---|---|
| Crash | M83 F3; soak device SIGKILLs |
| Reboot | M83 F4; J reboot; soak in-process reboots |
| Network outage | E2E outage; F lossy link; soak flaps |
| Backend downtime | F backend restart mid-outage; soak backend SIGKILLs |
| Repeated inference failure → safe | R `test_a_module_that_fails_on_every_frame_is_failed_never_protecting_and_recovers` |
| Temperature rise | M83 F20 (heat, throttle, hysteresis) |
| No false protecting state | R `test_no_false_protecting_state_through_a_fault_storm[1-3]`; every suite's InvariantMonitor; soak per-row check |
| No delayed expired stop | ACK `test_a_stop_the_device_cannot_receive_times_out_and_is_never_applied_late` |
| No duplicate physical alerts after reconnect | R `test_a_reconnect_never_re_sounds_an_alert_or_stores_it_twice`; `test_m83_events_sim` camera reconnect |

---

## 5. Storage integrity (D3)

| Requirement | Test | Result |
|---|---|---|
| Partial writes | `test_event_log_durability`, `test_m8_lifecycle_uplink` (torn spool), `test_m8_lifecycle_boot` (torn config) | |
| Corruption | `test_sim_reliability` damaged state files; lifecycle boot | |
| Disk exhaustion | S full card end to end; S every writer; S events.csv bound | see §7 |
| Interrupted upload | `test_m8_lifecycle_uplink` power cut after the backend stored a batch | |
| Duplicate delivery | F duplicating link; events backend replay | |
| Reordering | F reordering link | |
| Retention boundaries | S 90-day prune at the exact cut-off (by arrival); S spool capacity | |
| Version migration | S M8 spool + newer-firmware line + torn line; `test_config_migration` | |

---

## 6. Soaks

`scripts/sim/twin_soak.py`:

- **live**: real processes and real time — `python -m sim.backend` (uvicorn)
  and a virtual-device process (its real threads, dashboard, `/stream`) on a
  repeating 6-minute cycle: drowsiness, phone, distraction, a 20 s network
  flap, a 45 s outage with an alert raised offline, a 15 s camera freeze, a
  20 s detector failure, then a quiet tail in which the controller stops and
  resumes monitoring through the API. On top: device SIGKILL + restart (30 %,
  50 % of the run), backend SIGKILL + restart (40 %, 60 %), an in-process
  device reboot every 30 min, `/stream` read continuously, `GET /v1/devices`
  every 3 s checked against `sim.invariants.status_violations`, and a final
  150 s fault-free drain before counting.
- **simclock**: the same timeline on the SimClock with the InvariantMonitor,
  through `LinkFaults(loss=.05, answer_loss=.05, duplicate=.03, hold=.03)`,
  with stop/resume each cycle, device restarts, power cuts and backend restarts.

Results: §7.

---

## 7. Results

(Filled from the runs; evidence in `~/sensordyme-audit/2026-11-10/m84-evidence/twin/`.)

---

## 8. One command

```bash
make acceptance SIM_PY=~/.venvs/sensordyme-m01/bin/python [ACCEPTANCE_OUT=dir] [ACCEPTANCE_ARGS=--quick]
```

`scripts/sim/acceptance.py`: every suite above in its own process, a short
sim-clock soak (judged on event integrity and invariants), a short live
`/stream` measurement (judged only on completing with frames — no timing
thresholds) and the BLE harness where swiftc exists. Writes
`acceptance.json` / `acceptance.md` (per suite pass / fail / skip / xfail,
durations, exit codes, commit + dirty flag, platform, `simulated: true`, the
fidelity map). Exit status is nonzero on any failure; skips are listed.
`make sim-e2e` and `scripts/sim/reliability_suite.sh` are unchanged.
