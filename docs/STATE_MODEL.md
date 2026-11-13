# State model (M8.3)

The one thing the app may say about a device, the exact evidence behind each
state, and the transitions between them. It formalises the existing chain;
it adds no parallel status:

    device measurements ─▶ monitoring evaluation ─▶ health sample ─▶ backend status ─▶ app presentation
    services/runtime.py    core/monitoring_state.py  sd-health-v1     cloud/app.py       ios/Sources/*
    core/perception.py     (one evaluation feeds     (stamped,        _device_status     ConnectionTruth
    core/device_health.py   both the monitoring and   M8.1)            (APP_STATUS_       DeviceStatus
                            protection blocks)                         CONTRACT.md)       InstrumentState
                                                                                          ProtectionControl

Code: `core/state_model.py` (`derive`, `ALLOWED`, `transition_problem`) maps a
`GET /v1/devices` row to one state; `ios/Sources/InstrumentState.swift` is the
same table on the phone. The invariants that keep every layer consistent are
`sim/invariants.py` (Python, device sample and backend row) and
`ios/Sources/PresentationInvariants.swift` (Swift, what the screen shows).

The principle: **the app never claims a protection capability the hardware is
not delivering.** Every positive claim below needs live, fresh evidence; when
evidence is missing the state falls toward `unknown`, never toward `ready`.

## Device states

First rule that applies wins. "Device" = `services/runtime.py`; "backend" =
the `status` block of `GET /v1/devices`; "app" = `InstrumentState.Kind` and
the monitoring pill (`MonitoringStatusPresentation`).

| # | State | Device evidence | Backend evidence | App shows | Protecting? |
|---|---|---|---|---|---|
| 1 | **Offline** | nothing reaches the backend (no poll for 30 s) | `online: false`, `link_stale: true`; state/protection `unknown`; capabilities, perception, `health` null | `offline`, "Offline", wifi.slash; "Status unknown" | unknown |
| 2 | **Stale** | polling, but the health sample is ≥ 30 s old (frozen sampler) | `online: true`, `health_stale: true`, `health_freshness.state: stale`, `stale: true`; everything withheld as for offline | `stale` (presence `monitoringStale`), "Stale", "Status unknown" | unknown |
| 3 | **Unknown** | firmware that reports no monitoring state, or the phone's reply is old / its refresh failed | `monitoring_state: unknown` (or no `status`); ConnectionTruth `unknown(.refreshFailed / .notRefreshed)` | `unknown`, "Status unknown" | unknown |
| 4 | **Resuming** | stopped, with a resume not yet applied; or `starting` within 60 s of an applied resume | `protection: stopped` + `protection_command {action: resume, outcome: pending/timed_out}`; or `monitoring_state: starting` + `protection_command {resume, confirmed}` | the stopped readout, with the control's "Resuming…" / overdue note; then "Starting" | no |
| 5 | **Stopped** | `_stopped` set by a confirmed stop (app or device dashboard) | `protection: stopped`, `monitoring_state: stopped`, `monitoring_active: false`; analysis capabilities in `capabilities.paused` | `stopped`, "Stopped", pause.circle, neutral tone (never a warning) | no |
| 6 | **Faulted** | a role-required capability impaired, or nothing can be evaluated | `monitoring_state: failed`, `protection: not_protecting`, `reasons` | `failed`, "Not monitoring", xmark.octagon, critical | no |
| 7 | **Activation pending** | boot, a mode switch or calibration still bringing a needed capability up (bounded: 30 s, then impaired) | `monitoring_state: starting`, `capabilities.starting` | `starting`, "Starting", hourglass | no |
| 8 | **Degraded** | every role-required capability available, but some policy/event/output cannot work, or the SoC is throttling (M8.3) | `monitoring_state: degraded`, `monitoring_active: true`, `protection: protecting` (or `not_protecting` with reasons), `reasons` | `degraded`, "Degraded — reason", caution | yes when `protection: protecting` |
| 9 | **Ready (Active)** | every policy can evaluate every event, every output ready | `monitoring_state: ready`, `monitoring_active: true`, `protection: protecting`, `stale: false` | `ready`, "All clear", checkmark.shield, the only `ok` tone | yes |

`Active` / protecting is a property, not a separate state: `protecting` is
true only in Ready or Degraded with `protection == "protecting"` on live
evidence (`core/state_model.DerivedState.protecting`).

A pending **stop** is not a device state: the device keeps protecting until it
applies the stop, and the screen keeps the current state with "Stopping…".

## Capability states

Per capability (`status.capabilities`), `core/state_model.capability_state`:

| State | Evidence | App row |
|---|---|---|
| available | the device produced a result within the model's max age, with frames and a live loop | "… · producing results", ok |
| starting | not up yet, within the 30 s grace | the capability's own start-up words, caution |
| **paused** | monitoring stopped on purpose (M8.2) | "Paused — monitoring is stopped", neutral |
| impaired | with the device's reason | "Not working — reason", critical |
| not reported | stale, or the running mode does not use it | no row ("no capability is shown as working") |

Never `available` while: the device is stopped; `camera` or `inference` is
impaired (M8.3, `services/runtime.py` `_capabilities`); its perception
component is `failed`/`unavailable`; the status is stale.

## Stop / resume

    idle ──POST stop (confirm)──▶ pending ──device echo, state stopped────▶ confirmed  (Stopped)
                                     │    ──device echo, other state───────▶ mismatch   (device's reason)
                                     └────no echo in 45 s──────────────────▶ failed     (stop withdrawn at 30 s:
                                                                                         never applied late)
    idle ──POST resume──────────▶ pending ──device echo, state not stopped─▶ confirmed  ("monitoring again" only
                                     │                                                    when protecting)
                                     │    ──device echo, still stopped─────▶ mismatch
                                     └────no echo in 45 s──────────────────▶ overdue    (still delivered;
                                                                                         "still stopped")

Backend: `status.protection_command` (outcome latched from the first echo).
App: `ProtectionControl.phase`. Never success before the device's echo of THIS
id with the requested state; an older backend without `protection_command`
falls back to the echo and the app's own 45 s timer, and an echo with the
wrong state is still a mismatch.

## Transitions

`core/state_model.ALLOWED`. Losing or regaining liveness is always allowed (a
cable can be pulled at any time), so Offline, Stale and Unknown connect to
everything. Between live states:

| From \ To | Ready | Degraded | Faulted | Activation | Stopped | Resuming |
|---|---|---|---|---|---|---|
| Ready / Degraded / Faulted / Activation | ✓ | ✓ | ✓ | ✓ | guarded: a stop applied by the device | ✗ |
| Stopped | guarded | guarded | guarded | guarded | ✓ | ✓ (resume requested) |
| Resuming | guarded | guarded | guarded | guarded | ✓ (timed out / refused) | ✓ |

Guarded = `transition_problem`: entering Stopped needs a stop the device
applied (app command echoed, or its own dashboard); leaving Stopped for a
running state needs a resume the device applied or a restart (a new boot id;
stops are not persisted, so a restarted device comes back protecting).
`InvariantMonitor` (sim/invariants.py) checks every consecutive pair of states
the app observes in every virtual-device test that attaches it.

## Invariants

Python (`sim/invariants.py`), run on every device sample (`report_violations`)
and every backend row (`status_violations`) by `InvariantMonitor`, and over
generated reports by `tests/test_m83_truth_invariants.py`:

| Invariant | Rule | The contradiction it catches |
|---|---|---|
| `stale_withholds` | stale ⇒ state/protection unknown, active null, capabilities/perception/policies/actuators/rules.active/last_command_id/health null | device offline yet current protection shown |
| `stale_definition` | stale = link_stale ∨ health_stale; health_stale ⇔ freshness stale; verified health ≥ 30 s old is stale whatever `online` says | backend "online" while health telemetry is too old |
| `active_matches_state` | monitoring_active ⇔ state ∈ {ready, degraded}; null when unknown | |
| `protecting_requires_monitoring` | protecting ⇒ ready/degraded and active; the protection block's evaluation = the monitoring block's | a torn sample: failed beside protecting |
| `stopped_coherent` | protection stopped ⇔ state stopped | |
| `paused_not_analysing` | stopped ⇒ no analysis capability available, no component `ok` | paused yet analysis capabilities active |
| `pipeline_down_not_detecting` | camera/inference impaired ⇒ no analysis capability available, state not ready/degraded | inference stopped yet shown detecting; camera stale yet monitoring fully operational |
| `model_missing_not_available` | component failed/unavailable ⇒ its capability not available | model missing yet capability available |
| `capability_lists_disjoint` | one list per capability | |
| `ack_before_success` | outcome confirmed ⇒ applied (echoed) with the requested state | resume requested yet success shown before device ack |
| `state_transition` | consecutive states follow `ALLOWED` and the guards | stopped without a stop, running without a resume |

Swift (`ios/Sources/PresentationInvariants.swift`), over 20,000 generated
replies per run (`ios/Tests/PresentationInvariantsTests.swift`, half of them
contradictory on purpose):

| Rule | Claim | Needs |
|---|---|---|
| P1 | presence "Online" | `online`, not stale/link-stale/health-stale, a current reply, health < 30 s |
| P2 | "All clear", the ok tone, the ready readout | live and `monitoring_state: ready`, not stopped, camera and inference not impaired |
| P3 | live visuals (`isActive`), any monitoring phrase other than "Status unknown" | live |
| P4 | a capability row "producing results" | live, not stopped, camera and inference not impaired, its model not failed/missing |
| P5 | a perception component "Working" | live, not stopped |
| P6 | stop/resume "confirmed"; "monitoring again" | live, the device's echo (or latched confirmed) with the requested state; protecting |

## What M8.3 found and fixed

See the M8.3 report for line numbers. Device: the health sample's monitoring
and protection blocks came from two evaluations; `attention` stayed available
with the camera frozen or the loop dead; heat never degraded the state.
Backend: contradictory reports passed through; a stop/resume had no outcome
and a stop was delivered whenever the device came back. App: an echoed
refusal read "Stopped."; a resume while starting read "monitoring again";
stopped / camera-down / failed-model capabilities read "producing results";
"All clear" with the camera impaired; a module impairment hidden behind a
healthy model in the preset check; an alert on an offline device drew the
live visuals.

## Hardware-only

The model is software; what it cannot prove off a Pi is listed per case in
`docs/FAILURE_MATRIX.md` (real camera disconnect timing, Hailo loss, thermal
readings and throttle bits, BLE, Wi-Fi radio, systemd restart).
