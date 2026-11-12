# Monitoring state (M0.2)

What the device says about whether it is actually monitoring, how it decides,
and what it does when it cannot. Code: `core/monitoring_state.py` (the pure
decision), `services/runtime.py` (`_evaluate_monitoring`, `monitoring_report`,
`_monitoring_check`), `core/roles.py` (`required_capabilities`).

"The process is running" is never "the device is monitoring". Nothing here is
an accuracy or confidence figure: *available* means a component is producing
results now, not that the results are right.

## States and precedence

The first rule that applies wins:

| State | When |
|---|---|
| `stopped` | an explicit, confirmed stop is in force (`/api/protection`, the app) |
| `failed` | a capability the **role** requires is impaired, or no active policy can evaluate anything |
| `starting` | boot, resume or a module change is still bringing a needed capability up, and nothing has failed |
| `degraded` | some policy, event or alert output cannot work, but every role-required capability is available |
| `ready` | every active policy can evaluate every event it declares, and every alert output is ready |

`monitoring_active` (the field the app trusts) is true only for `ready` and
`degraded`. "Starting" is bounded: a capability not up 30 s after boot, resume
or a module change is reported impaired.

Transitions are evaluated every 2 s on the health monitor's thread, not the
inference thread, so a wedged inference loop is still detected. Each
transition is logged (journal + `events.csv` as `monitoring_<state>`) and
pushed to the dashboard and backend. Entering `failed` or `stopped` ends the
module's and rules' open alert episodes. A condition that is still true when
monitoring returns re-opens its episode once, with no duplicates.

## Capabilities

Vocabulary shared with `core/capabilities.py`:

| Capability | Source | Impaired when |
|---|---|---|
| `camera` | frame bus age | no frame for more than `max(2 s, 5 frame intervals)`; no frame ever after 10 s; capture stopped |
| `inference` | loop heartbeat, processed-frame age | loop silent for more than 10 s; `process()` failing; frames arrive but none analysed for more than 5 s; a stuck mode switch |
| `face`, `objects`, `hands`, `pose` | the module's `capability_status()` | the module says so (e.g. a detector failing and waiting to retry) |
| `alert_output`, `audible_alert` | `AlertManager.actuators()` | no configured output ready; no *audible* output ready |

A module declares what each event needs (`event_requirements()`); see
driver_attention: drowsiness → face; distraction / eyes down / eyes up /
glancing → face + `attention` (the driver's calibrated forward gaze, `starting`
until calibration settles); phone_use → face + attention + objects. A module that
declares nothing is judged only by whether it processes frames. Custom rules
need the capability that provides their target label. A rule naming an area
that has not been drawn yet is *waiting*, and reported as such.

## Roles

| Role | Required (loss → `failed`) |
|---|---|
| `driver_safety` | camera, inference, face, audible_alert |
| `property`, `occupancy` | camera, inference, objects |
| `child_monitoring`, `workplace` | camera, inference |
| `general` (default) | none: `failed` only when nothing can be evaluated |

Example: object detection fails on a driver unit. Phone use is unavailable and
the state is `degraded`. Drowsiness, distraction and attention-lost keep working
and keep alerting. Object detection retries on its own (1 s, doubling to 60 s)
and the state returns to `ready`.

## Alert outputs

Each output reports **software** readiness: it initialised, it says it is
ready, and its last command did not raise. An output that could not be set up
(no GPIO, not a Pi) stays listed, as unavailable, with the reason.
`physically_verified` is false for every output. No SensorDyme output has a
feedback line, so nothing can tell that a buzzer actually sounded.

While `failed`, the device plays the `impaired` pattern on every output that
still works: every 15 s, at most 20 repeats (5 min), then silent until
recovery. `degraded` is reported, never sounded, so an unexplained chirp for a
lost phone detector does not teach a driver to ignore the buzzer.

## Frames and time

Every frame carries its capture (publish) time (`FrameBus.wait_for_fresh_stamped`).
The runtime hands it to the module (`set_frame_time`). Trackers run on
*evidence time* (`modules/base.py:EvidenceClock`): a gap between frames longer
than `max_evidence_gap_s` (0.5 s) counts as only 0.5 s. `driver_attention`
uses a stricter clock (`StallAwareEvidenceClock`, M8): a stall counts as one
ordinary frame interval of the stream, and only a stream that is slow
throughout keeps the 0.5 s cap. A camera freeze or a
blocked loop can no longer become evidence of eye closure or looking down.
This is also the seam a deterministic replay would feed recorded timestamps
through.

## Recovery and restarts

- Detector errors: retried with bounded backoff (1 s, doubling to 60 s), never
  disabled for the run. A failed load holds no model. The accelerator is
  re-probed when the cached probe said "not usable".
- Detector missing at boot: a module that can still work without it (face-based
  driver monitoring) keeps running; one that cannot idles and reports why.
- Watchdog: a DOWN critical component withholds the systemd ping, so systemd
  restarts the process. Once the last-resort policy decides to stay up degraded
  (reboot suppressed or disabled, or a process restart already tried for this
  outage), the hold is released. The device stays up, reachable and honest
  instead of being restarted every 45 s.
- Configuration: an unreadable, malformed or non-mapping `config.yaml` exits 78
  (no restart, no reboot loop). A module that cannot be built boots `failed`
  with the reason, so health, the dashboard, the app link and BLE still work.

## Deployment preflight

`make deploy` runs `make deploy-preflight` first (`scripts/deploy_preflight.py`).
It reads the **device's** `config.yaml` (deploys never overwrite it) and checks:
- the schema and role floor;
- the role this deployment is for (`ROLE=driver_safety`);
- what that role needs;
- whether the device's `.hef` is the build the config pins.

A device with no `role:` runs as `general`. That is reported, and refused for a
protected-role deployment with the line to add; the preflight never edits it.
