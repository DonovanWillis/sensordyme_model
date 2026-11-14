# App status contract

What the backend tells the app about whether a device is actually watching,
and how it says no. Code: `cloud/app.py` (`_device_status`, `CodedError`).
Tests: `tests/test_app_status_contract.py`. App side: `ios/Sources/DeviceStatus.swift`.

Every field is derived from what the device itself reported
(`docs/MONITORING_STATE.md`: the health report's `monitoring` and `protection`
blocks, and the monitoring report pushed by `_publish_monitoring`). The
backend invents no state of its own except `unknown`, `stale`, and its own
record of a mode request it dropped.

## `status` on each device in `GET /v1/devices`

Always present. An app built before this contract ignores it; an app built
for it treats a missing `status` (an older backend) as "no truth available".

| Field | Type | Meaning |
|---|---|---|
| `monitoring_state` | `starting` \| `ready` \| `degraded` \| `stopped` \| `failed` \| `unknown` | The device's own monitoring state. `unknown` when the report is stale or absent, or the device's software does not report one. |
| `monitoring_active` | bool \| null | True only for `ready` and `degraded`. Null when the state is `unknown`. |
| `protection` | `protecting` \| `not_protecting` \| `stopped` \| `unknown` | Whether every safeguard the device's role requires is running. `unknown` when stale or not reported. |
| `reasons` | [string] | Why the state is not `ready`, in the device's words. Empty when ready. For `unknown`, the backend's reason (no report in 30 s, never reported, or not reported by this software). |
| `protection_reasons` | [string] | Why `protection` is `not_protecting` or `stopped`, in the device's words. |
| `mode` | `off` \| `standard` \| `ai_custom` \| `standard_plus_custom` \| null | What runs: a mode alone, custom rules replacing the mode (`ai_custom`), or custom rules beside the mode (`standard_plus_custom`, additive roles). The last reported value, kept when stale. Null when never reported. |
| `role` | string \| null | The deployment role (`core/roles.py`). Read from the dict or the bare-name shape. Null when never reported. |
| `role_title` | string \| null | The role's display name. |
| `allowed_presets` | [string] \| null | The only preset base names this role allows. Null means every preset is allowed, or no role has been reported. Kept when stale: it describes configuration, not liveness. |
| `capabilities` | object \| null | Null when stale. Otherwise the three lists below. |
| `capabilities.available` | [string] | Capabilities producing results now. "Available" is not a claim that the results are accurate. |
| `capabilities.impaired` | [{`name`, `reason`}] | Capabilities that are not working, each with the device's reason. |
| `capabilities.starting` | [string] | Capabilities still starting up. |
| `capabilities.paused` | [string] | M8.2, additive. Capabilities not running because monitoring was stopped on purpose; never a fault. Real faults found while stopped stay in `impaired`. |
| `rules` | object | Custom rules as the DEVICE runs them. |
| `rules.active` | [string] \| null | Rule ids the device reports running now. Null when stale. Never computed from what the backend accepted. |
| `rules.additive` | bool \| null | Whether this role runs custom rules beside its mode instead of replacing the mode. Null when no role is known. |
| `rules.rejected` | {`version`, `reasons`, `at`} \| null | Set when the device refused the rule set the backend currently wants to run (its rejection names that set's `rules_version`). The device keeps its previous rules (all or nothing). A refusal of an older set does not count. |
| `last_rejection` | {`kind`, `subject`, `reasons`, `code`, `at`} \| null | The device's most recent refusal of any kind (`preset`, `rules`, `protection`, `nodes`). `code` is the device's machine-readable code when it sends one, else null. The device never clears it, so check `at`. |
| `preset_rejection` | {`preset`, `code`, `reason`, `at`} \| null | A stored mode request that will not run, and why. `code` is `role_forbids_preset` when the backend dropped it for the device's role, or the device's own code (`device_refused` when it gave none) when the DEVICE refused it (M7: unknown mode, module build failure, role). Either way the request is no longer re-sent. The next accepted mode request clears it. |
| `nodes_rejection` | {`node`, `code`, `reasons`, `at`} \| null | A companion-node request (`POST /v1/devices/{id}/nodes`) the device refused; the request is no longer re-sent. The next node request for that node clears it. |
| `perception` | object \| null | M7 shared perception, exactly as the device reports it (`docs/M7_IMPLEMENTATION_MAP.md` wire contract): `schema`, `simulated`, `components` {name: {`state` `ok`\|`stale`\|`failed`\|`unavailable`\|`idle`\|`unknown`, `model`, `last_run_age_s`, `runs`, `errors`, `reason`, `consumers`}} and `nodes` [{`id`, `kind`, `version`, `preset`, `state` `active`\|`degraded`\|`inactive`\|`failed`\|`unknown`, `required_capabilities`, `missing_capabilities`, `validation`, `experimental`, `reason`}]. Null when stale or not reported by this software; never filled in by the backend. A state outside the vocabulary reads `unknown`. |
| `last_command_id` | string \| null | The id of the last protection command the device handled, applied or refused (`POST .../protection` returns it). Null when stale or not reported. An echo is an answer, not a success: compare the state, or read `protection_command.outcome`. |
| `protection_command` | object \| null | M8.3, additive. The last stop/resume asked for on this device and what the DEVICE did with it; null when none was ever sent. Not withheld when stale: it is a record of an acknowledgement, with its time. Fields below. |
| `protection_command.id` | string | The `command_id` `POST .../protection` returned. |
| `protection_command.action` | `stop` \| `resume` | |
| `protection_command.outcome` | `pending` \| `confirmed` \| `mismatch` \| `timed_out` | `confirmed`: the device echoed this id with the requested state (`stopped` for a stop, anything else for a resume). `mismatch`: it echoed this id with another state (it refused, or something changed it first). `timed_out`: no echo within `timeout_s`. Latched from the first report carrying the echo; a later restart or local change does not rewrite it. |
| `protection_command.delivering` | bool | Whether the backend still sends it. A stop is delivered for 30 s (`PROTECTION_STOP_DELIVERY_S`) and then withdrawn, so a stop is never applied minutes after the person gave up on it; a resume is delivered until applied (late protection is the safe direction). |
| `protection_command.age_s` | number \| null | Seconds since the request, backend clock. |
| `protection_command.timeout_s` | number | 45 (`PROTECTION_ACK_TIMEOUT_S`, the app's `ProtectionControl.timeout`). |
| `protection_command.applied_at` | number \| null | Backend receipt time of the first report echoing this id. |
| `protection_command.reported_state` | string \| null | The protection state that report carried. |
| `protection_command.reason` | string \| null | For `mismatch`: the device's refusal reason (a new `last_rejection` of kind `protection`), else the state it reported. |
| `camera_mode` | object | M8.4, additive (owner Decision C, `docs/PRIVACY.md`). Camera access while monitoring is stopped, as the DEVICE reports it, beside what the owner asked for. `supported` (bool: false = firmware before M8.4, which streams the raw camera while stopped and cannot be told not to; the app says "update it"), `camera_only` (bool \| null: the device's value), `remote_stream` (`monitoring` \| `camera_only` \| `off_while_stopped` \| `unavailable` \| null: what a remote viewer gets now; null when stale), `desired_camera_only` (bool: the owner's request, OFF unless set), `pending` (bool: the device has not reported the request yet), `command_id`, `stale`. Set with `POST /v1/devices/{id}/camera-mode` `{camera_only, confirm}` (owner only; `confirmation_required` to turn ON without `confirm`; `firmware_unsupported` 409 for a device without the setting). Decoded by `DeviceCameraMode` (ios/Sources/StreamLiveness.swift). |
| `corrections` | [string] | M8.3, additive. Claims the backend withdrew because the device's report contradicted its own rules (see "Coherence" below). Empty for a coherent report and whenever `stale`. |
| `policies` | [{`id`, `kind`, `name`, `can_evaluate`, `operational`, `unavailable_events`, `reasons`}] \| null | Each active policy (the mode, each custom rule) and whether the device can evaluate it now. Null when stale. |
| `actuators` | [{`name`, `kind`, `audible`, `ready`, `detail`, `commands_ok`, `commands_failed`, `physically_verified`}] \| null | Configured alert outputs, as far as software can tell (`physically_verified` is false until a person confirmed it). Null when stale. |
| `stale` | bool | True when the status's evidence cannot be trusted now: `link_stale` (the device has sent nothing for 30 s, `_HEALTH_STALE_S`, the same window as `online`, or has never reported) **or** `health_stale` (M8.1: the device is polling, but the health sample it sends is 30 s old or more). Before M8.1 only the first half existed; an app that reads only `stale` therefore falls to "unknown" on a frozen sample, which is the safe direction. |
| `reported_at` | number \| null | Backend receipt time (Unix, backend clock) of the device's last poll (`last_seen`). Link liveness, not an observation time: a poll carrying a re-sent sample moves it. The observation time is `health_freshness.observed_at`. |
| `link_stale` | bool | M8.1. True when no poll arrived in the last 30 s, or none ever. Whether the device is reachable; says nothing about whether it is monitoring. |
| `health_stale` | bool | M8.1. True when the stored health sample is stamped and 30 s old or more (`health_freshness.state == "stale"`), even if the device polled a second ago. |
| `health_freshness` | object | M8.1. How current the device's health evidence is, separately from the link. Fields below. |
| `health_freshness.state` | `fresh` \| `stale` \| `unverified` \| `absent` | `fresh`/`stale`: a stamped sample younger / not younger than `threshold_s`. `unverified`: firmware that does not stamp its health (before M8.1); freshness is judged by the link alone, exactly as before. `absent`: no health ever stored. |
| `health_freshness.verified` | bool | True only for a stamped sample (`fresh` or `stale`). |
| `health_freshness.age_s` | number \| null | Age of the sample now: `now - observed_at`, backend clock. Null unless verified. |
| `health_freshness.threshold_s` | number | 30 (`_HEALTH_STALE_S`). |
| `health_freshness.observed_at` | number \| null | When the device took the sample, on the BACKEND's clock: the receipt time of the first copy of this sample minus the age the device measured on its monotonic clock (`stamp.age_s`), lowered by any later copy that implies an earlier time, never raised. Null unless verified. |
| `health_freshness.received_at` | number \| null | Backend receipt time of the first copy of the current sample (a re-sent copy does not move it). For unverified firmware, the receipt time of the last report. |
| `health_freshness.device_generated_at` | number \| null | The device's own wall clock when it took the sample. Informational only: a device without an RTC can be hours off. Never used to judge freshness. |
| `health_freshness.boot_id` | string \| null | The device process's random boot id. Changes on every restart. |
| `health_freshness.generation` | int \| null | The sample's number within that boot. |
| `health_freshness.repeats` | int | How many times the current sample was received again unchanged. Many repeats on a live link is the frozen-snapshot signature. |
| `health_freshness.last_ignored` | {`reason`, `at`, `boot_id`, `generation`} \| null | The last health report NOT applied: `out_of_order` (older generation or revision of the current boot), `retired_boot` (from before a reboot the backend already saw) or `invalid` (a stamp that cannot be trusted; `detail` says why). The poll itself and its events were still accepted. |

Liveness rule: when `stale` is true, `monitoring_state` and `protection` are
`unknown`, and `monitoring_active`, `capabilities` and `rules.active` are null.
An old `ready` or `protecting` is never repeated. Since M8.1 this applies to
stale health evidence as much as to a silent device: a device that keeps
polling while re-sending a frozen sample is never "protecting". The `reasons`
then say "the device is connected, but its monitoring status has not been
updated for N s". `GET /v1/devices` `health` is withheld (null) on the same
rule; `online` stays link-only.

## Coherence (M8.3)

The backend invents no state, but it never forwards a claim the device's own
rules forbid (`_make_coherent` in `cloud/app.py`; the rules are
`core/monitoring_state.py` and `services/runtime.py`). Every correction only
WITHDRAWS a claim, is listed in `corrections`, and is applied only to a live
status:

| Report says | Status says | Device rule |
|---|---|---|
| `stopped` in one block, something else in the other | `stopped` in both | an explicit stop is authoritative |
| stopped, with an analysis capability (`face`, `objects`, `hands`, `pose`, `attention`, `inference`) `available` | moved to `paused`; perception components `ok`/`stale`/`idle` read `paused` | M8.2: nothing analyses while stopped |
| `camera` or `inference` impaired, an analysis capability `available` | moved to `impaired` ("no analysis: camera unavailable …") | nothing produces results without frames or a running loop |
| a perception component `failed` / `unavailable`, its capability `available` | removed from `available` (`failed`: into `impaired`; `unavailable`: left to the perception report, which `capability_states` reads as `unsupported`) | a missing model produces nothing |
| `ready` / `degraded` with `camera` or `inference` impaired | `failed`, with the reason | `BASE_REQUIREMENTS`: no policy can evaluate |
| `protecting` while not `ready`/`degraded` | `not_protecting`, reason "monitoring state is X" | `MonitoringReport.monitoring` |

`monitoring_active` is re-derived from the corrected state. A current device
sends coherent reports (the M8.3 device computes the monitoring and protection
blocks from ONE evaluation), so `corrections` is normally empty; it is there
for older firmware and torn samples. Tests: `tests/test_m83_truth_invariants.py`.

## Stop / resume acknowledgement (M8.3)

`POST /v1/devices/{id}/protection` answers `{ok, desired_protection, command_id, timeout_s}`.
Nothing is a success until the device echoes `command_id` with the requested
state; `status.protection_command` carries the outcome:

    pending ──echo + requested state──▶ confirmed
       │    ──echo + another state────▶ mismatch (reason)
       └────no echo within 45 s───────▶ timed_out
                                          stop: withdrawn after 30 s, never applied late
                                          resume: still delivered; confirmed if it lands later

The app (`ios/Sources/ProtectionControl.swift`) shows pending → confirmed /
mismatch / failed (a timed-out stop) / overdue (a timed-out resume, "still
stopped"). Tests: `tests/test_m83_truth_protection_ack.py`,
`ios/Tests/ProtectionControlTests.swift`.

## Health freshness (M8.1)

The defect: a device kept polling while re-sending an old, frozen health
sample (the sampling thread wedged, say), and the backend took every copy as
current, because it judged staleness from `last_seen`, the time of the last
poll. A successful poll proves the link, not the monitoring.

### Device side

`services/runtime.py` `_collect_health` stamps each sample where it is taken
(`core/health_stamp.py`):

    "stamp": {"schema": "sd-health-v1", "boot_id": "<16 hex, random per process>",
              "generation": 42, "revision": 0,
              "generated_wall": 1731400000.1, "generated_mono": 812.4,
              "age_s": 1.9}

`generation` rises by one per sample within a boot. `revision` rises when the
uplink refreshes only the monitoring keys of the held sample
(`refresh_health_monitoring`, on a stop/resume or monitoring transition): the
backend stores the newer content but the sample keeps its age. `age_s` is added
by `core/uplink.py` to the copy it sends, at send time:
`time.monotonic() - generated_mono`. The held sample is never modified, so a
sample that stops being refreshed is re-sent with a growing `age_s`.

### Backend side

`_store_health` (ingest) and `_health_freshness` (status), in `cloud/app.py`.
New `devices` columns, added by `init_db` like every other: `health_boot`,
`health_generation`, `health_revision`, `health_received_at`,
`health_observed_at`, `health_repeats`, `health_boots` (boot ids superseded by
a reboot, the last 8), `health_ignored`. The state is in the database, so a
backend restart changes nothing.

| Report (same device) | Outcome | Stored sample | Age |
|---|---|---|---|
| first stamped sample, or a newer generation of the current boot | `new` | replaced | from this receipt minus `age_s` |
| a boot id never seen | `reboot`: generation restart accepted, old boot id retired | replaced | as `new` |
| same generation, higher revision | `revised` | content replaced | kept (min with this copy's estimate) |
| same generation and revision | `repeat`: not new evidence | unchanged | keeps growing from the first receipt; `repeats` + 1 |
| lower generation or revision of the current boot | `out_of_order` | unchanged | unchanged; `last_ignored` set |
| a retired boot id | `retired_boot` | unchanged | unchanged; `last_ignored` set |
| a stamp that does not parse (wrong schema, no boot id, bad counter, negative or non-numeric age) | `invalid` | unchanged, so it never refreshes | unchanged; `last_ignored` set |
| no stamp at all | `unverified` (older firmware) | replaced | not computed |

Whatever the outcome, the poll is answered as before: `last_seen` moves, the
events in it are stored and notified, pending commands are delivered. Health
handling never refuses a poll.

Threshold: the same 30 s as the link (`_HEALTH_STALE_S`). A device samples
health every 5 s and polls every 2 s, so a healthy device's sample is normally
under 8 s old; 30 s is six missed samples.

### Clocks and the signed clock answer

Freshness uses only the backend's clock (receipt times) and the device's
monotonic clock (`age_s`, a difference within one process). The device wall
clock (`generated_wall`, reported as `device_generated_at`) is never compared
with anything, so a Pi without an RTC booting in 1970, or a clock that jumps
forwards or backwards under NTP, changes neither the age nor the order of
samples (order is `(boot_id, generation, revision)`, not time).

sd-auth-v1's signed clock answer (`clock_auth`, docs/DEVICE_AUTH_CONTRACT.md
§5) is orthogonal and composes without interaction: the learned offset is
applied to request timestamps only (`X-SD-Auth` `ts`), never to `stamp`
fields. With `cloud.command_auth: required` the whole body, `stamp.age_s`
included, is covered by the request signature, so it cannot be altered on the
path; a captured request replayed later is refused by the nonce cache before
ingest, and a replayed bearer-mode body carries a generation the backend has
already seen (`repeat` or `out_of_order`), which never refreshes the sample.
While the device's requests are refused for `clock_skew`, nothing is ingested:
the link goes stale after 30 s, and the sample ages on the backend's clock.
After the device adopts the signed time, its next poll carries the current
sample with its true monotonic `age_s`.

### Compatibility

| | backend before M8.1 | backend M8.1 |
|---|---|---|
| **device before M8.1** (no stamp) | as before | `health_freshness.state: "unverified"`; `stale`, `health`, `monitoring_state` and `protection` exactly as before (link only); `health_stale` false |
| **device M8.1** (stamped) | `stamp` is just more JSON inside `health`: stored and ignored; behaviour as before (a frozen sample is not detected) | full freshness |

An app built before M8.1 reads `stale`, which now also covers stale health,
and so shows "unknown" rather than an old "protecting". An app built for M8.1
on an older backend sees no `health_freshness` and keeps its earlier rules.

### Known limits

- `age_s` is measured at send time; time spent in transit (bounded by the
  10 s request timeout) is not included, so an age can be up to that much low
  on a slow link. The receipt-based floor still applies to every re-sent copy.
- A device that never stores a single sample (its health thread failing from
  boot) sends no `health`; the status then falls back to the monitoring block's
  state, as for firmware that predates health reports.
- The backend's own wall clock is trusted; a backend clock step moves every
  age by the same amount.

## Refusal codes

Refusals keep `detail` as a human sentence, which every client shows. A
machine-readable `code` sits next to it:
`{"detail": "...", "code": "role_forbids_preset"}`. The app branches only on
`code`, never on the status alone or on the wording.

| Code | Status | Endpoint | Meaning |
|---|---|---|---|
| `role_forbids_preset` | 409 | `POST /v1/devices/{id}/preset` | The device's role does not allow this mode. Checked before `draft_superseded`. |
| `draft_superseded` | 409 | `POST /v1/devices/{id}/preset` | The proposal being activated was replaced by a newer preview. Review it again. |
| `unknown_preset` | 400 | `POST /v1/devices/{id}/preset` | No such mode or variant. A planned mode (`GET /v1/presets`, `availability: "planned"`) is refused with this code too, and `detail` says it is planned. |
| `firmware_unsupported` | 409 | `POST /v1/devices/{id}/protection` | The device's software cannot stop or resume. |
| `confirmation_required` | 400 | `POST /v1/devices/{id}/protection` | A stop request must carry `confirm: true`. |
| `firmware_unsupported` | 409 | `POST /v1/devices/{id}/camera-mode` | M8.4. The device reports no camera block: its software cannot turn the camera off while monitoring is stopped. |
| `confirmation_required` | 400 | `POST /v1/devices/{id}/camera-mode`, `POST /v1/account/delete` | M8.4. Turning camera-only ON, and deleting an account, must carry `confirm: true`. |
| `wrong_password` | 403 | `POST /v1/account/delete` | M8.4. The password did not match. |
| `firmware_unsupported` | 409 | `POST /v1/devices/{id}/preset` | The request carries mode `settings`, and the device's software has not reported the `preset_settings` feature. The same mode without settings still works. |
| `invalid_setting` | 400 | `POST /v1/devices/{id}/preset` | A setting is unknown for that mode, of the wrong type, out of range or off its step. `detail` names the key. |
| `incompatible_policy` | 409 | `POST /v1/devices/{id}/preset` | The request asks for policies that cannot run together: `settings` with no mode (`preset: ""`), or `keep_rules: true` on a device whose custom rules replace the mode. Nothing was changed. |
| `rule_not_found` | 404 | `PATCH /v1/devices/{id}/rules/{rule_id}` | No custom rule with that id on this device. |
| `draft_superseded` | 409 | `PATCH /v1/devices/{id}/rules/{rule_id}` | The stored rules changed while the toggle was being saved. Reload them. |
| `device_not_found` | 404 | any `/v1/devices/{id}/...` | No such device on this account. |
| `device_already_claimed` | 409 | `POST /v1/devices/claim` | Another account owns the device. |
| `account_exists` | 409 | `POST /v1/signup` | The email is already registered. |

Other errors (401, 422, 429, 503) carry no `code`.

Per-preset availability codes (`role_forbids_preset`, `capability_impaired`,
`capability_unsupported`, `device_stale`, `planned`) are not refusals; they
are fields of `GET /v1/devices/{id}/presets`. See docs/PRESET_LIBRARY.md.

## Known limits

- A rule set is "delivered" once the device's uplink receives it, even if the
  device later refuses it. The rule count in the activation response
  (`rules_active`) comes from the backend's own validation. Whether the device
  runs the rules shows up on the next poll, in `rules.active` and
  `rules.rejected`.
- The backend cannot see a refusal that has not been reported yet. Until the
  device's next report arrives (within one uplink poll), `rules.rejected` is
  null.
- `allowed_presets` comes from the backend's role table when it knows the role,
  and otherwise from the device's own role description. A role that neither
  knows is treated as having no restriction.

## M8.2: `paused` capabilities while monitoring is stopped

When the owner stops monitoring (`protection: stopped`, `monitoring_state: stopped`), nothing is analysed, so models produce no fresh results. That is intended, not a fault. Before M8.2 those capabilities were reported `impaired` ("face landmarks stale"), and the app showed them as "Not working".

**Device**
- A capability that is only idle because the device is stopped is reported `{"state": "paused", "reason": "<name>: paused while monitoring is stopped"}`. This covers stale or missing results, the inference loop alive but not analysing, and module capabilities still starting (e.g. the driver calibration).
- In `monitoring.perception.components`, an `ok`, `stale` or `idle` component is reported `{"state": "paused", "reason": "monitoring is stopped"}`.
- Real faults keep their state:
  - camera faults;
  - a failed or unavailable component;
  - a module's own impairment;
  - an unavailable detector;
  - an unresponsive loop;
  - consecutive analysis failures.

**Backend**
- `status.capabilities.paused`: a new, additive list.
- `capability_states` and the preset readiness rows map it to `"state": "paused"`.
- A paused capability is never overridden by a merely stale component.

**App**
- The row reads "Paused — monitoring is stopped" in a neutral tone with a pause symbol, never "Not working".
- Paused components are not in the "Detection is impaired" banner.

**Compatibility**
- Older apps don't read `paused`, so a paused capability appears in none of `available`, `impaired` or `starting`, and is never shown as working. The headline still says stopped.
- An older backend sends no `paused`, and the new app decodes that as nil.

Tests: `tests/test_m82_paused_capabilities.py` (3 of 4 fail on `15bf35b`) and `ios/Tests/PerceptionStatusTests.swift` ("a stopped device is paused").
