# Event pipeline (M8.3)

How one detection travels from the vision module to the app. For each hop: the
code that does it, what identifies the event there, and what can go wrong. Line
numbers are as of the M8.3 commits; search for the named function if they drift.

Detecting an event and warning someone about it are different facts. The app
says "Alert sounded on the device" only when the device reported that a real
alert output played. A stored event is evidence of a detection and nothing more.

## 1. Lifecycle, hop by hop

| # | Hop | Code | Identity of the event here |
|---|-----|------|----------------------------|
| 1 | Inference signal: a module tracker emits one `Event` per episode of a condition | `modules/*` (e.g. `modules/driver_attention.py`), `core/events.py:54` `Event.now` | `Event.event_id`, 32 hex chars, minted when the Event is built (`core/events.py:45`). It is excluded from equality and repr. |
| 2 | Alert policy (M0.1 episodes) | `services/runtime.py:2696` `_dispatch_module_events` → `core/alerts.py:803` `AlertManager.notify`. Repeats and episode ends come from `observe` (`:877`). Custom rules go through `services/runtime.py:1322` `_emit_rule_alert`; "cannot monitor" goes through `:2321` `_signal_impaired` | Episode ref `<boot>-<n>` (`core/alerts.py:799`). `boot` is random per process, so episode 3 of one boot is never episode 3 of the next. The policy records what it did with each event as an `AlertDelivery`, looked up by event_id (`core/alerts.py:780`). |
| 3 | Alert decision | `notify`: **alerted** (a new episode, actuated now, `:866`), **merged** (joined an open episode, no actuation, `:837`), **suppressed** (a rolling-window notice inside its cooldown, `:862`) | `AlertDelivery.decision`, `.episode` |
| 4 | Device output | `core/alerts.py:951` `_actuate` → each sink's `play(pattern, event, category[, report=])`. The GPIO buzzer (`:416`) and the speaker (`:299`) render on one `_PatternPlayer` worker (`:182`). It calls `report("played" \| "failed" \| "skipped", error)` exactly once per actuation. `UnavailableSink` (`:246`) reports `skipped` with its reason. | Per output: played/failed/skipped/pending/unconfirmed counts and the last error. These are summarised by `AlertDelivery._output_locked` (`:627`). |
| 5 | Local record | `core/logger.py:192` `EventLogger.log` → `events.csv`. The dashboard flash comes from `core/dashboard.py:488`. | The CSV row (timestamp, type, …). event_id is not a CSV column yet (see §7). |
| 6 | Uplink queue + spool | `core/uplink.py:408` `push_event`, which takes the delivery from `delivery_lookup`. The runtime wires that up once, at `services/runtime.py:654`. Spool: `_spool_append` `:955`, `_spool_rewrite` `:980`, `_load_spool` `:910` (`uplink_spool.jsonl` beside `events.csv`, `services/runtime.py:650`). | `event_key` = `Event.event_id` (`core/uplink.py:470`). The internal keys `_boot` and `_mono` (the process and its monotonic time at queueing) stay in the spool and never go on the wire. |
| 7 | Delivery | `core/uplink.py:688` `sync_once`: batches of up to 100, with the in-flight batch tracked (`:712`). Wire format: `_wire_item` `:157` adds `age_s`. A late output report goes through `_alert_reported` `:535`. | `event_key`. The same key is used for every retry, spool replay and late alert report. |
| 8 | Backend ingest | `cloud/app.py:4617` `ingest`: `INSERT OR IGNORE` (`:4717`) under two partial unique indexes. `idx_events_dedup (device_id, event_key)` is at `:635`. `idx_events_episode (device_id, episode)` is new in M8.3, at `:642`. A re-sent key with a newer alert report updates only the report (`:4749`). Notifications go only to newly inserted rows (`:4962`). | Server row `id` (it only grows), plus `event_key` and `episode`. Times are `timestamp` (the device's claim, verbatim), `received_at` (the backend clock), and `occurred_at` + `time_basis` (`_event_time` `:4389`). `alert` is validated by `_alert_report` `:4345`; `simulated` comes from `_event_simulated` `:4421`. |
| 9 | Events API | `cloud/app.py:2482` `GET /v1/devices/{id}/events?limit=&before_id=`, newest stored first. `before_id` is a stable page cursor. Each row is built by `_event_row` `:4446`. | `id`, `event_key` |
| 10 | App timeline | `ios/Sources/BackendAPI.swift:670` → `EventItem` (`ios/Sources/EventModel.swift:36`) → `EventHistoryView` merges each fetch through `EventFeed.merge` (`EventHistoryView.swift:101`, `EventModel.swift:275`). Display is `ActivityTimeline` / `TimelineRow` (`ActivityViews.swift:216`, `:34`). | `EventItem.id` = `id:<row id>`, else `key:<event_key>`. Stable when a newer alert report arrives. |
| 11 | Detail / explain | `EventDetailView.swift:108` `deviceAlertCard` (delivery evidence, clock note, simulated label). `GET …/events/{id}/explain` (`cloud/app.py:4238`) uses `occurred_at` when the device clock was not the evidence. | row `id` |
| 12 | Notifications | Backend push: `ingest` → `gate.evaluate` → `_notify`, only for rows this request inserted. "(simulated)" is added to the title for simulated events (`cloud/app.py:4977`). App local notifications: `Notifications.swift:109` `announce` with per-device id marks (`NotificationPolicy.swift:90`), labelled simulated from the event too (`:175`). | row `id` |

## 2. Identity and idempotency

* **One event, one key.** `Event.event_id` → `event_key` → `UNIQUE(device_id,
  event_key)`. Every resend is the same key: a retry after a lost response, a
  spool replay after a power cut, a replayed request, a late output report.
  `INSERT OR IGNORE` stores it once, and only a stored row can notify.
* **One episode, one stored alert, at most one notification.** An event that the
  policy merged into an open episode is not queued if that episode's event was
  already queued (`core/uplink.py:456`, counted in `merged_events`). If it was
  not queued, the merged event becomes the episode's record. That happens for an
  episode the runtime re-opened after a recovery, because
  `_reopen_live_episodes` never uploads its synthetic event. The record carries
  what the episode actually actuated. The backend backs this up with
  `UNIQUE(device_id, episode)`: a second key for a stored episode is not stored
  and not notified.
* **A new episode is a new alert.** Episode refs carry the boot id. After a
  restart, or a module rebuild (a mode re-delivered from the app), a condition
  that is still live is a new episode. The device really alerted again, so the
  event is stored and may notify (subject to the 20 s cooldown). This is
  deliberate M0.1 semantics, not a duplicate.

## 3. Time: what the device says vs. what is true

| Field | Clock | Trust |
|-------|-------|-------|
| `timestamp` | the device's wall clock at detection, kept verbatim | A Pi has no RTC: it can be 1970 or the last shutdown time until NTP syncs |
| `age_s` (wire only) | the device's **monotonic** clock: seconds from queueing to sending, sent only for events queued in this boot | Exact, whatever the wall clock says |
| `received_at` | backend | Authoritative arrival time |
| `occurred_at`, `time_basis` | backend | `device_monotonic` = `received_at − age_s`. `device_clock` = `timestamp`, used only when it is no later than arrival + 120 s and no older than 30 days. Otherwise `received`, meaning the arrival time, and the app says so. |

The app shows `occurred_at` and falls back to `timestamp` for history. It flags
a device clock that was off by 2 minutes or more, and a delivery that arrived
more than a minute late. It orders the timeline by when events happened, with
the row id breaking ties.

## 4. Alert delivery evidence

**Device.** `core/alerts.py:530` `AlertDelivery`, one per event handed to the
policy:

```
decision  alerted | merged | suppressed
episode   "<boot>-<n>" (None for suppressed)
output    sounded      a REAL in-cabin output reported that its whole pattern rendered
          simulated    only a simulated output (sim/) recorded it; nothing made a sound
          pending      commanded; no report yet (the GPIO pattern takes up to 3.5 s)
          unconfirmed  commanded to an output that cannot report completion
          failed       commanded; every in-cabin output failed or skipped it
          none         no in-cabin output was commanded (console only, merged, suppressed)
rev       rises with every change of `output`
outputs   [{name, kind, audible, played, failed, skipped, pending, unconfirmed, error?, simulated?}]
```

* "In-cabin" means a sink with `audible` or `visual` set. The console log is not
  an alert output.
* "Played" for the GPIO buzzer means every pulse of the pattern was driven with
  no GPIO error. The pin was driven. Whether the buzzer is wired and audible is
  invisible to software (there is no feedback line). See §8.
* Repeats count: a failed onset followed by a repeat that played is `sounded`.
* The summary goes up with the event. If an output reports later, the event is
  re-sent under the same `event_key` with the higher `rev`. If the event is still
  queued, the queued item is updated in place. The backend keeps the report with
  the highest `rev`, whatever order the reports arrive in, and never notifies for
  an update.

**Backend.** `_alert_report` (`cloud/app.py:4345`) accepts known keys and values
only and bounds them (8 outputs, 160-character error). An unknown
`decision`/`output` means no evidence. The backend never derives `sounded`.

**App.** `AlertDeliveryEvidence` (`EventModel.swift:184`) is the only place the
wording is decided:

* `sounded` → **"Alert sounded on the device"**, with "It cannot measure whether
  anyone heard it."
* `simulated`, or `sounded` on a simulated event → "Simulated alert (no sound)".
* everything else, including no report at all → **"Detected; no alert output
  confirmed"**, with the reason (awaiting the device / the output's error / the
  output cannot report / no output).

**Compatibility.** Every field is additive and optional at every hop:

* Older firmware: no `alert`, `age_s` or `simulated`. The rows store NULL. The
  app says "Detected; no alert output confirmed" and uses the device timestamp,
  judged for plausibility.
* Older backend: drops the unknown keys, exactly as before.
* Older app: ignores the new JSON keys.
* Sinks without `reports_completion` keep the old `play(pattern, event,
  category)` signature and report `unconfirmed`.
* The ingest response, the auth and the request signing are unchanged. The
  signed body simply contains the new keys.

## 5. Real vs. synthetic

* **Virtual device (sim/).** `SimUplink.SIMULATED = True` (`sim/uplink.py:89`)
  marks every event `simulated: true`. This is a class attribute, not config, so
  a real device cannot label its alerts as simulated. The sim's buzzer reports
  `simulated` (`sim/runtime.py:45`), never `sounded`.
* **Backend.** `simulated` is also set when the report carrying the event says
  `simulated` (health or monitoring), or when the event's model ran on a
  `simulated` detector backend.
* **App demo (MockAPI).** Every demo event has `simulated: true`.
* **Labels.** The row shows "Simulated" (and VoiceOver reads it). The detail
  shows "Simulated event, not from a real camera". Local and push notifications
  get "(simulated)".

## 6. Robustness

| Case | Mechanism | Test |
|------|-----------|------|
| Duplicates (retry after a lost response) | same `event_key`; `INSERT OR IGNORE`; notify only inserted rows | `test_m83_events_backend.py::test_a_replayed_batch_creates_nothing`, `tests/test_m8_lifecycle_uplink.py` |
| Re-reported open episode (after a reconnect / tracker rebuild) | device: merged events not re-queued; backend: `UNIQUE(device_id, episode)` | `test_m83_events_sim.py::test_an_open_episode_re_reported_is_one_stored_alert`, `test_m83_events_backend.py::test_a_second_event_for_one_episode_is_not_a_second_alert`, `test_m83_events_delivery.py::test_an_event_merged_into_a_queued_episode_is_not_queued_again` |
| Late output report | same key, higher `rev`, report-only UPDATE | `test_a_late_output_report_updates_the_stored_alert_without_a_second_row_or_notification` (sim), `test_a_late_alert_report_updates_the_row_never_a_second_row_or_notification` (backend) |
| Missing events: torn spool append | a failed append marks the spool dirty; the next write rewrites it whole | `test_a_torn_append_does_not_cost_the_next_event` |
| Missing events: rewrite during an in-flight POST | `_spool_rewrite` writes the in-flight batch + the queue | `test_a_rewrite_while_a_batch_is_in_flight_keeps_it_on_disk` |
| Out-of-order arrival / restore order | the restore keeps the first position with the latest content; the app orders by `occurred_at` | `test_a_restore_keeps_first_position_and_latest_content`, `EventModelTests` "ordered by when it happened" |
| Replayed telemetry | request nonces (sd-auth-v1, `DEVICE_AUTH_CONTRACT.md` §5); event level: `event_key` and episode uniqueness | `test_a_replayed_batch_creates_nothing` |
| Network interruption | bounded queue + fsynced spool; a backlog delivered once, with its `age_s` | `test_an_alert_raised_offline_arrives_once_with_its_output_evidence_and_its_true_time` |
| Device restart / power cut | the spool is restored, with no `age_s` (another boot) → `device_clock` | `test_a_power_cut_mid_outage_delivers_the_alert_once_after_power_on` |
| Backend restart | SQLite commit before the answer; a resend is deduped; the cooldown is in memory | `test_a_backend_restart_between_storing_and_answering_is_harmless` |
| Device clock wrong | `age_s` → `occurred_at` on the backend clock; `timestamp` kept verbatim | `test_a_device_wall_clock_three_days_wrong_does_not_move_the_alert` (sim), `test_a_wrong_device_clock_never_moves_the_event` |
| Delayed delivery | `received_at − occurred_at` is shown when over a minute | backend + `EventModelTests` |
| Client refresh races | `LatestRequestGuard` + `EventFeed.merge`: one copy per event, never older than what is shown, ordered by occurrence | `EventModelTests` "the timeline merges refreshes…" |
| Paging while events arrive | `before_id` cursor | `test_pages_by_id_never_overlap_or_skip_while_events_arrive` |
| Camera reconnect mid-episode | the episode ends at FAILED and is re-opened (and re-sounded) locally without a new upload | `test_a_camera_reconnect_mid_episode_resounds_locally_but_stores_one_alert` |

## 7. Privacy

* The uplink carries event metadata only: key, type, times, confidence,
  free-text signals, duration, severity, source, model identity, and the alert
  report (output names, counts, a short error). No image field exists. This is
  pinned by `tests/test_m6_security_preflight.py::test_the_uplink_payload_has_no_image_field`,
  which M8.3 extended only with `age_s`, `alert` and `simulated`.
* Frames stay on the device:
  * Evidence snapshots are opt-in (`alerts.save_evidence: false`) and written to
    local disk only (`services/runtime.py:3024`).
  * Training capture is off by default.
  * The repo config is pinned by
    `test_the_repo_config_keeps_frame_capture_and_frame_upload_off`.
* The backend drops anything outside the alert report's schema (frames,
  extra keys). Ingest logs of skipped events carry only the key and type, as
  before.
* Not yet carried: `event_id` in `events.csv` (`core/logger.py`, a CSV schema
  change). Local rows correlate with stored events by timestamp and type.

## 8. Hardware-only (not provable in software)

* That the buzzer is **audible**: wiring, polarity, a dead buzzer. Software sees
  only that the GPIO writes succeeded, so `sounded` means the pin was driven.
* That the GPIO buzzer reports `played` / `failed` on the Pi as it does in tests:
  `gpiozero` on real hardware, a write to an unexported pin, or a pin claimed by
  another process.
* The pattern timing as heard.

These belong on the Monday runbook:

1. Trigger a drowsiness alert on the Pi. Check that the buzzer is heard **and**
   that the app shows "Alert sounded on the device".
2. Claim the buzzer pin from another process, or remove the `gpio_buzzer`
   output so it is `UnavailableSink`. Check that the app shows "Detected; no
   alert output confirmed" with the reason.

An unplugged buzzer still reads "sounded", because there is no feedback line.
That is the stated limit, not a bug.
