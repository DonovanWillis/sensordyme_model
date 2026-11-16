# Alert escalation across relapses: decision needed (M10.1)

**Status: implemented, OFF.** `alerts.relapse_escalation.enabled: false` in
`config.yaml`. Nothing about how the device sounds has changed. Turning it on is
the owner's decision, because it changes how a mandatory driver alert sounds.

## The defect (reproduced)

Alert episodes (`core/alerts.py`) work like this:
- An episode ends when its condition clears.
- A new episode starts at the onset pattern.
- A "warning" episode (distraction, eyes_down, eyes_up, phone_use) plays
  `double`, repeats every 5 s, and switches to `urgent` only after 10 s
  **within that one episode**.

So a driver who keeps relapsing in bursts shorter than 10 s never hears
`urgent`, however long it goes on.

`tests/test_m101_escalation.py` reproduces it on the production module and
the shipped alert config:
- **Scenario:** six 6 s look-aways at 45°, 2 s back on the road between them.
- **Result:** six separate episodes, each alerting for about 3.4 s (about 20 s
  alerted in 48 s).
- **Patterns heard:** `double, double, double, double, double, double`.

`test_relapsing_distraction_reaches_urgent_under_the_shipped_policy` is a
**strict xfail**: it fails today, by design. If the policy is enabled, it
passes unexpectedly and fails the suite, forcing this file and that test to be
updated with the decision.

## Five things kept separate

| Concept | Today | With the proposal |
|---|---|---|
| **Alert episode** | one per condition onset; ends when the condition clears | unchanged |
| **Cooldown** (`cooldown_s` 15) | de-duplicates rolling-window notices (fatigue_warning) only; never drops a new episode | unchanged |
| **Repeated distraction** | each relapse is a new episode starting at `double` | episodes stay separate, and a **run** of them is tracked |
| **Escalation window** | `escalate_after_s` 10 s inside one episode | the same 10 s, counted over the run's **alerted** time |
| **Genuine recovery** | any clearing (≥ 0.4 s back on the road) restarts escalation | `recovery_s` (proposed 60 s) with none of the run's types alerting |

## The proposed policy (`core/alerts.RelapseEscalation`)

- **Run:** consecutive episodes of `types` (proposed: distraction, eyes_down,
  eyes_up, phone_use). Looking down at a phone and turning away count as the
  same inattention.
- **Run time:** the union of those episodes' alerted time. Concurrent episodes
  count once.
- **Escalation:** a new episode in a run whose time has already reached the
  category's `escalate_after_s` (10 s, unchanged) plays `urgent` from its onset.
  An episode that crosses the bar while alerting escalates at its next repeat,
  exactly as a continuous episode does.
- **Recovery:** the run ends after `recovery_s` with none of `types` open.

**Bounded by construction.** It changes which pattern sounds, never whether or
when anything fires:
- the same events and the same episodes;
- the same actuation times and repeat interval;
- the same repeat cap and the same cooldowns.

`test_enabled_policy_changes_patterns_only_never_timing_or_count` pins this.
Critical (drowsiness, attention_lost) and notice categories are untouched, as
are types not listed. Configuration is validated (`recovery_s` must be 5–600 s),
and only the boolean `true` enables it.

**With it enabled:** the same six relapses play `double, double, double, urgent,
urgent, urgent`. After the third episode the run has 10.2 s alerted.

## Decision requested

1. **Enable relapse escalation?** Recommended: **not yet**. Escalation
   amplifies whatever the alert does, including false alerts. Today's false
   distraction rate per hour on real cabins is unmeasured, and a driver whose
   mirror checks are wrongly alerted would hear `urgent` sooner. Measure false
   alerts per hour first (`scripts/hw_validation/cabin_session.py`, preset
   `core`), then decide.
2. **`recovery_s`:** 60 s proposed. It matches the existing
   `escalation_window_s: 60` ("a repeat within this window → continuous").
   30 s escalates less often, 120 s more often.
3. **Types:** distraction, eyes_down, eyes_up, phone_use. `frequent_glancing`
   is left out because it is itself a count over a window.

To enable after approval:

```yaml
alerts:
  relapse_escalation:
    enabled: true
```

Then update `tests/test_m101_escalation.py`: drop the strict xfail and pin the
shipped behaviour.

Evidence level: synthetic scenarios only (L1). No real cabin, driver or
annoyance data exists.
