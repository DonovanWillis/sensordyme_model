# Preset library

What the app can offer, what each mode needs, how far its models have been
validated, which modes a given device can run now, and the custom rules it
runs beside or instead of them.

Code: `presets.yaml`, `core/presets.py`, `cloud/app.py` (`/v1/presets`,
`/v1/devices/{id}/presets`, `/v1/devices/{id}/preset`, `/v1/devices/{id}/rules`).
Tests: `tests/test_m6_preset_catalog.py`, `tests/test_m6_policy_api.py`,
`tests/test_m6_preset_settings_device.py`. Prompt-to-policy: docs/PROMPT_POLICIES.md.

## What can run together

A device runs **one vision module at a time**. A preset selects that module.
Beside it there can be:

- the role's mandatory floor (`core/roles.py`): for `driver_safety`, the
  driver module, its loud alert types and its threshold caps, always;
- N custom rules. On a role with `additive_custom_rules` (today only
  `driver_safety`) they run **beside** the preset (`standard_plus_custom`).
  On every other role they **replace** the preset (`ai_custom`): the device
  runs a general detector and only the rules.

Presets that need different modules are mutually exclusive. Every preset is
in the `vision_module` exclusive group. Asking for something that cannot run
together is refused with a code (`incompatible_policy`), never resolved by
silently dropping one side.

## Product modes

| Product mode | Preset | Availability | Notes |
|---|---|---|---|
| Driver Safety | `driver_safety` | available | `driver_attention` module. |
| Child Monitoring | `baby_monitoring` | available | Infant and young-child focus. Adult face and pose models are not reliable on infants, and the preset says so. |
| Property Monitoring | `security_monitoring` | available | `occupancy` module: areas, lingering, vehicles, animals, crowd. |
| Occupancy | `occupancy` | **planned** | The occupancy module runs today inside Security Monitoring (`served_by`). A standalone mode needs its own alert set and an entry in the app. |
| Workplace Safety | `workplace_safety` | **planned** | No PPE, hard-hat or hi-vis detector exists. `served_by` lists the parts that do exist: falls (Elderly) and areas (Security). |
| (none) | `elderly_monitoring` | available | Kept for existing installations. |

Planned entries live under `planned:` in presets.yaml. `load_presets()` never
returns them, so no command path can select one.

## `GET /v1/presets` (auth)

```json
{"catalog_version": "2026-11-11.m6", "presets": [ ... ]}
```

Each preset:

| Field | Type | Meaning |
|---|---|---|
| `id` | string | The preset name (`driver_safety`). Planned entries use their planned id. |
| `version` | int | Bumped whenever what the preset watches for changes. |
| `title`, `description` | string | Display text. |
| `product_mode` | string \| null | Which of the five product modes it serves. |
| `module` | string \| null | The vision module it runs. Null for a planned mode with no module. |
| `availability` | `available` \| `planned` | Catalogue-level. Per device: `GET /v1/devices/{id}/presets`. |
| `planned_reason` | string \| null | Why a planned mode is not available. |
| `served_by` | [string] | For planned modes: the available presets that cover part of it. |
| `variants` | [{`id`, `title`}] | Named tuning profiles, selectable as `preset:variant`. |
| `required_capabilities` | [string] | `core/monitoring_state.py` kinds the preset cannot run without (`camera`, `inference`, `face`, `objects`, `hands`, `pose`, `alert_output`, `audible_alert`). |
| `optional_capabilities` | [string] | Kinds whose loss degrades the preset instead of stopping it. Required and optional together, minus the alert outputs, equal what `core/model_registry.yaml` says the module consumes. A test enforces this. |
| `hardware` | {`camera`, `accelerator`, `audible_output`} | What the preset needs physically. |
| `alert_types` | [string] | Event types that alert in this mode. |
| `temporal_policies` | [{`event`, `description`, `sustain_s`?, `window_s`?, `setting`?}] | Each alert, and how long its condition must hold. `setting` names the setting that tunes it; `sustain_s` is that setting's default. |
| `alert_patterns` | [{`tone`, `events`, `description`}] | `tone` is `critical` \| `loud` \| `soft` (config.yaml `alerts:`). |
| `settings` | [Setting] | Owner-tunable values. See below. |
| `compatibility` | {`exclusive_group`, `module`, `custom_rules`} | `exclusive_group` is always `vision_module`. `custom_rules` is `additive_on_driver_safety_role` or `replace`. |
| `validation` | {`status`, `validated`, `models`, `evidence`, `references`} | See below. |
| `limitations` | [string] | Honest caveats. |

### Setting

| Field | Type | Meaning |
|---|---|---|
| `key` | string | **An existing module config key** (config.yaml `modules.<module>.<key>`). The default is that key's shipped value; a test enforces this. |
| `title` | string | Display label. |
| `type` | `number` \| `bool` \| `enum` | |
| `default` | number \| bool \| string | |
| `min`, `max`, `step` | number \| null | Numbers only. A value must lie in [min, max] and on the step grid from `min`. Values are refused, never clamped. |
| `unit` | string \| null | |
| `options` | [string] \| null | Enum only. |
| `description` | string \| null | |

Role floors win over settings. A setting the schema allows (for example
`drowsy_seconds` up to 3.0) is still refused on a `driver_safety`-role device
when it passes the floor (2.0).

### `validation`

**Derived, never declared.** `status` is the weakest
`core/model_registry.py` evaluation status along the preset's capability chain
(`model_registry.preset_view(...).weakest`):
`unevaluated` < `benchmarked_offline` < `evaluated_in_domain` < `validated_on_device`.
A planned mode with no module has `not_applicable`. presets.yaml cannot carry
a status: `catalog_problems()` flags one, and a test enforces it.

- `validated`: true only when every required model-backed link is at
  `validated_on_device` with event-level evidence. Nothing is validated today.
- `models`: the registry model ids along the chain.
- `evidence`: [{`model`, `kind`, `report_id`, `date`, `granularity`, `supports`, `shipped_config`}]
  from the registry cards. `supports` is null for evidence measured on a
  configuration the device does not ship.
- `references`: the documents behind the preset's limitations.

Today: `security_monitoring` is `benchmarked_offline` (the detector's offline
benchmark). `driver_safety`, `baby_monitoring` and `elderly_monitoring` are
`unevaluated`, because Face Mesh, Hands and Pose have no evidence.

## `GET /v1/devices/{id}/presets` (auth)

```json
{
  "device_id": 1, "catalog_version": "...", "active_preset": "driver_safety",
  "active_variant": "strict", "stale": false,
  "presets": [{"id": "driver_safety", "available": true, "code": null, "reason": null,
               "active": true, "validation_status": "unevaluated",
               "settings": {"drowsy_seconds": 1.5, "...": "..."}}]
}
```

| Field | Meaning |
|---|---|
| `active_preset` | Base id of the preset the device reports running. Null if the device has never reported, or runs custom rules **instead of** a preset (`ai_custom`). |
| `active_variant` | The active preset's variant, if any. *(Addition to the M6 brief.)* |
| `stale` | The status block's `stale`: no report in 30 s, or never. |
| `presets[].available` | True when `code` is null. |
| `presets[].code` | null, or why the preset cannot be selected (precedence order): `planned` (not runnable anywhere) > `role_forbids_preset` (the reported role forbids it; for a role the backend does not know, the device's own `allowed_presets`) > `device_stale` (what the device can run now is unknown) > `capability_unsupported` (a required capability is reported impaired because the device does not have it, e.g. "no audible alert output is configured") > `capability_impaired` (a required capability is reported impaired). |
| `presets[].reason` | Human sentence for `code`. |
| `presets[].active` | This preset is `active_preset`. |
| `presets[].validation_status` | `validation.status` from the catalogue. |
| `presets[].settings` | `{key: value}` that runs: default, then the active variant's override, then the owner's setting (active preset only). |

A capability the device does not report at all is not held against a preset.
A device reports only what its current module uses, so a driver device's
silence about `pose` says nothing about whether it could run pose.

## `POST /v1/devices/{id}/preset` (auth)

Unchanged for existing callers. New optional fields:

```json
{"preset": "security_monitoring:away", "settings": {"linger_s": 60}, "keep_rules": null}
```

- `settings`: validated against the mode's schema, then stored as the name
  the device applies, `mode:variant@key=value,...` (keys sorted). It rides the
  existing `desired_preset` -> `set_preset` -> `active_preset` channel, so
  convergence, restart persistence and the role check all work on it.
  `desired_preset` in the response is that name. Settings compare by value
  (`1.50` = `1.5`). `settings` replaces any suffix already in `preset`.
- `keep_rules`: custom rules beside the new mode. `true` on a device whose
  rules replace the mode is refused (`incompatible_policy`). Left null, rules
  are kept exactly when the device can keep them: an additive role, the
  mandatory module, and rules active. The switch then re-delivers the rules
  (new `rules_version`, returned with `rules_active`), because a device that
  switches mode exits custom monitoring before it re-applies them. `false`
  drops them, as every mode switch did before M6.

Refusals, in check order: `unknown_preset` (400; a planned mode too) ->
`role_forbids_preset` (409, the mode) -> `invalid_setting` (400, names the
key) -> `firmware_unsupported` (409, settings to a device without the
`preset_settings` feature) -> `role_forbids_preset` (409, a floor broken by a
setting) -> `incompatible_policy` (409) -> `draft_superseded` (409). Before
all of them, `settings` with `preset: ""` is `incompatible_policy`.

**Device side.** The device reports `features` in its monitoring report
(`core/presets.py DEVICE_POLICY_FEATURES`: `preset_settings`, `dwell_in_zone`).
`services/runtime.py` applies variant overrides plus settings through
`core/presets.preset_overrides` and refuses an invalid setting with a
recorded `last_rejection` (`kind: "preset"`). `main.py` restores the same
name after a restart. Older firmware would read `@...` as an unknown variant
and never converge, which is why settings go only to firmware that reports
the feature.

## `GET /v1/devices/{id}/rules` (auth)

```json
{"device_id": 1, "rules_version": "53ab44a3b510bc72", "additive": false,
 "custom_monitoring_wanted": true,
 "rules": [{"id": "r1a2b3c4d5e", "enabled": true, "summary": "Someone stays in the front door for 10 seconds",
            "targets": ["person"], "zone": "front door", "dwell_s": 10.0,
            "required_capabilities": ["camera", "inference", "objects"],
            "status": "running", "reason": null}]}
```

| Field | Meaning |
|---|---|
| `rules_version` | The version the device will report once it runs this set. Empty when custom rules are not wanted. |
| `additive` | The role runs rules beside its mode. False when no role is known. |
| `custom_monitoring_wanted` | Whether custom rules are meant to run now. *(Addition to the M6 brief.)* |
| `rules[].id` | Stable id. Assigned at preview (`/v1/ai/configure`) and kept by Activate. Rows from before ids existed get a deterministic id from content and position, written down on the first toggle. |
| `rules[].enabled` | The owner's on/off switch. |
| `rules[].summary`, `targets`, `zone`, `dwell_s` | From the stored rule. `dwell_s` appears only on dwell rules. |
| `rules[].required_capabilities` | `camera`, `inference`, and the perception kinds the targets come from. |
| `rules[].status` | `disabled`: switched off, or custom rules not wanted (a mode was chosen instead). `waiting`: its area is not drawn, so it is inert and never fires anywhere else. `unknown`: device stale, delivery not yet confirmed (the device's reported `version` differs), or the device does not report it. `rejected`: the device refused the current set (all or nothing). `running`: the device reports it in `custom_rules`. Precedence follows that order. |
| `rules[].reason` | Why, when not `running`. |

The device runs only the enabled rules, in stored order. The device-side id
`rule<i>:<predicate>:<target>` uses `i`, the position among enabled rules.
That index is how `running` is matched.

## `PATCH /v1/devices/{id}/rules/{rule_id}` (auth)

`{"enabled": false}` -> `{"ok": true, "rule": {...as above}, "rules_version": "...", "rules_active": 1}`.

The stored set changes, and the activation counter is bumped, so the version
is new every time. While custom rules are wanted, ingest delivers the enabled
rules on the next poll through the existing `set_rules` path. A disabled rule
is never delivered. Codes: `rule_not_found` (404), `device_not_found` (404),
`draft_superseded` (409, the set changed concurrently).

## Known limits

- Settings exist only for keys that already exist in config.yaml. No new
  thresholds were invented.
- No dedicated Occupancy or Workplace Safety mode exists. They are listed as
  planned with the reason.
- `capability_unsupported` is inferred from the device's own impairment
  wording ("not configured" or "not provided"). The device reports no
  separate "absent" state.
- Validation status is preset-level. A device running an unregistered
  detector binary (a different sha256) is not reflected here; the registry's
  `capability_evidence` covers that case.
