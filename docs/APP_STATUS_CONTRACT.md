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
| `rules` | object | Custom rules as the DEVICE runs them. |
| `rules.active` | [string] \| null | Rule ids the device reports running now. Null when stale. Never computed from what the backend accepted. |
| `rules.additive` | bool \| null | Whether this role runs custom rules beside its mode instead of replacing the mode. Null when no role is known. |
| `rules.rejected` | {`version`, `reasons`, `at`} \| null | Set when the device refused the rule set the backend currently wants to run (its rejection names that set's `rules_version`). The device keeps its previous rules (all or nothing). A refusal of an older set does not count. |
| `last_rejection` | {`kind`, `subject`, `reasons`, `at`} \| null | The device's most recent refusal of any kind (`preset`, `rules`, `protection`). The device never clears it, so check `at`. |
| `preset_rejection` | {`preset`, `code`, `reason`, `at`} \| null | A stored mode request the backend dropped because the device's role forbids it (`code` is `role_forbids_preset`). The next accepted mode request clears it. |
| `stale` | bool | True when the device has sent nothing for 30 s (`_HEALTH_STALE_S`, the same window as `online` and `health`), or has never reported. |
| `reported_at` | number \| null | Unix time of the device's last report (`last_seen`). |

Liveness rule: when `stale` is true, `monitoring_state` and `protection` are
`unknown`, and `monitoring_active`, `capabilities` and `rules.active` are null.
An old `ready` or `protecting` is never repeated.

## Refusal codes

Refusals keep `detail` as a human sentence, which every client shows. A
machine-readable `code` sits next to it:
`{"detail": "...", "code": "role_forbids_preset"}`. The app branches only on
`code`, never on the status alone or on the wording.

| Code | Status | Endpoint | Meaning |
|---|---|---|---|
| `role_forbids_preset` | 409 | `POST /v1/devices/{id}/preset` | The device's role does not allow this mode. Checked before `draft_superseded`. |
| `draft_superseded` | 409 | `POST /v1/devices/{id}/preset` | The proposal being activated was replaced by a newer preview. Review it again. |
| `unknown_preset` | 400 | `POST /v1/devices/{id}/preset` | No such mode or variant. |
| `firmware_unsupported` | 409 | `POST /v1/devices/{id}/protection` | The device's software cannot stop or resume. |
| `confirmation_required` | 400 | `POST /v1/devices/{id}/protection` | A stop request must carry `confirm: true`. |
| `device_not_found` | 404 | any `/v1/devices/{id}/...` | No such device on this account. |
| `device_already_claimed` | 409 | `POST /v1/devices/claim` | Another account owns the device. |
| `account_exists` | 409 | `POST /v1/signup` | The email is already registered. |

Other errors (401, 422, 429, 503) carry no `code`.

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
