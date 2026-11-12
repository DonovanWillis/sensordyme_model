# iOS app status (ORYN EDGE)

What every visible action in the iPhone app (`ios/`) does, as of the M6
audit (branch `m6/ios-app`, 2026-11-11), with the Mission 8 changes
(2026-11-12) in the section below. The rule this audit enforces: an action either
performs the real operation, shows an understandable error, or says the
capability is not supported yet. Silent success is allowed only in demo mode
(`MockAPI`, entered with "Try the demo" or `-orynDemo`), which every screen
labels as simulated.

Legend: **real**: calls the backend or device and shows the result.
**error**: failure is shown in words, with the backend's `detail` when it sent one.
**not-supported**: an older server lacks the endpoint (an uncoded 404/405); the
screen says "Not supported by this server yet". **demo-only**: exists only in
demo mode and is labelled simulated.

Line numbers refer to `ios/Sources/` at the commit that added this file.

## Mission 8 (Checkpoint F): what changed, and how it is checked

### Presence comes from fresh device telemetry (ConnectionTruth.swift)

ONLINE used to be the backend's `online` flag from whatever reply the screen
last kept. It is now decided by `ConnectionTruth.presence` from three things:
the device's own `last_seen` / `status.stale`, how old the reply carrying them
is on the phone's clock (`Device.receivedAt`, stamped by `APIClient.devices()`),
and whether the latest refresh failed.

| Presence | When | Shown as |
|---|---|---|
| Online | reply ≤ 30 s old, `online`, not `status.stale`, `last_seen` within 30 s + 30 s clock tolerance | green ONLINE |
| Stale | the reply says online but the telemetry is old (`status.stale`, or `last_seen` > 60 s ago: a cached or replayed reply) | amber STALE; card "No report for 2 minutes — showing its last report"; device screen banner "… may no longer be true" |
| Offline | the backend says it has not reported | OFFLINE; "Last seen 5 minutes ago" |
| Unknown | the refresh failed, or the reply is > 30 s old (polling paused, request hanging) | UNKNOWN; "Not updated for 45 seconds" |

Every screen reads the same function: the device card and the "N online"
summary (which used to count raw `online` flags even after a failed refresh),
the device screen's presence pill, its monitoring pill (never "All clear" on
telemetry that is not live), health and capability cards (hidden when not
live), mode rows, rules, nodes, protection, the preset details and Ask Jupiter
sheets, and the Connection doctor. A device the server no longer lists for the
account says so instead of polling its last reading for ever.

### Replies the app cannot use are said in words (ServerReply.swift)

| Reply | Headline | Message |
|---|---|---|
| 4xx/5xx with FastAPI `detail` (string) | "The server refused the request" / "The server reported a problem" | the detail, trimmed to 280 characters; never if it is itself markup or JSON |
| 422 validation list | "The server refused the request" | "The server couldn't accept that — name: field required." |
| `{"detail": {...}}`, `{"error": ...}`, `{"message": ...}` | as above | that sentence |
| HTML / plain text / empty error body | by status | 502/503/504 → gateway couldn't get an answer, try again in a moment; 500 → internal error; 404 uncoded → "Not supported by this server"; 403, 408, 413, 429 explained |
| 200 with a web page | "Something else answered at this address" | Wi-Fi sign-in page or another app; check Settings → Server |
| 200 empty | "The server's reply couldn't be read" | empty reply |
| 200 broken JSON | "The server's reply couldn't be read" | unchanged (ReplyError.unreadable) |

`getJSON`/`postJSON` (Ask Jupiter, explanations) used to let a raw `URLError`
through ("Could not connect to the server."); they now use the same
transport wording as every other request. Banners on the device list and
device screen take their headline from the error, so a reply the server did
send is never headed "Can't reach the server". Coded refusals from the device
(`last_rejection.code`: `role_forbids_preset`, `unknown_node`,
`invalid_setting`, `profile_fixed`, `firmware_unsupported`) and per-device
preset reasons written for developers ("preset 'x' would stop
driver_attention; allowed: …") are put into sentences
(RefusalText, PresetAvailability.readableReason).

A 401 while the app believes it is signed in ends the session with a reason,
even when no token could be sent (it used to show "missing bearer token" on
every refresh, for ever). The session token is read from the Keychain once
and kept in memory, instead of being re-read for every request.

### Capabilities and nodes (M8 DRIVER / NODES)

`attention` (the driver's calibrated forward gaze) is titled "Attention" and,
while STARTING, explained: "Calibrating to the driver — look ahead normally;
usually done within 30 seconds" (amber hourglass, never red; it does not block
activation). The status reason "attention: starting" reads "Attention:
calibrating to the driver …". Calibrated, the row says "Calibrated to the
driver's forward gaze". A node's `experimental_features` are listed under it
as "Experimental: Flailing (off), Choking hazard (off)".

### Notifications

A tapped notification opens its own device; when that device is not on the
account (or an old notification names none on a multi-device account) the
app says "Couldn't open that alert" with the reason instead of opening the
first device, which then ignored it. An alert the device screen can't find is
its own banner, no longer headed "Can't reach the server" (which also turned
the presence pill to Unknown).

### Accessibility

- 44 pt targets: Add device / Bluetooth header buttons, the device menu, the
  preset ⓘ buttons, the Ask Jupiter send button, banner "Try again", the
  Event history timeline dots (the drawn sizes are unchanged).
- Large text: the presence and monitoring pills wrap instead of truncating to
  "ON…"/"AL…"; the device screen's status row, the device card's pill and id,
  and the Ask Jupiter caption/Examples stack at accessibility sizes; event
  signals and mode-row reasons wrap in full.
- Eyebrow labels (SDEyebrow, pills): `Font.caption.weight(.medium).monospaced()`
  and no tracking on SDEyebrow. Three Simulator audit runs isolated the
  findings: tracking caused "text clipped"; "Dynamic Type partially
  unsupported" was the caption2 text style itself (pinned at 11 pt from Extra
  Small to Large), not the font construction (`system(_:design:)`,
  `.monospaced()` and `fontDesign` all gave the same finding). The labels are
  1 pt larger at the default size as a result.
- Contrast: unavailable-mode reasons and the sign-in "Welcome back" use
  textSecondary. The audit waits 3 s for late content to settle, and contrast
  findings on text under the translucent tab bar or the navigation bar's
  scroll edge are re-measured with the element scrolled clear
  (AccessibilityAuditUITests).
- VoiceOver: presence is spoken with its meaning ("Stale data, no report for
  2 minutes"); the Ask Jupiter title offers the Large Content Viewer.

### Journeys added or changed (ios/UITests, scripts/sim/uitest_control.py)

- 10 (Ask Jupiter): reveals the field clear of the tab bar, then requires the
  keyboard and keyboard focus, and that focus survives the screen's refreshes,
  then types into the focused element. What was found: with temporary
  focus/lifetime logging, nine runs showed the field and the device screen
  are never re-created by the 4 s refresh and focus holds until submit; one
  run showed XCUITest briefly unable to resolve the field ~20 s after the tap
  while it was still focused on screen; and the full run caught the same
  error on the sign-in screen (journey 02) with `typeText` issued 0.46 s after
  the tap: XCUITest waits for the app to go idle, not for focus, and SwiftUI
  makes a tapped field first responder a beat later. The old journey typed
  straight after the tap through a re-resolved element. Every journey that
  types now taps, waits for keyboard focus (one more tap after 5 s), and types
  into the focused element.
- 24 (companion node): flips the switch by its knob; the nodes section is its
  own accessibility element so the switch keeps its identifier.
- The controller gives its own logins a separate rate-limit key (the backend
  allows 10 logins per address per 5 minutes; it signs in once per launch).
- 26 Stale telemetry: the proxy replays the last device list (`/mode?m=freeze`);
  card, count and device screen go Stale together, never "All clear", and
  recover together.
- 27 Unreadable replies: `/mode?m=html|empty|error500|gateway` on the list,
  and `error500` on `/v1/ai/configure` only (`&path=`), each with a readable
  headline and message.

## Mission 8.1: the UI-test suite is order-independent

The M8 full run passed 26 of 27 with journey 03 (discovery) SKIPPED. Cause:
XCTest runs test classes alphabetically, so `AccessibilityAuditUITests` ran
first, and its setUp called `Stack.prepare()`, which claimed the single
virtual device for the shared demo account; 03 found the device already
claimed and skipped. The journeys also leaned on each other the other way
("03 claims the device every later journey uses").

Every test now starts from its own world. `Stack.fresh` (from the setUp of
both test classes) calls the controller's `POST /fresh`, which stops the
device and backend, deletes their data and starts a new backend and a newly
provisioned device with a new account, then claims the device and waits for
it to report ready (or leaves it unclaimed, for 03). About 13 s per test,
2 s for 03.

| Concern | How it is handled |
|---|---|
| Device pairing | a new device per test; claimed by the controller, or unclaimed for 03, which pairs it through Add device. 03 asserts the unclaimed start (it no longer skips) |
| Account and auth state | a new account per test (`uitest-<n>@sensordyme.local`); every launch passes `-orynSignOut` and then `-orynSession` with a token from a real `/v1/login` to that test's backend. A new account also runs the app's own per-account reset (`AppState.claimLocalState`: paired camera, announced alerts, SMS number) |
| Login limiter | the backend's limiters live in its process, which is replaced per test; the controller's own logins still use their own rate-limit key (`SENSORDYME_CLIENT_IP_HEADER`), the app's are limited as in production |
| Event history | a new database per test; the controller's event count is no longer capped at 50 (it stopped growing, so a "more events than before" wait could never finish) |
| Rules, modes, nodes, zones, Ask Jupiter drafts | a new database and a new device state directory per test; 24 now asserts the companion starts off instead of switching off one left by another journey |
| Proxy faults | `/fresh` puts the proxy back to `pass` and listening, and clears the replay cache used by `freeze` |
| Simulator state | before the run the script uninstalls the app and its runner and resets the Simulator keychain; each test's tearDown terminates the app |
| Shared fixtures (Stack.swift) | stateless apart from the controller; only transport failures are retried, a controller error fails the test with its message; with the controller missing, a scripted run FAILS (`ORYN_UITEST_REQUIRE_STACK=1`) instead of skipping |
| Parallel testing | off: one controller on fixed ports, one Simulator and one app install cannot serve two tests. XcodeGen's scheme leaves it off, and the script forces it off (`-parallel-testing-enabled NO` and `ParallelizationEnabled` in the .xctestrun) |
| Order | any: `ORDER=random scripts/sim/ios_uitest.sh` writes `TestExecutionOrdering=random` into the .xctestrun; summary.md lists the tests in the order they ran |
| Retries | none: `-retry-tests-on-failure` and `-test-iterations` are never passed. summary.md counts skipped and never-reported tests, and the script exits non-zero unless every test passed |

Health freshness (M8.1): a new presence state, `monitoringStale(age)`
(ConnectionTruth). The device is still polling, but its health sample is
older than the backend's 30 s (`status.health_stale`). It is never online
(`isOnline` is false and it is not counted in "N online"). The card shows the
amber Stale pill and "Connected — monitoring status not updated for N". The
device screen shows "Status unknown", never "All clear", and withholds the
health and capability rows.

Journey 28 (frozen health snapshot): `POST /health?freeze=1` sends SIGUSR1 to
the virtual device (`sim/virtual_device.py`). Its uplink keeps polling with the
last health sample, which is the same fault as
tests/test_m81_health_freshness_sim.py. The journey waits past the threshold
and checks, on the card: the spoken label "Connected, monitoring status not
updated for N", the card line, no Online/All clear/Protecting, and "0 online".
On the device screen it checks: presence Stale, monitoring "Status unknown",
the banner "It is connected, but its monitoring status hasn't been updated
for N, so whether it is watching is unknown", the health card withheld, and
the capabilities "Unknown" notice. After `freeze=0` it checks that Online and
"All clear" come back. The time the recovery took goes into the result bundle
(`28_recovery_time`).

The audit's re-check of text found under the bars was itself order- and
timing-dependent. It scrolled with slow swipes that glided about 400 pt and
overshot. It measured pixels while the page could still be moving, parked
text at the band's edge inside the tab bar's blur, followed element handles
that a lazily built list re-binds to other rows, and read each later finding
at its pre-scroll position. It now drags the page by the exact distance that
centres the flagged text, with no glide. It measures the page's real
displacement from two accessibility snapshots, finds the text by label where
it must now be, and audits it on a still screen. A failing re-check saves the
screenshot it measured. Journeys tap an element only once its frame has
stayed put for a second: XCUITest resolves a tap 0.6 s before it synthesizes
it, and content arriving late above the target once made journey 08's tap
miss.

Runs (`scripts/sim/ios_uitest.sh`, iPhone 17 Pro Simulator, each run one
xcodebuild invocation with no retries; evidence in
`~/sensordyme-audit/2026-11-10/m81-evidence/ios/`):

| Run | Commit | Order | Result |
|---|---|---|---|
| pre-merge | 71e1745 | alphabetical (audits first) | 27/27 |
| A | 3e52700 (+ health freshness, journey 28) | alphabetical | 26/28: both audits failed in the re-check (fixed above) |
| B | f41bea2 | alphabetical | 27/28: journey 08's tap missed (fixed above) |
| C | f22564e | alphabetical | 28/28 |
| D | f22564e | alphabetical (random was asked for; the script missed the build's format-1 .xctestrun, fixed) | 27/28: journey 06's Resume tap changed nothing (no resume reached the device) |
| E | (same build as C and D) | random | <!-- M81-RUN-E --> |

<!-- M81-RESULTS -->

## Screen-by-screen audit

### Sign in (AuthView)

| Action | Behaviour | Where |
|---|---|---|
| Sign in / Create account | real (POST /v1/login, /v1/signup); error shown; 401 on sign-in reads as wrong credentials, not "session expired" | AuthView.swift:149, :235; API.swift:110 |
| Show/hide password, switch sign-in/up | local UI | AuthView.swift:111, :184 |
| Try the demo | demo-only (MockAPI) | AuthView.swift:206 |

### Devices tab (DevicesView)

| Action | Behaviour | Where |
|---|---|---|
| Device list, 10 s refresh, pull to refresh | real (GET /v1/devices); failure keeps the last list and shows a "Can't reach the server" banner with Try again; a 401 signs out with the reason | DevicesView.swift:240, :51 |
| Card subtitle | from `status`: "Stopped · Driver Safety", "Not monitoring · …", never "All clear" unless the device reports `ready` | DevicesView.swift:287; DeviceStatus.swift |
| Add device (claim code) | real (POST /v1/devices/claim); `device_already_claimed` and other errors shown | DevicesView.swift:167, :452; SetupWizardView.swift:216 |
| Connect over Bluetooth | real BLE Wi-Fi provisioning; needs a physical device | DevicesView.swift:164; SetupFlowView.swift |
| Linked-camera banner | real mDNS/LAN probe of the camera | DeviceLinkBanner.swift:91 |

### Device screen (DeviceDetailView)

| Action | Behaviour | Where |
|---|---|---|
| Live feed | real MJPEG from the device on the LAN, with token; errors in the feed card; the demo feed is labelled "Demo · simulated feed" | FeedViews.swift; MJPEGStream.swift |
| Status pills | from `status` (MonitoringStatusPresentation); unknown when stale or offline | DeviceDetailView.swift:338 area, DeviceStatus.swift |
| **Device isn't reporting** banner (new) | when the server answers but the device is offline or stale: says when it was last heard from, and that changes are saved and apply when it reconnects | DeviceDetailView.swift:100, :870 |
| Ask Jupiter bar / Examples | opens Ask Jupiter (below) | DeviceDetailView.swift:487 |
| **Monitoring mode rows** | real (POST /v1/devices/{id}/preset); switching → active only when the device confirms; times out to "Didn't switch"; offline queues; a coded refusal shows its detail | DeviceDetailView.swift:712, :1034 |
| Mode list source (changed) | the **server catalog** (GET /v1/presets); the hardcoded `Preset.catalog` is used only as a fallback and is labelled "Built-in list: …" | DeviceDetailView.swift:606, :823 |
| Per-device availability (new) | GET /v1/devices/{id}/presets: rows the device can't run are disabled with the reason (role, impaired or unsupported capability, planned); `device_stale` is said but not blocked; not-supported → falls back to `status.allowed_presets` | DeviceDetailView.swift:833, :844; PresetLibrary.swift |
| Validation label per row (new) | "NOT YET VALIDATED" (amber) unless `validated_on_device`; hidden for the built-in list, which knows nothing about validation | DeviceDetailView.swift:712 area |
| Variant chips | real (POST preset `base:variant`) | DeviceDetailView.swift:793 |
| **ⓘ preset details / settings** (new) | sheet: description, availability, validation + evidence, required capabilities and hardware, when it alerts, limitations, settings form built from the schema; Activate/Save → POST preset with `settings`; waits up to 20 s for the device to report it active with those settings; errors (`invalid_setting`, `incompatible_policy`, `role_forbids_preset`, `unknown_preset`) shown with the backend detail | DeviceDetailView.swift:775; PresetDetailView.swift:318, :347 |
| **Custom rules card** (new) | GET /v1/devices/{id}/rules: status chips Running / Waiting for area / Waiting / Rejected (with the device's reason) / Off / Unknown (never Running on a silent device); toggle → PATCH, then re-reads, with no optimistic flip; `rule_not_found` and not-supported said; "Ask Jupiter" link | DeviceDetailView.swift:117, :886, :900; CustomRulesCard.swift:91 |
| Rejected-rules notice | from `status.rules.rejected` | DeviceDetailView.swift (mode card) |
| Event history | real (GET events); errors shown | DeviceDetailView.swift:523; EventHistoryView.swift |
| Detection areas | real (GET/POST /v1/devices/{id}/zones); add, undo, delete; errors shown | DeviceDetailView.swift:535; ZoneEditorView.swift:249, :281 |
| Report | see Reports | DeviceDetailView.swift:547 |
| **Stop / Resume monitoring** (new) | POST /v1/devices/{id}/protection; stop asks for confirmation first; shows "Stopping…" until the device echoes the `command_id` (`status.last_command_id`) or reports `stopped`; overdue after 45 s ("saved, will apply when it checks in"); `firmware_unsupported` → "update the device", disabled; endpoint missing → not-supported; no `status` (old backend) → disabled, because the result couldn't be confirmed | DeviceDetailView.swift:930, :967, :991; ProtectionControl.swift |
| Run a demo alert | demo-only; on a real device the API refuses it with a message | DeviceDetailView.swift:562; BackendAPI.swift (`triggerDemoEvent`) |
| Rename device (⋯ menu) | real (POST name); **fixed**: it was `try?` and failed silently; now an alert shows the error | DeviceDetailView.swift:183 |
| Device health card | the device's own health report; withheld when stale ("Health appears here while the device is online") | DeviceDetailView.swift:339 |
| **Detection capabilities** (new) | `status.capabilities`: impaired (with reason), starting, available ("Producing results", plus `evaluation` status when the backend sends it); stale, offline or null → no rows, "Unknown — the device isn't reporting" | DeviceDetailView.swift:381; DeviceStatus.swift (`CapabilityHealth`) |
| Tapped notification → alert sheet | real; "Couldn't open that alert" if it can't be found | DeviceDetailView.swift (onChange pendingEventID) |

### Ask Jupiter (AIConfigureView)

| Action | Behaviour | Where |
|---|---|---|
| Send request / examples | real (POST /v1/ai/configure); over-length refused with a message; errors shown; `source: "unavailable"` says nothing changed | AIConfigureView.swift:157, :620 |
| Rule editors (sensitivity, push/silent, durations) | edit the proposal sent on activation | AIConfigureView.swift |
| "Not yet possible" card | `proposal.unsupported` with reason, route and alternatives | AIConfigureView.swift |
| **If you activate this** (new) | lists what WILL be watched (rules; "once the X area is drawn"; or the mode) and what WON'T (unsupported requests, rules not turned on with a mode switch, the current mode that AI Custom replaces); notes an offline device | AIConfigureView.swift:503 |
| Activate / Switch (changed) | now asks for confirmation first, then POST preset with the edited rules and `draft_version`; `draft_superseded` → re-fetches with a notice; other codes shown | AIConfigureView.swift:444, :649 |
| After activation (changed) | says "Sent to <device>" and polls the device (status + rules list) for 20 s: **Active** only when the device reports the rules running or waiting for an area, or the mode active; **Refused** with the device's reason; otherwise "hasn't confirmed yet" | AIConfigureView.swift:549; DeviceRules.swift (`ActivationFollowUp`) |

### Alert detail (EventDetailView)

| Action | Behaviour | Where |
|---|---|---|
| Why did this fire | real (GET explain); an event without an id says the backend doesn't support explanations; errors shown | EventDetailView.swift:167 |

### Shop (ShopView)

| Action | Behaviour | Where |
|---|---|---|
| Quantity, address | local | ShopView.swift:73 |
| Send order request (changed) | real (POST /v1/orders) but it is **an order request, not a purchase**: the button reads "Send order request · est. $…", the screen says no payment is taken and someone will confirm, the result reads "Request #N received — not yet confirmed or charged" (it said "on the way"), and status "placed" shows as "Requested" | ShopView.swift:91, :149 |
| Order list | real (GET /v1/orders); errors shown; the last list is kept on failure | ShopView.swift:165 |

### Settings (SettingsView)

| Action | Behaviour | Where |
|---|---|---|
| Server status / Find my server | real (health probe + Bonjour); reasons shown | SettingsView.swift:56 |
| Manual server address | real: checked against /v1/health before saving; refuses localhost on a phone and cleartext to internet names | SettingsView.swift:79, :401 |
| Connection doctor | see Diagnostics | SettingsView.swift:89 |
| Device Wi-Fi setup (BLE) | real; physical device only | SettingsView.swift:104 |
| Live feed: Find my device, Device token | real mDNS; token kept in the Keychain | SettingsView.swift:122, :173 |
| Check device health | real (device /api/health on the LAN); errors shown | SettingsView.swift:202 |
| Text-message alerts → Save (changed) | real (POST /v1/account/phone). **Fixed**: the error was set but never displayed. It now says texts are sent only if SMS is set up on the server, and that the server **can't show saved settings back** (there is no GET /v1/account), so the field shows what this phone last entered | SettingsView.swift:222 |
| Alerts when the app is closed (ntfy) (changed) | real (POST /v1/account/ntfy). **Fixed**: the channel was regenerated on every visit to Settings, which silently replaced the subscribed one; it is now kept in the Keychain (cleared when a different account signs in), "New channel" is explicit, and the screen says whether this channel was saved and that the server can't read it back | BackgroundAlertsSection.swift:93, :145 |
| Exit demo / Log out | real; logout revokes the token server-side (best effort) | SettingsView.swift:30, :259 |

### Connection doctor (DiagnosticsView)

| Action | Behaviour | Where |
|---|---|---|
| Run all checks | real chain: server state → configured → reachable → session → **monitoring per device (new)** → device address → device reachable → live frame. The monitoring check passes only when the device reports `ready` with nothing impaired; unknown/stale/starting is "–", never ✓ | DiagnosticsView.swift:74, :101 |

### Reports (ReportsView)

| Action | Behaviour | Where |
|---|---|---|
| Score, charts (changed) | computed on the phone from up to 1,000 events. The "safety score" is now the **Activity score** with an "Estimate" tag and the line "Not a validated measure of safety"; shared text says the same. **Fixed**: a load failure left a spinner for ever (`try?`); it now shows an error with Try again | ReportsView.swift:84, :64 |
| Share | shares the estimate text | ReportsView.swift:76 |

### Demo alert rehearsal (DemoTriggerView)

| Action | Behaviour | Where |
|---|---|---|
| Fire scenario, Reset, Allow notifications | demo-only; refused with a message against a real backend | DemoTriggerView.swift:96 |

## The M6 backend contract, as the app uses it

Client: `APIClient` in API.swift (shared `request` path: relocation, cleartext
refusal, coded errors). Demo: `MockAPI` in BackendAPI.swift implements the same
calls. Decoding is lenient everywhere (PresetLibrary.swift, DeviceRules.swift):
missing or unknown fields degrade to "not said"; one malformed element is
dropped, not fatal.

| Endpoint | App behaviour against it | Against an older backend |
|---|---|---|
| GET /v1/presets | Mode card and detail sheet; titles; validation; settings schema | uncoded 404 → built-in list, labelled "Built-in list: this server doesn't publish its preset catalog…"; no settings and no validation shown |
| GET /v1/devices/{id}/presets | availability and reason per row; current settings; active | 404 → availability from `status.allowed_presets` (as before) |
| POST /v1/devices/{id}/preset `{preset, settings}` | settings sheet; codes mapped to sentences that keep the detail | settings are only offered when the catalog came from the server |
| GET /v1/devices/{id}/rules | Custom rules card, activation follow-up | 404 → "This server can't list rules yet…"; follow-up uses `status.rules.active` |
| PATCH /v1/devices/{id}/rules/{rule_id} | toggle, then re-read | 404 → "Turning rules on and off isn't supported by this server yet." |
| POST /v1/devices/{id}/protection | Stop/Resume with convergence on `status.last_command_id`, else on `status.protection` | 404 → not-supported; `firmware_unsupported` → update the device |
| `status.capabilities.evaluation` (optional) | shown next to each available capability | absent → nothing claimed |

"Uncoded" matters: `device_not_found` and `rule_not_found` are coded 404s from
a backend that does have the endpoint, and are reported as such
(`ServerSupport.endpointMissing`).

## Running the app in the iOS Simulator against a local backend

The virtual-device/backend side (seeding a device that reports health,
monitoring and rules) is documented by the virtual-device work; this is the
app side.

1. Start the backend on the Mac (from the repository root):
   `.venv/bin/uvicorn cloud.app:app --port 8787` (cloud/README.md).
2. Generate and build the app (the project is generated, and gitignored):
   ```
   cd ios && xcodegen generate
   xcodebuild -project SensorDyme.xcodeproj -scheme SensorDyme -configuration Debug \
     -sdk iphonesimulator -destination 'generic/platform=iOS Simulator' \
     -derivedDataPath .build-dd CODE_SIGNING_ALLOWED=NO build
   ```
3. Install and launch on a simulator:
   ```
   xcrun simctl boot "iPhone 17 Pro"
   xcrun simctl install booted ios/.build-dd/Build/Products/Debug-iphonesimulator/SensorDyme.app
   xcrun simctl launch booted com.sensordyme.app
   ```
4. Point it at the backend. A simulator build with no `SENSORDYME_API_BASE_URL`
   already defaults to `http://localhost:8787` (in the Simulator, localhost is
   the Mac). For any other address: **Settings → Manual server address →**
   e.g. `http://127.0.0.1:8787` → **Check and save**. The address is checked
   against `/v1/health` and is not saved if nothing answers. "Clear and go back
   to automatic" undoes it.
5. Create an account, claim the virtual device with its claim code (**+ → Add
   device**), and open it. Settings → Connection doctor checks the whole chain,
   including the device's monitoring state.
6. Clean up: `xcrun simctl uninstall booted com.sensordyme.app`,
   `xcrun simctl shutdown booted`, `rm -rf ios/.build-dd ios/SensorDyme.xcodeproj ios/Info.plist`.

Debug launch flags (DEBUG builds only, DebugLaunch.swift): `-orynDemo`,
`-orynOpenDevice`, `-orynOpenJupiter -orynPrompt "<text>"`, `-orynOpenHistory`,
`-orynScrollBottom`, `-orynTab shop|settings`.

## Tests

`bash ios/Tests/run.sh` runs the Foundation-only logic tests (each
`<Name>Tests.swift` is concatenated with `Sources/<Name>.swift`). New in M8:
`ConnectionTruthTests` (presence from telemetry age, reply age and refresh
outcome; the summary count), `ServerReplyTests` (every reply shape to a
headline and sentence), and additions to `NotificationPolicyTests` (routing),
`PresetLibraryTests` (developer-worded reasons), `DeviceStatusTests`
(attention calibrating) and `PerceptionStatusTests` (starting verdict,
experimental features, coded refusals). New in M6:

- `PresetLibraryTests`: catalog and per-device decoding (lenient, lossy),
  availability reasons per code, validation never overstated, the
  settings-form logic (defaults, snapping, range checks, payload, confirmation),
  refusal wording, uncoded vs coded 404, the built-in fallback.
- `DeviceRulesTests`: rules decoding, status chips (never Running when the
  device is silent), toggle errors, the post-activation follow-up.
- `ProtectionControlTests`: offered action, convergence by command id and by
  state, pending, overdue and confirmed phases, refusals.
- `DeviceStatusTests` (extended): `last_command_id`, `capabilities.evaluation`,
  capability health never green when stale or offline.

## Limitations and hardware dependencies

- Live feed, Bluetooth provisioning, "Find my device" and device health need a
  real SensorDyme on the same Wi-Fi (or the virtual device for the backend
  parts). The Simulator has no Bluetooth.
- The new endpoints were built against the contract while the backend was
  implemented in parallel (branch `m6/presets-policies`). Field names follow
  the contract; anything the backend adds or omits degrades to "not said".
- `status.last_command_id` is not passed through by the backend at the time of
  writing; stop/resume then converges on `status.protection`, which cannot
  tell two quick opposite requests apart.
- Notification settings (SMS number, ntfy channel) are write-only on the
  server; the app shows what this phone last saved and says so.
- Reports are phone-side arithmetic over loaded events, labelled as an
  estimate.
- Shop has no payment; orders are requests.
- Push alerts while the app is closed go through ntfy (no APNs without a paid
  Apple Developer membership).
- XCUITest journeys (ios/UITests) run the real app in the Simulator against
  the real backend and the virtual device: `scripts/sim/ios_uitest.sh`. They
  need the Simulator, so they are not part of `make verify-release`.
