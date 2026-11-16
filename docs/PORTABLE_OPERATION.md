# Portable device operation: what needs what (analysis, 2026-11-14)

The target: *"Plug in SensorDyme. Turn it on. Open the iPhone app. It reconnects
by itself. Start or resume monitoring. Use the camera and events."* The
development machines (MacBook, `oryn`, `quantcompute`) build models; they must
never be part of customer inference.

This document says, feature by feature, what each part needs to work today, and
where the code ties features together more than it has to. It then looks at a
direct-to-device Wi-Fi mode (the device runs its own access point and the phone
joins it). Every claim cites `file:line` and is labelled:

* **[test]**: a test pins it. The test is named.
* **[code]**: read from the code. No test pins it.
* **[unverified]**: platform behaviour (iOS, NetworkManager, avahi) or hardware
  behaviour that nothing in this repo has checked.

**Nothing here is hardware evidence.** The tests named here run on a Mac, against
the virtual device (`sim/`) or fakes. Earlier documents record some Phase 7/8
observations on a Pi. This analysis did not re-check them and relies on none of
them.

Line numbers are as of this commit, on a branch based on `5ed3e59`.

---

## 0. How the pieces actually talk today

```
device ──HTTP(S) POST /v1/ingest, outbound only──▶ backend          core/uplink.py:747-820
   events + health up; ONE pending command down, in the response   core/uplink.py:783-820
phone ──HTTP(S) /v1/*──▶ backend                                     ios/Sources/BackendAPI.swift:259-319
phone ──HTTP GET /stream, /api/health (X-Device-Token)──▶ device     ios/Sources/FeedViews.swift:261, DeviceLink.swift:244-270
phone ──BLE GATT (Wi-Fi credentials, "rescan"/"identify")──▶ device  services/ble_gatt.py:14, :604-611
```

* The app sends commands to the backend only. It calls exactly two device
  endpoints, `/api/health` and `/stream`
  (`ios/Sources/Discovery.swift:254`, `DeviceLink.swift:245`,
  `FeedViews.swift:261`). [code: a grep of `ios/Sources` for `api/` finds no
  other device endpoint]
* The device dashboard **already serves token-gated local commands**. These are
  `POST /api/protection` (stop or resume; a stop needs `"confirm": true`),
  `POST /api/preset` and `POST /setup/wifi`. It also serves
  `GET /api/history`, `/api/state`, `/api/presets`, `/api/health`,
  `/api/stream/stats` and `/stream` (`core/dashboard.py:452-562`). The
  runtime applies dashboard stop/resume and preset requests the same way as
  remote ones (`services/runtime.py:2738-2779`). [test:
  `tests/test_m6_security_device.py::test_writes_with_the_header_token_are_accepted`,
  `::test_a_query_token_cannot_authorize_a_write`, and
  `tests/test_m83_stream_runtime.py:143-147`, where a dashboard stop reaches
  the runtime]
* Rules, zones, camera mode and companion nodes arrive **only** through the
  uplink (`services/runtime.py:2784-2811`). [code]

---

## 1. Dependency matrix

Two backend deployments exist, and they change what "without the MacBook" means:

* **(M) Mac LaunchAgent** (`scripts/install_backend_mac.sh`). The backend is a
  process on the MacBook, found over mDNS as `sensordyme-backend.local`
  (`docs/NETWORKING.md`, `cloud/mdns.py`). For this deployment, **"without the
  MacBook" is the same as "without the backend"**. It is also "without the
  backend" whenever the phone or the device is not on the Mac's network. The
  device reaches the backend by `cloud.base_url` (`config.yaml:201-210`). The
  app finds it by Bonjour `_sensordyme-api._tcp`
  (`ios/Sources/BackendLocator.swift:26`).
* **(H) Hosted backend** (`fly.toml`, HTTPS, no mDNS). The MacBook does not
  matter. Here **"without internet" is the same as "without the backend"**, for
  whichever side has lost internet. I found nothing showing a Fly deployment is
  live. [unverified]

Key:

* **YES**: works.
* **DEGRADED**: works with less (named).
* **NO**: does not work.

*Fails closed* means the feature refuses and says so; it never pretends.

| Feature | no oryn | no quantcompute | no MacBook | no home Wi-Fi | no internet | no backend connectivity |
|---|---|---|---|---|---|---|
| **Local inference** | YES | YES | YES | YES | YES | YES [test] |
| **Local (buzzer) alerts** | YES | YES | YES | YES | YES | YES [test] |
| **Live video** | YES | YES | (M) **NO in the app** on a cold launch; transport YES ¹ | needs *some* LAN shared by phone and device ² | YES on a LAN | **NO in the app** on a cold launch; transport YES ¹ |
| **Commands** (stop/resume, preset) | YES | YES | (M) **NO** in the app ³ | (M) NO / (H) YES if the device has another uplink | (M) YES on the LAN / (H) NO | **NO** in the app; fails closed with a stop timeout ⁴ |
| **Custom rules / Ask Jupiter** | YES (not used by default) | YES | (M) **NO** | as Commands | (M) YES with Ollama on the Mac / (H) DEGRADED ⁵ | **NO** |
| **Event history** | YES | YES | (M) **NO** in the app; device keeps `events.csv` | as Commands | (M) YES on the LAN | **NO** in the app; the device records and spools ⁶ |
| **Account authentication** | YES | YES | (M) **NO** new sign-in; an existing session persists ⁷ | as Commands | (M) YES on the LAN | **NO** new sign-in; an existing session persists ⁷ |
| **Remote notifications** | YES | YES | (M) **NO** | as Commands | **NO** for ntfy and SMS; in-app local notifications only (M, app open) ⁸ | **NO** |
| **Wi-Fi provisioning (BLE)** | YES | YES | YES ⁹ | YES ⁹ | YES ⁹ | DEGRADED: the Wi-Fi hand-off works, the claim step does not ⁹ |
| **Device dashboard** (web page at :8899) | YES | YES | YES | needs a shared LAN | YES | YES ¹⁰ |

Evidence for the rows:

* **Local inference and alerts.**
  `tests/test_m8_lifecycle_e2e.py::test_lost_backend_local_alerts_continue_queue_bounded_and_flush_once`
  asserts `buzzer_actuations > 0`, `monitoring_state == "ready"` and a backlog
  of at least 1 while the uplink is down (`:145-172`) [test, virtual device].
  The uplink never raises into the device (`core/uplink.py:836-926`) [code].
  The Pi presets fetch nothing at runtime. The only runtime download in
  `core/`, `services/`, `modules/` and `main.py` is the openpilot engine
  (`modules/driver_attention_openpilot.py:699`), and the shipped presets alias
  that engine away (`presets.yaml:42-52`). The VLM judge (Anthropic) is
  `mac_max` only (`core/vlm_judge.py:1-33`). [code]
* **The dev machines.** No device or backend code path names oryn or
  quantcompute. Jupiter's default model server is `localhost:11434` on the
  **backend's** host (`jupiter/providers.py:94`, `jupiter/router.py:66-68`).
  [code]

Notes:

1. **Live video.** The transport is phone → device directly, gated by the
   dashboard token. It needs no backend
   (`core/dashboard.py:358-398`, `:540-562`). But the app can only show it
   inside a backend `Device` row:
   * `FeedCard(device:)` and `CameraSource(state:device:)` both take a backend
     `Device` (`ios/Sources/FeedViews.swift:5-17`,
     `CameraViews.swift:59-69`).
   * The device list is `@State` and empty at launch
     (`DevicesView.swift:17`). It is filled only by
     `state.backend.devices()` (`:322-347`).
   * Nothing caches it to disk. [code: grep]

   So on a cold launch with no backend there is no card and no way into the
   live view. On a warm launch (the list was already loaded), a failed refresh
   **keeps** the cards (`DevicesView.swift:160-161`, `:342-346`), so the feed
   stays reachable. [code]
2. Discovery is Bonjour `_sensordyme._tcp` (`ios/Sources/Discovery.swift:39`,
   `deploy/avahi/sensordyme.service`), so it needs multicast on a shared
   network. A phone hotspot qualifies if iOS's hotspot host sees its clients'
   Bonjour records [unverified].
3. **Commands.** The app sends stop/resume, preset, zones, nodes, camera mode
   and rules only to the backend (`ios/Sources/BackendAPI.swift:269-316`). The
   device answers these locally only on `/api/protection` and `/api/preset`
   (§0), and the app never calls them. [code]
4. A stop that the device cannot receive times out at 45 s. The app then says
   "didn't confirm the stop in time … It is still monitoring" (fail closed).
   [test: `tests/test_m83_truth_protection_ack.py`, `docs/FAILURE_MATRIX.md`
   row 10]
5. **Ask Jupiter** is `POST /v1/ai/configure` on the backend
   (`cloud/app.py:4612`). It tries these in order (`cloud/app.py:4551-4581`):
   1. Policy refusals (identity, emotion, …). No model.
   2. The deterministic area/duration parser. No model.
   3. The language model. By default this is local Ollama on the backend host,
      `qwen2.5:7b-instruct` (`jupiter/router.py:66-68`). An Anthropic retry
      happens **only** when `SENSORDYME_ALLOW_EXTERNAL_AI=1` or
      `cloud_fallback.enabled` is set (`jupiter/router.py:157-177`,
      `cloud/app.py:4186-4192`).
   4. The keyword fallback. It proposes a *mode*, never a rule, or refuses
      with `interpreter_unavailable` (`cloud/app.py:4240-4285`).

   So by default Jupiter makes **no external LLM API call**. It needs the
   backend plus a model server on the backend's machine. On (M) that is Ollama
   on the MacBook; it works offline. On (H), `cloud/Dockerfile` copies
   `jupiter/` but ships no model server, so step 3 is skipped unless the
   operator turns on the Anthropic fallback (which needs internet) or points
   `base_url` at a reachable server. Event explanations
   (`cloud/app.py:4674`) give a facts-only answer without a model
   (`jupiter_config.yaml.example`). [code; refusal/fallback shape: test
   `tests/test_m84_jupiter_grammar.py`, `tests/test_capabilities.py`]
6. **Event history.**
   * The device writes `data/events.csv` locally
     (`docs/EVENT_PIPELINE.md` §1, `core/logger.py`).
   * Undelivered events are spooled to `uplink_spool.jsonl` and delivered once
     on return. [test: `tests/test_m8_lifecycle_uplink.py`,
     `tests/test_m8_lifecycle_e2e.py:175`]
   * The app reads history from the backend only (`BackendAPI.swift:719`). The
     device's `GET /api/history` (`core/dashboard.py:532-537`, `:1012-1051`)
     is unused by the app.
   * Without NTP the Pi's wall clock can be wrong (`docs/EVENT_PIPELINE.md`
     §3).
7. **Account authentication.** Sign-in is `POST /v1/login` on the backend
   (`cloud/app.py:2581`). An existing session survives with no backend:
   * `isSignedIn = !storedToken.isEmpty` (`ios/Sources/API.swift:546`). There
     is no local expiry check.
   * The JWT lives 30 days (`cloud/app.py:84`). Only a server *rejection* ends
     the session (`API.swift:1206-1210`).
   * The whole app sits behind sign-in (`SensorDymeApp.swift:93-116`).

   [code]
8. **Remote notifications.**
   * Push goes backend → `ntfy.sh` (`cloud/app.py:85`, `:2115-2140`). SMS goes
     backend → Twilio (`:2143-2178`). Both need the backend and internet.
   * The app's own notifications come from polling the backend every 15 s, and
     only while the app runs (`ios/Sources/AlertWatcher.swift:11-16`,
     `:51-76`).
   * Polling stops while the backend is not `.connected` (`API.swift:615-619`).

   [code]
9. **Wi-Fi provisioning (BLE).** The BLE window opens while the device is
   offline and needs no network (`services/provisioning.py:1-35`). [test:
   `tests/test_provisioning.py::test_offline_boot_opens_ble_after_grace`] The
   app's BLE setup is reachable only once signed in. The "Connect over
   Bluetooth" button is in `DevicesView` (`:214`), and the sign-in screen
   offers only the demo (`ios/Sources/AuthView.swift:231-233`). So **a phone
   that has never signed in cannot provision Wi-Fi with no backend**. The claim
   step is a backend call (`ios/Sources/SetupFlow.swift:135-146`). [code]
10. **Device dashboard.** On loopback, no token is needed. On a LAN bind it
    fails closed without a token (`core/dashboard.py:360-371`). [test:
    `tests/test_m6_security_device.py::test_no_token_on_a_lan_bind_fails_closed`]

---

## 2. Separation of concerns as built, and where it is coupled more than needed

| Concern | Lives in | Depends on |
|---|---|---|
| Local inference | `services/runtime.py` inference loop, `core/` | camera, accelerator. Nothing networked |
| Local alerts | `core/alerts.py` sinks (console, sound, GPIO buzzer; `config.yaml:255-256`) | the runtime only |
| Live video | `core/dashboard.py` `/stream`; app `MJPEGView` | a shared LAN, the dashboard token, **and a backend `Device` row in the app** (C1) |
| Commands | backend queue → ingest response → `core/uplink.py:783-820` → runtime | backend reachable by **both** phone and device (C2) |
| Event history | device `events.csv` + spool → backend DB → app | the backend, for the app (C4) |
| Account auth | `cloud/app.py` JWT; app Keychain | the backend for sign-in only |
| Remote notifications | `cloud/app.py` → ntfy/Twilio; app poll → local notification | backend + internet |
| Device identity on the LAN | avahi TXT `id=SD-…`; app `FeedLink` | backend account ids *preferred*, not required (`FeedLink.swift:49-75`) |

Couplings that are stronger than the job needs, most consequential first:

* **C1. The live view needs the backend's device list.**
  * `FeedViews.swift:5-17`, `CameraViews.swift:59-69`, `DevicesView.swift:17`,
    `:322-347`.
  * The video itself never touches the backend, but a cold launch with no
    backend gives "Can't reach the server" and no way to reach the camera.
  * Worse, the LAN banner hides itself when the device *is* found
    (`DeviceLinkBanner.swift:23-25`, `.connected → EmptyView`). So a phone
    that can see the camera says nothing about it.

  [code]
* **C2. Every command goes through the backend.** The device already accepts
  token-gated `POST /api/protection` and `/api/preset`
  (`core/dashboard.py:463-488`), but the app has no local command path
  (`BackendAPI.swift:259-319`). Stop/resume needs phone → backend → device
  round trips, so it fails whenever either side cannot reach the backend, even
  with the phone next to the device on one LAN. [code]
* **C3. The dashboard token is pasted by hand.**
  * The only writers of `AppState.deviceToken` are the Settings text field
    (`ios/Sources/SettingsPages.swift:77-95`, footer: "Set dashboard.token in
    the device's config.yaml, then paste the same value here") and a debug
    launch argument (`SensorDymeApp.swift:53-56`).
  * Neither the backend nor BLE setup delivers it. A grep of `services/ble_*`,
    `cloud/app.py` and `ios/Sources/Setup*.swift` finds no token hand-off.
  * Until someone pastes it, `/stream` answers 401. "Automatically reconnect"
    cannot include video for a new customer.

  [code]
* **C4. History in the app is backend-only.** The device's `/api/history`
  exists but is unused. Its rows also have no `event_id`
  (`docs/EVENT_PIPELINE.md` §7), so local and backend history cannot be merged
  without duplicates. [code]
* **C5. BLE setup is behind account sign-in** (note 9). Getting a device onto
  Wi-Fi does not need an account, but the app requires one first. [code]
* **C6. In-app notifications stop whenever the backend is not connected**
  (`AlertWatcher.swift:68`). This is defensible for battery. But on a LAN with
  the device in view, the phone could poll the device's `/api/state`
  (`alert_active`, recent events; `core/dashboard.py:1077-1095`) and does not.
  [code]

Separations that are already good:

* The device never needs the backend to protect. [test, note on the rows]
* Video never transits the backend (`docs/VIDEO_TRANSPORT.md`).
* The uplink is outbound-only and NAT-friendly (`core/uplink.py:1-12`).
* Privacy-weakening commands (camera-only ON) need a signed envelope
  (`core/uplink.py:1070`).
* Role floors are enforced on the device for local and remote presets alike
  (`services/runtime.py:627`, `core/dashboard.py:470-475`).

---

## 3. Direct-device Wi-Fi mode (device as access point)

### 3.1 What exists

* **A legacy setup hotspot**, `scripts/provisioning_hotspot.sh`. It is
  installed only when `network.provisioning.method: hotspot`
  (`setup_pi.sh:179-199`); the shipped value is `ble` (`config.yaml:165`).
  * It runs once at boot, 45 s after NetworkManager fails to join
    (`:30-41`).
  * It is **not a portable operating mode**. It exists so that `/setup` can
    hand the device a home network, and the hotspot ends as soon as one is
    joined (`:43-47`).
  * Its comment says the hotspot is open ("no password", `:40`). Per the
    `nmcli` manual, `device wifi hotspot` with no password *generates* a
    random one. So in practice it is probably WPA-protected with a passphrase
    nobody is shown. [unverified on the Pi's NetworkManager] Either way it is
    unsuitable: an open AP, or one nobody can join.
* **The setup wizard** still has the hotspot path. It is labelled "older
  devices", with `http://10.42.0.1:8899/setup`
  (`ios/Sources/SetupWizardView.swift:218`, `:244`).

### 3.2 If the phone joined a device AP today, with no code change

Assume someone hand-configured an AP on the Pi: WPA2, NetworkManager
`ipv4.method shared` (10.42.0.1/24). The phone joins it. There is no other
uplink: the Pi 5 has one Wi-Fi radio, and simultaneous client + AP on it is
not something this repo has tested. [unverified]

| Function | Today on a device AP | Why |
|---|---|---|
| Device keeps protecting | **YES** | §1, rows 1-2 [test] |
| Discovery | **Probably YES** | avahi publishes the static service on every interface by default [unverified]. The app re-resolves on a network path change (`DeviceLink.swift:102-113`). Local Network permission is required (`ios/project.yml:51-52`). |
| Live stream | **Only if** the token was pasted **and** a `Device` card exists (warm launch, or a hosted backend reachable over cellular) | C1, C3. ATS allows plain HTTP (`ios/project.yml:45`) |
| Local stop/resume and preset | **NO from the app** (C2). YES from the on-device web page for presets (`core/dashboard.py:1302`), but it has no stop/resume button | The endpoints exist; no client uses them |
| Rules, zones, camera mode, Jupiter | **NO** | uplink-only (`services/runtime.py:2784-2811`), plus the backend for Jupiter |
| Event history | **NO** in the app (C4). The device keeps recording and spooling | |
| Account sign-in | **NO** for a new sign-in. An existing session persists (note 7) | |
| App with no backend at all | "Can't reach the server" banner, an empty list on a cold launch, no notifications | `DevicesView.swift:162-165`, `AlertWatcher.swift:68` |
| App with a hosted backend over **cellular** while on the AP Wi-Fi | Cards load; the device shows offline. A backend stop **times out** (fail closed, note 4) because the device has no uplink | iOS routing to cellular on a no-internet Wi-Fi is [unverified] |
| BLE recovery while in AP mode | **At risk.** `WifiManager.is_online()` counts `connected (local only)` and `(site only)` as online (`core/network.py:191-205`) [test: `tests/test_network_review_fixes.py:884-885`]. An AP-mode device would therefore read "online" and close BLE after `reconfigure_window_s` (default 300 s), unless `always_available` is set (`services/provisioning.py:132-137`, `:291-294`) [test: `tests/test_provisioning.py::test_bluetooth_setup_is_not_left_open_by_default`]. NetworkManager's exact state string in AP mode is [unverified]. | |
| Event timestamps | wall clock may be wrong with no NTP; the backend corrects for queued events using `age_s` and `occurred_at` | `docs/EVENT_PIPELINE.md` §3 |

**Verdict.** An access point alone does **not** make the product portable or
the app offline-capable. Today it gives a protecting device with an
unreachable app. Only the camera might work, and only for a user who pasted
the token and did not cold-launch. Most of the device half already exists:
token-gated local stop/resume, preset, history and stream. The missing pieces
are in the app (C1-C4) and in network-mode management on the device.

### 3.3 Smallest safe design

**Device**

1. **A direct-mode NetworkManager profile**, managed by `core/network.py`
   (still the single owner of nmcli).
   * `802-11-wireless.mode ap`, `ipv4.method shared`.
   * WPA2-PSK (CCMP) at minimum. Prefer WPA3-SAE in transition mode.
   * **Never open, never a shared default passphrase.**
   * SSID `SensorDyme-XXXX` (`core/identity.display_name()`).
   * Entered in two cases only:
     * explicitly: a new BLE command (behind PoP) or a token-gated
       `POST /api/network/direct`;
     * as a fallback: no saved client network seen for N minutes, *off* by
       default.
   * Left when a saved network returns, or on command.
   * Uses the same validate-then-switch-with-rollback pattern as
     `WifiManager.provision()` (`core/network.py:435-470`).
2. **The PSK comes from the device label.**
   * Random and per device, generated and stored the way the BLE setup code
     is: `scripts/ble_setup_code.py`, mode 0600, refused if loosely
     permissioned (`docs/BLE_PROOF_OF_POSSESSION.md`).
   * Printed beside the setup code and in the QR.
   * Not derived from the MAC or serial, both of which are broadcast.
3. **Keep the dashboard token** as the application credential, plus:
   * a firewall on the AP interface: only 8899, DHCP, DNS and mDNS;
   * no SSH on the AP.

   WPA keeps out anyone without the label. The token keeps the stream and
   commands from any client that has the PSK but was never paired.
4. **Treat AP mode as "not online" for provisioning.** Keep the BLE (PoP)
   window open for as long as direct mode is active. This is the recovery path:
   a broken AP must never strand the device (see 3.5).
5. **Add `event_id` to `events.csv`** so local history can later merge with the
   backend's without duplicates (`docs/EVENT_PIPELINE.md` §7).

**App**

1. **Persist the last device list** (id, name, hardware id), so a device card
   and its live view exist with no backend (fixes C1). Label it "Direct
   connection: not reporting to your account".
2. **A local control client.** When the backend is unreachable and the
   `DeviceLink` is `.connected`, send stop/resume and preset to the device's
   `/api/protection` and `/api/preset` with `X-Device-Token` (fixes C2).
   * A stop still requires explicit confirmation.
   * The device's role floor still applies (409 refusal,
     `core/dashboard.py:470-475`).
   * The result is confirmed by reading `/api/state` `policy.protection`,
     never assumed.
3. **Token delivery without pasting** (fixes C3): add a token read to the BLE
   PoP session, after the device has proved the code. The session is already
   AES-256-GCM (`services/ble_pop.py`). The backend never learns the token.
4. **Local history**: `GET /api/history` while direct (C4).
5. **In-foreground alerting while direct**: poll `/api/state` `alert_active`
   (C6). Background push stays a backend feature, and the UI says so.
6. **Allow BLE setup before sign-in**, Wi-Fi hand-off only; the claim stays
   behind sign-in (fixes C5).

**Backend**

* Nothing is required for direct mode.
* Already handled: events made while direct arrive later, once, with corrected
  time (spool, `age_s`), and a local stop or resume made while disconnected is
  reconciled. [test:
  `tests/test_m83_truth_protection_ack.py::test_a_later_local_change_does_not_rewrite_what_the_request_did`]
* Optional: surface `protection.by == "dashboard"` (`services/runtime.py:2546`)
  in the app's history as "stopped locally".

### 3.4 Security model

| Layer | Control | Existing evidence |
|---|---|---|
| Radio | WPA2-PSK, WPA3-SAE preferred; per-device random PSK from the label; never open | new (G5, G6) |
| Pairing | BLE proof of possession: PBKDF2 → HKDF → HMAC proofs → AES-GCM; 5 failures lock out, doubling up to 1 h | `docs/BLE_PROOF_OF_POSSESSION.md`, `tests/test_ble_pop.py` [test] |
| Application | `X-Device-Token` on every `/api` and `/stream`; constant-time compare; `?token=` accepted on GET `/stream` only; fail closed with no token on a LAN bind | `core/dashboard.py:358-398` [test: `tests/test_m6_security_device.py`] |
| Command semantics | a stop needs `confirm: true`; role floors refuse unsafe presets; camera-only ON only through a backend-signed envelope | `core/dashboard.py:484-486`, `:470-475`; `core/uplink.py:1070` |
| Resource exhaustion | at most 64 connections, 30 s socket timeouts, 64 KiB bodies | `core/dashboard.py:77-108`, `:353`, `:444` |

Known weaknesses to accept or fix:

* **One shared token per device.** There is no per-phone revocation, and
  rotating it means editing `config.yaml` on the device.
* **Plain HTTP inside the WPA link.** Any client holding the PSK can sniff the
  token. That is acceptable only because the PSK is per device and physically
  held.
* **Local commands are not signed (sd-auth-v1).** Their trust is "holds the
  PSK and the token".

### 3.5 Recovery requirements (must be tested before shipping)

* **R1. BLE PoP stays open whenever direct mode is active.** Unit test it in
  `tests/test_provisioning.py` with a `FakeWifi` reporting `connected (local
  only)`.
* **R2. Entering direct mode is transactional.** If the AP fails to come up
  within T seconds, roll back to the previous profile. Unit test it in the
  style of `tests/test_wifi_connect_failures.py`.
* **R3. A power cycle always tries saved client networks first.** Direct mode
  is never the sticky boot default, unless explicitly configured.
* **R4. Hardware session (Pi + iPhone), each step logged as evidence:**
  1. enter direct mode;
  2. the phone joins with the label PSK and finds the device over Bonjour;
  3. stream;
  4. stop/resume locally;
  5. leave direct mode, then return to home Wi-Fi;
  6. a deliberately broken AP config, recovered over BLE;
  7. confirm a wrong PSK is refused and that no open SSID is ever broadcast.

  No step may be run on the device before R1-R3 pass.

### 3.6 Gap list and size

Sizes: XS < ½ day, S ≈ 1 day, M ≈ 2-4 days, L ≈ 1 week. These are software
only; hardware verification is extra.

| # | Gap | Side | Size |
|---|---|---|---|
| G1 | Persist the device list and open a device and its live view without a backend (C1) | app | M |
| G2 | Local command client (stop/resume, preset) with truthful confirmation from `/api/state` (C2) | app | M |
| G3 | Token over BLE PoP after proof (C3) | device + app | M |
| G4 | Local history view from `/api/history`, plus `event_id` in `events.csv` (C4) | app + device | M |
| G5 | Direct-mode AP profile management with rollback, and an entry/exit policy | device | L |
| G6 | Label PSK generation and storage (like `ble_setup_code.py`) | device/factory | S |
| G7 | AP-interface firewall | device | S |
| G8 | BLE open while in direct mode (R1); AP not counted as online for provisioning | device | S |
| G9 | Foreground local alert polling while direct (C6) | app | S |
| G10 | BLE Wi-Fi setup before sign-in (C5) | app | S |
| G11 | Retire or harden `scripts/provisioning_hotspot.sh` (its "open" comment; it is not a portable mode) | device | XS |

G1 + G2 + G3 alone remove most of the coupling, **on any shared LAN** (home
Wi-Fi with a dead backend, a hotspot, a travel router) and before any AP work.
They are the highest-value first step. G5-G8 are what make the device its own
network.

### 3.7 Small defects found

* **Fixed in this commit: Jupiter's remedy text.** It told people to "set
  ANTHROPIC_API_KEY on the backend and restart it"
  (`cloud/app.py:4278-4279`). With the default configuration that does
  nothing: the interpreter is the local Ollama server, and the Anthropic path
  also needs `SENSORDYME_ALLOW_EXTERNAL_AI=1`
  (`jupiter/router.py:157-177`). The app shows this line as "To enable: …"
  (`ios/Sources/AIConfigureView.swift:537`). It now names both options. Pinned
  by `tests/test_capabilities.py::test_missing_language_service_is_stated_not_papered_over`.
* **Fixed in this commit: a stale networking doc.** `docs/NETWORKING.md` said
  queued events are kept "in memory". They are spooled to disk
  (`core/uplink.py:8-12`, `services/runtime.py:670-673`,
  `tests/test_m8_lifecycle_uplink.py`).
* **Not fixed: hard-coded "your Mac" advice.** For
  `URLError.cannotConnectToHost`, `ios/Sources/API.swift:159` says "Is the
  backend running on your Mac?" even for a hosted backend. Its own comment
  (`:147-150`) says host-specific advice should come from
  `BackendResolution`. It was not fixed because `API.swift` has no Foundation-only
  test harness (`ios/Tests/run.sh`), and an untested Swift edit is not "tiny
  and covered".
* **Not fixed: a silent banner.** When the device is found on the LAN but the
  backend is not (C1), `DeviceLinkBanner` shows nothing
  (`DeviceLinkBanner.swift:23-25`). The person sees only "Can't reach the
  server", with no word that the camera is right there. Fixing it belongs with
  G1.

---

## 4. What to demo where (investor demo)

Assumptions:

* Backend = the Mac LaunchAgent, with Ollama and the configured model pulled on
  the Mac (for Jupiter).
* The device has `cloud.base_url: http://sensordyme-backend.local:8787`, a name
  rather than an IP (`docs/NETWORKING.md`).
* The token is already pasted in the app.
* The app has been launched once with the backend reachable.

**Rehearse every cell on the real hardware first. None of it has been run on
the Pi.**

| Step | Home Wi-Fi + Mac backend | Phone hotspot (Pi + Mac + phone on it) | No internet (travel router, Mac on it) |
|---|---|---|---|
| Power on → device monitoring, buzzer alert | YES | YES | YES |
| Open app → reconnects automatically | YES | likely; the hotspot-host phone browsing Bonjour is [unverified], rehearse | YES; iOS "no internet" Wi-Fi handling [unverified] |
| Live camera | YES | likely (same caveat) | YES |
| Stop / resume, switch mode | YES (via backend) | YES | YES |
| Ask Jupiter (custom rule) | YES with Ollama on the Mac; otherwise mode suggestion or refusal | YES (same) | YES, local model, no internet needed |
| Event history and explanations | YES | YES | YES |
| Push notification (ntfy) / SMS | YES if the Mac has internet and ntfy is set up | YES (cellular internet) | **NO**: in-app notifications only, app in the foreground |
| BLE Wi-Fi setup | YES (signed in) | YES | YES (signed in) |
| Unplug the Mac mid-demo | device keeps alerting; app shows the server unreachable; the open live view keeps playing; commands time out truthfully | same | same |

Do **not** demo:

* direct-AP mode (not built);
* a cold launch of the app with the Mac off (empty device list, C1);
* video away from the device's network (by design: no relay).
