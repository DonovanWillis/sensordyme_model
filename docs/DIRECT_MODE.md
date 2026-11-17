# Direct mode: the device's own Wi-Fi access point (M12, device side)

In direct mode the device runs its own WPA access point, `SensorDyme-XXXX`. A
phone joins it with the Wi-Fi password printed on the device label and reaches
the dashboard (port 8899, still gated by the device token), with no other
network. The Pi 5 has one Wi-Fi radio, so while the access point is up the
device is on no client network and does not reach the backend.

This document covers the device half of `docs/PORTABLE_OPERATION.md` §3.3
(Device items 1-4) and gaps G5-G8 and G11. The app half (G1-G4, G9, G10) is
not here.

**Status: built and unit-tested on a Mac against a fake NetworkManager. Never
run on a Pi.** Nothing below is hardware evidence. Each claim is labelled:

* **[test]**: a test in this repository pins it. The test is named.
* **[unverified]**: NetworkManager, kernel, driver or iOS behaviour that no test
  here can check. Each one is a hardware gate (§8).

Direct mode is **off by default** (`network.direct_mode.enabled: false`). When
it is off, the runtime builds no manager, no service and no health component,
and passes nothing to provisioning, so a device runs what it ran before.
[test: `tests/test_direct_mode_runtime.py`, `tests/test_direct_mode_config.py::test_a_disabled_block_validates_exactly_like_no_block`]

---

## 1. Pieces

| Piece | File | What it does |
|---|---|---|
| Manager | `core/direct_mode.py` `DirectModeManager` | enter, leave, status, the fallback policy and the boot check |
| nmcli primitives | `core/network.py` | still the only module that runs nmcli: `ap_profile_args`, `ap_profile_problem`, `write_ap_profile`, `activate_profile`, `deactivate_profile`, `interface_ipv4`, `client_networks`, `profile_settings`, `set_autoconnect` |
| Label passphrase | `core/direct_mode.py` (`generate_psk`, `write_psk`, `load_psk`), `scripts/direct_mode_psk.py` | the per-device Wi-Fi password, stored 0600 |
| AP firewall | `core/ap_firewall.py` (ruleset text), `deploy/networkmanager/90-sensordyme-direct` (root dispatcher script) | only DHCP, DNS, mDNS and the dashboard on the AP interface |
| polkit grant | `deploy/polkit/11-sensordyme-direct-mode.rules` | `wifi.share.protected` for the netdev group. Never `wifi.share.open` |
| Service | `services/direct_mode.py` `DirectModeService` | calls `tick()` every 5 s; supervised; health component `direct_mode` |
| Provisioning hook | `services/provisioning.py` (`direct_mode_active`) | R1: Bluetooth setup stays open while direct mode is active |
| Operator CLI | `scripts/direct_mode.py` | `status`, `enter`, `leave`, `install`, `ruleset`; `--dry-run` |
| Install | `setup_pi.sh` (marked block) | runs `direct_mode.py install` only when direct mode is enabled |
| Config | `config.yaml` `network.direct_mode:`, `core/config_schema.py` | all defaults off |

### 1.1 Public interface (for the local API integration)

```python
from core.direct_mode import DirectModeManager

manager = DirectModeManager(wifi: WifiManager, config: dict | None)  # config = network.direct_mode
manager.status() -> dict        # JSON-safe, never the passphrase
manager.enter(reason: str, *, by: str = "explicit") -> tuple[bool, str]
manager.leave(reason: str) -> tuple[bool, str]
manager.tick(now: float | None = None) -> None   # monotonic seconds; never raises
manager.is_active() -> bool                      # for provisioning (R1)
manager.unavailable_reason() -> str | None
manager.plan_enter() / plan_leave() -> list[str] # the --dry-run text, passphrase redacted
```

`status()` returns exactly these keys
[test: `tests/test_direct_mode.py::test_the_status_is_json_with_exactly_the_agreed_keys`]:

| Key | Value |
|---|---|
| `mode` | `client`, `direct`, `entering`, `leaving` or `unavailable` |
| `ssid` | always the access point's name, `SensorDyme-XXXX` (what a phone would join), in every mode. It is broadcast, so it is not secret |
| `since` | wall-clock epoch seconds when the current mode began, when this process knows it (from the state file for `direct`; after a `leave` for `client`), else `null` |
| `last_error` | the last failure, or, in `unavailable`, why direct mode cannot be entered |
| `fallback_enabled` | `enabled and fallback_enabled` |

Differences from the interface in the brief:

* `enter` has an extra keyword `by` (`"explicit"` or `"fallback"`). Callers can
  leave it out.
* `ssid` is the AP's name in every mode, not the client network's name.
* `unavailable` means direct mode **cannot be entered** (disabled, no
  NetworkManager, firewall not installed, or the passphrase file missing or
  loosely permissioned). If the AP is up anyway, `mode` is `direct`:
  NetworkManager is the truth.
* `enter` blocks for up to `ap_timeout_s` plus at most 45 s of rollback, and
  `leave` for up to 45 s. A local API should run them off the request thread.
  Over the home network, the phone loses its connection to the device the
  moment `enter` switches the radio, so the API should answer (202) **before**
  calling `enter`.
* Only one change runs at a time. A second `enter`/`leave` during a change
  returns `(False, "a direct-mode change is already in progress")` at once
  [test: `::test_only_one_change_runs_at_a_time`].

`manager.is_active()` asks NetworkManager whether the profile
`sensordyme-direct` is the active Wi-Fi connection. Because of that, the
service and the CLI (two processes) agree [test:
`::test_a_direct_mode_started_by_another_process_is_seen_and_left`].

---

## 2. How it works

### 2.1 Enter (transactional, R2)

1. Refuse before touching anything: when it is disabled, when there is no
   nmcli, when the firewall files are missing (`/etc/sensordyme/direct_mode.nft`
   or the dispatcher script), or when the passphrase file is unusable. Nothing is
   run [test: `::test_enter_refuses_and_touches_nothing_when_it_cannot_be_safe`].
2. Record the active client profile ("previous").
3. Delete any existing `sensordyme-direct` profile and create it fresh:

   ```
   nmcli connection add type wifi con-name sensordyme-direct ifname wlan0 ssid SensorDyme-XXXX \
     connection.autoconnect no 802-11-wireless.mode ap 802-11-wireless.band bg \
     802-11-wireless.channel 6 ipv4.method shared ipv4.addresses 10.42.0.1/24 ipv6.method disabled \
     wifi-sec.key-mgmt wpa-psk wifi-sec.proto rsn wifi-sec.pairwise ccmp wifi-sec.group ccmp \
     wifi-sec.psk <label passphrase>
   ```

   A profile someone edited by hand is replaced, never trusted
   [test: `::test_a_hand_edited_profile_is_replaced_not_trusted`].
4. Read it back (`connection.autoconnect`, `802-11-wireless.mode`,
   `802-11-wireless-security.key-mgmt`, `...proto`, `ipv4.method`). Anything but
   `no / ap / wpa-psk (or sae) / rsn / shared` deletes the profile and refuses.
   The client connection has not been touched yet
   [test: `::test_a_profile_that_reads_back_open_is_deleted_and_never_brought_up`].
5. `nmcli connection up id sensordyme-direct`.
6. Wait up to `ap_timeout_s` (30 s) for all three: the profile active,
   `10.42.0.1/24` on `wlan0`, and the firewall marker
   `/run/sensordyme-direct/firewall-active` (§3.3).
7. If any is missing at the deadline: `connection down` the AP and
   `connection up` the previous profile. The answer says which part failed and
   where the device went. [tests: `::test_an_access_point_that_does_not_come_up_rolls_back_to_the_previous_network`,
   `::test_an_access_point_without_its_firewall_is_rolled_back`,
   `::test_an_access_point_without_an_address_is_rolled_back`,
   `::test_a_device_with_no_previous_network_says_so_on_rollback`]

The state file (`state_path`, default `data/direct_mode.json`) records
`{mode, previous, since, reason, by}`. It never holds the passphrase.

### 2.2 Leave

`connection down id sensordyme-direct`. Then `connection up` the recorded
previous profile if it is still saved. If it was forgotten or is out of range,
NetworkManager's own autoconnect takes over, and Bluetooth setup opens while
the device is offline. [tests: `::test_leave_restores_the_saved_client_network`,
`::test_leave_when_the_previous_network_was_forgotten_leaves_it_to_networkmanager`,
`::test_leave_when_the_previous_network_is_out_of_range_still_stops_the_ap`]

A Bluetooth (or `/setup/wifi`) credential write while in direct mode also leaves
it. It goes through `WifiManager.provision()`, which brings up the new network.
If the new network refuses, `provision()` rolls back to the AP, its "previous".
[unverified on hardware; the code path is `core/network.py` `provision()`]

### 2.3 Power cycle (R3)

* The profile is always written with `connection.autoconnect no`, so
  NetworkManager never raises it at boot and saved client networks get the
  radio [test: `::test_a_power_cycle_comes_back_to_the_client_network`].
* On the first tick, a profile whose autoconnect was turned on by hand is set
  back to `no`. If that profile came up at boot because of it, it is taken down
  [test: `::test_a_profile_made_sticky_by_hand_is_unstuck_and_left_at_boot`].
* An AP that is up with autoconnect off was started on purpose in this boot (by
  the CLI, or before a service restart). It is left alone
  [test: `::test_a_direct_mode_started_on_purpose_survives_a_service_restart`].
* There is no setting that makes direct mode the boot default. The only
  automatic entry is the fallback below, and it waits `fallback_after_s` after
  every boot.

### 2.4 Fallback policy (off by default)

With `fallback_enabled: true`:

* The device enters direct mode (`by: fallback`) after `fallback_after_s`
  (600 s) with no connectivity at all.
* While in a fallback direct mode, every `return_check_s` (300 s) it scans. If a
  saved client network is visible, it leaves direct mode for that network.
* A failed start is retried only after another full interval.
* An explicit direct mode (CLI or API) is never left by the policy.

[tests: `::test_the_fallback_is_off_by_default`,
`::test_the_fallback_enters_after_the_offline_interval_and_leaves_when_home_returns`,
`::test_the_fallback_never_fires_while_online`,
`::test_an_explicit_direct_mode_is_not_left_by_the_fallback_policy`,
`::test_a_failed_fallback_start_waits_a_full_interval_before_retrying`,
`::test_a_disabled_manager_runs_no_nmcli_on_tick`]

Whether the Pi's radio can scan while it is an access point is [unverified]. If
it cannot, the scan finds nothing and the AP stays up until a command, a
Bluetooth hand-off or a power cycle.

---

## 3. Security model

| Layer | Control | Evidence |
|---|---|---|
| Radio | WPA2-PSK, RSN only, CCMP only (`security: wpa2`, the default), or WPA3-SAE with PMF required (`security: wpa3`) | [test: `::test_the_access_point_profile_is_wpa2_ccmp_shared_and_never_autoconnects`, `::test_wpa3_is_sae_with_management_frame_protection`] |
| Never open | four separate places: the profile builder has no open option and refuses a missing or short passphrase; the read-back refuses anything but WPA; the schema has no open value; the polkit grant is `wifi.share.protected` only | [test: `::test_no_code_path_can_build_an_open_or_unusable_access_point`, `::test_a_profile_that_reads_back_open_is_deleted_and_never_brought_up`, `tests/test_direct_mode_config.py::test_bad_values_are_errors_naming_the_key`, `tests/test_direct_mode_cli.py::test_the_polkit_grant_is_protected_sharing_only`] |
| Passphrase | random per device, from the label (§3.1) | §3.1 |
| Network | AP-interface firewall: DHCP, DNS, mDNS, TCP 8899 only; no SSH; no forwarding (§3.2) | [test: `tests/test_direct_mode_firewall.py`] |
| Application | the dashboard token, unchanged (`core/dashboard.py`, fails closed on a LAN bind with no token) | existing tests; the schema warns when direct mode is on and no token is set |
| Recovery | Bluetooth proof of possession stays open while direct (R1) | §4 |

### 3.1 The label passphrase (G6)

* Generated by `secrets.choice`: 16 symbols from
  `23456789abcdefghjkmnpqrstuvwxyz` (lower case, no `0 o 1 l i`), printed as
  `xxxx-xxxx-xxxx-xxxx`, about 79 bits. It is never derived from the MAC, the
  serial or the device ID [test: `::test_a_generated_passphrase_is_random_long_and_typable`,
  `::test_the_passphrase_is_not_derived_from_the_board`].
* Written once, atomically, mode 0600. `write_psk` refuses to replace an
  existing file. `load_psk` refuses a symlink, a non-regular file, any
  group/other permission bit, another owner, or anything that is not the
  generated form, and never echoes the content
  [test: `::test_the_passphrase_file_is_written_once_0600_and_loaded_only_if_private`,
  `::test_a_file_that_is_not_a_generated_passphrase_is_refused_without_echoing_it`].
  These are the same rules as the BLE setup code (`services/ble_pop.py`).
* Never logged, never in `status()`, never in the state file, redacted in
  `--dry-run` [test: `::test_the_passphrase_is_never_logged_or_reported`]. The
  passphrase is in nmcli's argv for the one `connection add` call. That matches
  `provision()` today; on the Pi only local accounts could read it, and only
  for that moment.
* Not settable in `config.yaml`: a `psk`/`password`/`passphrase` key is a schema
  error [test: `tests/test_direct_mode_config.py::test_a_passphrase_in_config_yaml_is_refused`].
* Factory step, on the device as the service user:

  ```
  .venv/bin/python scripts/direct_mode_psk.py generate --out /etc/sensordyme/direct_mode_psk
  ```

  It prints, for the label printer only:

  ```
  SensorDyme SD-XXXXXXXX
  Direct Wi-Fi: SensorDyme-XXXX
  Wi-Fi password: xxxx-xxxx-xxxx-xxxx
  Wi-Fi QR: WIFI:T:WPA;S:SensorDyme-XXXX;P:xxxx-xxxx-xxxx-xxxx;;
  ```

  `label --psk-file …` reprints it. Print it beside the BLE setup code.
  [test: `tests/test_direct_mode_cli.py::test_generate_writes_once_0600_and_prints_the_label`]

### 3.2 The AP-interface firewall (G7)

The service runs unprivileged (no capabilities under the hardened unit), so it
cannot load nftables rules. A **root NetworkManager dispatcher script** loads
them instead (`deploy/networkmanager/90-sensordyme-direct`, POSIX sh):

* It acts only on `CONNECTION_ID=sensordyme-direct`.
* On `pre-up` and `up`: `nft -f /etc/sensordyme/direct_mode.nft`, then it
  writes `/run/sensordyme-direct/firewall-active`.
* If nft fails: it removes the marker, runs `nmcli connection down id
  sensordyme-direct` in the background, and exits 1.
* On `pre-down` and `down`: it removes the marker and runs `nft delete table
  inet sensordyme_direct`.

[test: `tests/test_direct_mode_firewall.py`, with fake `nft`/`nmcli`/`logger`]

The manager does not report the AP as up until the marker exists. Without the
marker it rolls back. Fail closed, twice.

The ruleset (exact text pinned by
`tests/test_direct_mode_firewall.py::test_the_ruleset_is_exactly_the_reviewed_text`;
print it with `scripts/direct_mode.py ruleset`):

* Table `inet sensordyme_direct`, replaced atomically on every load.
* Input chain, priority -10, policy accept for every other interface. On
  `wlan0`: established/related accepted, invalid dropped. DHCP (UDP 68→67), DNS
  (UDP and TCP 53), mDNS (UDP 5353) and TCP `dashboard.port` are accepted.
  Everything else gets `counter drop`, SSH included.
* Forward chain: everything from or to `wlan0` is dropped, so AP clients are
  never routed anywhere, including to Ethernet.
* A drop in this table is final whatever NetworkManager's own shared-mode table
  accepts. That is nftables semantics for multiple base chains on one hook.

`sudo scripts/direct_mode.py install` renders the ruleset for the configured
interface and dashboard port. It checks the ruleset with `nft -c -f` before
replacing anything (when nft exists), then installs the ruleset (0644), the
dispatcher (0755) with a `pre-up.d/` symlink, and the polkit rule (0644), all
owned by root. `setup_pi.sh` runs it only when `network.direct_mode.enabled:
true` [test: `tests/test_direct_mode_setup.py`, `tests/test_direct_mode_cli.py::test_install_writes_the_ruleset_the_dispatcher_and_its_pre_up_link`].

### 3.3 Known weaknesses

* **There may be a short window before the firewall loads.** NetworkManager
  waits for `pre-up` scripts before it finishes activating, but it may already
  be beaconing and serving DHCP during activation. A client would still need
  the passphrase. [unverified; R4 step 7 measures it]
* **Plain HTTP inside the WPA link.** Anyone holding the passphrase can sniff the
  dashboard token. This is acceptable only because the passphrase is per device
  and printed on the device (PORTABLE_OPERATION §3.4).
* **One shared dashboard token per device.** Unchanged.
* **The polkit grant.** With direct mode installed, the netdev group may raise
  protected shared Wi-Fi connections.
* **The PSK is in argv during `connection add`** (§3.1).

---

## 4. Recovery requirements

* **R1. Bluetooth (PoP) stays open whenever direct mode is active.** In AP mode
  NetworkManager reports `connected (local only)` (or `site only`), and
  `WifiManager.is_online()` counts that as online. With direct mode enabled, the
  runtime hands `ProvisioningService` the manager's `is_active`. The provisioning
  policy then reads "online and not direct", so the window opens past the
  reconfigure window and stays open. An exception from the check also keeps it
  open. [tests: `tests/test_provisioning.py::test_bluetooth_setup_stays_open_for_as_long_as_direct_mode_is_active`,
  `::test_the_same_connectivity_without_direct_mode_still_closes_the_window`,
  `::test_leaving_direct_mode_for_a_working_network_closes_the_window`,
  `::test_an_unknown_direct_mode_state_keeps_the_recovery_path_open`,
  `::test_direct_mode_on_networkmanager_reads_offline_for_provisioning`]
  The check uses the active profile's name, not NetworkManager's state string,
  so it does not depend on which `connected (...)` string NetworkManager prints
  in AP mode.
* **R2. Entering is transactional.** See §2.1.
* **R3. A power cycle tries saved client networks first.** See §2.3.
* **R4. Hardware session.** See §6. No step may run on the device before the R1-R3
  tests pass on the commit being installed.

---

## 5. Configuration (`network.direct_mode:`)

| Key | Default | Notes |
|---|---|---|
| `enabled` | `false` | only YAML `true` enables it |
| `security` | `wpa2` | `wpa2` or `wpa3`. There is no open value |
| `channel` | `6` | 2.4 GHz, 1-11 (band `bg`) |
| `ap_timeout_s` | `30` | 5-120 s |
| `passphrase_path` | `/etc/sensordyme/direct_mode_psk` | the 0600 label file. Named so the release gate's credential scan does not read it as a secret |
| `state_path` | `data/direct_mode.json` | relative to the checkout (the service's WorkingDirectory) |
| `interface` | `wlan0` | |
| `fallback_enabled` | `false` | |
| `fallback_after_s` | `600` | 60-86400 s |
| `return_check_s` | `300` | 30-86400 s |

The AP address is fixed at `10.42.0.1/24`, NetworkManager's shared-mode
default, which the app's legacy hotspot path also uses. The schema warns about
an unknown key, about direct mode enabled with `provisioning.method` other than
`ble` (no Bluetooth recovery path), and about direct mode enabled with no
`dashboard.token` (every request would get 401).

**State file under the hardened unit.** `deploy/systemd/sensordyme.service.hardened`
bind-mounts `/var/lib/sensordyme` over `data/`. A CLI run outside the unit sees
a different `data/`. To share the state file, pass a config whose
`state_path` is `/var/lib/sensordyme/direct_mode.json`. If they disagree,
nothing unsafe follows: NetworkManager is the truth for `mode`. Only `since`
and the "previous network" for `leave` are lost, and NetworkManager's
autoconnect then picks the network.

---

## 6. R4 hardware session (Pi + iPhone)

Log every step as evidence: the command, its output, `journalctl -u sensordyme
-u NetworkManager --since <start>`, and `scripts/direct_mode.py status`.
Prerequisites:

* The R1-R3 tests pass on the commit being installed.
* `network.provisioning.auth: pop` with a setup code.
* `dashboard.token` set and pasted in the app.

Steps:

0. **Setup.** As the service user, run `scripts/direct_mode_psk.py generate`
   and keep the printed label. Set `network.direct_mode.enabled: true`, run
   `sudo .venv/bin/python scripts/direct_mode.py install`, and restart the
   service. Check:
   * `sudo nft -c -f /etc/sensordyme/direct_mode.nft` succeeds.
   * `ls -l /etc/NetworkManager/dispatcher.d/90-sensordyme-direct
     /etc/NetworkManager/dispatcher.d/pre-up.d/` shows root-owned files.
   * `status` says `client`.
1. **Dry run, then enter.** `sudo -u <user> .venv/bin/python
   scripts/direct_mode.py enter --dry-run`, then `… enter`. Record the time to
   `direct mode on`. Then record:
   * `nmcli -f GENERAL.STATE,IP4.ADDRESS device show wlan0` and
     `nmcli -t -f STATE general`. Record the exact string: the R1 logic does not
     depend on it, but PORTABLE_OPERATION asks for it.
   * `sudo nft list table inet sensordyme_direct`.
   * `ls -l /run/sensordyme-direct/`.
   * `nmcli -t -f 802-11-wireless-security.key-mgmt,802-11-wireless-security.proto connection show id sensordyme-direct`.
2. **Join and discover.** From another device, run a Wi-Fi scan and record
   `SensorDyme-XXXX` and its security (WPA2, or WPA2/WPA3 if NetworkManager
   added SAE). On the iPhone, join by scanning the label's Wi-Fi QR, then again
   by typing the password. Record the "No Internet Connection" banner. Open the
   app and record whether Bonjour finds `_sensordyme._tcp` on the AP.
3. **Stream.** Live view in the app with the token. Record the time to first frame.
4. **Stop and resume locally.** Only once the app's local command client (G2)
   exists. Until then, use `curl -H 'X-Device-Token: …' -X POST
   http://10.42.0.1:8899/api/protection -d '{"enabled": false, "confirm": true}'`
   from a laptop on the AP, then the resume.
5. **Leave.** `… leave`. The device must return to the home network by itself
   (`status` says `client`; the app sees it online) and `nft list tables` must
   no longer show `sensordyme_direct`.
6. **A broken AP, recovered over Bluetooth.**
   * Break it: `sudo mv /etc/sensordyme/direct_mode.nft{,.bak}`, then `enter`.
     It must refuse ("firewall is not installed"). Restore the file.
   * Then make the ruleset unloadable (root ignores file modes, so chmod does
     not do it): `sudo cp /etc/sensordyme/direct_mode.nft /root/nft.bak; echo
     garbage | sudo tee /etc/sensordyme/direct_mode.nft`, then `enter`. The AP
     must be taken down by the dispatcher and rolled back within `ap_timeout_s`
     ("firewall did not load"), with the device back on the home network.
     Restore the file from `/root/nft.bak`.
   * Then enter for real, and from the app (Bluetooth, PoP) hand the device the
     home network while it is in direct mode. It must join, and Bluetooth must
     have been open the whole time (`/api/health` → `provisioning`: "direct
     mode … kept open").
   * Power cycle while in direct mode: the device must come back on the home
     network, not the AP.
7. **Wrong password and no open SSID.**
   * Join with a wrong password: refused.
   * Throughout the session, a scanner (`sudo iw dev <if> scan` on a laptop, or
     `nmcli -f SSID,SECURITY dev wifi` on another Pi) must never show an
     `SensorDyme-XXXX` with empty security.
   * From a laptop on the AP: `nmap -sT -p- 10.42.0.1` must show 8899 open and
     22 closed or filtered. `nmap -sU -p 53,67,5353 10.42.0.1` is optional.
   * Right after `enter`, repeat `nc -zv 10.42.0.1 22` in a loop to look for the
     pre-up window in §3.3.

Any failure at steps 1-7 stops the session. Record it, `leave` (or power
cycle), and set `enabled: false`.

---

## 7. If a Pi loses Wi-Fi (recovery procedure)

In order. Each step needs nothing from the step after it.

1. **Power cycle.** Direct mode is never the boot default (R3), so the device
   tries every saved network. If none answers, it is offline and Bluetooth setup
   opens at once (`grace_s: 0`).
2. **Bluetooth setup from the app** (needs the label's setup code under
   `auth: pop`). Hand it a reachable network, or a phone hotspot. This works
   whether or not direct mode is active (R1).
3. **Direct mode, if enabled.** With `fallback_enabled: true` the AP appears
   after `fallback_after_s`. Otherwise run `enter` from a console. Join with
   the label Wi-Fi password, then open `http://10.42.0.1:8899/setup?token=…` and
   hand it a network. `/setup/wifi` goes through `provision()`, which rolls
   back to the AP if the new network refuses.
4. **Ethernet.** Plug in a cable. The firewall applies to `wlan0` only, so the
   dashboard and SSH are reachable on `eth0` as usual.
5. **Console (HDMI and keyboard) or SSH over Ethernet**, as the service user:
   * `.venv/bin/python scripts/direct_mode.py status`, then `… leave`.
   * Last resort, as root:
     * `nmcli connection down id sensordyme-direct`
     * `nmcli connection delete id sensordyme-direct`
     * `nft delete table inet sensordyme_direct`
     * remove `/etc/NetworkManager/dispatcher.d/90-sensordyme-direct` and its
       `pre-up.d/` link, and `/etc/polkit-1/rules.d/11-sensordyme-direct-mode.rules`.
   * `nmcli device wifi connect <ssid> password <psk>` joins a network by hand.
6. **SD card.** In `config.yaml`, set `network.direct_mode.enabled: false`. If
   the profile file exists, delete
   `/etc/NetworkManager/system-connections/sensordyme-direct.nmconnection`.

---

## 8. What is verified by tests, and what is a hardware gate

Verified on a Mac (fake NetworkManager, fake nft, `/bin/sh` and `bash`):

* the exact nmcli argv for the AP profile, its read-back check, and the
  transactional enter with rollback for every failure point (no `up`, no
  address, no firewall marker, open read-back, no previous network);
* leave, and the restore paths;
* the R3 boot check;
* the fallback policy and its default-off;
* passphrase generation, storage, refusals and redaction;
* the exact ruleset text and the dispatcher's load, unload and fail-closed
  behaviour;
* the R1 provisioning policy over `FakeWifi` and over `WifiManager` driven by
  the fake nmcli;
* the runtime wiring, enabled and disabled;
* the config schema;
* the CLI (`--dry-run` runs only read-only queries) and `install` under a
  scratch root;
* the `setup_pi.sh` block;
* the hardened legacy hotspot script.

Hardware gates (all [unverified]; R4 checks them):

1. That the Pi 5's Wi-Fi (brcmfmac, CYW43455) runs a NetworkManager AP with
   `wpa-psk`/RSN/CCMP on channel 6, and how long it takes. Whether it supports
   AP-mode WPA3-SAE (`security: wpa3`). If it does not, enter rolls back.
2. **WPA2/WPA3 transition mode.** It is not configured explicitly, because
   nmcli has no transition `key-mgmt` for an access point. Some
   NetworkManager/wpa_supplicant versions may add SAE alongside WPA-PSK for a
   `wpa-psk` AP; whether Pi OS Bookworm's does is unverified. Record the
   security a scanner reports (R4 step 2).
3. The terse output format of `nmcli -t -f <fields> connection show id …`
   (`field:value` lines) that the read-back parses. If it differs, enter
   **refuses** (fails closed); it never opens.
4. The exact `nmcli general` STATE string in AP mode. R1 does not depend on it.
5. Dispatcher behaviour:
   * that NetworkManager runs the script through the `pre-up.d/` symlink;
   * that it passes `CONNECTION_ID`;
   * whether a failing `pre-up` script aborts activation (the script takes the
     AP down itself either way);
   * the length of the window before the rules load (§3.3);
   * that `nft -f` of this ruleset loads on Bookworm's nftables (the fleet
     check below ran `nft -c` only where nft existed).
6. That NetworkManager asks polkit for `wifi.share.protected` when activating
   the AP, and that polkitd picks up the new rule without a restart.
7. NetworkManager's `ipv4.method shared` dnsmasq answering DHCP and DNS through
   the firewall. That avahi publishes `_sensordyme._tcp` on the AP interface.
8. Scanning while in AP mode (the fallback's return check).
9. iOS:
   * joining a WPA2 network with no internet, and the "No Internet Connection"
     behaviour;
   * whether iOS keeps routing to the AP or prefers cellular;
   * Local Network permission and Bonjour on the AP;
   * joining from the `WIFI:` QR with the system camera.
10. That a `provision()` hand-off from direct mode, and its rollback to the AP,
    work on hardware.

**Fleet check:** see §10.

---

## 9. Integration notes (for the owners of files this workstream did not edit)

* **Local API** (`core/dashboard.py`): add token-gated
  `GET /api/network/direct` → `manager.status()`, and
  `POST /api/network/direct` `{"action": "enter"|"leave", "confirm": true}`.
  Run the action on a worker thread after answering 202 (§1.1). The runtime
  exposes the manager as `runtime.direct_mode` (None when disabled; then answer
  404 or `{"mode": "unavailable"}`).
* **BLE command** (`services/ble_gatt.py`, `services/provisioning.py`
  `_on_command`): entering direct mode over Bluetooth must be behind proof of
  possession. Today `_on_command` also receives unauthenticated commands
  (`rescan`, `identify`) in legacy mode, so do not add `direct_mode` there
  without a PoP check.
* **App** (`ios/`):
  * `SetupWizardView`'s hotspot step says to join `SensorDyme-XXXX` without
    mentioning a password. The legacy hotspot now uses the label Wi-Fi password
    when one exists, otherwise a random one nobody is shown.
  * G1-G4, G9 and G10 remain app work.
* **Docs:**
  * `docs/PORTABLE_OPERATION.md` §3.1 (the hotspot comment is fixed, and it
    now uses the label password), §3.2 ("BLE recovery while in AP mode: at
    risk" is now addressed when direct mode is enabled), §3.6 (G5-G8 and G11
    built, hardware-unverified), and §4 ("direct-AP mode (not built)").
  * `docs/FRIEND_HARDWARE_SETUP.md`: the factory passphrase step.
  * `README.md`: a pointer here.

---

## 10. Fleet check

One read-only job ran on the Linux worker `oryn` (job `20261115-232412-6eb6bd`,
source `acc6951`, WSL2 kernel 6.18). It changed no networking. Results:

* `dash -n deploy/networkmanager/90-sensordyme-direct` passed. Pi OS runs
  dispatcher scripts with dash.
* `bash -n scripts/provisioning_hotspot.sh` passed under bash 5.2.21.
* The ruleset rendered there byte-identical to the pinned text.
* `nft` and `nmcli` are **not installed** on the worker, so the ruleset was not
  checked by nftables (`nft -c -f`). Installing packages on a worker is not
  allowed. Two checks remain:
  * `sudo nft -c -f /etc/sensordyme/direct_mode.nft` on the Pi (R4 step 0). It
    also runs automatically inside `direct_mode.py install`, which refuses to
    install a ruleset nft rejects.
  * The nmcli argument syntax, on the Pi.
