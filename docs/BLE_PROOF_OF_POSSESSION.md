# Bluetooth provisioning: proof of possession (M6.5, 4B)

Closes the "BLE provisioning is unauthenticated" production blocker in
`docs/SECURITY_REVIEW_M6.md` for the **credential write**, behind config. Status:
**software-ready, simulator/vector-verified, hardware-pending**. Nothing here was run
over a physical Bluetooth link.

## What changes

| | `auth: legacy` (default today) | `auth: pop` |
|---|---|---|
| Who can hand the device a network | anyone in range while the window is open | only a phone that proves it holds the device's setup code |
| Credentials on the air | cleartext JSON on 5d1e0003 | AES-256-GCM, per-session key, on 5d1e0007 |
| 5d1e0003 (plain write) | accepted | refused (logged) |
| Phone checks the device | no | yes: the device proves the code back before anything secret is sent |
| `deployment: production` | **refused** (schema error; the window stays shut) | allowed |

The setup code is 12 Crockford base32 symbols (60 bits), printed as `XXXX-XXXX-XXXX`
and as a QR code carrying `SDSETUP:1:<device id>:<code>`. The app connects only to
the device whose device-info `id` matches the label. Being discoverable proves nothing.

## Protocol v1

The full description is in `services/ble_pop.py`. It uses PBKDF2-HMAC-SHA256 (600 000
iterations, salted with the device id) to derive the code key, HKDF-SHA256 over both
nonces for the session keys, HMAC-SHA256 proofs over a transcript (version, device id,
both nonces), and AES-256-GCM for the credentials. Established primitives only:
`cryptography` on the device, CryptoKit and CommonCrypto on iOS.

The device-side state machine is explicit: idle → advertising → challenged → proven
→ credentials_received → applied/failed. `TRANSITIONS` lists every legal move; any
other move raises.

- **Timeouts:** challenge 30 s, proof 120 s, join outcome 120 s.
- **Retries:** a wrong code spends its challenge, so each challenge allows one guess.
- **Lockout:** 5 failures lock the device out for 30 s, doubling per further failure up
  to 1 h. The lockout is held by the device, so reconnecting does not reset it.

## Device setup (factory / bring-up, on the device as the service user)

```
python scripts/ble_setup_code.py generate --out /etc/sensordyme/ble_setup_code --device-id <SD-…>
```

- The command writes the file atomically with mode 0600 and prints the label text and
  QR payload. Send that output to the label printer only, never to tickets or logs.
- It refuses to overwrite an existing code. There is no rotation command.
- The device refuses a code file that is group- or world-readable, a symlink, or owned
  by another user, and keeps Bluetooth shut until the file is fixed (health shows why).

Config (`network.provisioning`):

```yaml
auth: pop
setup_code_path: /etc/sensordyme/ble_setup_code
```

Optionally also set top-level `deployment: production`, which makes `auth: legacy` a
schema error.

## Rollout order (do not flip a device before the app can follow)

1. Ship an app build whose setup screen takes the label and uses
   `BLEProofOfPossession.Session` over characteristic 5d1e0007.
   - Done in software (M8, see "First-time onboarding" below): the app reads the
     device info `"auth"` and asks for the setup code (typed, or the QR text
     pasted) only for a `pop` device. Not yet run on a phone against a Pi.
2. Generate and print the codes for existing units.
3. Set `auth: pop` per device, then verify on hardware:
   - a wrong code is refused and the lockout engages;
   - a plain write to 5d1e0003 is refused;
   - a correct code joins.
4. Set `deployment: production`.

Rollback: set `auth: legacy` and leave `deployment` unset (development). No data
migration is involved.

## Verification

- Python: `pytest tests/test_ble_pop.py`. It covers the protocol, replay, tampering,
  lockout, timeouts, every illegal transition, the config gate, GATT refusal, the
  provisioning service end-to-end with a fake radio, and no secrets in logs. It also
  checks the shared vectors against hashlib/hmac.
- Swift: `ios/Tests/run.sh` (or concatenate `ios/Sources/BLEProofOfPossession.swift`
  and `ios/Tests/BLEProofOfPossessionTests.swift` and run with `swift`). This checks
  the same vectors (`tests/ble_pop_vectors.json`, embedded verbatim) byte for byte.
- To regenerate the vectors, run `python scripts/ble_setup_code.py vectors --out
  tests/ble_pop_vectors.json`, then re-embed them in the Swift test. The Python test
  fails until both copies agree.

## Known limits

- **Offline guessing is still possible.** Someone who records a session, or runs a fake
  device to collect a proof, can try codes offline. The 60-bit code combined with
  600 000 PBKDF2 iterations is the defence. A PAKE (SPAKE2/OPAQUE) would remove the
  attack, but no vetted library is available on both CryptoKit and the device.
- **Lockout denial of service.** A stranger in range can trigger the lockout, which
  costs the owner at most an hour, or a power cycle (the counter is in memory).
- **Link-layer pairing is still BLE "Just Works".** The protection is at the
  application layer.
- **Dependency.** `cryptography` must be on the device (`python3-cryptography` on
  Raspberry Pi OS, or pinned in `requirements-device.txt`; that file is the release
  stream's). Without it, `auth: pop` fails closed: Bluetooth stays shut.
- **A setup-code file that cannot be read** (mode 0000, not ASCII, a directory) is a
  `SetupCodeError` like every other bad file: Bluetooth stays shut, health says
  "setup code unusable: …", and the 30 s retry picks up a repaired file. (Before
  M8 these escaped as PermissionError/UnicodeDecodeError and crashed the
  provisioning thread; found by the security review.)
- **Other open items.** `scripts/ble_provision_client.py` (the bleak test client) still
  speaks only legacy. The persistent device id in the BLE name and the PSK in nmcli
  argv remain as documented in the security review.

## First-time onboarding (M8, Checkpoint D)

The app's journey, on the protocol above and nothing else. Status:
**software-ready, verified against the device's own Python code over a simulated
link; not run on a phone or a Pi.**

    find -> connect -> prove the setup code -> choose Wi-Fi -> send (encrypted)
         -> the device joins -> add it to the account (claim) -> it reports in
         -> choose a monitoring mode -> done

### States and transitions (`ios/Sources/SetupOnboarding.swift`)

`Onboarding.next(state, event)` lists every legal move. Anything else is refused
and leaves the state as it was, the same discipline as the device side's
`TRANSITIONS`.

| State | Event | Next |
|---|---|---|
| discovering | deviceChosen | connecting |
| connecting / identifying | identified(requiresCode) | enteringSetupCode (pop) / choosingNetwork (legacy) |
| connecting / identifying | linkFailed | failed(connect: search again) |
| discovering / connecting / identifying | bluetoothOff | bluetoothUnavailable (bluetoothOn -> discovering) |
| enteringSetupCode | proofStarted | proving |
| proving | proven | choosingNetwork |
| proving | proofRefused(message, lockedUntil) | enteringSetupCode with the device's reason and wait |
| enteringSetupCode / proving | linkFailed (wrong device, device cannot prove the code, drop) | failed(prove: search again) |
| choosingNetwork | networkSent | sending |
| sending | deviceJoining | joining |
| sending / joining | deviceJoined | associating |
| sending / joining | joinRefused(reason) | choosingNetwork(reason), link kept |
| sending | proofRefused | enteringSetupCode (a lapsed proof that could not be renewed) |
| sending / joining | linkFailed | failed(join: check online / search again) |
| associating | alreadyOnAccount | waitingForDevice (no claim) |
| associating | needsClaim | enteringClaimCode |
| associating | noAccount(found) | finished (Wi-Fi only) / failed(online: keep waiting) |
| enteringClaimCode -> claiming | claimRefused / claimed | enteringClaimCode(reason) / waitingForDevice |
| waitingForDevice | deviceOnline / deviceSilent / claimRefused (the claim added another unit) | choosingMode / failed(online) / failed(account) |
| choosingMode | modeChosen / modeSkipped | activatingMode / finished |
| activatingMode | modeActivated / modeRefused | finished / failed(mode: choose again, skip) |
| failed | recover(r) for an offered r; searchAgain always | per recovery |

What the phone does underneath (`BLEProvisioner`, `BLEPoPFlow`):

- **Prove first, then send.** The proof runs before the network list, so a wrong
  code or a lockout is reported before anyone types a Wi-Fi password. A wrong
  code spends the challenge, and the link stays up for the next try.
- **Lockout, truthfully.** The device's `retry_after_s` (on the failure that starts
  the lockout and on every `locked`) is stored per device id as "accepts codes
  again at", counted down on screen, and the Continue button waits for it. The
  number is the device's; the app never invents one. A power cycle clears the
  device's counter, so the device stays the authority.
- **Renewal.** Credentials go on the proven session. If the device says the proof
  lapsed (`unexpected`, after 120 s) or the previous join failed, the app runs one
  fresh handshake with the same code key and new nonces, then sends them.
- **Refusals that end the link.** A challenge naming another device, a device
  proof that does not verify, or an unreadable reply: the app disconnects and
  sends nothing.
- **A QR label for another device** is refused on the phone and nothing is written.

### How the claim code reaches the app (decision)

**The claim code stays a separate code, entered from the card in the box.** This
is the existing mechanism: `POST /v1/devices/claim`, issued by `cloud/manage.py
create-device`. It is not read over Bluetooth and not added to the setup-code QR.

- The device does not hold its claim code. The backend generates it at the factory
  step and prints it, and the device holds only its device key. Sending it over
  BLE would need a factory change, a new device file and a new protocol message.
- The two codes prove different things to different parties. The setup code
  proves physical possession *to the device*, and the device proves it back. The
  claim code proves entitlement *to the account*. Folding the claim code into
  the QR (`SDSETUP:1:<id>:<code>`) would change the label format that the parser
  and the v1 vectors fix.
- The app links the claim to the unit that was just set up. It matches the
  account row by the SD- id the device reported over Bluetooth (its health
  `device_id` → `hardware_id`), never by name or "the newest". A claim code that
  adds a different unit is reported, naming both ids. If the account already has
  the unit (moving a device to a new network), the app does not ask for a code.
- **No backend change was made.** Follow-ups worth considering (SECURITY stream):
  print both codes on one label, and have `POST /v1/devices/claim` return the
  row's `hardware_id` so the app can confirm the match without waiting for the
  device to report.

### Simulated Bluetooth (development only)

The iOS Simulator has no Bluetooth. `sim/ble_peripheral.py` runs the device's
real `ProvisioningService`, `PopAuthenticator` and `BLEProvisioningServer` with an
in-memory GATT table, ATT-MTU chunking and a simulated Wi-Fi radio, listening on
**127.0.0.1 only**. The D-Bus characteristics now call the server's
`gatt_read`/`gatt_write`, so the simulator and the Pi serve the same bytes. The
app reaches the simulator through `BLESimulatedTransport`, which exists only in
DEBUG builds and only when launched with `-orynSimulatedBLE`. A Release build has
CoreBluetooth and nothing else.

```
python -m sim.backend --port 8787                                  # prints factory.json
python -m sim.ble_peripheral --factory <dir>/factory.json --start-device \
    --backend http://127.0.0.1:8787 --print-label                  # label = what you type
# then launch a Debug build in the Simulator with:  -orynSimulatedBLE 127.0.0.1:8790
```

Options: `--auth legacy`, `--online SSID` (an installed device in its reconfigure
window), `--network SSID[:PSK[:SIGNAL]]`, `--max-failures`, `--lockout-base`,
`--challenge-ttl`, `--proven-ttl`, `--transcript FILE` (redacted).

### Verification

- The device side over the link: `pytest tests/test_m8_onboarding_peripheral.py`.
- The app's Swift code against the device's Python code, message by message:
  `pytest tests/test_m8_onboarding_crosslang.py` or `python -m sim.ble_harness --all`.
  Scenarios: success, wrong code, lockout, another device's label, impostor id,
  tampered device proof, tampered credentials, late proof, expired proof, refused
  password.
- First-time virtual pairing end to end, with evidence:
  `python scripts/ble_onboarding_e2e.py` (also `pytest tests/test_m8_onboarding_e2e.py`).
- Swift logic: `ios/Tests/run.sh` (SetupOnboardingTests, SetupFlowTests,
  BLEProvisioningTests, BLEPoPFlowTests).

### On the hardware (Monday)

1. On the Pi, as the service user: run `python scripts/ble_setup_code.py generate
   --out /etc/sensordyme/ble_setup_code --device-id <SD-…>` and keep the printed
   label. Set `network.provisioning.auth: pop` and restart. Health must not say
   "setup code unusable". Leave `deployment` unset.
2. On the iPhone (a build launched **without** `-orynSimulatedBLE`): Devices → + →
   Set up over Bluetooth. The device must appear as `SensorDyme-XXXX` within about
   10 s of an offline boot. Record the time.
3. Connect. The debug panel must show its SD- id and "setup code". Enter a wrong
   code once: the app says "The setup code was wrong" and the link stays up.
   Record the Pi journal line; it must not contain the code.
4. Lockout: enter 5 wrong codes. The 5th says "refuse for N s", and the countdown
   shows the device's N (30 s). Try the right code during it: refused as locked.
   After it: proven. A power cycle resets the counter (known limit).
5. Paste the QR text of a different device's label: the phone refuses it, and the
   Pi journal shows no auth write.
6. Enter the right code, then pick a network from the list (the Pi's scan) and
   give a wrong Wi-Fi password. The device's reason shows and the list returns
   without reconnecting. Then give the right one: "Device joins", and Bluetooth
   closes about 10 s later (outcome linger).
7. With a plain BLE tool (nRF Connect), write `{"ssid":"x","psk":"y"}` to 5d1e0003.
   It is acknowledged and refused (journal), and nothing joins.
8. Claim: enter a wrong claim code (refused and explained), then the card's. The
   app waits for the device to report, which needs the Pi's uplink to reach the
   backend, then offers modes. Pick one; the device list shows it in that mode.
9. Move it: reboot the Pi (5-minute reconfigure window) and run setup again from
   the app, to a phone hotspot. No claim is asked, and the device row keeps its id.
10. Measure and record: the BLE MTU iOS negotiated (journal `WriteValue` offsets),
    the time from Continue to "proven" (PBKDF2 on the phone plus round trips), and
    whether any notification was lost (status polling covers it, but say so).
