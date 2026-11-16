# Release record: M12 unified development candidate

**Status:** a software-verified candidate.

* Not pushed, not merged.
* **No hardware gate has passed**: the Raspberry Pi 5 + Hailo-8L and a physical
  iPhone have run none of it.
* It supersedes the M11 candidate `6df82b7` for the first physical install.
  M11 stays a valid baseline for rollback.

| Item | Value |
|---|---|
| Branch | `feat/m12-unified-development` (local; based on the pushed `feat/m11-hardware-activation` @ `92743be`) |
| Release commit (`$RC`) | `0ff8f45388354ed5bcf67569c25d33ba3f06a4ca` |
| Transfer package | `~/sensordyme-release/transfer-0ff8f4538835/` (59 files, 354.6 MB) |
| **sha256 of `SHA256SUMS`** (send this to the technician separately) | `b5b22a45d9ff347cd8528cc74fff60132bfadf22e2d7c6e7e9da4241f2b98f28` |
| Bundle | `bundle/sensordyme-0ff8f4538835.tar.gz`, sha256 `481a35e2c26b021236570375f3ff5dd73d49137634ccd251a6dc0da95c58f8eb` |
| Bundle manifest | `bundle/sensordyme-0ff8f4538835.manifest.json`, sha256 `14539aa29add75665fd4c54576e194f6f7e88082fcbe33306ca7c04dbfa20b88` |
| Production model | `models/damoyolo_tinynasL20_T.hef`, sha256 `e69136d3ebf3cc8ebaf3da5417a59962b139f3d3d67d715e9a9b1ae10963960e` (18,554,430 bytes). Decoder `legacy`. **Unchanged** |

## What M12 adds

The device and the backend:

* **sd-local-v1 local access** (`docs/LOCAL_ACCESS.md`):
  * each phone has its own P-256 key;
  * a phone is enrolled over Bluetooth PoP, through the owner's account (a
    signed relay), or by the holder of the legacy token;
  * the device signs every session, and every write is signed over a
    single-use challenge;
  * revocation ends sessions and open streams at once;
  * a phone idle for 180 days must enrol again.
* **`GET /api/events`**: the local event history, with the backend's
  `event_key` and an honest sync state.
* **Direct mode** (`docs/DIRECT_MODE.md`): the device runs its own Wi-Fi
  access point. **Off by default.**

The iPhone app:

* Paired cameras are cached in the Keychain, with the phone's key in the
  Secure Enclave on a real phone.
* The app reaches the camera directly, so a cold launch works with the
  backend off.
* Video authorisation is automatic: nothing to paste.
* Stop, resume and mode are signed and confirmed by the device.
* Local history is merged with the account's history without duplicates.
* Foreground alerts keep working while the backend is unreachable.
* Bluetooth setup works before sign-in.
* A simulated event never sounds.

Development: `docs/DEVELOPER_SETUP.md` (ASUS for Python and backend, oryn
through `sdrun`, the MacBook for Xcode), `scripts/dev/bootstrap_python.sh` and
`make test-backend`.

**Unchanged:** the production `.hef`, the legacy decoder, the detection
thresholds, the M10.1 driver-association and low-fps fatigue fixes, shadow
isolation, and relapse escalation (still OFF). Nothing experimental was
promoted.

## Verification at `0ff8f45`

| Gate | Result |
|---|---|
| ruff | clean |
| pyright | 0 errors |
| pytest (full) | **4,763 passed**, 6 skipped, 1 strict xfail (the documented escalation decision, OFF), 0 failed |
| Swift logic tests (`make test-ios`) | PASSED: 43 files, 1,955 checks |
| iOS release-configuration audit | PASS |
| `make verify-release REF=HEAD` | **PASS** (the systemd step is SKIP on macOS) |
| `make acceptance-full` | **PASS: 897 / 897**, including the owner-journey matrix (simulated) and **29 / 29 XCUITest journeys in one run**, with no retries. New: `test29_OfflineLocalCamera` (cold launch with the backend down, local live view, a stop confirmed by the device) |
| Transfer package | `TRANSFER PACKAGE OK`; independently re-verified with `shasum -a 256 -c SHA256SUMS`: 59 / 59 OK |
| Off-Mac reproducibility | the backend smoke tests (344) and the full suite on oryn's Linux worker through `sdrun` (`docs/DEVELOPER_SETUP.md`). One failure, caused by the exported job tree having no `.git` |

Evidence is in `~/sensordyme-audit/2026-11-10/m12-evidence/`:

* `pytest-full.txt`, `test-ios.txt`, `verify-release.txt`;
* `acceptance-0ff8f45/`;
* `devenv/`.

## Integrating it on the friend's Pi

**Prerequisite:** `docs/FRIEND_HARDWARE_SETUP.md` steps 1–8 have passed. The
owner has decided between a fresh install and an upgrade.

1. The owner gives the technician the package directory: a USB drive or a
   private share. In chat, the owner sends **only** `$RC` and the `SHA256SUMS`
   hash above.
2. The technician verifies the package on the Pi with `sha256sum -c SHA256SUMS`,
   and checks the sha256 of `SHA256SUMS` against the hash from chat.
3. Install with the package's own `INSTALL.md`, which pins every command for
   this release:
   * §A for a fresh device (`setup_pi.sh`);
   * §B for an adopted device (`release_guard.py stage` → preflight →
     `activate` → `status`).

   Then continue with the friend guide's steps 9–28 as written. Step 24 and the
   new **step 24b** (the camera with the MacBook off) cover M12.
4. **New on first start:** the service creates `data/local_identity.pem` and
   `data/local_clients.json` (both 0600), plus `data/local_events.jsonl`.
   * `data/` is shared by every release, so these survive updates and
     rollback.
   * Check: `curl -s localhost:8899/api/local/hello` answers with
     `"contract": "sd-local-v1"` and a challenge, and with no secret.
5. **Leave direct mode off** (`network.direct_mode.enabled: false`, the
   default). Its hardware session (`docs/DIRECT_MODE.md` §6) runs only after
   the owner approves it, with SSH or console recovery access confirmed. It
   changes the Pi's networking.
6. **Rollback:** `release_guard.py switch --to previous` (INSTALL.md
   "Rollback"). The M11 release ignores the new `data/` files.

## Installing the app on the owner's iPhone

On the MacBook (README Phase 5 has the full text):

1. Xcode → Settings → Accounts: sign in with the Apple ID of the personal team.
2. Connect the iPhone by cable, unlock it, then tap **Trust This Computer**.
3. Turn on Developer Mode on the phone (Settings → Privacy & Security). The
   phone restarts.
4. Open `ios/SensorDyme.xcodeproj` (after `cd ios && xcodegen generate`). Pick
   the iPhone and press **Run**. Or use the `xcodebuild … -allowProvisioningUpdates`
   and `xcrun devicectl device install app` commands in README Phase 5.
5. On the phone, trust the developer certificate (Settings → General → VPN &
   Device Management).
6. Allow Local Network, Bluetooth and Notifications when the app asks.
7. Pair (README Phase 6): Bluetooth setup with the setup code, then add the
   camera with the claim code. The iPhone enrols itself with the camera; there
   is nothing to paste.

Notes:

* The **free personal team** signature lasts 7 days. Reinstall before any demo.
* The Secure Enclave key needs no extra entitlement, so free signing still
  works.

## What this release does not claim

* **Hardware.** INT8 parity, Pi latency, thermal behaviour, and an audible
  buzzer: hardware gates H1–H3 and step 21.
* **Real-device local access.** Bluetooth enrolment on a physical iPhone, the
  Secure Enclave key, Bonjour on an iPhone Personal Hotspot, an IP change or a
  power cycle on real hardware, a revocation cutting off a real stream:
  `docs/LOCAL_ACCESS.md` §10.
* **Direct mode on the Pi radio.** WPA3/SAE support, the nftables ruleset under
  real `nft`, NetworkManager's dispatcher and polkit behaviour, iOS joining a
  network with no internet. There is also no app UI to switch direct mode:
  `docs/DIRECT_MODE.md` §8.
* **Transport.** It is plain HTTP on the LAN. A sniffer on the same network can
  watch for a session's lifetime; it cannot command (`docs/LOCAL_ACCESS.md` §8).
* **Accuracy.** There is no real-world accuracy figure
  (`docs/REAL_DATA_EVALUATION_PROTOCOL.md`).
