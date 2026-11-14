# Installing ORYN EDGE on a personal iPhone (M8.5)

Status, blockers and the exact procedure for putting the real app on the
owner's phone with legitimate Apple development signing — a personal team,
no paid membership. Nothing here weakens Release: the device build is the
same app, and `ios/Tests/release_config_audit.sh` proves a Release product
carries no debug affordance.

## (a) Current status — NOT YET BUILT FOR DEVICE

Attempted on 2026-11-13 at commit `c8f6bb2` on this Mac (Xcode 26.1.1 / 17B100):

```
xcodegen generate   # in ios/
xcodebuild -project ios/SensorDyme.xcodeproj -scheme SensorDyme \
  -configuration Debug -destination 'generic/platform=iOS' \
  -derivedDataPath <dd> -allowProvisioningUpdates build
```

It failed with exactly:

```
error: No Account for Team "D6ZR59ZR7Q". Add a new account in Accounts settings
       or verify that your accounts have valid credentials.
       (in target 'SensorDyme' from project 'SensorDyme')
error: No profiles for 'com.sensordyme.app' were found: Xcode couldn't find any
       iOS App Development provisioning profiles matching 'com.sensordyme.app'.
       (in target 'SensorDyme' from project 'SensorDyme')
** BUILD FAILED **
```

What this means, precisely:

- The Mac **does** hold one valid code-signing identity:
  `Apple Development: <owner's Apple ID> (63ZWF9B3GY)` — the certificate half
  of signing is fine.
- Xcode has **no signed-in Apple ID session** for team `D6ZR59ZR7Q`, and there
  is **no provisioning profile** for `com.sensordyme.app` anywhere on the Mac
  (`~/Library/Developer/Xcode/UserData/Provisioning Profiles/` is empty).
  A profile is the one artefact that cannot be created non-interactively: with
  a personal (free) team it is minted by Apple's developer services when a
  signed-in Xcode sees the app ID **and a registered device**, which also means
  the iPhone generally has to be connected at least once.
- The phone itself is known to this Mac but was not connected:
  `MonishIphone (iOS 26.5.2), UDID 00008150-0006590C2652401C` shows in
  `xctrace list devices` under *Devices Offline*.

No certificate was created, nothing was disabled, and signing was not bypassed.

## (b) Minimal user actions (one-time, ~5 minutes)

1. **Sign into Xcode**: Xcode → Settings… → Accounts → `+` → Apple Account →
   the Apple ID that owns personal team `D6ZR59ZR7Q` (the one on the
   certificate above). This is the step no script can do.
2. **Connect the iPhone by cable** (first time; Wi-Fi works after pairing).
   Unlock it and tap **Trust This Computer** when asked.
3. **Developer Mode** on the phone: Settings → Privacy & Security →
   Developer Mode → on → restart the phone → confirm. (iOS only shows this
   switch after a development install or paired Xcode; if it is not visible
   yet, run step (c) once — the failed install makes it appear.)
4. If Xcode prompts about signing in the project editor, the team is already
   set in `ios/project.yml` (`DEVELOPMENT_TEAM: D6ZR59ZR7Q`,
   `CODE_SIGN_STYLE: Automatic`); nothing to change. Regenerating the project
   with `xcodegen generate` preserves it.

## (c) Build + install once connected

The build half of this invocation is what was exercised above (with
`-destination 'generic/platform=iOS'`, since no phone was attached); the only
changes for a connected phone are the destination id and the install step.
It cannot have been run to success without step (b), so treat it as the exact
command to run, not as verified output:

```sh
cd ios && xcodegen generate && cd ..
xcodebuild -project ios/SensorDyme.xcodeproj -scheme SensorDyme \
  -configuration Debug \
  -destination 'id=00008150-0006590C2652401C' \
  -derivedDataPath /tmp/sd-device-dd \
  -allowProvisioningUpdates build

xcrun devicectl device install app \
  --device 00008150-0006590C2652401C \
  /tmp/sd-device-dd/Build/Products/Debug-iphoneos/SensorDyme.app
```

Or simply: open `ios/SensorDyme.xcodeproj` in Xcode, pick *MonishIphone* in
the run-destination bar, press **Run** (⌘R). The first launch on the phone
needs one more tap: Settings → General → VPN & Device Management → trust the
developer certificate.

Check free disk space first (`df -k /` — keep > 3 GB), and delete the
DerivedData folder afterwards.

## (d) Personal-team expiry: 7 days

A free personal team's provisioning profile is valid for **7 days** and at
most 3 app IDs. After a week the icon stays but the app refuses to launch
("app is no longer available"). To re-install: plug in (or be on the same
Wi-Fi as a paired Xcode), press Run again — same command as (c). Data
survives: the Keychain session, the paired camera and settings all persist
across re-installs of the same bundle id (`com.sensordyme.app`). Only
deleting the app loses them.

## (e) What to expect on the phone's first launch

The app ships with **no backend address baked in** (`SENSORDYME_API_BASE_URL`
is empty in `ios/project.yml`, deliberately — a baked-in LAN IP is the one
config that reliably breaks the app when the network changes):

1. **Local Network prompt.** On first use the app browses Bonjour for
   `_sensordyme-api._tcp` (the backend, published by `cloud/mdns.py` /
   `scripts/publish_mdns.py` on the Mac) — iOS asks for Local Network
   permission with the text from `NSLocalNetworkUsageDescription`. Decline it
   and discovery is impossible; the app says so rather than searching forever.
2. **Sign-in screen.** Discovery runs behind it (`BackendLocator` confirms a
   candidate by `GET /v1/health` reporting `service: sensordyme-cloud` before
   trusting it — an answering stranger on the same port is not the backend).
   The phone and the Mac running the backend must be on the same Wi-Fi.
   If nothing is found: Settings (gear on the sign-in screen) → *Manual server
   address* — note the app refuses to send credentials over plain `http://` to
   anything but a LAN/loopback-scoped host (`BackendResolution.cleartextPermitted`);
   an internet-name backend must be `https://`.
   "Explore the demo device" works with no backend and no account, clearly
   labelled simulated.
3. **Note: `localhost` means the phone.** On the Simulator the app falls back
   to `http://localhost:8787`; on a real phone it deliberately does not — an
   unconfigured phone build starts with discovery only.
4. **Device setup** (pairing a real SensorDyme) uses real CoreBluetooth —
   iOS asks for Bluetooth with `NSBluetoothAlwaysUsageDescription` — and the
   proof-of-possession flow when the device declares `auth: pop`
   (docs/BLE_PROOF_OF_POSSESSION.md). The live feed is discovered via
   `_sensordyme._tcp` on the LAN.
5. **Notifications** prompt appears when alerts are first armed (local
   notifications only; no push entitlement, nothing to configure).

## Scope note

This procedure signs a **development** build onto the owner's own phone with
their own Apple ID — the legitimate personal-team path. Distribution to anyone
else's phone (TestFlight/App Store) needs the paid Developer Program and is
out of scope for M8.5.
