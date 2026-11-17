# iPhone stages for FIRST_PHYSICAL_BOOT.md (M14 stream C draft)

**For the integrator:** merge these stages into `docs/FIRST_PHYSICAL_BOOT.md`
at the numbers shown, then delete this file. Stage numbers match the mission
plan; the Pi stages between them belong to the other streams.

**Status:** written from the code and from Simulator runs only. **None of these
stages has run on the physical iPhone.** At the time of writing
`xcrun devicectl list devices` reports the owner's iPhone as *unavailable*, and
no app has been installed on it.

Conventions:

* `<UDID>` is the iPhone's identifier from stage 4. Never paste it into Git.
* `$REPO` is the repository checkout on the Mac.
* **Approval** says who must agree before the stage runs. "Owner" means the
  person who owns the phone, the Apple ID and the Pi.
* Every stage records one result: `PASS`, `FAIL` or `NOT RUN`, with the
  evidence named in the stage.
* Safety: stage 19 is **stationary only**. Never test an alert in a moving
  vehicle or on a public road.

---

## Stage 3: connect the iPhone to the Mac by data cable

| | |
|---|---|
| **Action** | Plug the iPhone into the Mac with a USB-C **data** cable (a charge-only cable shows nothing in stage 4). Unlock the phone. When iOS asks **Trust This Computer?**, tap **Trust** and enter the passcode. |
| **Expected** | The phone shows it is charging. Finder's sidebar lists the iPhone. |
| **If it fails** | No Trust prompt: unlock the phone and re-plug. Prompt dismissed by mistake: Settings → General → Transfer or Reset iPhone → Reset → **Reset Location & Privacy**, then re-plug (this resets every app's location and privacy permissions on the phone; ask the owner first). Not in Finder: try another cable or port; a hub can drop data. |
| **Recovery** | Unplug, unlock, re-plug. |
| **Approval** | Owner (it is their phone and their passcode). |

## Stage 4: confirm Xcode sees the iPhone

| | |
|---|---|
| **Command** | `xcrun devicectl list devices` then `xcrun xctrace list devices` |
| **Expected** | `devicectl` lists the iPhone with **State `available (paired)`** and its identifier (the `<UDID>`). `xctrace` lists it under *Devices*, not *Devices Offline*. |
| **If it fails** | `unavailable`: the phone is locked, untrusted (stage 3), or on a cable without data. `connected (no DDI)` or "preparing": wait for Xcode to copy the developer disk image (Xcode → Window → Devices and Simulators shows progress; minutes on the first connection). "Developer Mode disabled": stage 14, step 3. |
| **Recovery** | Unlock, re-plug, open Xcode → Window → Devices and Simulators and wait for it to finish preparing the phone. |
| **Approval** | None (read-only). |

## Stage 14: build and install the app

**Prerequisites. These are the owner's actions; no script can do them.**

1. **Apple ID in Xcode.** Xcode → Settings… → Accounts → `+` → Apple Account,
   signed in as the Apple ID that owns development team `D6ZR59ZR7Q`. This is
   the team already set in `ios/project.yml`. On 2026-11-13 the Mac had that
   team's signing certificate, but no signed-in account and no provisioning
   profile (`docs/IOS_DEVICE_BUILD.md` §a).
2. **Team.** Nothing to change: `ios/project.yml` sets
   `DEVELOPMENT_TEAM: D6ZR59ZR7Q` with `CODE_SIGN_STYLE: Automatic`. Pick
   another team only if the owner signs in with a different Apple ID. Then
   change it in `project.yml`; do not edit the generated project.
3. **Developer Mode** on the phone: Settings → Privacy & Security → Developer
   Mode → On → Restart → Turn On. The switch appears only after the phone has
   been paired with Xcode (stage 4), or after a first install attempt fails.
4. **Disk.** `df -h /System/Volumes/Data` must show at least 3 GB free.

**Command** (from `docs/IOS_DEVICE_BUILD.md` §c):

```sh
cd $REPO/ios && xcodegen generate && cd $REPO
xcodebuild -project ios/SensorDyme.xcodeproj -scheme SensorDyme \
  -configuration Debug -destination 'id=<UDID>' \
  -derivedDataPath /tmp/sd-device-dd -allowProvisioningUpdates build
xcrun devicectl device install app --device <UDID> \
  /tmp/sd-device-dd/Build/Products/Debug-iphoneos/SensorDyme.app
xcrun devicectl device info apps --device <UDID> | grep -i sensordyme
rm -rf /tmp/sd-device-dd
```

Or in Xcode: open `ios/SensorDyme.xcodeproj`, choose the iPhone as the run
destination, and press Run (⌘R).

The first launch needs one more step on the phone: Settings → General → VPN &
Device Management → the developer app entry for the Apple ID → **Trust**.

| | |
|---|---|
| **Expected** | `** BUILD SUCCEEDED **`; `devicectl` reports the app installed; `device info apps` shows `com.sensordyme.app` with the build's version. The ORYN EDGE icon appears and the app opens on the sign-in screen. |
| **If it fails** | `No Account for Team "D6ZR59ZR7Q"`: do prerequisite 1. `No profiles for 'com.sensordyme.app'`: the phone is not registered yet. Keep it connected and unlocked and rerun with `-allowProvisioningUpdates`. "Developer Mode disabled": prerequisite 3. "Untrusted Developer" at launch: the Trust step above. "No space left on device": free disk space and delete `/tmp/sd-device-dd`. |
| **Recovery** | Fix the named cause and run the same command again. A failed install changes nothing on the phone. |
| **Limits** | A free personal team's profile lasts **7 days** (`docs/IOS_DEVICE_BUILD.md` §d): after that the app will not open until it is reinstalled. Distribution to other phones (TestFlight or the App Store) needs the paid programme. It is not part of this boot. |
| **Approval** | Owner: signs in to Xcode, enables Developer Mode and taps Trust. |

## Stage 15: pair the camera over Bluetooth

On a real phone this is the **real CoreBluetooth** flow against the Pi's BLE
provisioning service (`services/ble_pop.py`, `docs/BLE_PROOF_OF_POSSESSION.md`).
It has only been run over the simulated transport.

| | |
|---|---|
| **Action** | 1. The Pi's Bluetooth window must be open: it opens when the Pi has no network, for `reconfigure_window_s` after boot, or when `always_available` is set. 2. On the iPhone, sign in (or skip sign-in), then on *Your Devices* tap **Connect over Bluetooth** (the radio-waves button). 3. Allow **Bluetooth** when iOS asks (`NSBluetoothAlwaysUsageDescription`). Allow **Local Network** when asked (`NSLocalNetworkUsageDescription`). 4. If the device uses `auth: pop`, enter the setup code from the label (`XXXX-XXXX-XXXX`). 5. Pick the Wi-Fi network and enter its password, or keep the Pi's current Wi-Fi. 6. Signed in: enter the claim code from the card. |
| **Expected** | The app finds exactly the device whose id matches the label. The device proves the code back before the password is sent. The phone enrols its own key over the proven session (`docs/LOCAL_ACCESS.md` §4.1). On the Pi, `scripts/local_clients.py list` shows this iPhone with `source: ble`. The app lists the camera as **Connected directly on this network** or, when signed in, as the account device. |
| **If it fails** | "Waiting for Bluetooth to be turned on": turn Bluetooth on (Control Center is not enough if it is off in Settings). "Still nothing after N s": the Pi's window is closed, or the phone is more than a few metres away. Check `journalctl -u sensordyme` on the Pi for the BLE service state. Wrong code: each wrong code spends a challenge. 5 failures lock the device out for 30 s, doubling. Wait, do not retry rapidly. Join failed: wrong Wi-Fi password, or a 5 GHz-only network the Pi cannot see. |
| **Recovery** | Close and reopen the setup sheet. To start clean: *Forget this camera on this iPhone* on the camera's screen, and `scripts/local_clients.py revoke <client_id>` on the Pi. |
| **Approval** | Owner (the setup code, the claim code and the Wi-Fi password are theirs). |

## Stage 16: open the real live feed

| | |
|---|---|
| **Action** | Open the camera from *Your Devices*. |
| **Expected** | The picture shows the Pi's camera and **Live video** is in the corner. **No "Simulated" badge**: the badge appears only when the overlay says `sim: true`, so on real hardware it must be absent. The feed runs over this iPhone's own sd-local-v1 session: no token is typed anywhere. |
| **If it fails** | Placeholder "Live picture on your Wi-Fi": the phone and the Pi are not on the same network, or Local Network permission was declined (Settings → Privacy & Security → Local Network → ORYN EDGE). "This iPhone needs to be paired again": the Pi answered 403 `unknown_client`; repeat stage 15. **A "Simulated" badge on a real feed is a release blocker: stop and report it.** |
| **Recovery** | Same Wi-Fi on both, permission on, reopen the screen. |
| **Approval** | None. |

## Stage 17: AI view and Present

| | |
|---|---|
| **Action** | On the live camera tap **AI view** (bottom left of the picture), then **Present**. |
| **Expected** | AI view off by default. When on, it shows the device-drawn analysis state and the device's measured fps. It **never shows percentages and never uses the word "gaze"** (`docs/LIVE_OVERLAY.md`). Present opens full screen with the AI view on, the distracted-driving panel, and the mode chips. When the account server is off, the presentation state says **Direct connection**. |
| **If it fails** | "Not analysed": the device sends no overlay for this frame (camera fallback). Check the Pi's perception health. No fps: the overlay header is missing. Check the device runs a build with sd-overlay/1. |
| **Recovery** | Close Present (top-left ✕) and reopen it. |
| **Approval** | None. |

## Stage 18: activate Distracted Driving

| | |
|---|---|
| **Action** | In Present (or on the camera's *Mode* list), choose Driver Safety, the Distracted Driving product, in its Standard or Strict variant. |
| **Expected** | The chip shows a spinner and becomes selected **only when the Pi reports the new mode**. The app never selects a mode on its own. The Pi's journal (`journalctl -u sensordyme`) records the preset change, sent by `local_app` over the signed local path. A mode the device's role does not allow is refused with the device's own reason (409). |
| **If it fails** | The spinner ends with the old chip still selected: the device refused or did not confirm. Read the message under the chips. |
| **Recovery** | Choose the mode again. Nothing changes on the device unless it confirms. |
| **Approval** | Owner (it changes what the device alerts on). |

## Stage 19: safe stationary event test

**Never on a public road, never in a moving vehicle.** The vehicle is parked
with the engine off, or the Pi is on a desk. The person playing the driver
consents, and the buzzer is announced before every trigger.

| | |
|---|---|
| **Action** | With Driver Safety active and the iPhone app **open in the foreground**, the seated person looks clearly away from the road (head turned) for longer than the mode's threshold, then looks back. Use the stationary cases in `docs/SUNDAY_DEVICE_ACTIVATION.md` §5. |
| **Expected** | The Present panel shows "Head turned away" developing, then confirmed. The Pi records one `distraction` event (`data/events.csv`, `/api/events`). The why line under the picture names that event. |
| **If it fails** | No event: check the calibration and the driver seat region, and the Pi's perception health (FaceMesh). An event the person did not cause: record it as a false positive with the time; do not tune thresholds during the boot. |
| **Recovery** | Wait for the next cycle; the event history keeps everything. |
| **Approval** | Owner and the person in the seat (consent to being filmed and to the buzzer). |

## Stage 20: device buzzer and iPhone foreground alert

| | |
|---|---|
| **Action** | Repeat stage 19 once with the phone's ringer **on** and the app in the foreground. A person listens at the Pi. |
| **Expected** | **The Pi's buzzer sounds**: this is the safety alert, and it never depends on the phone. Within about 5 s (the local poll), the iPhone shows **one** banner "Driver alert" for the event. Because this is a real, non-simulated event under 60 s old, **the iPhone plays the standard notification sound exactly once**, if *Alert sound on this iPhone* is on (Settings → Alerts). The same event reached again through the account later shows **no second banner and no second sound** (`AlertSound.decide`, `docs/LOCAL_ACCESS.md` §7). |
| **Not expected** | A sound for a simulated, experimental or log-only event, or for one older than 60 s. Any sound while the app is in the background: local notifications are posted only by the app in the foreground. There is no push (APNs). |
| **If it fails** | No banner: notification permission was declined (Settings → Notifications → ORYN EDGE). No sound but a banner: Silent Mode, a Focus, or *Alert sound on this iPhone* is off. iOS rules, and the app does not override them. **No buzzer: stop. The product's core claim fails** (Sunday OC §4). |
| **Recovery** | Fix the setting and repeat stage 19. |
| **Approval** | Owner, and everyone within earshot is warned about the buzzer. |

## Stage 22: the iPhone reconnects by itself after a Pi restart

| | |
|---|---|
| **Action** | With the app open on the camera, restart the Pi (`sudo reboot`, with the owner's approval) or restart its service (`sudo systemctl restart sensordyme`). Do not touch the phone. |
| **Expected** | Within seconds the screen says the camera **isn't answering on this network**. It never keeps showing "protecting" or a frozen picture as live. When the Pi is back (about 1–2 min for a reboot), the app opens a new device-signed session by itself, **Live video** returns, and monitoring reads fresh from the Pi. No re-pairing. Events recorded before the restart are not announced again; after a reboot the history marks them with the camera clock note. |
| **If it fails** | Still "isn't answering" after 3 min: the Pi changed IP and Bonjour is not reaching the phone (check `_sensordyme._tcp` with `dns-sd -B _sensordyme._tcp` on the Mac), or the service did not start (`systemctl status sensordyme`). "Pair again": the Pi's `data/local_identity.pem` or `local_clients.json` changed. That is a fault on the Pi: do not delete them. |
| **Recovery** | Leave and reopen the camera screen. If the Pi's IP changed, keep the phone on the same Wi-Fi; discovery finds the new address. |
| **Approval** | Owner (a reboot stops monitoring for its duration). |

---

### What the Simulator already proves for these stages (and what it does not)

* `test32_BackendDownCustomerJourney` (ios/UITests) runs stages 16–18, the app's
  half of stages 19–20 and stage 22, with the account server stopped. It runs
  against the real device protocol on the **simulated** device: scripted camera
  and models, an `SD-SIM-` identity, no Bluetooth, no Hailo, no buzzer. Its
  events are flagged simulated, so the journey checks that the phone decides
  **once** and stays **silent**. The once-only sound for a real event is
  checked in `ios/Tests/LocalEventsTests.swift` and `AlertSoundTests.swift`.
* Not proven anywhere yet:
  * installing on the phone;
  * real CoreBluetooth pairing;
  * the Secure Enclave key (`docs/LOCAL_ACCESS.md` §10 gate 6);
  * an audible buzzer or phone sound;
  * the Local Network permission prompt on a phone;
  * reconnecting across a real IP change.
