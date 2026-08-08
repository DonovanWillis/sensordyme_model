# SensorDyme iOS app

SwiftUI app for SensorDyme devices: sign up / log in, pair devices with a claim code,
watch live status, switch monitoring modes remotely, and browse alert history.

## Run it (simulator, 2 minutes)

```bash
# 1. start the cloud backend locally
.venv/bin/uvicorn cloud.app:app --port 8787

# 2. provision a device and start it (see cloud/README.md), then:
cd ios
xcodegen generate          # brew install xcodegen (once); regenerates SensorDyme.xcodeproj
open SensorDyme.xcodeproj  # Xcode: pick an iPhone simulator, Run
```

In the app: create an account → `+` → enter the device's claim code → pick a mode.
The simulator reaches your Mac's backend at the default `http://localhost:8787`
(change under Settings for a deployed backend).

## On a real iPhone

Set your Team under Signing & Capabilities in Xcode. You should not need to set a
backend address: the server announces itself on the local network
(`_sensordyme-api._tcp`, see cloud/mdns.py) and the app finds it. That is what
lets a brand-new phone sign in with nothing but an email and a password, and it
is also what heals the address when DHCP moves the server.

Settings → Cloud backend remains, for the one case discovery cannot cover: a
backend that is not on this network — a deployed public URL, say. An address
typed in there is used exactly as written and is never replaced while it answers;
discovery only takes over once it stops.

Worth being precise about the limit: mDNS is link-local, so this makes the
address irrelevant *on the server's network* and does nothing off it. Signing in
over cellular still needs a backend with a real public address.

## Device connection features

- **Find my device** (Settings → Live feed): Bonjour discovery of
  `_sensordyme._tcp` — fills in the device's feed address automatically.
- **Live feed auto-reconnect**: the MJPEG reader retries with backoff after
  any drop (device reboot, Wi-Fi blip) and shows a "reconnecting…" badge.
- **Device token**: if the device's config.yaml sets `dashboard.token`, enter
  the same value in Settings — it's sent as X-Device-Token on /stream and /api.
- **Bluetooth Wi-Fi setup** (Settings → Device Wi-Fi setup): provisions a
  device that has no network — scans for `SensorDyme-XXXX`, sends SSID +
  password over BLE, watches until the device comes online. Pairs with
  services/ble_gatt.py on the device.
- **Device health** (Settings): reads /api/health — per-component status
  (camera, inference, accelerator, Wi-Fi, …) straight from the on-device
  health monitor.

## Push notifications

Today: loud alerts push through ntfy.sh — install the free ntfy app, subscribe to a
secret topic, and enter the same topic in SensorDyme's Settings. For the App Store
build, replace with native APNs (requires an Apple Developer account: an APNs key,
a device-token registration endpoint on the backend, and swapping `send_push` in
cloud/app.py) — the provider is a single function by design.
