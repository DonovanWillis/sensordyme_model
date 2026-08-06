# Verifying a deployment, step by step

## 0. On the Mac (before deploying)

```bash
.venv/bin/python -m pytest -q          # full suite must pass
.venv/bin/python scripts/preflight.py  # config valid; camera/accel checks run
.venv/bin/python main.py --no-display  # runtime comes up; Ctrl+C stops cleanly
```

## 1. Deploy to the Pi

```bash
rsync -av --exclude='.venv' --exclude='data' --exclude='__pycache__' \
  ./ pi@sensordyme.local:~/sensordyme/
ssh pi@sensordyme.local 'cd ~/sensordyme && ./setup_pi.sh'
```

(`make deploy` for subsequent updates — it rsyncs and restarts the service.)

## 2. Boot verification (on the Pi)

```bash
sudo systemctl status sensordyme        # active (running), Type=notify READY
cat ~/sensordyme/data/preflight.json    # camera ok:true, accelerator present
journalctl -u sensordyme -n 50          # startup log: camera, accel, dashboard
rpicam-hello --list-cameras             # if preflight said camera failed
hailortcli fw-control identify          # if preflight said accelerator failed
```

Reboot test: `sudo reboot`, wait, confirm the service came back with no login
or manual step (`systemctl status sensordyme` again).

## 3. Health + recovery

```bash
curl -s localhost:8899/api/health | python3 -m json.tool
# camera + inference "ok"; accelerator explains CPU/Hailo; wifi shows SSID+IP

# Watchdog/recovery drill — kill -STOP wedges the process without killing it;
# the monitor stops feeding the watchdog and systemd restarts the service:
sudo kill -STOP $(pgrep -f 'main.py --no-display'); sleep 60
systemctl show sensordyme -p NRestarts   # incremented
curl -s localhost:8899/api/health        # healthy again after restart
```

Camera recovery: unplug the ribbon → `/api/health` camera goes degraded/down
with restarts attempted; replug → recovers without touching the service.

## 4. Networking workflow

- **Provisioned boot**: device joins saved Wi-Fi; `avahi-browse -r _sensordyme._tcp`
  from the Mac finds it.
- **Unprovisioned** (simulate: `sudo nmcli connection delete "<your-wifi>"`,
  reboot): within ~60s the device advertises BLE. In the app: Settings →
  "Set up device Wi-Fi over Bluetooth" → scan → SensorDyme-XXXX → send SSID +
  password → status walks connecting → online. Confirm the profile SAVED:
  `nmcli connection show` on the Pi, then reboot — it must rejoin by itself.
- **Wi-Fi loss while running**: take the network down; detection keeps running
  (events still land in data/events.csv); BLE window reopens; cloud events
  queue and flush when the network returns.

## 5. App connection flow

1. Phone on the same Wi-Fi as the device.
2. Settings → Live feed → **Find my device** → tap the discovered entry
   (fills `http://<ip>:8899` via mDNS).
3. Device card shows the live feed. Latency should be sub-second; the stream
   runs at `dashboard.stream_fps` (12 by default) and quality 60.
4. Kill test: restart the service while watching — the feed shows
   "reconnecting…" and comes back by itself (exponential backoff, no user action).
5. Settings → **Check device health** — overall state, uptime, per-component
   details straight from /api/health.
6. If `dashboard.token` is set in config.yaml, put the same value in Settings →
   "Device token" — the stream and health calls send X-Device-Token.

## 6. Cloud + alerts (unchanged paths, re-verify)

```bash
.venv/bin/uvicorn cloud.app:app --port 8787   # on the Mac / your server
```
App → sign up → claim device → trigger a detection (look away 4s in
driver_attention) → alert appears in the app timeline; push via ntfy if
configured; `set_preset` from the app switches the device's mode within ~5s
(uplink poll interval).
