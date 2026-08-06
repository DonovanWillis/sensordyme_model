# SensorDyme cloud backend

FastAPI service connecting devices to the iOS app: accounts (JWT), device pairing by
claim code, alert relay + history, remote mode commands, and push notifications.
Events only — frames never leave the device.

## Run locally

```bash
.venv/bin/pip install -r cloud/requirements.txt   # already installed on the dev Mac
.venv/bin/uvicorn cloud.app:app --port 8787
```

## Provision a device

```bash
python cloud/manage.py create-device "Car unit"
#  -> device_key : goes on the device, in config.yaml under cloud.device_key
#  -> claim_code : the customer types this into the iOS app to pair
```

Then on the device's config.yaml:

```yaml
cloud:
  enabled: true
  base_url: https://your-backend.example.com   # or http://<your-mac-ip>:8787 while testing
  device_key: dk_...
```

## Deploy

Anything that runs Python works (Fly.io, Railway, a small VPS). Set real secrets:

```bash
export SENSORDYME_JWT_SECRET=$(openssl rand -hex 32)
export SENSORDYME_DB=/data/sensordyme.db
uvicorn cloud.app:app --host 0.0.0.0 --port 8787
```

SQLite is fine to start; the storage layer is one file (`app.py`) to swap for
Postgres when real traffic arrives. Push notifications use ntfy.sh today (no
accounts needed); `send_push()` is the single function to replace with APNs for
the production App Store build.
