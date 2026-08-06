# SensorDyme cloud backend

FastAPI service connecting devices to the iOS app: accounts (JWT), device pairing by
claim code, alert relay + history, remote mode commands, and push notifications.
Events only — frames never leave the device.

## Run locally

```bash
.venv/bin/pip install -r cloud/requirements.txt   # already installed on the dev Mac
.venv/bin/uvicorn cloud.app:app --port 8787
```

That binds loopback only, which is right for the simulator. A real phone needs the
Mac's LAN address, so bind all interfaces — and once the port is reachable from the
network, sign tokens with a real secret instead of the dev default in `app.py`:

```bash
mkdir -p .secrets && openssl rand -hex 32 > .secrets/jwt_secret   # once; gitignored
SENSORDYME_JWT_SECRET="$(cat .secrets/jwt_secret)" \
  .venv/bin/uvicorn cloud.app:app --host 0.0.0.0 --port 8787
```

Then point the app at `http://<your-mac-ip>:8787` in Settings → Cloud backend.
Changing the secret invalidates existing tokens — the app will ask you to sign in again.

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

Anything that runs Python works (Fly.io, Railway, a small VPS).

```bash
export SENSORDYME_DB=/data/sensordyme.db
uvicorn cloud.app:app --host 0.0.0.0 --port 8787
```

**Do not** generate the JWT secret inline on the command line
(`export SENSORDYME_JWT_SECRET=$(openssl rand -hex 32)`, as this file used to
advise). That mints a new signing key on every restart, which invalidates every
session issued before it — the app reports `invalid token: Signature
verification failed` on a token that was valid when issued. `app.py` now
generates a key once and persists it to `cloud/.jwt_secret` (gitignored), so
sessions survive restarts on their own.

Set `SENSORDYME_JWT_SECRET` explicitly only when several server processes must
share one key — generate it *once*, store it in your host's secret manager, and
inject the same value every boot. `SENSORDYME_JWT_SECRET_FILE` overrides where
the persisted key lives (useful when the DB is on a read-only or ephemeral
volume).

SQLite is fine to start; the storage layer is one file (`app.py`) to swap for
Postgres when real traffic arrives. Push notifications use ntfy.sh today (no
accounts needed); `send_push()` is the single function to replace with APNs for
the production App Store build.
