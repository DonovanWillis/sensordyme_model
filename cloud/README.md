# SensorDyme cloud backend

FastAPI service connecting devices to the iOS app: accounts (JWT), device pairing by
claim code, alert relay + history, remote mode commands, and push notifications.
Events only — frames never leave the device.

## Run locally

```bash
.venv/bin/pip install -r cloud/requirements.txt   # already installed on the dev Mac
.venv/bin/uvicorn cloud.app:app --port 8787
```

That runs in the foreground of one terminal window: close the window, sleep the
Mac funny, or reboot, and the backend is gone — silently, with no restart. From
the device and the app this looks *exactly* like a Wi-Fi problem (device online,
DHCP fine, mDNS fine, "can't reach backend" anyway), which makes it the single
most common false alarm in this project. If that is ever what you are debugging,
run `.venv/bin/python scripts/diagnose_backend_mac.py <device-hostname>` first —
it checks the backend process itself, separately from everything the device and
app can already tell you about themselves.

For anything other than a five-minute test, install it as a background service
instead so it survives all of that on its own:

```bash
./scripts/install_backend_mac.sh
```

This registers a launchd agent that starts the backend at login and restarts it
if it ever crashes (see `deploy/launchd/com.sensordyme.backend.plist.template`).
Re-run it any time after `git pull` to pick up code changes. Logs land in
`cloud/backend.log`; `launchctl bootout gui/$(id -u)/com.sensordyme.backend`
stops it.

That binds loopback only, which is right for the simulator. A real phone needs the
Mac's LAN address, so bind all interfaces — and once the port is reachable from the
network, bind all interfaces. (Token signing needs nothing from you: there is no
longer a dev default to replace — `_load_jwt_secret` generates a secret on first
start and persists it beside the database, so sessions survive restarts.)

```bash
mkdir -p .secrets && openssl rand -hex 32 > .secrets/jwt_secret   # once; gitignored
SENSORDYME_JWT_SECRET="$(cat .secrets/jwt_secret)" \
  .venv/bin/uvicorn cloud.app:app --host 0.0.0.0 --port 8787
```

You should not have to tell the app where that is. Bound to all interfaces, the
server announces itself on the LAN as `_sensordyme-api._tcp` (cloud/mdns.py) and
the app finds it — which is what lets a new phone sign in with only an email and a
password, and what stops a DHCP change from looking like a broken app. Settings →
Cloud backend is still there for a server discovery cannot reach, such as a
deployed public URL, and an address typed in there always wins while it answers.

Two environment variables shape the announcement:

| Variable | Default | Why you would set it |
| --- | --- | --- |
| `SENSORDYME_PORT` | `8787` | uvicorn owns the real port and does not tell the app which one it chose. **Run on a non-default port and you must set this too** — announcing the wrong port is worse than not announcing, because it sends the app somewhere definite and wrong. The server now checks a couple of seconds after startup whether anything is listening on the port it announced, and logs a warning naming the mismatch if not, so this is no longer a silent failure. |
| `SENSORDYME_MDNS` | `1` | Set to `0` on a deployed server, where announcing to the datacentre LAN is pointless. |

Announcing is best-effort: if `zeroconf` is missing or the network refuses, the
server logs a warning, serves normally, and the app falls back to Settings.

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

## TLS

Plain HTTP is the default and stays that way until you change `base_url` — nothing
in this section is required to keep the existing LAN setup working exactly as it
does today.

For anything a real customer's phone will reach — the investor demo, a public
deploy — put a real, CA-signed certificate in front of the backend instead of
anything below: Fly.io/Railway terminate TLS for you, or run Caddy/Let's Encrypt
in front of uvicorn. iOS validates certificates against the system trust store, so
a real cert is trusted with **zero app changes**; that's the recommended path.

For local development — exercising HTTPS itself (e.g. the Pi's uplink talking to a
Mac-hosted backend over TLS) before a real deployment target exists — generate a
self-signed cert and pin the device to it directly:

```bash
.venv/bin/python scripts/generate_dev_cert.py        # writes .secrets/dev-cert.{pem,key}
.venv/bin/uvicorn cloud.app:app --host 0.0.0.0 --port 8787 \
    --ssl-keyfile .secrets/dev-cert.key --ssl-certfile .secrets/dev-cert.pem
```

Then on the device's config.yaml:

```yaml
cloud:
  base_url: https://your-mac.local:8787
  ca_cert_path: /path/to/.secrets/dev-cert.pem   # trust exactly this cert, nothing else
```

Verification stays fully on by default — `ca_cert_path` pins trust to one specific
cert rather than turning verification off. `tls_insecure_skip_verify: true` is also
available (genuinely insecure — see config.yaml's comment on it) for the rare case
where even generating a dev cert is in the way; the device logs a warning once at
startup whenever it's on.

**The iOS app will not trust a self-signed cert.** `URLSession` validates against
the system trust store regardless of the app's `NSAllowsArbitraryLoads` (which
relaxes ATS's connection *requirements* — TLS version, forward secrecy — not
certificate trust, a separate check). Making the app accept a specific self-signed
cert needs a `URLSessionDelegate` that pins to it, which is real client code, not a
config change, and does not exist yet — this is genuinely the one piece TLS support
does not close, and the reason the real-cert path above is the one to use for
anything the app must reach.

## Deploy

Anything that runs Python works (Fly.io, Railway, a small VPS). The repo ships a
Fly.io deployment: `fly.toml` (root) and `cloud/Dockerfile`. Fly terminates TLS
with a real certificate for `<app>.fly.dev`, which is the whole reason to prefer
it here — iOS trusts it with no app changes, and the device keeps
`verify=True`. `tests/test_deployment.py` pins the invariants below.

```bash
export SENSORDYME_DB=/data/sensordyme.db
uvicorn cloud.app:app --host 0.0.0.0 --port 8080
```

Port 8080 here, not the 8787 used for local development: `cloud/Dockerfile` and
`fly.toml`'s `internal_port` both use 8080, and `tests/test_deployment.py` pins
them to agree. This snippet said 8787 and so contradicted the two files it is
describing.

### Fly.io, first time

```bash
brew install flyctl && fly auth login          # interactive; needs a payment method on file
fly launch --no-deploy --copy-config --name sensordyme-api --region iad
fly volumes create sensordyme_data --region iad --size 1   # see "the volume" below
fly deploy --remote-only                       # builds on Fly; no local Docker needed
fly scale count 1                              # SQLite has exactly one writer
```

`app` in `fly.toml` must be globally unique. If the name is taken, pick another
— and remember the hostname travels with it, into both the app's
`SENSORDYME_API_BASE_URL` and the device's `cloud.base_url`.

### The volume is not optional

`/data` holds the database **and** the JWT signing key. On ephemeral storage
every deploy mints a new key, and every existing session then fails with
`invalid token: Signature verification failed` on a token that was valid when it
was issued — while the server itself looks perfectly healthy. That is the single
most confusing way to get this wrong, so `fly.toml` mounts a volume and
`tests/test_deployment.py` asserts both paths are on it.

### Bringing an existing database with you

Deploy first, import second, and point the device at it last: with no traffic
yet there is nothing to race.

```bash
# 1. Back up locally. Use SQLite's own backup API -- a `cp` of a WAL-mode
#    database that is being written to is not guaranteed to be consistent.
python - <<'EOF'
import sqlite3
src, dst = sqlite3.connect("cloud/sensordyme.db"), sqlite3.connect(".secrets/backups/pre-fly.db")
src.backup(dst); dst.close(); src.close()
EOF

# 2. Upload it (put it somewhere the backend is not serving from)
fly ssh sftp shell
  put .secrets/backups/pre-fly.db /data/import.db

# 3. Check, then import. --dry-run reports what it found and writes nothing.
fly ssh console -C "python /app/scripts/fly_import_db.py /data/import.db --dry-run"
fly ssh console -C "python /app/scripts/fly_import_db.py /data/import.db"
fly apps restart
```

`scripts/fly_import_db.py` refuses anything that is not a healthy SensorDyme
database — the realistic mistake is uploading the freshly-created *empty* file
and wiping the deployment with it — and backs up whatever it replaces on the
volume before writing.

The JWT secret is deliberately **not** migrated: a fresh key is generated on the
volume, so everyone signs in once more. Accounts, devices, device keys and event
history all come across; only sessions do not. Copy `cloud/.jwt_secret` to
`/data/.jwt_secret` (mode 0600) if you would rather keep existing sessions alive.

### Rate limiting behind the proxy

`SENSORDYME_CLIENT_IP_HEADER = "Fly-Client-IP"` in `fly.toml` is load-bearing.
Rate limiting keys on the caller's address, and behind a proxy the TCP peer is
the proxy for *every* request — so without it all users share one
10-attempts-per-5-minutes bucket and any single caller can lock everybody out of
login with ten tries. `app.py` trusts no header unless this names one, because a
header the proxy does not overwrite is spoofable by exactly the caller being
limited.

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
