# Taking SensorDyme somewhere else

## First: is the backend even running?

Before chasing Wi-Fi, rule out the simplest explanation, because it produces the
identical symptom: the device joins the new network fine, the app says "Can't
reach backend" anyway, and nothing about Wi-Fi is actually wrong.

If the backend is a self-hosted Mac (`cloud/README.md`'s "Run locally" path) it
is one `uvicorn` process in one terminal window. It does not survive that
terminal closing, the Mac sleeping oddly, or a reboot — and unlike the device
side (a systemd service with `Restart=always` and a watchdog), nothing restarts
it automatically unless you installed it with `scripts/install_backend_mac.sh`.
A device sitting on a perfectly good network with nothing answering at the other
end reports exactly the same "can't reach backend" as a genuine network problem.

Check it directly rather than guessing:

```bash
.venv/bin/python scripts/diagnose_backend_mac.py sensordyme.local
```

This checks the backend process itself (listening, answering, persisted JWT
secret, advertising over mDNS) and, given a device hostname, that device's own
wifi/cloud_uplink health in one report — so you don't have to reason about which
side is actually broken. If it is not running: `./scripts/install_backend_mac.sh`
starts it and keeps it running from then on.

### Two ways the backend can be "running" and still unreachable

Both of these were live on the development Mac and neither shows up as a network
problem, so `diagnose_backend_mac.py` checks for them explicitly.

**The LaunchAgent starts the wrong program.** `launchctl print` reporting
`state = running` answers "is the job alive", not "is the job the backend". The
installed plist had been hand-edited to run `main.py` — the camera/detector
entrypoint — while a `uvicorn` started before that edit was still answering on
:8787. So the port was open, `/v1/health` answered, launchd said running, and the
next login would have started a detector and left the API server simply absent.
The diagnostic now compares what is installed against the shipped template:

```
✗ Agent starts the backend: the installed agent does NOT start the backend --
  it runs `/…/.venv/bin/python /…/main.py`. Reinstall it: ./scripts/install_backend_mac.sh
```

Reinstalling keeps a timestamped copy of whatever was there before, so a
deliberate edit is recoverable rather than lost.

**The mDNS record has gone stale.** The address and hostname were read once, at
startup, and repeated for the life of the process. A laptop is the worst possible
host for that assumption: it changes Wi-Fi, tethers to a phone, gets a new DHCP
lease, and macOS renames it (`…-MacBook-Pro-2` → `-4`) on any name collision.
Because the LaunchAgent keeps the backend up for weeks, the process that survives
all of those changes is exactly the one still advertising where it used to be —
found in practice announcing `172.20.10.2` (a hotspot address from days earlier)
while the Mac had long since moved to a different address on a different network.

That is worse than not advertising at all: a phone that finds nothing keeps
looking, while a phone handed a dead address stops and reports the server down.
`cloud/mdns.py` now re-checks its own address and hostname every
`REFRESH_INTERVAL_S` (15s) and corrects the record — in place for an address
change, by re-registering for a rename — and withdraws it entirely when the
machine has no LAN address, rather than repeating the last one it had. The
diagnostic verifies the record still matches this machine:

```
✗ mDNS advertising is current: STALE: advertised at ['172.20.10.2'] but this
  machine is at 11.33.7.x -- the app is being sent to an address that no
  longer exists.
```

A backend started before this fix landed will still be advertising staleness
until it is restarted:

```bash
launchctl kickstart -k gui/$(id -u)/com.sensordyme.backend
```

### The app no longer depends on any of this being right

The phone health-checks every address before it believes it, and tries them in
order rather than committing to the first. A build-time URL used to outrank
discovery outright, so `SENSORDYME_API_BASE_URL=http://11.22.7.64:8787` in
`ios/project.yml` meant every request went to a dead address forever and could
not be repointed without a rebuild. Now a candidate that does not answer
`/v1/health` with `{"service": "sensordyme-cloud"}` is skipped and the next one
is tried, so a stale pinned address yields to a server that is really there while
a healthy deployed URL is never abandoned for whatever is on the local Wi-Fi.
`ios/project.yml` ships that setting empty, and a test enforces that it is never
a LAN address again.

## Second: is `cloud.base_url` a fixed IP address?

The next most likely reason a working device appears to break is that
`cloud.base_url` on the Pi holds a **fixed IP address**. That address was
correct on the network where it was configured and means nothing anywhere else.

It is a confusing failure because the Pi keeps running perfectly — camera,
inference, local alerts all fine — while the app shows it offline. Nothing is
broken; the device is talking to an address that no longer exists.

It is *especially* confusing because most home routers hand out `192.168.1.x`,
so the old address and the new one look like the same network when they are two
different LANs that happen to share a private range.

## The fix: use a name, not an address

A name follows the machine between networks. An IP does not.

On the Pi, in `~/sensordyme/config.yaml`:

```yaml
cloud:
  enabled: true
  base_url: http://your-macbook.local:8787   # NOT http://192.168.1.251:8787
  device_key: "dk_..."
```

Find the Mac's mDNS name with:

```
scutil --get LocalHostName
```

and append `.local`. Then restart the device:

```
sudo systemctl restart sensordyme
```

That is the whole fix. Both machines can now join any network together and the
link re-forms with no reconfiguration.

## Why this works

Both macOS and Raspberry Pi OS run mDNS (Bonjour / Avahi) out of the box, so
`.local` names resolve on the local network with no DNS server and no setup.
Raspberry Pi OS ships `avahi-daemon`; macOS advertises its `LocalHostName`
automatically.

## When it does not work

mDNS is link-local and uses multicast, which some networks block:

- **Guest and hotel Wi-Fi** often enable client isolation, which stops devices
  seeing each other at all. Neither names nor IPs work there.
- **Corporate networks** frequently filter multicast.
- **Separate bands or VLANs** — a Pi on 2.4GHz and a Mac on a 5GHz guest SSID
  may be on different segments of the "same" Wi-Fi.

If `.local` will not resolve, the fallbacks in order of effort:

1. **Phone hotspot.** Put the Pi and the Mac on the same hotspot. Reliable, free,
   and the most predictable choice for a demo in an unfamiliar building.
2. **Ethernet.** A cable between the Pi and the Mac's network beats any radio.
3. **A tunnel** (Tailscale's free tier, or similar) gives both machines stable
   addresses on any network. Worth it once the device lives somewhere permanent.
4. **A hosted backend** with a public URL. This is the real production answer and
   the only one that works with the Pi and the phone on different networks
   entirely — but it costs money and is not needed for a demo.

## Diagnosing it

The device says so itself now. On its first failed sync it prints the address it
could not reach, and if that address is a bare IP it explains why that is the
likely cause:

```
cloud_uplink: cannot reach http://192.168.1.251:8787 (...)
cloud_uplink: base_url is a fixed IP address. If this device or the backend has
moved to a different network, that address now points at nothing. Use the
backend host's mDNS name instead — e.g. http://your-macbook.local:8787.
```

Read it with:

```
sudo journalctl -u sensordyme -n 50 --no-pager | grep cloud_uplink
```

## What is unaffected

Losing the backend costs you the app and remote configuration. It does **not**
stop the device working: capture, inference, rule evaluation, local alerts and
the on-device dashboard all continue, and events queue in memory until the
uplink returns. That is by design — a safety device that stops watching because
a server is unreachable would be worse than useless.
