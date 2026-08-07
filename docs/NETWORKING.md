# Taking SensorDyme somewhere else

The single most likely reason a working device appears to break is that
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
