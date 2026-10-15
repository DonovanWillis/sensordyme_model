#!/usr/bin/env bash
# First-boot provisioning hotspot for the SensorDyme device.
#
# If the Pi can't reach any known Wi-Fi within 45s of boot, it opens its own
# hotspot "SensorDyme-XXXX" (last 4 of the MAC). The customer's phone joins it,
# the app's pairing wizard opens http://10.42.0.1:8899/setup, they pick their
# home Wi-Fi, and the device joins it — after which the hotspot shuts down and
# the device shows up online in the app.
#
# Install (setup_pi.sh runs this): copies itself into a systemd oneshot that
# runs after NetworkManager on every boot.
set -euo pipefail

# Whether NetworkManager reports ANY connectivity. Same rule as
# core/network.py's WifiManager.is_online: the state STARTS WITH "connected".
#
# nmcli prints full connectivity as plain "connected" -- there is no
# "connected (global)" string. The pattern used to require one, so a device
# with a perfectly good internet connection was treated as offline and, 45s
# later, had its Wi-Fi taken over by the hotspot on every boot.
nm_online() {
  nmcli -t -f STATE general 2>/dev/null | grep -q '^connected'
}

main() {
  local suffix hotspot_name
  suffix=$(cat /sys/class/net/wlan0/address 2>/dev/null | tr -d ':' | tail -c 5 || echo "0000")
  hotspot_name="SensorDyme-${suffix^^}"

  # give NetworkManager a chance to join a known network first
  for _ in $(seq 1 45); do
    if nm_online; then
      echo "provisioning: already online, no hotspot needed"
      exit 0
    fi
    sleep 1
  done

  echo "provisioning: no known Wi-Fi — starting hotspot ${hotspot_name}"
  # open hotspot (no password: it's physically-local, short-lived, and serves only /setup)
  nmcli device wifi hotspot ifname wlan0 ssid "${hotspot_name}" band bg || true

  # The dashboard server (running via sensordyme.service) serves /setup on :8899.
  # When /setup/wifi succeeds, nmcli switches wlan0 to the customer's network,
  # which tears the hotspot down automatically. The page needs the dashboard
  # token (open it as /setup?token=...): /setup/wifi is gated like every write.
  echo "provisioning: hotspot up — waiting for the app at http://10.42.0.1:8899/setup"
}

# Run only when executed, so tests can source nm_online without side effects.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
