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
# runs after network-online.target on every boot.
set -euo pipefail

SUFFIX=$(cat /sys/class/net/wlan0/address 2>/dev/null | tr -d ':' | tail -c 5 || echo "0000")
HOTSPOT_NAME="SensorDyme-${SUFFIX^^}"

# give NetworkManager a chance to join a known network first
for _ in $(seq 1 45); do
  if nmcli -t -f STATE general | grep -q "connected (site\|connected (global"; then
    echo "provisioning: already online, no hotspot needed"
    exit 0
  fi
  sleep 1
done

echo "provisioning: no known Wi-Fi — starting hotspot ${HOTSPOT_NAME}"
# open hotspot (no password: it's physically-local, short-lived, and serves only /setup)
nmcli device wifi hotspot ifname wlan0 ssid "${HOTSPOT_NAME}" band bg || true

# The dashboard server (running via sensordyme.service) serves /setup on :8899.
# When /setup/wifi succeeds, nmcli switches wlan0 to the customer's network,
# which tears the hotspot down automatically.
echo "provisioning: hotspot up — waiting for the app at http://10.42.0.1:8899/setup"
