#!/usr/bin/env bash
# LEGACY first-boot provisioning hotspot for the SensorDyme device.
#
# A setup aid for installs without Bluetooth, NOT a portable operating mode: it
# exists only so that /setup can hand the device a home network, and it ends as
# soon as one is joined. The shipped method is BLE (services/provisioning.py);
# this runs only with `network.provisioning.method: hotspot` (setup_pi.sh). The
# device's real access-point mode, with its firewall, rollback and recovery, is
# direct mode (core/direct_mode.py, docs/DIRECT_MODE.md).
#
# If the Pi can't reach any known Wi-Fi within 45s of boot, it opens a hotspot
# "SensorDyme-XXXX" (last 4 of the MAC). The customer's phone joins it, the
# app's pairing wizard opens http://10.42.0.1:8899/setup, they pick their home
# Wi-Fi, and the device joins it -- after which the hotspot shuts down and the
# device shows up online in the app.
#
# The hotspot is never open. `nmcli device wifi hotspot` always secures it with
# WPA: with the label passphrase below when the factory wrote one (the same
# per-device secret direct mode uses), otherwise with a random passphrase
# NetworkManager generates (nmcli(1)) -- which nobody is shown, so a phone
# cannot join it and Bluetooth (or a power cycle onto a saved network) is the
# only way in. Neither passphrase is ever printed to the journal.
#
# Install (setup_pi.sh runs this): a systemd oneshot that runs after
# NetworkManager on every boot.
set -euo pipefail

# The direct-mode label passphrase (scripts/direct_mode_psk.py), mode 0600.
PSK_FILE="${SENSORDYME_HOTSPOT_PSK_FILE:-/etc/sensordyme/direct_mode_psk}"

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

# The label passphrase on stdout, or failure when there is no usable one. Only a
# regular file holding exactly the generated form (xxxx-xxxx-xxxx-xxxx) counts.
label_psk() {
  local psk c g
  [ -f "$PSK_FILE" ] && [ ! -L "$PSK_FILE" ] || return 1
  psk=$(head -n 1 "$PSK_FILE" 2>/dev/null | tr -d '\r\n') || return 1
  # Spelled out, not a range: [a-z] matches capitals in some locales.
  c='[23456789abcdefghjkmnpqrstuvwxyz]'
  g="$c$c$c$c"
  # shellcheck disable=SC2254 # $g is a pattern on purpose
  case "$psk" in
    $g-$g-$g-$g) printf '%s' "$psk" ;;
    *) return 1 ;;
  esac
}

# Start the WPA hotspot `$1`. nmcli's own output is discarded: it is not needed,
# and nothing it says about the hotspot's secret belongs in the journal.
start_hotspot() {
  local name="$1" psk
  if psk=$(label_psk); then
    nmcli device wifi hotspot ifname wlan0 ssid "$name" band bg password "$psk" >/dev/null 2>&1
  else
    nmcli device wifi hotspot ifname wlan0 ssid "$name" band bg >/dev/null 2>&1
  fi
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

  echo "provisioning: no known Wi-Fi — starting the WPA setup hotspot ${hotspot_name}"
  if label_psk >/dev/null; then
    echo "provisioning: the hotspot uses the Wi-Fi password on the device label"
  else
    echo "provisioning: no label Wi-Fi password on this device; the hotspot has a random one" \
         "nobody is shown — use Bluetooth setup instead"
  fi
  start_hotspot "${hotspot_name}" || echo "provisioning: the hotspot did not start (nmcli failed)"

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
