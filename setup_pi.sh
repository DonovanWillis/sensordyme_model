#!/usr/bin/env bash
# One-shot Raspberry Pi setup: apt packages, pip installs, and the systemd service.
# Run this on the Pi itself (e.g. via `make deploy`, which rsyncs the repo first).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "==> Installing apt packages (picamera2, libcamera, gpio, audio, BLE, mDNS)"
sudo apt-get update
sudo apt-get install -y \
  python3-venv \
  python3-picamera2 \
  python3-libcamera \
  python3-dbus \
  python3-gi \
  bluez \
  avahi-daemon \
  libcap-dev \
  libatlas-base-dev \
  libopenjp2-7 \
  alsa-utils

# AI HAT+ (Hailo) support: driver + HailoRT runtime + tools. Available on
# Pi OS Bookworm's repos; if this fails the device still runs on CPU and
# /api/health explains why.
echo "==> Installing Hailo support for the AI HAT+ (optional but recommended)"
sudo apt-get install -y hailo-all || echo "WARNING: hailo-all not installed — AI HAT+ will be unused (CPU fallback)"

echo "==> Creating virtualenv (with access to system site-packages for picamera2)"
python3 -m venv --system-site-packages .venv
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install gpiozero==2.0.1 lgpio==0.2.2.0

mkdir -p data/reports

# Wi-Fi provisioning is BLE-based and runs INSIDE the main service now
# (services/provisioning.py). The legacy hotspot oneshot is only installed when
# config.yaml says `network.provisioning.method: hotspot`.
if grep -q "method: hotspot" config.yaml 2>/dev/null; then
  echo "==> Installing legacy provisioning hotspot (config method: hotspot)"
  chmod +x scripts/provisioning_hotspot.sh
  sudo tee /etc/systemd/system/sensordyme-provisioning.service > /dev/null <<UNIT
[Unit]
Description=SensorDyme first-boot Wi-Fi provisioning hotspot
After=NetworkManager.service
[Service]
Type=oneshot
ExecStart=$(pwd)/scripts/provisioning_hotspot.sh
[Install]
WantedBy=multi-user.target
UNIT
  sudo systemctl enable sensordyme-provisioning.service
else
  sudo systemctl disable sensordyme-provisioning.service 2>/dev/null || true
fi

echo "==> Advertising the device over mDNS (app auto-discovery: _sensordyme._tcp)"
# Generated, not copied: the record carries this board's stable device ID, which
# is how the app knows a device found at a new IP is one it already has.
sudo python3 scripts/publish_mdns.py /etc/avahi/services/sensordyme.service
sudo systemctl restart avahi-daemon

echo "==> Allowing the health monitor's last-resort reboot (single sudoers rule)"
sudo install -m 0440 deploy/sudoers/sensordyme-reboot /etc/sudoers.d/sensordyme-reboot

echo "==> Installing systemd service (Type=notify + watchdog + preflight)"
sudo cp sensordyme.service /etc/systemd/system/sensordyme.service
sudo sed -i "s#__SENSORDYME_DIR__#$(pwd)#g" /etc/systemd/system/sensordyme.service
sudo systemctl daemon-reload
sudo systemctl enable sensordyme.service
sudo systemctl restart sensordyme.service

echo ""
echo "Setup complete. SensorDyme is running as a systemd service (Restart=always)."
echo "Check status: sudo systemctl status sensordyme"
echo "Tail logs:    journalctl -u sensordyme -f"
