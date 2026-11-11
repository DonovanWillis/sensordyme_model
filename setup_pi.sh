#!/usr/bin/env bash
# One-shot Raspberry Pi setup: apt packages, pip installs, and the systemd service.
# Run this on the Pi itself (e.g. via `make deploy`, which rsyncs the repo first).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# The account the service runs as. Raspberry Pi OS Bookworm images have no
# default `pi` user (the Imager makes you choose a name), and the unit file and
# sudoers rule used to hardcode `pi` -- on any other name the service could not
# start and the reboot/btmgmt rules applied to nobody. Override with
# SENSORDYME_USER=name; otherwise the user who ran this script (via sudo or not).
SERVICE_USER="${SENSORDYME_USER:-${SUDO_USER:-$(id -un)}}"
if [ "$SERVICE_USER" = "root" ]; then
  echo "ERROR: refusing to run the SensorDyme service as root." >&2
  echo "  Run this as the device's normal user, or set SENSORDYME_USER=<name>." >&2
  exit 1
fi
if ! id "$SERVICE_USER" >/dev/null 2>&1; then
  echo "ERROR: service user '$SERVICE_USER' does not exist." >&2
  exit 1
fi
echo "==> Service user: $SERVICE_USER"

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
# Pi OS Bookworm's repos. If this fails the device has NO detector at all --
# there is no CPU fallback, because ultralytics (AGPL-3.0) is deliberately not
# installed. The service still boots and serves health, the dashboard and BLE
# provisioning, but reports monitoring_active: false and says why. That is
# intended: a safety device that cannot see must say so rather than appear well.
# (This comment used to promise a CPU fallback. That was true before Phase 4
# removed ultralytics from the device image; leaving it would have told an
# installer their failed install was harmless.)
echo "==> Installing Hailo support for the AI HAT+ (REQUIRED — there is no CPU fallback)"
sudo apt-get install -y hailo-all || echo "WARNING: hailo-all not installed — this device will have NO object detector and will boot degraded (monitoring_active: false)"

echo "==> Creating virtualenv (with access to system site-packages for picamera2)"
python3 -m venv --system-site-packages .venv
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip
# requirements-device.txt, NOT requirements.txt: this is a device we convey, and
# the developer set pins ultralytics, which is AGPL-3.0 (docs/LICENSING.md).
# Detection here runs on the AI HAT+ via core/accel.py, which needs no inference
# framework — just hailo-all, installed above. Dropping ultralytics also drops
# torch + torchvision, ~800MB that never had a use on this board.
pip install -r requirements-device.txt
pip install gpiozero==2.0.1 lgpio==0.2.2.0

# --- object-detection model -------------------------------------------------
# This step exists because without it a fresh install produced a device that
# could not detect anything. `models/` and `*.hef` are both gitignored and
# nothing here fetched a model, so the detector only ever worked by falling back
# to ultralytics, which auto-downloaded yolo11n.pt. Both are AGPL-3.0, so that
# path is now deliberately closed (requirements-device.txt) -- which turns the
# missing model from a silent licence problem into a loud startup failure. This
# provisions the model properly instead.
#
# Pinned by CHECKSUM, not just URL: a model file is the one binary in this
# product whose contents decide whether a fall is detected, and "whatever S3
# served today" is not a supply chain. A mismatch fails the install.
MODEL_DIR="models"
MODEL_FILE="$MODEL_DIR/damoyolo_tinynasL20_T.hef"
MODEL_URL="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/damoyolo_tinynasL20_T.hef"
MODEL_SHA256="e69136d3ebf3cc8ebaf3da5417a59962b139f3d3d67d715e9a9b1ae10963960e"
# The build Phase 3/4 actually measured. Its provenance could NOT be established
# -- it matches no Model Zoo release (v2.11-v2.15 all differ in size), and the
# script that fetched it did not survive. It is structurally identical to the
# pinned build above (same input, same 6 raw-tensor outputs, same shapes), so
# integration is unaffected, but the WEIGHTS differ and the accuracy numbers in
# docs/ were measured on this one. Re-run benchmarks/accuracy_state_farm.py
# against the pinned build before treating those numbers as describing it.
MODEL_SHA256_EVALUATED="4c71eab80976a2c9f1b134877890abce02871f6d4d92f53ac5a3f8bf2f24add2"

mkdir -p "$MODEL_DIR"
echo "==> Provisioning the object-detection model"
if [ -f "$MODEL_FILE" ]; then
  have="$(sha256sum "$MODEL_FILE" | cut -d' ' -f1)"
  if [ "$have" = "$MODEL_SHA256" ]; then
    echo "    already present and matches the pinned build"
  elif [ "$have" = "$MODEL_SHA256_EVALUATED" ]; then
    echo "    present: the Phase 3/4 EVALUATED build (provenance unverified, see above)."
    echo "    Keeping it — it is the file the shipped accuracy numbers describe."
  else
    echo "ERROR: $MODEL_FILE is present but matches no known build." >&2
    echo "  found:    $have" >&2
    echo "  expected: $MODEL_SHA256 (pinned) or $MODEL_SHA256_EVALUATED (evaluated)" >&2
    echo "  Refusing to guess. Delete the file to re-provision, or verify its origin." >&2
    exit 1
  fi
else
  echo "    downloading $(basename "$MODEL_FILE") from the Hailo Model Zoo"
  if ! curl -fSL --retry 3 --connect-timeout 20 -o "$MODEL_FILE.part" "$MODEL_URL"; then
    rm -f "$MODEL_FILE.part"
    echo "ERROR: could not download the detection model from:" >&2
    echo "  $MODEL_URL" >&2
    echo "Without it this device cannot detect people or objects and the service" >&2
    echo "will fail to start. Fetch it manually onto the device as $MODEL_FILE." >&2
    exit 1
  fi
  got="$(sha256sum "$MODEL_FILE.part" | cut -d' ' -f1)"
  if [ "$got" != "$MODEL_SHA256" ]; then
    rm -f "$MODEL_FILE.part"
    echo "ERROR: checksum mismatch on the downloaded model — NOT installing it." >&2
    echo "  expected: $MODEL_SHA256" >&2
    echo "  got:      $got" >&2
    exit 1
  fi
  mv "$MODEL_FILE.part" "$MODEL_FILE"
  echo "    installed and checksum-verified"
fi

mkdir -p data/reports

# Wi-Fi provisioning is BLE-based and runs INSIDE the main service now
# (services/provisioning.py). The legacy hotspot oneshot is only installed when
# config.yaml says `network.provisioning.method: hotspot`.
# Matches the live key only: a commented-out "# method: hotspot" must not
# install a hotspot that competes with in-process BLE provisioning.
if grep -Eq '^[[:space:]]*method:[[:space:]]*"?hotspot"?([[:space:]]|#|$)' config.yaml 2>/dev/null; then
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

echo "==> Allowing the device to manage its own Wi-Fi (polkit)"
# The polkit rule below grants to the netdev group; BlueZ's D-Bus policy lets the
# bluetooth group register GATT applications. Make sure the service user is in
# both (no-ops when already a member or when a group does not exist).
for group in netdev bluetooth; do
  if getent group "$group" >/dev/null 2>&1; then
    sudo usermod -aG "$group" "$SERVICE_USER"
  fi
done
# Without this the service runs as `pi` under systemd with no login session, so
# polkit refuses settings.modify.system and saving a Wi-Fi profile fails with
# "Insufficient privileges" — the device accepts credentials over Bluetooth and
# then cannot act on them.
sudo install -m 0644 deploy/polkit/10-sensordyme-network.rules \
  /etc/polkit-1/rules.d/10-sensordyme-network.rules
sudo systemctl restart polkit

echo "==> Allowing the last-resort reboot and BLE advertising (exact sudoers commands)"
# Rendered for SERVICE_USER and checked BEFORE it goes anywhere near
# /etc/sudoers.d: one syntax error in a drop-in breaks sudo for every user on
# the device, including the one who would have to fix it.
SUDOERS_TMP="$(mktemp)"
sed "s/__SENSORDYME_USER__/${SERVICE_USER}/g" deploy/sudoers/sensordyme-reboot > "$SUDOERS_TMP"
if command -v visudo >/dev/null 2>&1; then
  if ! sudo visudo -cf "$SUDOERS_TMP"; then
    rm -f "$SUDOERS_TMP"
    echo "ERROR: the rendered sudoers rule failed visudo -cf; NOT installing it." >&2
    exit 1
  fi
else
  echo "WARNING: visudo not found; cannot validate the sudoers rule (not on a Pi?)" >&2
fi
sudo install -m 0440 "$SUDOERS_TMP" /etc/sudoers.d/sensordyme-reboot
rm -f "$SUDOERS_TMP"

echo "==> Installing the bounded start-limit recovery unit (no unbounded reboot loop)"
# sensordyme.service's OnFailure= target. The guard runs as root, so it is
# installed root-owned OUTSIDE this (service-user-writable) checkout.
sudo install -D -m 0755 -o root -g root scripts/start_limit_guard.py \
  /usr/local/lib/sensordyme/start_limit_guard.py
sudo install -m 0644 -o root -g root deploy/systemd/sensordyme-recover.service \
  /etc/systemd/system/sensordyme-recover.service

echo "==> Installing systemd service (Type=notify + watchdog + preflight)"
sudo cp sensordyme.service /etc/systemd/system/sensordyme.service
sudo sed -i "s#__SENSORDYME_DIR__#$(pwd)#g; s#__SENSORDYME_USER__#${SERVICE_USER}#g" \
  /etc/systemd/system/sensordyme.service
sudo systemctl daemon-reload
sudo systemctl enable sensordyme.service
sudo systemctl restart sensordyme.service

echo ""
echo "Setup complete. SensorDyme is running as a systemd service (Restart=always)."
echo "Check status: sudo systemctl status sensordyme"
echo "Tail logs:    journalctl -u sensordyme -f"
