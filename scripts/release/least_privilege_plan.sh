#!/usr/bin/env bash
# PRINT the steps that move a Pi from the stock unit to the least-privilege unit.
# Executes NOTHING: it writes the plan to stdout for an operator to review and run
# by hand, on a bench device first (docs/LEAST_PRIVILEGE.md, "Bring-up on a Pi").
#
#   bash scripts/release/least_privilege_plan.sh [--from-dir /home/<admin>/sensordyme_model]
#
# The stock unit runs main.py from the installing admin user's checkout. The
# hardened unit needs the code root-owned under /opt/sensordyme and the state in
# /var/lib/sensordyme (bind-mounted onto /opt/sensordyme/data).
set -euo pipefail
FROM="${HOME}/sensordyme_model"
while [ $# -gt 0 ]; do
  case "$1" in
    --from-dir) FROM="${2:?--from-dir needs a directory}"; shift 2 ;;
    -h|--help) sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

cat <<PLAN
# ---- 0. record the current state (rollback needs it)
sudo systemctl cat sensordyme.service > ~/sensordyme.service.before
sudo sha256sum /etc/systemd/system/sensordyme.service ${FROM}/config.yaml

# ---- 1. identity: a system user with no login and no home of its own
sudo groupadd --system hailo 2>/dev/null || true
for g in gpio video audio netdev bluetooth; do getent group "\$g" >/dev/null || echo "MISSING GROUP \$g: stop here"; done
sudo useradd --system --user-group --no-create-home --home-dir /var/lib/sensordyme \\
     --shell /usr/sbin/nologin --groups video,gpio,audio,netdev,bluetooth,hailo sensordyme

# ---- 2. code: root-owned, read-only to the service (the venv keeps --system-site-packages)
sudo systemctl stop sensordyme.service
sudo mkdir -p /opt/sensordyme
sudo rsync -a --exclude data/ --exclude .git/ ${FROM}/ /opt/sensordyme/
sudo chown -R root:root /opt/sensordyme && sudo chmod -R go-w /opt/sensordyme
sudo mkdir -p /opt/sensordyme/data            # empty mount point for BindPaths=
# The venv records its own path: rebuild it in place rather than trusting the copy.
sudo python3 -m venv --system-site-packages --clear /opt/sensordyme/.venv
sudo /opt/sensordyme/.venv/bin/pip install --require-hashes -r /opt/sensordyme/requirements-device.lock

# ---- 3. config and credentials: readable by the service, writable only by root
sudo chown root:sensordyme /opt/sensordyme/config.yaml && sudo chmod 0640 /opt/sensordyme/config.yaml

# ---- 4. state: move data/ to /var/lib/sensordyme (StateDirectory= takes ownership on start)
sudo mkdir -p /var/lib/sensordyme
sudo rsync -a ${FROM}/data/ /var/lib/sensordyme/
sudo chown -R sensordyme:sensordyme /var/lib/sensordyme && sudo chmod 0750 /var/lib/sensordyme

# ---- 5. grants: hailo udev, reboot polkit (the network polkit rule is already installed)
sudo install -m 0644 /opt/sensordyme/deploy/udev/60-sensordyme-hailo.rules /etc/udev/rules.d/
sudo udevadm control --reload && sudo udevadm trigger
sudo install -m 0644 /opt/sensordyme/deploy/polkit/20-sensordyme-reboot.rules /etc/polkit-1/rules.d/
ls -l /dev/hailo0 /dev/video* /dev/media* /dev/gpiochip* /dev/snd /dev/dma_heap /dev/vcio
cat /proc/devices   # check the DeviceAllow= class names in the unit exist on this kernel

# ---- 6. unit: replace the stock unit (recover unit and guard stay as installed)
sudo install -m 0644 /opt/sensordyme/deploy/systemd/sensordyme.service.hardened /etc/systemd/system/sensordyme.service
# Pi 5 only, and only after reading the trade-off in the file:
# sudo install -D -m 0644 /opt/sensordyme/deploy/systemd/sensordyme.service.d/btmgmt-advertising.conf \\
#      /etc/systemd/system/sensordyme.service.d/btmgmt-advertising.conf
sudo systemd-analyze verify /etc/systemd/system/sensordyme.service
sudo systemctl daemon-reload
systemd-analyze security sensordyme.service | tail -1
sudo systemctl start sensordyme.service
journalctl -u sensordyme.service -b --no-pager | tail -50
# Then run the hardware checklist in docs/LEAST_PRIVILEGE.md ("Bring-up on a Pi").

# ---- ROLLBACK (any failed check above)
sudo systemctl stop sensordyme.service
sudo rm -f /etc/systemd/system/sensordyme.service.d/btmgmt-advertising.conf
sudo bash ${FROM}/scripts/release/install_units.sh --user <admin user>   # reinstalls the stock unit
sudo rsync -a /var/lib/sensordyme/ ${FROM}/data/ && sudo chown -R <admin user>: ${FROM}/data
sudo systemctl daemon-reload && sudo systemctl start sensordyme.service
PLAN
