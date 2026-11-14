# Least privilege for the device service (M6.5, 4D) and the device dependency split (4E)

**Status: software-ready, hardware-pending.** The least-privilege unit is prepared and
validated off-device; it is **not installed or enabled on any Pi**. The stock
`sensordyme.service` (admin user, sudoers rules) stays the installed unit until the
bring-up checklist below passes on a bench Pi.

| File | What it is |
|---|---|
| `deploy/systemd/sensordyme.service.hardened` | the least-privilege unit (replaces `sensordyme.service` when adopted) |
| `deploy/systemd/sensordyme.service.d/btmgmt-advertising.conf` | optional drop-in: CAP_NET_ADMIN + AF_BLUETOOTH, Pi 5 BLE advertising only |
| `deploy/polkit/20-sensordyme-reboot.rules` | reboot-only polkit grant for user `sensordyme` (replaces the sudoers reboot line) |
| `deploy/udev/60-sensordyme-hailo.rules` | `/dev/hailo0` → group `hailo`, 0660 |
| `core/privilege.py` | `SENSORDYME_PRIVILEGE=sudo` (default) or `direct` (set by the hardened unit) |
| `scripts/release/least_privilege_plan.sh` | prints the migration and rollback; executes nothing |
| `scripts/release/verify_systemd_units.sh` | `systemd-analyze verify` + offline `security` on all units |
| `tests/test_least_privilege_unit.py` | static checks; the verify test runs where systemd exists |

Validation done (2026-11-11): `systemd-analyze verify` clean on systemd 255 (oryn job
`20261111-060654-3b7a25`) for the stock unit, the hardened unit, and hardened + drop-in.
Offline exposure score (`systemd-analyze security --offline`): stock **9.0 UNSAFE** →
hardened **1.7 OK**. Not validated: anything that needs the Pi's kernel, devices or
BlueZ (see the checklist).

## Access matrix (derived from the code)

| Need | Code that does it | Resource | Grant in the hardened unit |
|---|---|---|---|
| Camera (Picamera2/libcamera) | `core/camera.py` | `/dev/video*`, `/dev/media*`, `/dev/v4l-subdev*`, `/dev/dma_heap/*` | group `video`; `DeviceAllow=char-video4linux`, `char-media`, `char-dma_heap` |
| Firmware temperature/throttle | `core/device_health.py`, `core/thermal.py` (`vcgencmd`) | `/dev/vcio` | group `video`; `DeviceAllow=/dev/vcio` |
| Thermal zones, meminfo, cpuinfo, MAC, machine-id | `core/thermal.py`, `core/device_health.py`, `core/identity.py` | `/sys/class/thermal`, `/sys/class/hwmon`, `/proc/{meminfo,cpuinfo}`, `/sys/class/net/*/address`, `/etc/machine-id` | read-only (ProtectSystem/ProtectKernelTunables keep them readable) |
| AI HAT (HailoRT) | `core/accel.py` | `/dev/hailo0`; HailoRT log file | group `hailo` + udev rule; `DeviceAllow=/dev/hailo0`; `HAILORT_LOGGER_PATH=/var/log/sensordyme` |
| GPIO buzzer | `core/alerts.py` (`gpiozero`) | `/dev/gpiochip*`, `/dev/gpiomem*` | group `gpio`; `DeviceAllow=char-gpiochip`, `char-rpi-gpiomem` |
| Audio alerts | `core/alerts.py` (`simpleaudio`, `aplay`) | `/dev/snd/*`; a temp WAV | group `audio`; `DeviceAllow=char-alsa`; `PrivateTmp` |
| Wi-Fi join/scan/hotspot | `core/network.py` (`nmcli` → NetworkManager over D-Bus) | system D-Bus; polkit NM actions | group `netdev` + existing `10-sensordyme-network.rules`; `AF_UNIX` |
| BLE GATT provisioning | `services/ble_gatt.py`, `services/provisioning.py` (BlueZ over D-Bus) | system D-Bus; BlueZ policy | group `bluetooth`; `AF_UNIX` |
| BLE advertising fallback (Pi 5) | `services/ble_gatt.py` (`btmgmt`) | kernel mgmt socket | **optional drop-in**: CAP_NET_ADMIN + `AF_BLUETOOTH` (stock unit: 5 exact sudoers lines) |
| Last-resort reboot | `services/runtime.py` `_issue_reboot` | logind `reboot` | polkit `20-sensordyme-reboot.rules` (stock: sudoers `systemctl reboot`) |
| Cloud uplink, dashboard (port 8899), mDNS | `core/uplink.py`, `core/dashboard.py` | TCP/UDP | `AF_INET`, `AF_INET6`; no bind capability (port > 1024) |
| Interface queries | Python/glibc `getifaddrs`, `socket.gethostname` | netlink | `AF_NETLINK` |
| systemd notify + watchdog | `core/sdnotify.py` | `$NOTIFY_SOCKET` | `AF_UNIX`, `NotifyAccess=main` |
| Config, presets, models (read) | `config.yaml`, `presets.yaml`, `models/*.hef`, `*.onnx` | `/opt/sensordyme` | read-only; `config.yaml` root:sensordyme 0640 |
| State and logs (write) | `data/{events,telemetry}.csv`, `device_state.json`, `monitoring.json`, `.reboot_state.json`, `preflight.json`, `evidence/`, `training/` | `data/` | `StateDirectory=sensordyme` bind-mounted onto `/opt/sensordyme/data` |
| Library caches | matplotlib (`core/reports.py`) | `$MPLCONFIGDIR`, `$XDG_CACHE_HOME` | `CacheDirectory=sensordyme` |
| Journal | stdout/stderr | journald | default |

Denied by the hardened unit and needed by nothing in the code: root, sudo, any
capability (without the drop-in), block devices, `/dev/mem`, I2C/SPI/serial, raw
packets (`AF_PACKET`), kernel modules/tunables/logs, the clock, the hostname,
namespaces, realtime scheduling, writes anywhere outside state/cache/logs.

Deliberately **not** set (and why): `PrivateUsers=` (unmaps the device groups);
`MemoryDenyWriteExecute=` (XNNPACK in mediapipe and onnxruntime may generate code;
verify on a Pi first); `IPAddressAllow=` (the device needs LAN and cloud); the memory
limits (`MemoryHigh=1200M`, `MemoryMax=1600M`) are first bounds, **not measured** on a Pi.

### Trade-off: Pi 5 BLE advertising

Under `NoNewPrivileges`, sudo cannot gain root, so the five exact `btmgmt` sudoers
lines stop working. The drop-in replaces them with an ambient CAP_NET_ADMIN for the
whole process — broader than five fixed command lines (a compromised service could
change interfaces, routes and firewall rules). Without the drop-in a Pi 5 cannot
advertise for BLE provisioning; Wi-Fi joined earlier keeps working. The narrower
design, left as follow-up: a root `Type=oneshot` unit that only advertises
(`sensordyme-ble-advertise.service`), started by the service through a polkit rule
restricted to that unit name.

### Credentials (coordination with 4C)

`cloud.device_key` lives in `config.yaml`, which the hardened unit reads but cannot
write. If the credential-storage work moves the key to its own file, put it under the
state directory (`/var/lib/sensordyme`, 0600, owned by `sensordyme`) if the device
must rotate it itself, or root-owned 0640 under `/opt/sensordyme` if only an operator
rotates it.

## Bring-up on a Pi (not done; needs an authorised hardware session)

`bash scripts/release/least_privilege_plan.sh --from-dir <checkout>` prints every
step, including rollback. After `systemctl start`, check each row of the matrix:

1. `journalctl -u sensordyme -b`: no `Permission denied`, `EPERM`, or `DeviceAllow` warnings;
   `cat /proc/devices` lists `video4linux`, `media`, `dma_heap`, `gpiochip`, `alsa` (fix the
   class names in the unit if this kernel names them differently; `rpi-gpiomem` is the least certain).
2. Camera frames and AI HAT detections (`/api/health`: camera ok, accelerator present).
3. `sensordyme-cli`/dashboard: `monitoring_active: true`; the watchdog does not fire for 10 min.
4. GPIO buzzer and audio alert self-test (physical observation).
5. BLE provisioning from the app (Pi 5: with the drop-in), Wi-Fi join, reconnect.
6. Health-monitor reboot path: `allow_reboot` test as in `docs/HAILO_HARDWARE_VALIDATION.md`.
7. Peak RSS under load (`systemctl show -p MemoryPeak sensordyme`) — set `MemoryMax` from it.
8. `systemd-analyze security sensordyme.service` on the device.

Any failure: run the ROLLBACK block of the plan (reinstalls the stock unit and copies
the state back).

## 4E: device dependencies

* `requirements-device.txt` holds only what the device runtime imports. **Moved out**
  to `requirements.txt` (dev + CI, which still includes the device file): `pytest`
  (nothing on the device imports it) and `anthropic` (only `core/vlm_judge.py`, reached
  only from the `mac_max` profile behind `require_mps`). No existing pin changed.
* **Added** `cryptography==50.0.2` (Apache-2.0 OR BSD-3-Clause) for the BLE proof-of-possession
  workstream (`services/ble_pop.py`, 4B: HKDF, HMAC, PBKDF2, AES-GCM, imported on use). The
  security workstream's `core/secret_store.py` and `core/command_auth.py` need only the stdlib.
  Until `ble_pop.py` is merged the pin is unused (the checker allows an unused pin).
  The integrator should confirm the BLE agent tested against this version.
* `scripts/release/check_device_deps.py` checks both directions statically: every
  third-party import in `main.py`, `scripts/preflight.py`, `core/`, `services/`,
  `modules/` and the vendored DashSentinel code it loads is pinned, apt-provided
  (picamera2, libcamera, hailo_platform, gpiozero, dbus, gi) or a listed Mac-only
  import (torch, ultralytics, anthropic, each confined to its file); and no
  test/dev/research package is pinned. `tests/test_device_dependencies.py` enforces it.
* `requirements-device.lock`: hash-locked for CPython 3.11 aarch64 (Bookworm), from
  metadata only. **M8.5: the only device install** (`--require-hashes
  --only-binary=:all:`) in `setup_pi.sh`, `make deploy`'s staged install and the
  rollback (docs/DEPLOYMENT.md §5); gpiozero/lgpio are in it. Its first install
  on a real Pi is still hardware-pending.
* Pre-existing finding, not changed: mediapipe 0.10.14 also pulls
  `opencv-contrib-python==4.11.0.86` beside the pinned `opencv-python==4.10.0.84`.
  Both install a `cv2` package and the last one installed wins, so the device may not
  run the opencv version CI pins. Resolve in a dedicated change (e.g. pin
  `opencv-contrib-python` alone), with a device test.
