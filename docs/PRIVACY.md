# Privacy (M8.4)

This page covers what the camera shows, to whom, and when. It also covers what is
kept, for how long, and what deleting an account does. It describes the code as of
M8.4 (`feat/m8.4`, base `c9029fa`). The security review is in
`docs/SECURITY_REVIEW_M6.md` §7.

## 1. Camera access: owner Decision C

**While monitoring is stopped, no camera frame leaves the device unless the owner
turned on camera-only mode.**

| Monitoring | Camera-only (owner) | Remote viewer (the app on the LAN) | Viewer on the device itself (loopback) |
|---|---|---|---|
| running (ready / degraded / failed) | any | analysed view; the raw camera labelled `camera` only while a running module cannot draw | same |
| **stopped** | **off (default)** | **no frames.** Heartbeats say `camera_off_while_stopped`, and the app shows "Camera off — monitoring stopped" | raw camera, labelled `camera_local` (local maintenance), unless `dashboard.local_camera_while_stopped: false` |
| stopped | on | raw camera, labelled `camera_only`. The app shows "LIVE · Camera only — not monitoring" | same as remote |

* **Who can turn camera-only on.** Only the device owner can turn it on. The path is
  `POST /v1/devices/{id}/camera-mode` with `{"camera_only": true, "confirm": true}`,
  which is owner-scoped. Another account gets the same 404 as a device that does not
  exist.
* **How the setting reaches the device.** The backend delivers it as `set_camera_mode`
  in the signed ingest response (sd-auth-v1).
  * The device turns camera-only **on** only if the envelope verifies, whatever
    `cloud.command_auth` says.
  * Turning it **off** is the safe direction, and is accepted like any other command.
* **There is no dashboard endpoint for this setting, and no `config.yaml` key.** The
  dashboard token is a LAN credential that may be shared with people who are not the
  owner.
* **Separate from monitoring.** Stopping or resuming never changes the setting.
  Turning camera-only on never stops monitoring.
* **What the app shows.** The app shows what the device reports
  (`status.camera_mode`: `camera_only`, `remote_stream`), with the owner's request
  beside it as `desired_camera_only` / `pending`. A device that has not confirmed is
  still shown as off.
* **Across restarts and updates.**
  * The setting is stored in `data/device_state.json` under `camera_privacy`.
    Anything other than exactly `{"schema": 1, "camera_only": true}` reads as off.
    That covers every device upgraded from M8.3, an unreadable file, another schema,
    and a non-boolean value.
  * The backend column `devices.camera_mode` is NULL after migration, which means off.
  * The backend re-sends the owner's choice whenever the device reports something else.
  * A deploy rollback never touches `data/`.
* **A new owner never inherits camera-only.** Claiming an unowned device, and deleting
  an account, both set it explicitly off.

**Why loopback stays allowed.** A process on the device can already open the camera
directly. `scripts/hw_validation/capture_calibration_frames.py` does so, with the
service stopped. Serving `/stream` to `127.0.0.1` therefore grants nothing new, and the
dashboard token is still required wherever one is configured. No runbook step and no
maintenance tool reads `/stream` remotely. In the virtual device, the iOS Simulator
connects over loopback. To see what a phone sees, run it with
`dashboard.local_camera_while_stopped: false`.

**Old firmware.**
* An M8.3 device has no camera block. It streams the raw camera while stopped, and the
  owner cannot switch that off. The backend refuses the setting with
  `409 firmware_unsupported`, and the app says "This device's software can't turn the
  camera off while monitoring is stopped. Update it."
* A pre-M8.3 device freezes on its last frame.

## 2. Footage

* **Frames are never written to disk.** The exception is opt-in evidence snapshots
  (`save_evidence: false` by default). When enabled, the newest 50 are kept on the
  device and they are never uploaded.
* **The backend never receives video.** The live picture goes device → phone over the
  LAN only, as HTTP MJPEG. It is not encrypted on the Wi-Fi (production item:
  device TLS, `SECURITY_REVIEW_M6.md` §6).

## 3. Retention

| Where | What | Kept |
|---|---|---|
| backend `events` | type, time, confidence, signals (may include rule text the owner wrote) | 90 days by `received_at` (`SENSORDYME_EVENT_RETENTION_DAYS`) and at most 100,000 per device. Pruned when an ingest triggers it. |
| backend `devices` | owner's names, areas, AI rules and instructions, last health | until the account is deleted (§5) |
| device `data/events.csv` | every event | **not rotated**: the device's life (gap: §6) |
| device `data/uplink_spool.jsonl` | events not yet delivered | bounded (`queue_max`, max 1000) |
| device telemetry CSV | performance counters | 25 MB, one rotated copy |
| device `data/device_state.json`, `monitoring.json` | mode, areas, rules, camera-only | until changed or the device is reset |

## 4. Research evidence

Dataset capture (`core/dataset.py`, `docs/DATASET_COLLECTION.md`) is off unless it is
configured. Captured frames stay on the machine that captured them. People who appear
in research recordings must have agreed to it. That is an operator obligation; the
code does not enforce it.

## 5. Deleting an account

`POST /v1/account/delete` needs `{"password": ..., "confirm": true}`. It runs as one
transaction, and the response counts what it did.

Deleted:
* every event of every device the account owns;
* its orders;
* the user row, which holds the email, the password hash, the phone number and the
  ntfy topic.

Every session ends.

Each owned device is **released**:
* its owner is cleared;
* the owner's configuration is cleared (areas, AI rules and drafts, instructions,
  pending commands, the stored monitoring report);
* its name is reset;
* camera-only is set explicitly off.

The device can then be claimed again, and a new owner sees none of the old history.

**Not covered (stated, not hidden):**
* **Copies on the device itself.** `data/events.csv`, saved rules, areas and the
  camera-only setting stay until the device is reset by hand. No remote wipe command
  exists.
* **The device key.** It is not rotated, so the device keeps working for its next owner.
* **SQLite free pages.** They may still hold deleted rows until SQLite reuses them.
  There is no `secure_delete` and no VACUUM.
* **Backups.** Copies of the database made before the deletion are outside this code.
* **The app.** It does not offer the action yet. The IOS stream must wire it.

## 6. Known gaps

* `data/events.csv` on the device is never rotated.
* There is no remote wipe of a released device.
* The live picture and the dashboard token cross the LAN in cleartext.
