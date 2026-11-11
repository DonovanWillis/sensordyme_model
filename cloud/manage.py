"""Admin CLI for the SensorDyme cloud backend.

    python cloud/manage.py create-device "Car unit"
        -> prints the device_key (goes in the device's config.yaml under cloud:)
           and the claim_code (the customer types this into the iOS app to pair)

    python cloud/manage.py list-devices

    python cloud/manage.py rotate-key <device id>
        -> issues a new device key; the old one stays valid for SIGNED requests
           (sd-auth-v1) as `previous` until revoked. docs/DEVICE_AUTH_CONTRACT.md §6.

    python cloud/manage.py revoke-previous-key <device id>
        -> ends the rotation: only the current key works from now on.
"""

from __future__ import annotations

import secrets
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from cloud.app import db, init_db
from core.command_auth import key_id


def _new_key() -> str:
    return "dk_" + secrets.token_hex(24)


def create_device(name: str) -> None:
    init_db()
    claim_code = "-".join(secrets.token_hex(2) for _ in range(3)).upper()
    device_key = _new_key()
    with db() as conn:
        conn.execute(
            "INSERT INTO devices (name, claim_code, device_key, auth_kid, created_at) VALUES (?, ?, ?, ?, ?)",
            (name, claim_code, device_key, key_id(device_key), time.time()),
        )
    print(f"device created: {name}")
    print(f"  device_key (config.yaml cloud.device_key): {device_key}")
    print(f"  claim_code (typed into the app to pair):   {claim_code}")


def list_devices() -> None:
    init_db()
    with db() as conn:
        for r in conn.execute("SELECT * FROM devices"):
            rotating = f"  rotating-from={r['previous_auth_kid']}" if r["previous_auth_kid"] else ""
            print(
                f"[{r['id']}] {r['name']}  owner={r['owner_id']}  preset={r['active_preset']}  "
                f"claim={r['claim_code']}  kid={r['auth_kid']}{rotating}"
            )


def rotate_key(device_id: int) -> str:
    """Issue a new key for one device; the old key becomes `previous`.

    Refused while a rotation is already open: a second rotation would silently
    drop the key the device may still be using.
    """
    init_db()
    new_key = _new_key()
    with db() as conn:
        row = conn.execute("SELECT * FROM devices WHERE id = ?", (device_id,)).fetchone()
        if row is None:
            raise SystemExit(f"no device {device_id}")
        if row["previous_device_key"]:
            raise SystemExit(f"device {device_id} is already rotating from kid {row['previous_auth_kid']}; "
                             "run revoke-previous-key first")
        conn.execute(
            "UPDATE devices SET previous_device_key = device_key, previous_auth_kid = auth_kid,"
            " device_key = ?, auth_kid = ? WHERE id = ?",
            (new_key, key_id(new_key), device_id),
        )
    print(f"device {device_id}: new key issued (kid {key_id(new_key)}); previous kid {row['auth_kid']} "
          "still accepted for signed requests until `revoke-previous-key`.")
    print(f"  device_key (config.yaml cloud.device_key): {new_key}")
    print("  On the device: move the old value to cloud.previous_device_key, set the new one, restart.")
    return new_key


def revoke_previous_key(device_id: int) -> None:
    init_db()
    with db() as conn:
        row = conn.execute("SELECT previous_auth_kid FROM devices WHERE id = ?", (device_id,)).fetchone()
        if row is None:
            raise SystemExit(f"no device {device_id}")
        conn.execute("UPDATE devices SET previous_device_key = NULL, previous_auth_kid = NULL WHERE id = ?",
                     (device_id,))
    print(f"device {device_id}: previous key {row['previous_auth_kid'] or '(none)'} revoked")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "create-device":
        create_device(sys.argv[2])
    elif len(sys.argv) >= 2 and sys.argv[1] == "list-devices":
        list_devices()
    elif len(sys.argv) >= 3 and sys.argv[1] == "rotate-key":
        rotate_key(int(sys.argv[2]))
    elif len(sys.argv) >= 3 and sys.argv[1] == "revoke-previous-key":
        revoke_previous_key(int(sys.argv[2]))
    else:
        print(__doc__)
