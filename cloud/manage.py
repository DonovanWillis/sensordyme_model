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

    python cloud/manage.py revoke-device <device id>
        -> refuses every request from the device (current and previous key);
           the app shows its credentials as revoked. Undo with rotate-key,
           which issues a fresh key (the revoked one never works again).

    python cloud/manage.py require-signed <device id> [--force]
        -> this device must use signed requests (sd-auth-v1), whatever
           SENSORDYME_DEVICE_AUTH says. Refused until the device has been seen
           signing, unless --force. `allow-bearer <device id>` undoes it.
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
            if r["auth_revoked_at"]:
                rotating += "  REVOKED"
            rotating += f"  auth={r['last_auth'] or '-'}{'(required)' if r['auth_mode'] == 'required' else ''}"
            print(
                f"[{r['id']}] {r['name']}  owner={r['owner_id']}  preset={r['active_preset']}  "
                f"claim={r['claim_code']}  kid={r['auth_kid']}{rotating}"
            )


def _remember_revoked(conn, device_id: int, secret, kid) -> None:
    """Keep a revoked key so a request made with it is reported as revoked (the
    backend never authenticates anything with it)."""
    if secret and not kid:
        try:
            kid = key_id(secret)
        except ValueError:  # a pre-sd-auth-v1 key under 16 bytes: nothing to recognise it by
            return
    if secret:
        conn.execute("INSERT OR REPLACE INTO revoked_device_keys (kid, device_id, device_key, revoked_at)"
                     " VALUES (?, ?, ?, ?)", (kid, device_id, secret, time.time()))


def rotate_key(device_id: int) -> str:
    """Issue a new key for one device; the old key becomes `previous`.

    Refused while a rotation is already open: a second rotation would silently
    drop the key the device may still be using. On a revoked device the old key
    is revoked outright instead, and the device is accepted again with the new one.
    """
    init_db()
    new_key = _new_key()
    with db() as conn:
        row = conn.execute("SELECT * FROM devices WHERE id = ?", (device_id,)).fetchone()
        if row is None:
            raise SystemExit(f"no device {device_id}")
        if row["auth_revoked_at"]:
            _remember_revoked(conn, device_id, row["device_key"], row["auth_kid"])
            conn.execute(
                "UPDATE devices SET device_key = ?, auth_kid = ?, previous_device_key = NULL,"
                " previous_auth_kid = NULL, auth_revoked_at = NULL, auth_refusal = NULL WHERE id = ?",
                (new_key, key_id(new_key), device_id),
            )
            print(f"device {device_id}: re-instated with a new key (kid {key_id(new_key)}); "
                  f"revoked kid {row['auth_kid']} stays refused.")
            print(f"  device_key (config.yaml cloud.device_key): {new_key}")
            return new_key
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
        row = conn.execute("SELECT previous_device_key, previous_auth_kid FROM devices WHERE id = ?",
                           (device_id,)).fetchone()
        if row is None:
            raise SystemExit(f"no device {device_id}")
        _remember_revoked(conn, device_id, row["previous_device_key"], row["previous_auth_kid"])
        conn.execute("UPDATE devices SET previous_device_key = NULL, previous_auth_kid = NULL WHERE id = ?",
                     (device_id,))
    print(f"device {device_id}: previous key {row['previous_auth_kid'] or '(none)'} revoked")


def revoke_device(device_id: int) -> None:
    """Refuse every request from one device until it is given a new key.

    The current key stays on the row (so its requests are recognised and the
    app is told "revoked", not "offline"); any previous key is revoked too.
    """
    init_db()
    with db() as conn:
        row = conn.execute("SELECT * FROM devices WHERE id = ?", (device_id,)).fetchone()
        if row is None:
            raise SystemExit(f"no device {device_id}")
        _remember_revoked(conn, device_id, row["previous_device_key"], row["previous_auth_kid"])
        conn.execute("UPDATE devices SET auth_revoked_at = ?, previous_device_key = NULL,"
                     " previous_auth_kid = NULL WHERE id = ?", (time.time(), device_id))
    print(f"device {device_id}: credentials revoked (kid {row['auth_kid']}). "
          "Every request from it is refused; `rotate-key` re-instates it with a new key.")


def set_auth_mode(device_id: int, required: bool, *, force: bool = False) -> None:
    """Per-device enforcement of signed requests (it can only tighten the backend mode)."""
    init_db()
    with db() as conn:
        row = conn.execute("SELECT last_auth FROM devices WHERE id = ?", (device_id,)).fetchone()
        if row is None:
            raise SystemExit(f"no device {device_id}")
        if required and row["last_auth"] != "signed" and not force:
            raise SystemExit(f"device {device_id} has not been seen signing (last: {row['last_auth'] or 'never'}); "
                             "set cloud.command_auth: required on it first, or pass --force")
        conn.execute("UPDATE devices SET auth_mode = ? WHERE id = ?", ("required" if required else None, device_id))
    print(f"device {device_id}: " + ("signed requests required" if required else "follows SENSORDYME_DEVICE_AUTH"))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "create-device":
        create_device(sys.argv[2])
    elif len(sys.argv) >= 2 and sys.argv[1] == "list-devices":
        list_devices()
    elif len(sys.argv) >= 3 and sys.argv[1] == "rotate-key":
        rotate_key(int(sys.argv[2]))
    elif len(sys.argv) >= 3 and sys.argv[1] == "revoke-previous-key":
        revoke_previous_key(int(sys.argv[2]))
    elif len(sys.argv) >= 3 and sys.argv[1] == "revoke-device":
        revoke_device(int(sys.argv[2]))
    elif len(sys.argv) >= 3 and sys.argv[1] == "require-signed":
        set_auth_mode(int(sys.argv[2]), True, force="--force" in sys.argv[3:])
    elif len(sys.argv) >= 3 and sys.argv[1] == "allow-bearer":
        set_auth_mode(int(sys.argv[2]), False)
    else:
        print(__doc__)
