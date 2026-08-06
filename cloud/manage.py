"""Admin CLI for the SensorDyme cloud backend.

    python cloud/manage.py create-device "Car unit"
        -> prints the device_key (goes in the device's config.yaml under cloud:)
           and the claim_code (the customer types this into the iOS app to pair)

    python cloud/manage.py list-devices
"""

from __future__ import annotations

import secrets
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from cloud.app import db, init_db  # noqa: E402


def create_device(name: str) -> None:
    init_db()
    claim_code = "-".join(secrets.token_hex(2) for _ in range(3)).upper()
    device_key = "dk_" + secrets.token_hex(24)
    with db() as conn:
        conn.execute(
            "INSERT INTO devices (name, claim_code, device_key, created_at) VALUES (?, ?, ?, ?)",
            (name, claim_code, device_key, time.time()),
        )
    print(f"device created: {name}")
    print(f"  device_key (config.yaml cloud.device_key): {device_key}")
    print(f"  claim_code (typed into the app to pair):   {claim_code}")


def list_devices() -> None:
    init_db()
    with db() as conn:
        for r in conn.execute("SELECT * FROM devices"):
            print(
                f"[{r['id']}] {r['name']}  owner={r['owner_id']}  preset={r['active_preset']}  "
                f"claim={r['claim_code']}"
            )


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "create-device":
        create_device(sys.argv[2])
    elif len(sys.argv) >= 2 and sys.argv[1] == "list-devices":
        list_devices()
    else:
        print(__doc__)
