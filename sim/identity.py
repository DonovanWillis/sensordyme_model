"""The virtual device's persistent identity: hardware id, device key, claim code.

A real SensorDyme's identity is its board (core/identity.py hashes the SoC
serial into SD-XXXXXXXX); its device key and claim code are issued at the
factory (`python cloud/manage.py create-device`) and flashed into config.yaml /
printed on the box. The virtual device keeps the equivalent in
<state-dir>/identity.json, so a restart is the same device: same id, same key,
still paired to the same account.

Its hardware id is "SD-SIM-" + 8 hex digits, so nothing can mistake it for a
board. (The backend only stores ids shaped exactly SD-XXXXXXXX, so for a
virtual device GET /v1/devices reports `hardware_id: null`; the id is still in
the health block it reports, beside `simulated: true`.)
"""

from __future__ import annotations

import json
import secrets
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

PREFIX = "SD-SIM-"
FILE_NAME = "identity.json"


@dataclass(frozen=True)
class SimIdentity:
    hardware_id: str
    device_key: str
    claim_code: str
    created_at: float

    @property
    def display_name(self) -> str:
        return f"Virtual SensorDyme {self.hardware_id.removeprefix(PREFIX)[:4]}"


def new_hardware_id() -> str:
    return PREFIX + secrets.token_hex(4).upper()


def load(state_dir: Path) -> Optional[SimIdentity]:
    path = Path(state_dir) / FILE_NAME
    try:
        raw = json.loads(path.read_text())
        return SimIdentity(str(raw["hardware_id"]), str(raw["device_key"]), str(raw.get("claim_code", "")),
                           float(raw.get("created_at", 0.0)))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def load_or_create(state_dir: Path, *, device_key: Optional[str] = None,
                   claim_code: Optional[str] = None) -> SimIdentity:
    """The identity in `state_dir`, created on first boot.

    A device key given on a later boot that differs from the stored one is a
    re-provisioning (a new backend row): the hardware id is kept, the key and
    claim code are replaced.
    """
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    existing = load(state_dir)
    if existing is not None and (device_key is None or device_key == existing.device_key):
        return existing
    if device_key is None:
        raise ValueError(
            f"{state_dir} has no virtual device identity yet: pass --device-key (from "
            "`python cloud/manage.py create-device`) or --factory (written by `python -m sim.backend`)"
        )
    identity = SimIdentity(
        hardware_id=existing.hardware_id if existing else new_hardware_id(),
        device_key=device_key, claim_code=claim_code or "", created_at=time.time(),
    )
    path = state_dir / FILE_NAME
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(identity), indent=2))
    tmp.replace(path)
    return identity
