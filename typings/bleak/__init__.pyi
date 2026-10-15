# Minimal stub for an optional, dev-machine-only module.
#
# bleak is used only by scripts/ble_provision_client.py, a developer tool for
# exercising BLE provisioning from a laptop. It is deliberately in no
# requirements file (`pip install bleak` to run that script), so pyright
# reported the import as unresolved and failed the blocking typecheck job.
#
# Any-typed on purpose, like the other stubs here: this exists only to make
# the import resolve, not to claim knowledge of bleak's API.
#
# See pyrightconfig.json -> stubPath.

from typing import Any

def __getattr__(name: str) -> Any: ...
