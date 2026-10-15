"""Test-suite setup: importable repo root, and isolation from the real runtime state.

The repo root is ALSO where a running installation keeps its state —
cloud/sensordyme.db (accounts, devices, events), cloud/.jwt_secret,
cloud/.backend_id and data/monitoring.json, all gitignored. Without this, any
test that imported cloud.app with default settings opened (or created) the real
database and secret, the health test minted a backend identity there, and the
preset-switch tests overwrote the real monitoring.json. On a checkout with live
state that is a test run silently editing production data.

Every one of those locations is pointed at a per-session temp directory BEFORE
any test module is imported (cloud.app and cloud.mdns read them at import).
Tests that need their own paths still set them per test, as they always have.
"""

import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

_STATE = Path(tempfile.mkdtemp(prefix="sensordyme-test-state-"))
for _var, _name in (
    ("SENSORDYME_DB", "sensordyme.db"),
    ("SENSORDYME_JWT_SECRET_FILE", ".jwt_secret"),
    ("SENSORDYME_BACKEND_ID_FILE", ".backend_id"),
    ("SENSORDYME_MONITORING_PATH", "monitoring.json"),
):
    os.environ.setdefault(_var, str(_STATE / _name))
# Never announce a test backend on the developer's real network.
os.environ.setdefault("SENSORDYME_MDNS", "0")
