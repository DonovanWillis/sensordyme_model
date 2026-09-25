"""What the deployed backend has to be true of, pinned so a config edit cannot
quietly undo it.

Three of these guard failures that are invisible until they are expensive:

  * a deploy without a persistent volume looks perfect and loses every account
    on the next restart, because the JWT signing key is regenerated with the
    database it sits beside;
  * mDNS left on announces a datacentre-internal address to nobody, and made the
    app's discovery path look like it still had a job to do; and
  * rate limiting behind a proxy silently degrades from per-caller to global —
    ten requests from anyone locks every account out of login.

The rest pin the plain requirements: HTTPS, a real health route, one writer.
"""

from __future__ import annotations

import importlib
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
FLY_TOML = REPO / "fly.toml"
DOCKERFILE = REPO / "cloud" / "Dockerfile"
DOCKERIGNORE = REPO / ".dockerignore"


@pytest.fixture(scope="module")
def fly_config() -> dict:
    """fly.toml, parsed. tomllib is stdlib from 3.11."""
    import tomllib

    return tomllib.loads(FLY_TOML.read_text())


# --------------------------------------------------------------- durable state

def test_the_database_and_jwt_secret_live_on_a_mounted_volume(fly_config):
    """The single most damaging way to get this wrong, and the quietest.

    cloud/app.py's _load_jwt_secret() persists the signing key next to the
    database unless told otherwise. Put either on the container's ephemeral
    filesystem and every redeploy mints a new key, so tokens that were valid
    when issued come back `Signature verification failed` -- while the server
    itself looks completely healthy.
    """
    mounts = fly_config["mounts"]
    assert len(mounts) == 1, "exactly one volume is expected"
    destination = mounts[0]["destination"]

    env = fly_config["env"]
    for key in ("SENSORDYME_DB", "SENSORDYME_JWT_SECRET_FILE"):
        assert env[key].startswith(destination + "/"), (
            f"{key}={env[key]!r} is not on the persistent volume at {destination}"
        )


def test_only_one_machine_writes_to_the_sqlite_database(fly_config):
    """SQLite on one volume has exactly one writer. Two machines sharing it is
    corruption, not throughput -- so nothing here may quietly scale it out."""
    service = fly_config["http_service"]
    assert service["min_machines_running"] == 1
    assert service["auto_stop_machines"] is False, (
        "a device polls every 2s and a stopped machine is a missed alert"
    )


# ---------------------------------------------------------------------- HTTPS

def test_the_deployment_forces_https(fly_config):
    """iOS trusting the backend with no app changes, and the device keeping
    core/uplink.py's `verify=True`, both depend on real TLS being the only way
    in. Fly terminates it; this is what stops plain HTTP being served beside it."""
    assert fly_config["http_service"]["force_https"] is True


def test_health_checks_use_the_route_that_exists(fly_config):
    """`/` returns 404 by design -- there is no root route. A health check
    pointed at it reports a working service as failing, forever."""
    checks = fly_config["http_service"]["checks"]
    assert [c["path"] for c in checks] == ["/v1/health"]


def test_the_published_port_matches_what_the_container_serves(fly_config):
    """An internal_port that disagrees with uvicorn's --port is a deploy that
    builds, boots, passes no health check, and reports nothing useful."""
    internal = fly_config["http_service"]["internal_port"]
    assert f"--port\", \"{internal}\"" in DOCKERFILE.read_text().replace("'", '"')


# ----------------------------------------------------------------------- mDNS

def test_mdns_is_disabled_in_the_cloud_deployment(fly_config):
    """Link-local announcement from a datacentre reaches nothing. Leaving it on
    also costs a couple of seconds of startup probing for a name nobody browses."""
    assert fly_config["env"]["SENSORDYME_MDNS"] == "0"


def test_the_image_defaults_match_fly_toml(fly_config):
    """The Dockerfile states the same three values. They are the contract, and a
    disagreement between the two files is how a missing mount or a live mDNS
    advertiser survives review."""
    dockerfile = DOCKERFILE.read_text()
    for key in ("SENSORDYME_DB", "SENSORDYME_JWT_SECRET_FILE", "SENSORDYME_MDNS"):
        assert f"{key}={fly_config['env'][key]}" in dockerfile, f"{key} disagrees with fly.toml"


# ------------------------------------------------------------------- no secrets

def test_the_image_cannot_ship_the_database_or_the_signing_key():
    """The Dockerfile copies cloud/ wholesale, and on a developer's machine that
    directory holds the live accounts database, the JWT signing key and the
    server log. .dockerignore is the only thing standing between those and
    anyone who can pull the image."""
    ignored = DOCKERIGNORE.read_text()
    for pattern in ("cloud/*.db", "cloud/.jwt_secret", "cloud/*.log", ".secrets/"):
        assert pattern in ignored, f".dockerignore does not exclude {pattern}"


def test_no_tracked_deployment_file_carries_a_secret():
    """Device keys, tokens and passwords are per-deployment state. The repo has
    a test for this on config.yaml already; the deploy files need the same one."""
    for path in (FLY_TOML, DOCKERFILE, DOCKERIGNORE):
        text = path.read_text()
        assert "dk_" not in text, f"{path.name} looks like it carries a device key"
        for marker in ("SENSORDYME_JWT_SECRET=", "TWILIO_AUTH_TOKEN=", "password"):
            assert marker not in text, f"{path.name} looks like it carries a secret ({marker})"


# ------------------------------------------------------ rate limiting behind a proxy

@pytest.fixture()
def backend(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSORDYME_AI_PROVIDER", "unavailable")
    monkeypatch.setenv("SENSORDYME_DB", str(tmp_path / "test.db"))
    monkeypatch.delenv("SENSORDYME_CLIENT_IP_HEADER", raising=False)
    for mod in ("cloud.app", "cloud", "cloud.ratelimit"):
        sys.modules.pop(mod, None)
    return importlib.import_module("cloud.app")


class _FakeRequest:
    """Just enough Request for _client_ip: a peer and some headers."""

    class _Client:
        def __init__(self, host):
            self.host = host

    def __init__(self, peer, headers=None):
        self.client = None if peer is None else self._Client(peer)
        self.headers = headers or {}


def test_no_header_is_trusted_unless_one_is_named(backend):
    """The default, and every LAN deployment: a client-supplied header is
    trivially spoofed by exactly the caller being rate limited."""
    request = _FakeRequest("203.0.113.9", {"X-Forwarded-For": "1.2.3.4", "Fly-Client-IP": "5.6.7.8"})
    assert backend._client_ip(request) == "203.0.113.9"


def test_the_named_header_is_used_when_configured(backend, monkeypatch):
    """Behind Fly, the TCP peer is the proxy for every request in the world.
    Without this the login limiter has ONE bucket for all users, and ten
    attempts from anyone locks everybody out."""
    monkeypatch.setenv("SENSORDYME_CLIENT_IP_HEADER", "Fly-Client-IP")
    same_proxy = "66.51.120.4"
    first = _FakeRequest(same_proxy, {"Fly-Client-IP": "198.51.100.7"})
    second = _FakeRequest(same_proxy, {"Fly-Client-IP": "198.51.100.8"})
    assert backend._client_ip(first) == "198.51.100.7"
    assert backend._client_ip(second) == "198.51.100.8"
    assert backend._client_ip(first) != backend._client_ip(second), (
        "two phones behind the same proxy must not share a rate-limit bucket"
    )


def test_a_proxy_chain_reports_the_original_client(backend, monkeypatch):
    monkeypatch.setenv("SENSORDYME_CLIENT_IP_HEADER", "X-Forwarded-For")
    request = _FakeRequest("10.0.0.1", {"X-Forwarded-For": "198.51.100.7, 70.1.1.1, 10.0.0.1"})
    assert backend._client_ip(request) == "198.51.100.7"


def test_a_missing_or_empty_header_falls_back_to_the_peer(backend, monkeypatch):
    """Configured but absent means the request did not arrive through the proxy
    it was promised to. Returning "" would make every such caller one client."""
    monkeypatch.setenv("SENSORDYME_CLIENT_IP_HEADER", "Fly-Client-IP")
    assert backend._client_ip(_FakeRequest("203.0.113.9", {})) == "203.0.113.9"
    assert backend._client_ip(_FakeRequest("203.0.113.9", {"Fly-Client-IP": "  "})) == "203.0.113.9"
    assert backend._client_ip(_FakeRequest(None, {})) == "unknown"


def test_the_deployment_names_the_header_its_proxy_actually_sets(fly_config):
    """Fly's proxy sets Fly-Client-IP and overwrites whatever the client sent,
    which is the whole reason it is safe to believe. Naming a header the proxy
    does NOT control would hand every caller a spoofable rate-limit key."""
    assert fly_config["env"]["SENSORDYME_CLIENT_IP_HEADER"] == "Fly-Client-IP"


# ------------------------------------------------------------- database import

def _seed(path: Path) -> None:
    """A minimal but real SensorDyme database."""
    sys.modules.pop("cloud.app", None)
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT, created_at REAL);
        CREATE TABLE devices (id INTEGER PRIMARY KEY, name TEXT, owner_id INTEGER, device_key TEXT);
        CREATE TABLE events (id INTEGER PRIMARY KEY, device_id INTEGER, type TEXT);
        INSERT INTO users VALUES (1, 'owner@example.com', 0.0);
        INSERT INTO devices VALUES (1, 'unit 1', 1, 'dk_test');
        INSERT INTO events VALUES (1, 1, 'fall');
        """
    )
    conn.commit()
    conn.close()


def _import(*args) -> subprocess.CompletedProcess:
    # The S603 suppression: a fixed interpreter and a path inside this repo.
    # Running it out-of-process is the point -- the script's SystemExit(1)
    # paths are part of what is being tested.
    return subprocess.run(  # noqa: S603
        [sys.executable, str(REPO / "scripts" / "fly_import_db.py"), *map(str, args)],
        capture_output=True, text=True, check=False,
    )


def test_import_preserves_accounts_devices_and_history(tmp_path):
    """The migration's whole purpose: the account, the claimed device (and so
    its device_key, and so the Pi's credential) and the event history all land."""
    source, target = tmp_path / "import.db", tmp_path / "sensordyme.db"
    _seed(source)
    result = _import(source, "--target", target)
    assert result.returncode == 0, result.stderr

    conn = sqlite3.connect(target)
    assert conn.execute("SELECT email FROM users WHERE id = 1").fetchone()[0] == "owner@example.com"
    assert conn.execute("SELECT owner_id, device_key FROM devices WHERE id = 1").fetchone() == (1, "dk_test")
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
    conn.close()


def test_import_backs_up_what_it_replaces(tmp_path):
    """Non-destructive by construction: a wrong import must be recoverable on
    the machine, not only from whatever the operator happened to keep."""
    source, target = tmp_path / "import.db", tmp_path / "sensordyme.db"
    _seed(source)
    _seed(target)
    assert _import(source, "--target", target).returncode == 0
    assert list(tmp_path.glob("sensordyme.db.replaced-*")), "the replaced database was not backed up"


def test_a_dry_run_writes_nothing(tmp_path):
    source, target = tmp_path / "import.db", tmp_path / "sensordyme.db"
    _seed(source)
    assert _import(source, "--target", target, "--dry-run").returncode == 0
    assert not target.exists()


def test_import_removes_the_stale_wal_of_the_database_it_replaces(tmp_path):
    """cloud/app.py runs in WAL mode. A -wal left beside a DIFFERENT database is
    a corruption waiting to be opened, not a harmless leftover."""
    source, target = tmp_path / "import.db", tmp_path / "sensordyme.db"
    _seed(source)
    _seed(target)
    stale = tmp_path / "sensordyme.db-wal"
    stale.write_bytes(b"stale")
    assert _import(source, "--target", target).returncode == 0
    assert not stale.exists()


@pytest.mark.parametrize(
    ("make", "expected"),
    [
        (lambda _p: None, "does not exist"),
        (lambda p: p.write_bytes(b""), "is empty"),
        (lambda p: p.write_bytes(b"this is not a database" * 10), "not a readable SQLite"),
    ],
)
def test_a_bad_upload_is_refused_before_anything_is_written(tmp_path, make, expected):
    """An import that lands the wrong file is worse than one that fails: the
    failure is visible immediately, the wrong file is not."""
    source, target = tmp_path / "import.db", tmp_path / "sensordyme.db"
    _seed(target)
    before = target.read_bytes()
    make(source)
    result = _import(source, "--target", target)
    assert result.returncode != 0
    assert expected in result.stderr
    assert target.read_bytes() == before, "the live database was touched by a refused import"


def test_a_database_from_some_other_application_is_refused(tmp_path):
    source, target = tmp_path / "import.db", tmp_path / "sensordyme.db"
    conn = sqlite3.connect(source)
    conn.execute("CREATE TABLE unrelated (id INTEGER)")
    conn.commit()
    conn.close()
    _seed(target)
    result = _import(source, "--target", target)
    assert result.returncode != 0 and "does not look like" in result.stderr


def test_an_empty_database_never_replaces_a_live_one(tmp_path):
    """The realistic mistake: uploading the freshly-created empty file instead
    of the real one, and wiping the deployment with it."""
    source, target = tmp_path / "import.db", tmp_path / "sensordyme.db"
    _seed(source)
    conn = sqlite3.connect(source)
    conn.execute("DELETE FROM users")
    conn.commit()
    conn.close()
    _seed(target)
    result = _import(source, "--target", target)
    assert result.returncode != 0 and "no user accounts" in result.stderr


def test_the_real_backup_taken_for_this_migration_passes_every_check():
    """Not a synthetic fixture: the actual file that will be uploaded.

    Skipped when the backup is absent, so this is a check on a real migration
    rather than a permanent dependency on one developer's machine.
    """
    backups = sorted((REPO / ".secrets" / "backups").glob("sensordyme-*.db"))
    if not backups:
        pytest.skip("no migration backup present")
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "sensordyme.db"
        shutil.copy2(backups[-1], Path(tmp) / "import.db")
        result = _import(Path(tmp) / "import.db", "--target", target)
        assert result.returncode == 0, result.stderr
        conn = sqlite3.connect(target)
        assert conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] > 0
        assert conn.execute("SELECT COUNT(*) FROM devices WHERE owner_id IS NOT NULL").fetchone()[0] > 0
        conn.close()
