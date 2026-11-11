"""Device credentials at rest (core/secret_store.py, scripts/migrate_device_key.py)
and the config checks that go with them (schema, deploy preflight)."""

from __future__ import annotations

import json
import logging
import os
import pickle
import stat
from pathlib import Path

import pytest
import yaml

from core import secret_store as ss

REPO = Path(__file__).resolve().parents[1]
KEY = "dk_" + "88" * 24
OLD = "dk_" + "99" * 24


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


# ------------------------------------------------------------------ write / read
def test_written_0600_and_read_back(tmp_path):
    path = tmp_path / "creds" / "credentials.json"
    previous_umask = os.umask(0o000)  # the most permissive umask still gives 0600
    try:
        ss.write_credentials(path, {"device_key": KEY})
    finally:
        os.umask(previous_umask)
    assert _mode(path) == 0o600
    creds = ss.load_credentials(path)
    assert creds.get("device_key") == KEY and creds.warnings == [] and not creds.recovered_from_backup
    assert [p.name for p in path.parent.iterdir()] == ["credentials.json"], "no temp file left behind"


def test_an_update_keeps_the_previous_version_as_a_0600_backup(tmp_path):
    path = tmp_path / "credentials.json"
    ss.write_credentials(path, {"device_key": OLD})
    ss.write_credentials(path, {"device_key": KEY, "previous_device_key": OLD})
    assert ss.load_credentials(path).get("previous_device_key") == OLD
    assert _mode(ss.backup_path(path)) == 0o600
    assert ss.load_credentials(ss.backup_path(path), allow_backup=False).get("device_key") == OLD


def test_an_interrupted_write_leaves_the_old_file_intact(tmp_path, monkeypatch):
    path = tmp_path / "credentials.json"
    ss.write_credentials(path, {"device_key": OLD})

    def crash(*_a, **_k):
        raise OSError("disk full")
    monkeypatch.setattr(ss.os, "replace", crash)
    with pytest.raises(OSError, match="disk full"):
        ss.write_credentials(path, {"device_key": KEY})
    monkeypatch.undo()
    assert ss.load_credentials(path, allow_backup=False).get("device_key") == OLD
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


@pytest.mark.parametrize("secrets", [{}, {"device_key": ""}, {"device_key": KEY, "password": "x"},
                                     {"device_key": 5}])
def test_bad_contents_are_refused_at_write(tmp_path, secrets):
    with pytest.raises(ss.CredentialError):
        ss.write_credentials(tmp_path / "c.json", secrets)


# ------------------------------------------------------------------ permissions
@pytest.mark.parametrize("mode", [0o640, 0o644, 0o604, 0o666])
def test_a_readable_file_is_refused_in_strict_mode(tmp_path, mode):
    path = tmp_path / "c.json"
    ss.write_credentials(path, {"device_key": KEY})
    os.chmod(path, mode)
    with pytest.raises(ss.CredentialError, match="must be 0600") as err:
        ss.load_credentials(path, allow_backup=False)
    assert KEY not in str(err.value)
    loose = ss.load_credentials(path, strict=False, allow_backup=False)
    assert loose.get("device_key") == KEY and "0600" in loose.warnings[0]


def test_a_symlink_is_refused_even_when_not_strict(tmp_path):
    real = tmp_path / "real.json"
    ss.write_credentials(real, {"device_key": KEY})
    link = tmp_path / "link.json"
    link.symlink_to(real)
    for strict in (True, False):
        with pytest.raises(ss.CredentialError, match="symlink"):
            ss.load_credentials(link, strict=strict, allow_backup=False)


def test_a_missing_file_is_an_error_not_an_empty_key(tmp_path):
    with pytest.raises(ss.CredentialError, match="does not exist"):
        ss.load_credentials(tmp_path / "absent.json")


# ------------------------------------------------------------------ integrity and recovery
@pytest.mark.parametrize("damage", [
    lambda b: b[: len(b) // 2],                        # truncated mid-write
    lambda b: b"",                                     # zero-length after a crash
    lambda b: b.replace(b"88", b"87", 1),              # one character edited
    lambda b: b"\xff\xfe" + b,                         # not UTF-8
    lambda b: json.dumps({"format": "other"}).encode(),
])
def test_a_damaged_file_falls_back_to_the_backup_and_says_so(tmp_path, damage):
    path = tmp_path / "c.json"
    ss.write_credentials(path, {"device_key": OLD})
    ss.write_credentials(path, {"device_key": KEY})
    path.write_bytes(damage(path.read_bytes()))
    os.chmod(path, 0o600)
    creds = ss.load_credentials(path)
    assert creds.recovered_from_backup and creds.get("device_key") == OLD
    assert "backup" in creds.warnings[0] and KEY not in creds.warnings[0] and OLD not in creds.warnings[0]
    ss.restore_from_backup(path)
    assert ss.load_credentials(path, allow_backup=False).get("device_key") == OLD and _mode(path) == 0o600


def test_both_copies_damaged_is_an_explicit_error(tmp_path):
    path = tmp_path / "c.json"
    ss.write_credentials(path, {"device_key": OLD})
    ss.write_credentials(path, {"device_key": KEY})
    for p in (path, ss.backup_path(path)):
        p.write_bytes(b"{")
    with pytest.raises(ss.CredentialError, match="backup unusable too"):
        ss.load_credentials(path)


def test_a_damaged_file_never_overwrites_a_good_backup(tmp_path):
    path = tmp_path / "c.json"
    ss.write_credentials(path, {"device_key": OLD})
    ss.write_credentials(path, {"device_key": KEY})  # backup = OLD
    path.write_bytes(b"garbage")
    ss.write_credentials(path, {"device_key": "dk_" + "aa" * 24})
    assert ss.load_credentials(ss.backup_path(path), allow_backup=False).get("device_key") == OLD


# ------------------------------------------------------------------ redaction and leaks
def test_a_secret_never_prints_itself():
    secret = ss.Secret(KEY)
    for text in (repr(secret), str(secret), f"{secret}", f"{secret:>80}", "%s" % secret,  # noqa: UP031 - %-formatting is one of the paths tested repr([secret]),
                 repr({"k": secret})):
        assert KEY not in text and "redacted" in text
    assert secret.reveal() == KEY and secret == ss.Secret(KEY) and secret != ss.Secret(OLD)
    with pytest.raises(TypeError):
        pickle.dumps(secret)


def test_credentials_repr_has_no_secret(tmp_path):
    path = tmp_path / "c.json"
    ss.write_credentials(path, {"device_key": KEY, "previous_device_key": OLD})
    text = repr(ss.load_credentials(path))
    assert KEY not in text and OLD not in text


def test_redact_and_the_logging_filter(caplog):
    assert ss.redact(f"key={KEY}; old={OLD}", [ss.Secret(KEY), OLD]) == "key=[REDACTED]; old=[REDACTED]"
    logger = logging.getLogger("sd.test.redact")
    logger.addFilter(ss.RedactingFilter([KEY]))
    with caplog.at_level(logging.INFO, logger="sd.test.redact"):
        logger.info("connecting with %s", KEY)
        logger.info("no secret here")
    assert KEY not in caplog.text and "[REDACTED]" in caplog.text and "no secret here" in caplog.text


def test_config_permission_warnings(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("x: 1\n")
    os.chmod(path, 0o644)
    assert ss.config_permission_warnings(path, {"cloud": {"device_key": ""}}) == []
    warning = ss.config_permission_warnings(path, {"cloud": {"device_key": KEY}})
    assert len(warning) == 1 and "0644" in warning[0] and KEY not in warning[0]
    os.chmod(path, 0o600)
    assert ss.config_permission_warnings(path, {"cloud": {"device_key": KEY}, "dashboard": {"token": "t"}}) == []


def test_resolve_cloud_credentials(tmp_path):
    store = tmp_path / "c.json"
    ss.write_credentials(store, {"device_key": KEY, "previous_device_key": OLD})
    cloud = {"enabled": True, "device_key": "", "device_key_file": str(store)}
    resolved, warnings = ss.resolve_cloud_credentials(cloud)
    assert resolved["device_key"] == KEY and resolved["previous_device_key"] == OLD and warnings == []
    assert cloud["device_key"] == "", "the caller's config is not mutated"
    assert ss.resolve_cloud_credentials({"device_key": KEY}) == ({"device_key": KEY}, [])
    os.chmod(store, 0o644)
    with pytest.raises(ss.CredentialError):
        ss.resolve_cloud_credentials(cloud)


# ------------------------------------------------------------------ migration (dry run and apply)
def _repo_config_copy(tmp_path, key=KEY, mode=0o644) -> Path:
    text = (REPO / "config.yaml").read_text()
    assert text.count('device_key: ""') == 1
    path = tmp_path / "config.yaml"
    path.write_text(text.replace('device_key: ""', f'device_key: "{key}"'))
    os.chmod(path, mode)
    return path


def test_the_dry_run_writes_nothing_and_prints_no_key(tmp_path, capsys):
    from scripts.migrate_device_key import main

    config = _repo_config_copy(tmp_path)
    before = config.read_bytes()
    assert main(["--config", str(config), "--store", str(tmp_path / "store.json")]) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and KEY not in out and ss.Secret(KEY).fingerprint() in out
    assert config.read_bytes() == before and sorted(p.name for p in tmp_path.iterdir()) == ["config.yaml"]


def test_apply_moves_the_key_and_keeps_every_comment(tmp_path, capsys):
    from scripts.migrate_device_key import main

    config = _repo_config_copy(tmp_path)
    original = yaml.safe_load(config.read_text())
    comments_before = [line for line in config.read_text().splitlines() if line.lstrip().startswith("#")]
    store = tmp_path / "secrets" / "store.json"
    assert main(["--config", str(config), "--store", str(store), "--apply"]) == 0
    assert KEY not in capsys.readouterr().out
    migrated = yaml.safe_load(config.read_text())
    assert KEY not in config.read_text()
    assert migrated["cloud"]["device_key"] == "" and migrated["cloud"]["device_key_file"] == str(store)
    assert {k: v for k, v in migrated.items() if k != "cloud"} == {k: v for k, v in original.items() if k != "cloud"}
    assert [line for line in config.read_text().splitlines() if line.lstrip().startswith("#")] == comments_before
    assert _mode(config) == 0o644, "the config keeps its own mode"
    assert ss.load_credentials(store).get("device_key") == KEY and _mode(store) == 0o600
    backup = config.with_name("config.yaml.pre-migration")
    assert _mode(backup) == 0o600 and KEY in backup.read_text()
    resolved, _ = ss.resolve_cloud_credentials(migrated["cloud"])
    assert resolved["device_key"] == KEY
    from core.config_schema import validate
    errors, _ = validate(migrated)
    assert errors == []
    assert main(["--config", str(config), "--store", str(store)]) == 1, "already migrated"


def test_migration_refuses_what_it_cannot_edit_safely(tmp_path):
    from scripts.migrate_device_key import plan

    config = _repo_config_copy(tmp_path, key="")
    with pytest.raises(ValueError, match="nothing to migrate"):
        plan(config, tmp_path / "s.json")
    config.write_text(f"cloud:\n  device_key: |\n    {KEY}\n")
    with pytest.raises(ValueError, match="re-parse"):
        plan(config, tmp_path / "s.json")


def test_restore_backup_cli(tmp_path):
    from scripts.migrate_device_key import main

    store = tmp_path / "s.json"
    ss.write_credentials(store, {"device_key": OLD})
    ss.write_credentials(store, {"device_key": KEY})
    store.write_bytes(b"{")
    assert main(["--restore-backup", str(store)]) == 0
    assert ss.load_credentials(store, allow_backup=False).get("device_key") == OLD


# ------------------------------------------------------------------ schema and deploy preflight
@pytest.mark.parametrize(("cloud", "needle"), [
    ({"command_auth": "on"}, "command_auth"),
    ({"revoked_key_ids": ["XYZ"]}, "revoked_key_ids"),
    ({"previous_device_key": 5}, "previous_device_key"),
    ({"device_key": KEY, "device_key_file": "/x"}, "both set"),
])
def test_the_schema_refuses_bad_auth_settings(cloud, needle):
    from core.config_schema import validate

    config = yaml.safe_load((REPO / "config.yaml").read_text())
    config["cloud"].update(cloud)
    errors, _ = validate(config)
    assert any(needle in e for e in errors), errors


def test_the_schema_accepts_a_key_file_instead_of_a_key():
    from core.config_schema import validate

    config = yaml.safe_load((REPO / "config.yaml").read_text())
    config["cloud"].update(enabled=True, device_key="", device_key_file="/etc/sensordyme/credentials.json",
                           command_auth="required", revoked_key_ids=["0123456789abcdef"])
    errors, warnings = validate(config)
    assert errors == [] and not any("device_key is empty" in w for w in warnings)


@pytest.mark.parametrize(("mode", "refused", "warned"), [
    ("off", True, False), ("permissive", False, True), (None, False, True), ("required", False, False),
])
def test_the_deploy_preflight_refuses_unsigned_commands(tmp_path, mode, refused, warned):
    from scripts.deploy_preflight import _device_profile_and_exposure

    cloud = {"enabled": True, "base_url": "https://b.example", "device_key": KEY}
    if mode is not None:
        cloud["command_auth"] = mode
    errors, warnings = _device_profile_and_exposure({"profile": "pi", "cloud": cloud})
    assert any("command_auth" in e for e in errors) == refused
    assert any("command_auth" in w for w in warnings) == warned
