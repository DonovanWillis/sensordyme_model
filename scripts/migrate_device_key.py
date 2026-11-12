#!/usr/bin/env python3
"""Move the device key out of config.yaml into a 0600 credentials file.

    python scripts/migrate_device_key.py --config config.yaml --store /etc/sensordyme/credentials.json
        dry run (the default): says what would change, writes nothing
    ... --apply
        writes the credentials file (0600, atomic), then rewrites config.yaml so
        `cloud.device_key` is "" and `cloud.device_key_file` points at the store
    python scripts/migrate_device_key.py --restore-backup <store>
        replaces a damaged credentials file with its .bak

The config is edited line by line, not re-dumped, so its comments survive; the
result is re-parsed and must equal the original except for those keys, or
nothing is written. The original config is kept beside it as
`<config>.pre-migration` (mode 0600, it still holds the key: delete it once the
device has reconnected). Key values are never printed: only their key ids.

Plan and rollback: docs/CREDENTIAL_STORAGE.md. Nothing here rotates a key.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

from core.secret_store import (
    CredentialError,
    Secret,
    load_credentials,
    restore_from_backup,
    write_credentials,
)

_MIGRATED = ("device_key", "previous_device_key")


def _cloud_block(lines: list[str]) -> tuple[int, int]:
    """[start, end) line range of the top-level `cloud:` mapping."""
    starts = [i for i, line in enumerate(lines) if re.match(r"^cloud:\s*(#.*)?$", line)]
    if len(starts) != 1:
        raise ValueError(f"expected one top-level `cloud:` line, found {len(starts)}")
    start = starts[0]
    end = next((i for i in range(start + 1, len(lines)) if re.match(r"^\S", lines[i])), len(lines))
    return start, end


def plan(config_path: Path, store: Path) -> tuple[str, dict, list[str]]:
    """(new config text, secrets to store, human-readable steps). Raises ValueError."""
    text = config_path.read_text()
    try:
        original = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        # Line and column only: PyYAML's own message quotes the offending line,
        # which may be the key itself (M8).
        mark = getattr(exc, "problem_mark", None)
        where = f" (line {mark.line + 1}, column {mark.column + 1})" if mark is not None else ""
        raise ValueError(f"{config_path} is not valid YAML{where}; fix it first. Nothing written.") from None
    cloud = (original or {}).get("cloud") if isinstance(original, dict) else None
    if not isinstance(cloud, dict):
        raise ValueError("config has no `cloud:` mapping")
    if cloud.get("device_key_file"):
        raise ValueError(f"already migrated: cloud.device_key_file is {cloud['device_key_file']!r}")
    secrets = {name: cloud[name] for name in _MIGRATED if isinstance(cloud.get(name), str) and cloud[name]}
    if "device_key" not in secrets:
        raise ValueError("cloud.device_key is empty: nothing to migrate")
    lines = text.splitlines(keepends=True)
    start, end = _cloud_block(lines)
    steps = []
    key_line = None
    for name in _MIGRATED:
        hits = [i for i in range(start + 1, end) if re.match(rf"^(\s+){name}:", lines[i])]
        if name in secrets and len(hits) != 1:
            raise ValueError(f"expected one `{name}:` line under cloud:, found {len(hits)}")
        for i in hits:
            indent = re.match(r"^(\s+)", lines[i]).group(1)
            lines[i] = f'{indent}{name}: "" # moved to cloud.device_key_file (docs/CREDENTIAL_STORAGE.md)\n'
            if name == "device_key":
                key_line = i
    indent = re.match(r"^(\s+)", lines[key_line]).group(1)
    lines.insert(key_line + 1, f"{indent}device_key_file: {store}\n")
    new_text = "".join(lines)
    try:
        reparsed = yaml.safe_load(new_text)
    except yaml.YAMLError:
        reparsed = None
    expected = dict(original)
    expected["cloud"] = {**cloud, **{name: "" for name in _MIGRATED if name in cloud}, "device_key_file": str(store)}
    if reparsed != expected:
        raise ValueError("the edited config does not re-parse to the expected values; nothing written")
    for name, value in secrets.items():
        steps.append(f"store cloud.{name} (kid {Secret(value).fingerprint()}) in {store} (0600)")
    steps.append(f"set cloud.device_key_file: {store}; blank the key line(s) in {config_path}")
    steps.append(f"keep the original as {config_path}.pre-migration (0600; delete after the device reconnects)")
    return new_text, secrets, steps


def apply(config_path: Path, store: Path) -> None:
    new_text, secrets, _ = plan(config_path, store)
    write_credentials(store, secrets)
    stored = load_credentials(store, strict=True, allow_backup=False)
    if any(stored.get(k) != v for k, v in secrets.items()):
        raise CredentialError("the credentials file did not read back identically; config left unchanged")
    backup = config_path.with_name(config_path.name + ".pre-migration")
    fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as out, open(config_path, "rb") as src:
        shutil.copyfileobj(src, out)
    mode = os.stat(config_path).st_mode & 0o777
    tmp = config_path.with_name(f".{config_path.name}.migrate.tmp")
    # The config keeps its own mode: it no longer holds the key, and a service
    # user may read it through its group (deploy/systemd least-privilege unit).
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.fchmod(fd, mode)
    with os.fdopen(fd, "w") as out:
        out.write(new_text)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, config_path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--store", type=Path)
    parser.add_argument("--apply", action="store_true", help="write (default: dry run)")
    parser.add_argument("--restore-backup", type=Path, metavar="STORE")
    args = parser.parse_args(argv)
    try:
        if args.restore_backup:
            restore_from_backup(args.restore_backup)
            print(f"restored {args.restore_backup} from its backup")
            return 0
        if not args.config or not args.store:
            parser.error("--config and --store are required")
        _, _, steps = plan(args.config, args.store)
        print(("APPLYING" if args.apply else "DRY RUN (nothing written; add --apply)") + ":")
        for step in steps:
            print(f"  - {step}")
        if args.apply:
            apply(args.config, args.store)
            print("done. Restart the service and confirm the device reconnects; then delete the .pre-migration copy.")
        return 0
    except (ValueError, CredentialError, OSError) as exc:
        print(f"migrate_device_key: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
