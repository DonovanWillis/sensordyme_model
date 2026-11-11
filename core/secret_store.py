"""Device credentials at rest (docs/CREDENTIAL_STORAGE.md).

The device key used to live only in config.yaml, a file whose mode follows the
umask and which `make deploy` deliberately never touches. This module keeps
secrets in their own file instead (`cloud.device_key_file`):

* **permissions**: the file must be a regular file (not a symlink), owned by
  the service user, with no group/other bits (0600). Looser is refused when
  loading in strict mode and reported otherwise;
* **atomic updates**: a new file is written beside the old one with O_EXCL and
  mode 0600 from the first byte, fsync'd, renamed over it, and the directory
  fsync'd; the previous good version is kept as `<file>.bak` (also 0600);
* **integrity and recovery**: the file carries a SHA-256 of its secrets, so a
  truncated or hand-damaged file is detected; loading then falls back to the
  backup and says so. Both bad -> CredentialError, and the caller runs the
  device standalone rather than crash-looping;
* **redaction**: secrets are held in `Secret`, whose repr/str/format never show
  the value; `redact()` and `RedactingFilter` scrub them from free text and logs.

Nothing here rotates a key. Rotation is an operator procedure
(docs/DEVICE_AUTH_CONTRACT.md §6).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import stat
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

FORMAT = "sensordyme-credentials"
VERSION = 1
#: The secrets a credentials file may hold.
FIELDS = ("device_key", "previous_device_key")


class CredentialError(RuntimeError):
    """A credentials file cannot be used. The message never contains a secret."""


class Secret:
    """A secret string that does not print itself."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("a secret must be a string")
        self._value = value

    def reveal(self) -> str:
        return self._value

    def fingerprint(self) -> str:
        """Public key id (sd-auth-v1 kid) when the value is a usable device key."""
        from core.command_auth import key_id

        try:
            return key_id(self._value)
        except ValueError:
            return "unusable"

    def __repr__(self) -> str:
        return f"Secret(<redacted, kid {self.fingerprint()}>)"

    __str__ = __repr__

    def __format__(self, _spec: str) -> str:
        return repr(self)

    def __eq__(self, other) -> bool:
        import hmac

        return isinstance(other, Secret) and hmac.compare_digest(self._value.encode(), other._value.encode())

    def __hash__(self) -> int:
        return hash(("Secret", hashlib.sha256(self._value.encode()).digest()))

    def __reduce__(self):
        raise TypeError("a Secret is not picklable")


@dataclass
class Credentials:
    secrets: dict[str, Secret]
    path: Path
    recovered_from_backup: bool = False
    warnings: list[str] = field(default_factory=list)

    def get(self, name: str) -> Optional[str]:
        secret = self.secrets.get(name)
        return secret.reveal() if secret is not None else None


# ------------------------------------------------------------------ redaction
def redact(text: str, secrets: Iterable[Union[str, Secret, None]]) -> str:
    """`text` with every occurrence of each secret replaced by a marker."""
    out = str(text)
    for secret in secrets:
        value = secret.reveal() if isinstance(secret, Secret) else secret
        if value and len(value) >= 4:
            out = out.replace(value, "[REDACTED]")
    return out


class RedactingFilter(logging.Filter):
    """A logging filter that scrubs the given secrets from every record it passes."""

    def __init__(self, secrets: Iterable[Union[str, Secret, None]]) -> None:
        super().__init__()
        self._secrets = [s for s in secrets if s]

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        cleaned = redact(message, self._secrets)
        if cleaned != message:
            record.msg, record.args = cleaned, ()
        return True


# ------------------------------------------------------------------ permissions
def permission_problems(path: Union[str, Path]) -> list[str]:
    """Why `path` is not safe to hold a secret (empty list = fine)."""
    path = Path(path)
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return [f"{path} does not exist"]
    problems = []
    if stat.S_ISLNK(st.st_mode):
        problems.append(f"{path} is a symlink; a credentials file must be a regular file")
    elif not stat.S_ISREG(st.st_mode):
        problems.append(f"{path} is not a regular file")
    if hasattr(os, "geteuid") and st.st_uid != os.geteuid():
        problems.append(f"{path} is owned by uid {st.st_uid}, not by this service (uid {os.geteuid()})")
    if st.st_mode & 0o077:
        problems.append(f"{path} has mode {stat.S_IMODE(st.st_mode):04o}; it must be 0600 "
                        f"(`chmod 600 {path}`)")
    return problems


def config_permission_warnings(path: Union[str, Path], config: dict) -> list[str]:
    """Warnings for a config.yaml that holds a secret but is readable by others."""
    cloud = config.get("cloud") if isinstance(config.get("cloud"), dict) else {}
    dash = config.get("dashboard") if isinstance(config.get("dashboard"), dict) else {}
    holds = [name for name, value in (("cloud.device_key", cloud.get("device_key")),
                                      ("cloud.previous_device_key", cloud.get("previous_device_key")),
                                      ("dashboard.token", dash.get("token"))) if value]
    if not holds:
        return []
    try:
        mode = stat.S_IMODE(os.stat(path).st_mode)
    except OSError:
        return []
    if mode & 0o077:
        return [(f"{path} holds {', '.join(holds)} and has mode {mode:04o}: other local users can read it. "
                 f"`chmod 600 {path}`, or move the key out with scripts/migrate_device_key.py "
                 f"(docs/CREDENTIAL_STORAGE.md).")]
    return []


# ------------------------------------------------------------------ file format
def _checksum(secrets: dict) -> str:
    return hashlib.sha256(json.dumps(secrets, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _encode(secrets: dict[str, str]) -> bytes:
    clean = {k: v for k, v in secrets.items() if v is not None}
    unknown = set(clean) - set(FIELDS)
    if unknown:
        raise CredentialError(f"unknown credential field(s): {sorted(unknown)}")
    if not isinstance(clean.get("device_key"), str) or not clean["device_key"]:
        raise CredentialError("a credentials file needs a non-empty device_key")
    if not all(isinstance(v, str) and v for v in clean.values()):
        raise CredentialError("credential values must be non-empty strings")
    doc = {"format": FORMAT, "version": VERSION, "secrets": clean, "sha256": _checksum(clean)}
    return (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode()


def _decode(data: bytes, path: Path) -> dict[str, str]:
    try:
        doc = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise CredentialError(f"{path} is not valid JSON (truncated or damaged)") from None
    if not isinstance(doc, dict) or doc.get("format") != FORMAT:
        raise CredentialError(f"{path} is not a SensorDyme credentials file")
    if doc.get("version") != VERSION:
        raise CredentialError(f"{path} has unsupported version {doc.get('version')!r}")
    secrets = doc.get("secrets")
    if not isinstance(secrets, dict) or not isinstance(doc.get("sha256"), str):
        raise CredentialError(f"{path} is missing its secrets or checksum")
    if _checksum(secrets) != doc["sha256"]:
        raise CredentialError(f"{path} failed its checksum (damaged or edited by hand)")
    if set(secrets) - set(FIELDS) or not isinstance(secrets.get("device_key"), str) or not secrets["device_key"]:
        raise CredentialError(f"{path} has unexpected or empty fields")
    return secrets


# ------------------------------------------------------------------ write
def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_write(path: Path, data: bytes) -> None:
    """Write `data` to `path` atomically, mode 0600 from the first byte."""
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    _fsync_dir(path.parent)


def backup_path(path: Union[str, Path]) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".bak")


def write_credentials(path: Union[str, Path], secrets: dict[str, str]) -> None:
    """Store `secrets` at `path` (0600, atomic). The current file, if it is
    valid, becomes `<path>.bak` first, so an interrupted update still leaves one
    good copy."""
    path = Path(path)
    data = _encode(secrets)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            current = _decode(path.read_bytes(), path)
        except CredentialError:
            current = None  # never back up a damaged file over a good backup
        if current is not None:
            _atomic_write(backup_path(path), _encode(current))
    _atomic_write(path, data)


# ------------------------------------------------------------------ read
def _load_one(path: Path, strict: bool) -> tuple[dict[str, str], list[str]]:
    problems = permission_problems(path)
    if problems and (strict or not path.exists() or path.is_symlink()):
        raise CredentialError("; ".join(problems))
    return _decode(path.read_bytes(), path), problems


def load_credentials(path: Union[str, Path], *, strict: bool = True, allow_backup: bool = True) -> Credentials:
    """Read a credentials file, falling back to its backup when it is damaged.

    strict: refuse a file that is not 0600 / not ours (the production setting).
    A missing or damaged primary with a good backup loads the backup and says so
    in `warnings`; `restore_from_backup` repairs the primary.
    """
    path = Path(path)
    try:
        secrets, warnings = _load_one(path, strict)
        return Credentials({k: Secret(v) for k, v in secrets.items()}, path, warnings=warnings)
    except (CredentialError, OSError) as primary:
        backup = backup_path(path)
        if not allow_backup or not backup.exists():
            raise CredentialError(str(primary)) from None
        try:
            secrets, warnings = _load_one(backup, strict)
        except (CredentialError, OSError) as second:
            raise CredentialError(f"{primary}; backup unusable too: {second}") from None
        warnings = [(f"{primary}; using the backup {backup}. Repair with "
                     f"`python scripts/migrate_device_key.py --restore-backup {path}`"), *warnings]
        return Credentials({k: Secret(v) for k, v in secrets.items()}, path, True, warnings)


def restore_from_backup(path: Union[str, Path]) -> None:
    """Replace a damaged credentials file with its (valid) backup, atomically."""
    path = Path(path)
    backup = backup_path(path)
    secrets, _ = _load_one(backup, strict=True)
    _atomic_write(path, _encode(secrets))


def resolve_cloud_credentials(cloud: dict, *, strict: bool = True) -> tuple[dict, list[str]]:
    """`cloud` with device_key / previous_device_key filled from cloud.device_key_file.

    Returns (config, warnings). Without a key file the config is returned
    unchanged (the legacy in-config key keeps working). Raises CredentialError
    when a key file is configured but cannot be used -- the caller then runs
    without the uplink rather than with no key or a wrong one.
    """
    key_file = cloud.get("device_key_file") if isinstance(cloud, dict) else None
    if not key_file:
        return cloud, []
    creds = load_credentials(key_file, strict=strict)
    resolved = dict(cloud)
    for name in FIELDS:
        value = creds.get(name)
        if value is not None:
            resolved[name] = value
    return resolved, list(creds.warnings)
