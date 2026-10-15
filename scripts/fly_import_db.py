#!/usr/bin/env python3
"""Import an uploaded SQLite database into the deployed backend's volume.

Runs ON the Fly machine, after `fly ssh sftp shell` has put a copy of the
developer's `cloud/sensordyme.db` at /data/import.db:

    fly ssh console -C "python /app/scripts/fly_import_db.py /data/import.db"

Why a script and not `mv`:

  * **A move can produce a database that opens and is wrong.** The source is
    copied off a machine where the backend may have been mid-write, and SQLite
    in WAL mode (cloud/app.py sets `journal_mode=WAL`) keeps committed pages in
    a `-wal` sidecar that a single-file copy leaves behind. Restoring through
    SQLite's own backup API instead of the filesystem reads the source as a
    database, not as bytes, so what lands is a consistent snapshot with the WAL
    already applied.

  * **The destination is not empty.** `init_db()` runs at import time, so the
    first boot already created the schema. A move would leave the `-wal` and
    `-shm` files of the file that used to be there beside a different
    database, which is a real way to corrupt one. Restoring THROUGH SQLite
    (the backup API, below) has no such problem: the sidecars stay the live
    database's own and SQLite keeps them coherent. They are therefore left
    alone -- deleting them threw away commits still in the WAL, from the
    backup as well, since that was a plain copy of the main file.

  * **An import that silently lands the wrong file is worse than a failed
    one.** Every check below runs before anything is written; the existing
    database is backed up first regardless.

  * **Sessions do not survive it.** A token names an account by row id, and
    the imported file numbers its accounts differently: a session someone
    opened on the fresh deployment would otherwise carry on as whoever has that
    row id in the imported data. The backend now also binds tokens to the
    account's email (cloud/app.py `_issue_token`), and this script rotates the
    signing key on the volume, so every phone signs in once more. Pass
    --keep-sessions only when the key you are keeping is the one the IMPORTED
    database's sessions were issued with (you copied its `.jwt_secret` over).

Idempotent, and safe to run more than once: each run re-backs-up whatever is
currently there before replacing it.
"""

from __future__ import annotations

import argparse
import os
import secrets
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import NoReturn

#: Tables the backend cannot run without. A file that opens as SQLite but is
#: some other application's database would otherwise import cleanly and fail
#: later, a long way from the cause.
REQUIRED_TABLES = {"users", "devices", "events"}


def _fail(message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def inspect(path: Path) -> dict:
    """Everything worth knowing about a candidate file, or exit non-zero."""
    if not path.is_file():
        _fail(f"{path} does not exist. Upload it first with `fly ssh sftp shell`.")
    if path.stat().st_size == 0:
        _fail(f"{path} is empty — the upload did not complete.")

    # `connect` on a non-database succeeds: SQLite does not read the file until
    # the first statement, so the check has to be a query, not the open. Caught
    # here rather than left to propagate — an operator staring at a DatabaseError
    # traceback has to work out that it means "you uploaded the wrong file".
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        try:
            integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        except sqlite3.DatabaseError as exc:
            _fail(f"{path} is not a readable SQLite database ({exc}).")
        if integrity != "ok":
            _fail(f"{path} failed its integrity check: {integrity}")

        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        missing = REQUIRED_TABLES - tables
        if missing:
            _fail(
                f"{path} is missing {', '.join(sorted(missing))} — this does not look like "
                f"a SensorDyme database."
            )

        # The S608 suppression below: `tables` comes from sqlite_master on the
        # file being inspected, not from user input, and SQLite has no bind
        # parameter for an identifier. The quoting keeps an odd name safe.
        counts = {
            name: conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]  # noqa: S608
            for name in sorted(tables)
        }
        # A database with no users cannot be the one being migrated to preserve
        # an account, and importing it would replace a working deployment with
        # an empty one.
        if counts.get("users", 0) == 0:
            _fail(f"{path} contains no user accounts. Refusing to import it over the live database.")
        owned = conn.execute("SELECT COUNT(*) FROM devices WHERE owner_id IS NOT NULL").fetchone()[0]
    finally:
        conn.close()

    return {"counts": counts, "claimed_devices": owned}


def restore(source: Path, target: Path) -> None:
    """Replace `target`'s contents with `source`, through SQLite itself."""
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    dst = sqlite3.connect(target)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()


def back_up(target: Path, backup: Path) -> None:
    """A consistent copy of the LIVE database, WAL included.

    Not `shutil.copy2`: that copies the main file only, and a WAL-mode
    database keeps its most recent commits in `-wal` until a checkpoint --
    measured, an account created moments before the import was missing from
    the "backup" this script promised. The backup API reads the database as
    SQLite sees it, so what lands is everything committed.
    """
    src = sqlite3.connect(target)
    dst = sqlite3.connect(backup)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    os.chmod(backup, 0o600)


def jwt_secret_path(target: Path) -> Path:
    """Where the backend keeps its signing key -- the same rule cloud/app.py's
    `_load_jwt_secret` uses: SENSORDYME_JWT_SECRET_FILE, else beside the DB."""
    configured = os.environ.get("SENSORDYME_JWT_SECRET_FILE", "").strip()
    return Path(configured) if configured else target.parent / ".jwt_secret"


def rotate_jwt_secret(path: Path) -> None:
    """Write a fresh signing key at `path`, mode 0600, atomically.

    Atomic and 0600 before the key is written, for the reasons cloud/app.py
    gives for its own copy: the key is never briefly readable, and a crash
    mid-write never leaves an empty file (which the backend would replace
    with yet another key on the next start, which is harmless but confusing).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".jwt_secret.")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(secrets.token_hex(32))
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", default="/data/import.db", help="uploaded database (default: /data/import.db)")
    parser.add_argument(
        "--target",
        default=os.environ.get("SENSORDYME_DB", "/data/sensordyme.db"),
        help="database the backend serves (default: $SENSORDYME_DB)",
    )
    parser.add_argument("--dry-run", action="store_true", help="check the upload and report, changing nothing")
    parser.add_argument(
        "--keep-sessions", action="store_true",
        help="do not rotate the JWT signing key (only if it is the key the imported database's "
             "sessions were issued with)",
    )
    args = parser.parse_args()

    source, target = Path(args.source), Path(args.target)
    facts = inspect(source)

    print(f"{source} looks like a SensorDyme database:")
    for name, count in facts["counts"].items():
        print(f"  {name:16} {count:7} rows")
    print(f"  devices claimed by an account: {facts['claimed_devices']}")

    if args.dry_run:
        print("\ndry run — nothing written.")
        return 0

    if target.exists():
        backup = target.with_name(f"{target.name}.replaced-{time.strftime('%Y%m%d-%H%M%S')}")
        back_up(target, backup)
        print(f"\nexisting database backed up to {backup}")

    restore(source, target)
    os.chmod(target, 0o600)

    if args.keep_sessions:
        print("\n--keep-sessions: the JWT signing key was left as it is.")
    elif os.environ.get("SENSORDYME_JWT_SECRET", "").strip():
        print(
            "\nwarning: the backend's signing key comes from SENSORDYME_JWT_SECRET, which this "
            "script cannot change. Rotate it in your secret store (fly secrets set ...) so "
            "sessions opened before the import end."
        )
    else:
        secret = jwt_secret_path(target)
        rotate_jwt_secret(secret)
        print(f"\nJWT signing key rotated at {secret}: every phone signs in once more after the restart.")

    verified = inspect(target)
    print(f"\nimported into {target}:")
    for name, count in verified["counts"].items():
        print(f"  {name:16} {count:7} rows")
    print("\nRestart the app so it reopens the database: fly apps restart")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
