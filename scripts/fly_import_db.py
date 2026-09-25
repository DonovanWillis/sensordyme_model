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
    first boot already created the schema. A move would leave stale `-wal` and
    `-shm` files belonging to the file that used to be there, which is a real
    way to corrupt a database rather than a theoretical one.

  * **An import that silently lands the wrong file is worse than a failed
    one.** Every check below runs before anything is written; the existing
    database is backed up first regardless.

Idempotent, and safe to run more than once: each run re-backs-up whatever is
currently there before replacing it.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", default="/data/import.db", help="uploaded database (default: /data/import.db)")
    parser.add_argument(
        "--target",
        default=os.environ.get("SENSORDYME_DB", "/data/sensordyme.db"),
        help="database the backend serves (default: $SENSORDYME_DB)",
    )
    parser.add_argument("--dry-run", action="store_true", help="check the upload and report, changing nothing")
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
        shutil.copy2(target, backup)
        print(f"\nexisting database backed up to {backup}")
        # The sidecars belong to the file being replaced, and a stale WAL beside
        # a different database is a corruption, not a leftover.
        for suffix in ("-wal", "-shm"):
            sidecar = target.with_name(target.name + suffix)
            if sidecar.exists():
                sidecar.unlink()

    restore(source, target)
    os.chmod(target, 0o600)

    verified = inspect(target)
    print(f"\nimported into {target}:")
    for name, count in verified["counts"].items():
        print(f"  {name:16} {count:7} rows")
    print("\nRestart the app so it reopens the database: fly apps restart")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
