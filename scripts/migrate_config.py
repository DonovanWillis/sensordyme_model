#!/usr/bin/env python3
"""Add new config keys to a device's config.yaml without touching its own values.

A deployed device's config.yaml is not a copy of the repo's. It carries things
only true of that unit: its `device_key`, the address of its backend, whether
its camera is mounted upside down, where its .hef lives. Overwriting it during a
deploy silently unpairs the device and un-rotates the camera, and the damage
surfaces much later as "the app stopped seeing it" — a long way from the cause.

So `make deploy` excludes config.yaml and runs this instead. The rule is narrow
and worth stating plainly:

    a key the device does not have  ->  added, with the repo's default
    a key the device already has    ->  left exactly as it is

New features therefore arrive configured, and nothing a person set on this
device is ever lost. Run it as often as you like; it converges.
"""

from __future__ import annotations

import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"

# Keys that must exist on a device, as (dotted path, literal line to insert
# under its parent). Only ever appended when absent.
MIGRATIONS: list[tuple[str, str, str]] = [
    (
        "network.provisioning.grace_s",
        "    grace_s: 0",
        "seconds offline before the Bluetooth setup window opens",
    ),
    (
        "network.provisioning.reconfigure_window_s",
        "    reconfigure_window_s: 300",
        "keep Bluetooth open this long after boot even when Wi-Fi is fine, so a"
        " device can be moved to a new network without being taken offline first",
    ),
    (
        "cloud.ca_cert_path",
        '  ca_cert_path: ""',
        "opt-in TLS: path to a self-signed cert's PEM file to trust for cloud.base_url,"
        " or leave empty. Off (empty) is identical to not having this key at all.",
    ),
    (
        "cloud.tls_insecure_skip_verify",
        "  tls_insecure_skip_verify: false",
        "opt-in TLS: skips certificate verification entirely when true -- genuinely"
        " insecure, dev-only. See config.yaml's fuller comment on this key.",
    ),
]

# A NEW MEMBER of an EXISTING flow-sequence value, as (dotted path to the key,
# member name to ensure is present, why). Unlike MIGRATIONS above (which only
# ever adds a key that's entirely missing), this handles a key the device
# ALREADY has whose *value* needs a new item -- e.g. a preset's alert type
# that was silently unable to ever produce a phone notification because it
# was never added to `alerts.loud_types` on already-deployed devices (a repo
# fix to config.yaml's own loud_types list is invisible to a device that
# already has that key, by MIGRATIONS' own "only when entirely absent" rule).
# A member already present (anywhere in the sequence, not just at the top
# level of a fresh repo checkout) is left untouched -- idempotent, like the
# rest of this script.
LIST_MIGRATIONS: list[tuple[str, str, str]] = [
    ("alerts.loud_types", "unusual_stillness",
     "baby/elderly presets' alert type was never in loud_types, so it could never produce a phone"
     " notification (severity defaulted to soft) -- see config.yaml's fuller comment"),
    ("alerts.loud_types", "person_absent",
     "same gap as unusual_stillness above"),
    ("alerts.loud_types", "flailing",
     "same gap -- confirmed live via a real flailing event that reached Recent Activity but never notified"),
    ("alerts.loud_types", "person_lingering",
     "security_monitoring's alert type had the identical gap"),
    ("alerts.loud_types", "vehicle_in_zone",
     "security_monitoring's alert type had the identical gap"),
    ("alerts.loud_types", "animal_in_zone",
     "security_monitoring's alert type had the identical gap"),
]


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip())


def has_key(lines: list[str], dotted: str) -> bool:
    """Whether `a.b.c` exists as a nested mapping key."""
    parts = dotted.split(".")
    depth = 0
    expected_indent = 0
    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = _indent_of(line)
        if depth and indent < expected_indent:
            # Dedented out of the parent without finding the child.
            depth = 0
            expected_indent = 0
        key = stripped.split(":", 1)[0].strip()
        if key == parts[depth] and (depth == 0 or indent >= expected_indent):
            if depth == len(parts) - 1:
                return True
            expected_indent = indent + 1
            depth += 1
    return False


def parent_block_end(lines: list[str], dotted: str) -> int | None:
    """Index just past the last real setting in `a.b`'s block.

    Deliberately *not* the last line of the block. A section's final lines are
    usually the comment introducing the next section, and inserting after those
    puts the new key under a paragraph describing something else — the YAML
    still parses, so nothing complains, and the file quietly becomes misleading.
    """
    parts = dotted.split(".")[:-1]
    depth = 0
    parent_indent = -1
    start = None
    last_setting = None
    for i, raw in enumerate(lines):
        line = raw.rstrip("\n")
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = _indent_of(line)
        key = stripped.split(":", 1)[0].strip()
        if start is not None and indent <= parent_indent:
            break
        if start is not None:
            last_setting = i
        if depth < len(parts) and key == parts[depth]:
            if depth == len(parts) - 1:
                start = i
                parent_indent = indent
            depth += 1
    if start is None:
        return None
    return (last_setting + 1) if last_setting is not None else start + 1


def _key_line_index(lines: list[str], dotted: str) -> int | None:
    """Index of the line that DEFINES the final segment of `a.b.c` (where
    `c: ...` or `c: [` starts), at the correct nesting depth. Shares
    has_key's depth-tracking so the two can never disagree about which line
    a dotted path actually refers to."""
    parts = dotted.split(".")
    depth = 0
    expected_indent = 0
    for i, raw in enumerate(lines):
        line = raw.rstrip("\n")
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = _indent_of(line)
        if depth and indent < expected_indent:
            depth = 0
            expected_indent = 0
        key = stripped.split(":", 1)[0].strip()
        if key == parts[depth] and (depth == 0 or indent >= expected_indent):
            if depth == len(parts) - 1:
                return i
            expected_indent = indent + 1
            depth += 1
    return None


def _list_span(lines: list[str], dotted: str) -> tuple[int, int] | None:
    """(open_line_index, close_line_index) of a `key: [ ... ]` flow sequence,
    however many lines it spans -- located by bracket-depth counting from the
    line the key is defined on, so it works whether the sequence is on one
    line or many."""
    open_idx = _key_line_index(lines, dotted)
    if open_idx is None:
        return None
    depth = 0
    for i in range(open_idx, len(lines)):
        # Comments can contain stray brackets (prose, code snippets) -- only
        # count brackets in the code portion of the line, before any '#'.
        code = lines[i].split("#", 1)[0]
        depth += code.count("[") - code.count("]")
        if depth <= 0:
            return open_idx, i
    return None  # unterminated -- malformed YAML, caller should not touch it


def _list_contains_member(lines: list[str], open_idx: int, close_idx: int, member: str) -> bool:
    """Whether `member` appears as a whole token within the sequence's own
    code (comments stripped), not merely as a substring of a longer name or
    mentioned in passing in a comment."""
    code = "".join(line.split("#", 1)[0] for line in lines[open_idx : close_idx + 1])
    return re.search(rf"(?<![\w]){re.escape(member)}(?![\w])", code) is not None


def append_list_member(lines: list[str], dotted: str, member: str) -> bool:
    """Add `member` to an existing flow-sequence value if it isn't already
    present anywhere in it. Returns whether a change was made.

    Inserted as its own new line, immediately before the closing bracket's
    line, rather than editing that line's existing text in place -- a flow
    sequence does not care which physical line an item sits on, and this
    sidesteps needing to parse past whatever trailing comment already sits
    on the closing-bracket line.
    """
    span = _list_span(lines, dotted)
    if span is None:
        print(f"  skip {dotted}+={member}: key or its closing bracket not found", file=sys.stderr)
        return False
    open_idx, close_idx = span
    if _list_contains_member(lines, open_idx, close_idx, member):
        return False
    # Indent new items to match the sequence's own continuation style: the
    # line right after the opening one, if the sequence is multi-line;
    # otherwise two spaces deeper than the key itself.
    if close_idx > open_idx:
        template = lines[open_idx + 1]
        indent = template[: _indent_of(template)]
    else:
        indent = " " * (_indent_of(lines[open_idx]) + 2)
    lines.insert(close_idx, f"{indent}{member},\n")
    return True


def migrate(path: Path) -> list[str]:
    lines = path.read_text().splitlines(keepends=True)
    added: list[str] = []
    for dotted, literal, why in MIGRATIONS:
        if has_key(lines, dotted):
            continue
        insert_at = parent_block_end(lines, dotted)
        if insert_at is None:
            print(f"  skip {dotted}: parent section not found", file=sys.stderr)
            continue
        # Match the comment's indent to the key's own, not a hardcoded 4 --
        # that was only ever right by coincidence, for keys nested exactly
        # two levels deep (network.provisioning.*). A one-level key (cloud.*)
        # inserted at indent 2 got a comment sitting at indent 4 above it.
        indent = literal[: len(literal) - len(literal.lstrip(" "))]
        block = [f"{indent}# {why}\n", literal + "\n"]
        lines[insert_at:insert_at] = block
        added.append(dotted)
    for dotted, member, _why in LIST_MIGRATIONS:
        if append_list_member(lines, dotted, member):
            added.append(f"{dotted}+={member}")
    if added:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        shutil.copy2(path, path.with_suffix(f".yaml.bak-{stamp}"))
        path.write_text("".join(lines))
    return added


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config.yaml")
    if not target.exists():
        print(f"error: {target} not found", file=sys.stderr)
        return 1
    added = migrate(target)
    if added:
        print(f"config: added {', '.join(added)} (backup alongside)")
    else:
        print("config: already current, nothing changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
