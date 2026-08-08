"""The mode registry, and the names modes used to have.

Deliberately dependency-free — yaml and the standard library, nothing else.
Both the device and the backend need to agree on what a preset name means, and
the backend runs on machines that have never had mediapipe or picamera2
installed. Putting this in main.py made it unreachable from cloud/app.py, and
the two ended up comparing names by exact string equality instead, which is
what let a renamed mode silently never converge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "presets.yaml"


def _load(path: Union[str, Path, None]) -> dict:
    path = Path(path) if path is not None else DEFAULT_PATH
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def load_presets(path: Union[str, Path, None] = None) -> dict:
    """The modes on offer: name -> {title, module, alert_types?, variants?}."""
    return _load(path).get("presets", {}) or {}


def load_preset_aliases(path: Union[str, Path, None] = None) -> dict:
    """Superseded mode names -> their replacement."""
    return _load(path).get("aliases", {}) or {}


def resolve_preset(name: str, presets: Optional[dict] = None,
                   aliases: Optional[dict] = None) -> str:
    """Map a possibly-superseded preset name onto one that exists.

    Renaming a mode strands every device already holding the old name: an
    unknown preset is a device stuck in a mode it cannot switch out of, and the
    name can be anywhere — a deployed config.yaml, a `desired_preset` sitting in
    the backend, a phone that has not been updated.

    A variant suffix ("driver_distraction:teen_driver") is carried across, since
    a variant belongs to the preset rather than to the name it used to have —
    but only when the replacement actually defines it, because a variant that no
    longer exists would fail later, a long way from the rename that caused it.
    """
    if not name:
        return name
    presets = load_presets() if presets is None else presets
    base, _, variant = name.partition(":")
    if base in presets:
        return name
    target = (load_preset_aliases() if aliases is None else aliases).get(base)
    if target is None or target not in presets:
        return name
    if variant and variant in (presets[target].get("variants") or {}):
        return f"{target}:{variant}"
    return target


def same_preset(a: Optional[str], b: Optional[str], presets: Optional[dict] = None) -> bool:
    """Whether two preset names refer to the same mode once aliases are applied.

    The comparison that convergence depends on. A device handed a superseded
    name applies — and correctly reports — the current one, so comparing the raw
    strings never matches: the backend concludes the command was never obeyed
    and re-sends it on every poll, fighting any mode the user picks locally.
    """
    if not a or not b:
        return a == b
    presets = load_presets() if presets is None else presets
    return resolve_preset(a, presets) == resolve_preset(b, presets)
