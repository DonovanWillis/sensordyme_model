"""The app must only offer modes and variants the device can actually apply.

The app listed "Newborn" and "Toddler" for baby monitoring; the device has
neither. Tapping one queued a variant no device could apply, and because the
backend only ever validated the base name, it accepted the request and re-sent
it on every poll for ever.

Nothing checked this, which is exactly why it drifted.
"""

import re
from pathlib import Path

from main import load_presets

CATALOG = Path("ios/Sources/Models.swift")


def app_catalog() -> dict[str, set[str]]:
    """Parse Preset.catalog out of the Swift source.

    Reading the source is ugly, but the alternative is a hand-maintained copy of
    the same list in Python — a third place to drift.
    """
    text = CATALOG.read_text()
    start = text.index("static let catalog:")
    body = text[start:text.index("\n    }", start)]
    catalog: dict[str, set[str]] = {}
    current = None
    for line in body.splitlines():
        preset = re.search(r'Preset\(name: "([a-z_]+)"', line)
        if preset:
            current = preset.group(1)
            catalog[current] = set()
        for variant in re.findall(r'\.init\(name: "([a-z_]+)"', line):
            if current:
                catalog[current].add(variant)
    return catalog


def test_every_mode_the_app_offers_exists_on_the_device():
    device, app = set(load_presets()), set(app_catalog())
    assert app <= device, f"app offers modes the device does not have: {app - device}"


def test_every_variant_the_app_offers_exists_on_the_device():
    presets = load_presets()
    for mode, variants in app_catalog().items():
        available = set((presets.get(mode) or {}).get("variants") or {})
        missing = variants - available
        assert not missing, f"{mode}: app offers {missing}, device has {sorted(available)}"


def test_the_app_offers_every_mode_the_device_supports():
    """A mode nobody can select is a feature that does not exist."""
    device, app = set(load_presets()), set(app_catalog())
    assert device <= app, f"device has modes the app never shows: {device - app}"


def test_there_is_exactly_one_driver_mode_in_the_app_too():
    assert [m for m in app_catalog() if "driver" in m] == ["driver_safety"]
