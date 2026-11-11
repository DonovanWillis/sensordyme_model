"""presets.yaml is hand-edited and has been silently corrupted once before.

A stray edit split the file's header comment across two places and left a
duplicate `aliases:` and a duplicate `presets:` key — one of them a bare
string, not a mapping. The file still loaded, because PyYAML's default
SafeLoader accepts duplicate mapping keys and simply keeps the last one; a
stricter parser, or a linter run over the file, would not have been so quiet
about it. These tests parse the file with duplicate-key detection turned on,
so a repeat of that mistake fails here instead of shipping.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

PRESETS_PATH = Path(__file__).resolve().parent.parent / "presets.yaml"


class _DuplicateKeyError(ValueError):
    pass


class _StrictUniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader, except a repeated key at the same mapping level is an error.

    This is the standard recipe for the check PyYAML does not do by default:
    override construct_mapping and inspect the keys as they're built, rather
    than silently letting a later key win.
    """


def _construct_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False):
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise _DuplicateKeyError(f"duplicate key {key!r} at line {key_node.start_mark.line + 1}")
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_StrictUniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def _load_strict() -> dict:
    with PRESETS_PATH.open() as f:
        return yaml.load(f, Loader=_StrictUniqueKeyLoader)  # noqa: S506 - a yaml.SafeLoader subclass


def test_file_exists():
    assert PRESETS_PATH.exists()


def test_parses_with_no_duplicate_keys_anywhere():
    """The regression test: the corrupted file had two `aliases:` and two
    `presets:` keys at the top level, and PyYAML's default loader silently
    accepted it. This must fail loudly on any duplicate key, not just at the
    top level — a duplicated preset name or variant name is the same defect."""
    try:
        _load_strict()
    except _DuplicateKeyError as exc:
        pytest.fail(f"presets.yaml has a duplicate key: {exc}")


def test_top_level_shape():
    data = _load_strict()
    # catalog_version and planned: the M6 preset catalogue (docs/PRESET_LIBRARY.md).
    assert {"aliases", "presets"} <= set(data.keys()) <= {"aliases", "presets", "catalog_version", "planned"}
    assert isinstance(data.get("planned", {}), dict)
    assert isinstance(data["aliases"], dict)
    assert isinstance(data["presets"], dict)
    # The exact defect that shipped: a stray line turned `presets:` into a
    # scalar string instead of a mapping. Assert the type explicitly rather
    # than relying on downstream code to fail in a way that points back here.
    for name, preset in data["presets"].items():
        assert isinstance(preset, dict), f"preset {name!r} is not a mapping: {preset!r}"


def test_every_preset_has_a_module():
    data = _load_strict()
    for name, preset in data["presets"].items():
        assert preset.get("module"), f"preset {name!r} has no module"


def test_every_alias_points_at_a_real_preset():
    data = _load_strict()
    presets = data["presets"]
    for old_name, target in data["aliases"].items():
        assert target in presets, f"alias {old_name!r} points at unknown preset {target!r}"


def test_no_alias_points_at_another_alias():
    """An alias chain (a -> b -> c) is a device stuck one hop further out —
    resolve_preset() only follows one level."""
    data = _load_strict()
    aliases = data["aliases"]
    presets = data["presets"]
    for old_name, target in aliases.items():
        assert target not in aliases, (
            f"alias {old_name!r} points at {target!r}, which is itself an alias — "
            "resolve_preset() only follows one hop"
        )
        assert target in presets
