"""Auto-discovers and imports every module in this package so their
`@register_module` decorators run — main.py never needs to import them by name.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from modules.base import VisionModule, available_modules, get_module_class, register_module


def _discover() -> None:
    """Import every sibling module in this package (except base.py itself)."""
    package_dir = Path(__file__).parent
    for info in pkgutil.iter_modules([str(package_dir)]):
        if info.name == "base":
            continue
        importlib.import_module(f"{__name__}.{info.name}")


_discover()

__all__ = ["VisionModule", "register_module", "get_module_class", "available_modules"]
