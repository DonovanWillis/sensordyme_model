"""Unit tests for the module registry / auto-discovery pattern."""

import pytest

from modules import available_modules, get_module_class
from modules.base import VisionModule, register_module


def test_all_three_product_modules_are_registered():
    names = available_modules()
    assert {"driver_attention", "occupancy", "fall_detection"}.issubset(set(names))


def test_get_module_class_returns_a_vision_module_subclass():
    cls = get_module_class("driver_attention")
    assert issubclass(cls, VisionModule)


def test_get_module_class_raises_for_unknown_name():
    with pytest.raises(KeyError):
        get_module_class("not_a_real_module")


def test_register_module_adds_new_entries():
    @register_module("_test_dummy_module")
    class _Dummy(VisionModule):
        name = "_test_dummy_module"

        def process(self, frame):
            return []

    assert "_test_dummy_module" in available_modules()
    assert get_module_class("_test_dummy_module") is _Dummy
