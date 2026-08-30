# Minimal stub for a Raspberry Pi OS-only module.
#
# This package does not exist on macOS and is not pip-installable there: on the
# Pi it comes from apt (python3-picamera2, python3-libcamera, python3-dbus,
# python3-gi) or from `apt install hailo-all`. Every import of it in this repo
# is inside a try/except ImportError with a non-Pi fallback, so the code is
# correct without it — but pyright, running on a Mac, reported each one as an
# unresolved import and buried the real type errors.
#
# These stubs exist ONLY to make the import resolve. They are Any-typed on
# purpose: pretending to know these APIs would produce confidently wrong type
# errors. The Pi code paths are verified by running on the device
# (docs/TESTING.md), not by this stub.
#
# See pyrightconfig.json -> stubPath.

from typing import Any

def __getattr__(name: str) -> Any: ...
