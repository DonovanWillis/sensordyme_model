# Minimal stub for `onnx`, covering ONLY what this repo uses.
#
# onnx is not a dependency of the device, the backend or the test suite: it is
# needed by one worker-side tool, scripts/hw_validation/compare_float_vs_int8.py
# (which imports it lazily, inside build_session), and by one test that does
# pytest.importorskip("onnx") first. It is in no requirements file, so on a dev
# machine without it pyright reported both imports as unresolved.
#
# The protobuf message types (ModelProto, GraphProto, NodeProto, ...) are left
# as Any: they are generated classes and this repo only walks their fields.
# If onnx is ever added to requirements-dev.txt, delete this directory: the
# package ships its own inline types, which this stub would otherwise shadow.
#
# See pyrightconfig.json -> stubPath.

from typing import Any

from . import helper as helper

__version__: str

class TensorProto:
    FLOAT: int

def load_model_from_string(s: bytes) -> Any: ...
