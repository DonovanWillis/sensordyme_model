# Minimal stub for `onnx.helper`: the builders that tests/test_m5_hw_validation.py
# and scripts/hw_validation/compare_float_vs_int8.py call. See __init__.pyi.

from collections.abc import Sequence
from typing import Any

def make_tensor_value_info(
    name: str,
    elem_type: int,
    shape: Sequence[str | int | None] | None,
    doc_string: str = ...,
    shape_denotation: list[str] | None = ...,
) -> Any: ...
def make_node(
    op_type: str,
    inputs: Sequence[str],
    outputs: Sequence[str],
    name: str | None = ...,
    doc_string: str | None = ...,
    domain: str | None = ...,
    **kwargs: Any,
) -> Any: ...
def make_graph(
    nodes: Sequence[Any],
    name: str,
    inputs: Sequence[Any],
    outputs: Sequence[Any],
    initializer: Sequence[Any] | None = ...,
    doc_string: str | None = ...,
    value_info: Sequence[Any] | None = ...,
) -> Any: ...
def make_model(graph: Any, **kwargs: Any) -> Any: ...
def make_opsetid(domain: str, version: int) -> Any: ...
