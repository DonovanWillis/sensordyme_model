"""A stand-in for HailoRT's `hailo_platform`, shaped like damoyolo_tinynasL20_T.hef.

NOTHING THIS PRODUCES IS A HARDWARE RESULT. It exists so that the device's own
inference code (core/accel.py HailoYOLO: HEF parsing, head pairing, UINT8 input,
FLOAT32 outputs, letterbox, decode, NMS, class mapping, frame mapping) can be
exercised without a Raspberry Pi:

  * CI on a Mac (tests/test_m83_inference_path*.py), with synthetic head tensors;
  * dry runs of every scripts/hw_validation stage (`hwv_stage.py --fake`);
  * the fleet inference-path job (research/m83/inference_path_val.py), where the
    outputs come from the FLOAT ONNX the .hef was compiled from.

What it models, and where each property comes from:

  output layout  six outputs, NHWC float32 (1, G, G, C): grids 80/40/20 with a
                 68-channel box head (4 sides x 17 DFL bins, raw logits) and an
                 81-channel class head (80 COCO sigmoid probabilities + 1
                 extra channel) at each grid, no on-chip NMS. The float ONNX
                 class heads are 81 channels as well (fleet job
                 20261113-043758-b69ebe), so the 81st is the network's own
                 mmdet-style background slot, not compiler padding; every
                 decoder drops it. Source: the
                 `hailortcli parse-hef` expectation in
                 docs/HAILO_HARDWARE_VALIDATION.md (step 6: "six outputs at grids
                 80/40/20 with 68 and 81 channels") and core/accel.py's notes
                 from the real .hef (68-channel logits, 81-channel probabilities).
  output names   `damoyolo_tinynasL20_T/convNN`. The real names are NOT known
                 here; the decoder pairs heads by SHAPE, never by name, and the
                 tests shuffle the order to prove it.
  input          one NHWC uint8 (640, 640, 3) input.
  exclusivity    one VDevice per process, like the real chip.

What it does NOT model: quantisation (outputs are whatever the backend returns),
timing, the PCIe link, firmware, thermal behaviour. A pass under this fake says
the code path is wired correctly; it says nothing about the accelerator.

Usage::

    import fake_hailo
    fake_hailo.install(backend)        # sys.modules["hailo_platform"] = fake
    ...                                # core.accel.HailoYOLO(...) now loads
    fake_hailo.uninstall()

`backend(batch_uint8_nhwc) -> {output_name: float32 (1, G, G, C)}`. The default
returns all-zero class scores (no detections) and neutral box logits.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

INPUT_NAME = "damoyolo_tinynasL20_T/input_layer1"
INPUT_SHAPE = (640, 640, 3)
#: (name, (H, W, C)) in the order get_output_vstream_infos() returns them. The
#: order is deliberately NOT finest-grid-first: the decoder must not depend on it.
OUTPUTS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("damoyolo_tinynasL20_T/conv77", (20, 20, 81)),
    ("damoyolo_tinynasL20_T/conv53", (80, 80, 68)),
    ("damoyolo_tinynasL20_T/conv66", (40, 40, 81)),
    ("damoyolo_tinynasL20_T/conv76", (20, 20, 68)),
    ("damoyolo_tinynasL20_T/conv54", (80, 80, 81)),
    ("damoyolo_tinynasL20_T/conv65", (40, 40, 68)),
)
BOX_CHANNELS = 68
CLASS_CHANNELS = 81  # 80 COCO + 1 padding

Backend = Callable[[np.ndarray], dict]


def zero_backend(batch: np.ndarray) -> dict:
    """No objects anywhere: class scores 0, box logits 0 (uniform DFL)."""
    if batch.shape != (1, *INPUT_SHAPE) or batch.dtype != np.uint8:
        raise RuntimeError(f"fake HEF: input must be uint8 (1, 640, 640, 3), got {batch.dtype} {batch.shape}")
    return {name: np.zeros((1, *shape), dtype=np.float32) for name, shape in OUTPUTS}


@dataclass
class _State:
    backend: Backend = zero_backend
    outputs: tuple = OUTPUTS
    vdevice_open: bool = False
    infers: int = 0
    inputs_seen: list = field(default_factory=list)
    keep_inputs: bool = False
    fail_next_infer: Optional[str] = None


state = _State()


class _Info:
    """Only what core.accel and hwv_common read: name and shape (plus None quant)."""

    def __init__(self, name: str, shape: tuple) -> None:
        self.name, self.shape = name, tuple(shape)
        self.quant_info = None
        self.format = None


class HEF:
    def __init__(self, path: str) -> None:
        self.path = path

    def get_input_vstream_infos(self):
        return [_Info(INPUT_NAME, INPUT_SHAPE)]

    def get_output_vstream_infos(self):
        return [_Info(name, shape) for name, shape in state.outputs]


class _Activation:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _NetworkGroup:
    def create_params(self):
        return {}

    def activate(self, params=None):
        return _Activation()


class VDevice:
    def __init__(self, *args, **kwargs) -> None:
        if state.vdevice_open:
            raise RuntimeError("fake HEF: HAILO_OUT_OF_PHYSICAL_DEVICES (the chip is exclusive)")
        state.vdevice_open = True

    def configure(self, hef, params):
        return [_NetworkGroup()]

    def release(self) -> None:
        state.vdevice_open = False


class InferVStreams:
    def __init__(self, network_group, input_params, output_params) -> None:
        self.output_params = output_params

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def infer(self, feed: dict) -> dict:
        if state.fail_next_infer:
            message, state.fail_next_infer = state.fail_next_infer, None
            raise RuntimeError(message)
        if list(feed) != [INPUT_NAME]:
            raise RuntimeError(f"fake HEF: expected one input {INPUT_NAME!r}, got {list(feed)}")
        batch = np.asarray(feed[INPUT_NAME])
        if state.keep_inputs:
            state.inputs_seen.append(batch.copy())
        state.infers += 1
        return state.backend(batch)


class _Device:
    @staticmethod
    def scan():
        return ["FAKE-0000:00:00.0 (simulated, not hardware)"]


def module() -> types.ModuleType:
    """A module object with the hailo_platform surface core/accel.py uses."""
    fake = types.ModuleType("hailo_platform")
    fake.__version__ = "fake-0 (simulated; NOT HailoRT)"  # type: ignore[attr-defined]
    fake.SIMULATED = True  # type: ignore[attr-defined]
    namespace = {
        "HEF": HEF, "VDevice": VDevice, "InferVStreams": InferVStreams, "Device": _Device,
        "ConfigureParams": types.SimpleNamespace(create_from_hef=lambda hef, interface=None: {}),
        "HailoStreamInterface": types.SimpleNamespace(PCIe="pcie"),
        "FormatType": types.SimpleNamespace(UINT8="uint8", FLOAT32="float32", AUTO="auto"),
        "InputVStreamParams": types.SimpleNamespace(make=lambda ng, format_type=None: {}),
        "OutputVStreamParams": types.SimpleNamespace(make=lambda ng, format_type=None: {}),
    }
    for key, value in namespace.items():
        setattr(fake, key, value)
    return fake


def install(backend: Backend = zero_backend, *, outputs: Optional[tuple] = None,
            keep_inputs: bool = False) -> types.ModuleType:
    """Put the fake in sys.modules["hailo_platform"] with a fresh device state."""
    global state  # noqa: PLW0603 - one simulated chip per process, like the real one
    state = _State(backend=backend, outputs=tuple(outputs or OUTPUTS), keep_inputs=keep_inputs)
    fake = module()
    sys.modules["hailo_platform"] = fake
    return fake


def uninstall() -> None:
    mod = sys.modules.get("hailo_platform")
    if mod is not None and getattr(mod, "SIMULATED", False):
        del sys.modules["hailo_platform"]


def is_simulated() -> bool:
    mod = sys.modules.get("hailo_platform")
    return bool(mod is not None and getattr(mod, "SIMULATED", False))


# ---------------------------------------------------------------- planted objects
def encode_distance(d: float, bins: int = 17) -> np.ndarray:
    """DFL logits whose softmax expectation is exactly `d` (in stride units, 0..bins-1).

    Two adjacent bins share the mass linearly; every other bin gets a logit so
    low that its probability underflows to 0 in float32.
    """
    if not 0.0 <= d <= bins - 1:
        raise ValueError(f"distance {d} is outside the DFL range [0, {bins - 1}]")
    lo = int(np.floor(d))
    hi = min(lo + 1, bins - 1)
    frac = d - lo
    logits = np.full(bins, -1e4, dtype=np.float32)
    if hi == lo or frac == 0.0:
        logits[lo] = 0.0
    else:
        logits[lo] = np.log(1.0 - frac)
        logits[hi] = np.log(frac)
    return logits


def planted_heads(objects, *, anchor_offset: float = 0.5, background: float = 0.0) -> dict:
    """Head tensors that decode to `objects`, keyed by the fake .hef's output names.

    `objects`: iterable of (class_id, score, x1, y1, x2, y2, stride) in MODEL-INPUT
    pixels (the 640x640 letterboxed tensor). Each is encoded at the anchor whose
    point is nearest the box centre under `anchor_offset` (0.5 = the production
    "legacy" convention, 0.0 = DAMO-YOLO's own), so decoding with the matching
    convention returns the box exactly (to float32 precision).
    """
    out = {name: np.zeros((1, *shape), dtype=np.float32) for name, shape in state.outputs}
    for name, (_g, _, c) in state.outputs:
        if c == CLASS_CHANNELS and background:
            out[name][...] = background
    by_grid = {}
    for name, (g, _, c) in state.outputs:
        by_grid.setdefault(g, {})["box" if c == BOX_CHANNELS else "cls"] = name
    for cls_id, score, x1, y1, x2, y2, stride in objects:
        grid = INPUT_SHAPE[0] // stride
        names = by_grid[grid]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        col = int(np.clip(round(cx / stride - anchor_offset), 0, grid - 1))
        row = int(np.clip(round(cy / stride - anchor_offset), 0, grid - 1))
        px, py = (col + anchor_offset) * stride, (row + anchor_offset) * stride
        dists = ((px - x1) / stride, (py - y1) / stride, (x2 - px) / stride, (y2 - py) / stride)
        logits = np.concatenate([encode_distance(d) for d in dists])
        out[names["box"]][0, row, col, :] = logits
        out[names["cls"]][0, row, col, int(cls_id)] = float(score)
    return out


def planted_backend(objects, **kwargs) -> Backend:
    heads = planted_heads(objects, **kwargs)

    def run(batch: np.ndarray) -> dict:
        zero_backend(batch)  # validates the input exactly as the zero backend does
        return {k: v.copy() for k, v in heads.items()}

    return run


# ---------------------------------------------------------------- float ONNX backend
def nchw_heads_to_hef(outputs: list[np.ndarray], *, pad_value: float = 0.0) -> dict:
    """Float-ONNX head tensors -> the six NHWC tensors the .hef emits, by name.

    The ONNX end nodes (Mul_224, Sigmoid_244, ...) are NCHW (1, C, G, G); the
    .hef's FLOAT32 vstreams are NHWC (1, G, G, C). A class head with 80 channels
    is padded to the .hef's 81 with `pad_value` (the decoder must drop it).
    Pairs by (grid, role), exactly like the decoder, and refuses anything else.
    """
    by_key: dict[tuple[int, str], np.ndarray] = {}
    for array in outputs:
        a = np.asarray(array, dtype=np.float32)
        if a.ndim != 4 or a.shape[0] != 1:
            raise RuntimeError(f"ONNX head has shape {a.shape}; expected (1, C, G, G) or (1, G, G, C)")
        nhwc = a if a.shape[1] == a.shape[2] else a.transpose(0, 2, 3, 1)
        if nhwc.shape[1] != nhwc.shape[2]:
            raise RuntimeError(f"ONNX head {a.shape} is not a square grid in NCHW or NHWC")
        role = "box" if nhwc.shape[3] == BOX_CHANNELS else "cls"
        if role == "cls" and nhwc.shape[3] < CLASS_CHANNELS:
            pad = np.full((*nhwc.shape[:3], CLASS_CHANNELS - nhwc.shape[3]), pad_value, dtype=np.float32)
            nhwc = np.concatenate([nhwc, pad], axis=3)
        by_key[(nhwc.shape[1], role)] = np.ascontiguousarray(nhwc)
    result = {}
    for name, (grid, _, channels) in state.outputs:
        role = "box" if channels == BOX_CHANNELS else "cls"
        if (grid, role) not in by_key:
            raise RuntimeError(f"ONNX outputs have no grid-{grid} {role} head")
        result[name] = by_key[(grid, role)]
    return result


def onnx_backend(session, *, pad_value: float = 0.0, swap_rb: bool = False) -> Backend:
    """A backend running the float ONNX heads model on the exact uint8 tensor the device sends.

    The network's normalisation is inside the graph (mean 0, std 1 on 0..255), so
    the uint8 NHWC batch is fed as float32 NCHW unchanged. `swap_rb` feeds the
    channels reversed: a deliberate fault, used only to confirm the channel order.
    """
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    def run(batch: np.ndarray) -> dict:
        if batch.shape != (1, *INPUT_SHAPE) or batch.dtype != np.uint8:
            raise RuntimeError(f"fake HEF: input must be uint8 (1, 640, 640, 3), got {batch.dtype} {batch.shape}")
        x = batch[..., ::-1] if swap_rb else batch
        tensor = np.ascontiguousarray(x.transpose(0, 3, 1, 2)).astype(np.float32)
        return nchw_heads_to_hef(session.run(output_names, {input_name: tensor}), pad_value=pad_value)

    return run
