"""ONNX detector wrappers with explicit, documented pre- and post-processing.

Every wrapper maps an RGB PIL image to detections in NORMALIZED image coordinates
[(label, score, x1, y1, x2, y2)], keeping only the classes we evaluate (COCO
`person` -> "person", COCO `cell phone` -> "phone"). COCO has no hand class, so
no pretrained COCO detector can be scored on hands.

DAMO-YOLO is decoded three ways so that each production choice can be measured
separately (see core/accel.py HailoYOLO._decode_raw, and the Hailo Model Zoo's
own nanodet_split postprocessing, whose anchor offset for DAMO-YOLO is 0.0):
  production         centered letterbox, pad 114, anchor at (cell+0.5)*stride,
                     one class per anchor (argmax), NMS IoU 0.50 -- the device's code
  offset0            the same, with anchors at cell*stride (the Model Zoo's decode)
  offset0_multilabel the same, scoring every class at every anchor (a phone that
                     shares an anchor with a higher-scoring class is not dropped)
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, overload

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
if os.environ.get("SD_PYLIB"):
    sys.path.insert(0, os.environ["SD_PYLIB"])

from core.accel import _nms, letterbox_transform

COCO_TO_OURS = {0: "person", 67: "phone"}
DFL_BINS = 17
SCORE_FLOOR = 0.05  # detections below this are never kept (AP is computed above it)
MAX_DETS = 100


@dataclass
class Timing:
    pre_ms: list[float] = field(default_factory=list)
    infer_ms: list[float] = field(default_factory=list)
    post_ms: list[float] = field(default_factory=list)


def letterbox(image, size: int, *, centered: bool, pad: int) -> tuple[np.ndarray, float, int, int]:
    """RGB uint8 HxWx3 of size x size, plus the scale and offsets to map boxes back."""
    from PIL import Image

    w, h = image.size
    if centered:
        scale, left, top = letterbox_transform(w, h, size, size)  # the production geometry
    else:
        scale, left, top = min(size / w, size / h), 0, 0
    new_w, new_h = round(w * scale), round(h * scale)
    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = np.full((size, size, 3), pad, dtype=np.uint8)
    canvas[top:top + new_h, left:left + new_w] = np.asarray(resized)
    return canvas, scale, left, top


def preprocess(image, size: int, mode: str) -> tuple[np.ndarray, float, float, int, int]:
    """RGB canvas plus (sx, sy, left, top) to map input pixels back to the image.

    letterbox114  production: centered letterbox, pad 114 (core/accel.py)
    topleft0      Hailo Model Zoo eval: top-left letterbox, pad 0 (base/damoyolo.yaml)
    stretch       DAMO-YOLO's own test transform: resize to size x size, keep_ratio False
    """
    from PIL import Image

    if mode == "stretch":
        w, h = image.size
        canvas = np.asarray(image.resize((size, size), Image.Resampling.BILINEAR))
        return canvas, size / w, size / h, 0, 0
    if mode not in ("letterbox114", "topleft0"):
        raise ValueError(f"unknown preprocess mode {mode!r}")
    canvas, scale, left, top = letterbox(image, size, centered=mode == "letterbox114",
                                         pad=114 if mode == "letterbox114" else 0)
    return canvas, scale, scale, left, top


@overload
def decode_dfl(heads: list[tuple[np.ndarray, np.ndarray, int]], *, conf: float, offset: float,
               multilabel: bool, nms_iou: float, num_classes: int = 80,
               with_argmax: Literal[False] = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
@overload
def decode_dfl(heads: list[tuple[np.ndarray, np.ndarray, int]], *, conf: float, offset: float,
               multilabel: bool, nms_iou: float, num_classes: int = 80,
               with_argmax: Literal[True]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
def decode_dfl(heads: list[tuple[np.ndarray, np.ndarray, int]], *, conf: float, offset: float,
               multilabel: bool, nms_iou: float, num_classes: int = 80, with_argmax: bool = False,
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """DAMO-YOLO split heads -> (boxes_xyxy in input pixels, scores, class ids[, anchor argmax class]).

    ``heads``: per stride, (box logits HxWx(4*17), class probabilities HxWxC, stride).
    ``with_argmax`` also returns, for each kept box, the class that won its anchor:
    a box whose class differs from it exists only because of multilabel scoring.
    """
    all_boxes, all_scores, all_classes, all_argmax = [], [], [], []
    for box_raw, cls_raw, stride in heads:
        grid = box_raw.shape[0]
        scores = cls_raw.reshape(-1, cls_raw.shape[-1])[:, :num_classes]
        if multilabel:
            anchor_idx, class_idx = np.nonzero(scores >= conf)
            best = scores[anchor_idx, class_idx]
        else:
            class_all = scores.argmax(axis=1)
            best_all = scores.max(axis=1)
            anchor_idx = np.nonzero(best_all >= conf)[0]
            class_idx, best = class_all[anchor_idx], best_all[anchor_idx]
        if anchor_idx.size == 0:
            continue
        dfl = box_raw.reshape(-1, 4, DFL_BINS)[anchor_idx]
        dfl = dfl - dfl.max(axis=2, keepdims=True)
        prob = np.exp(dfl)
        prob /= prob.sum(axis=2, keepdims=True)
        dist = (prob * np.arange(DFL_BINS, dtype=np.float32)).sum(axis=2)
        ys, xs = np.divmod(anchor_idx, grid)
        cx, cy = (xs + offset) * stride, (ys + offset) * stride
        all_boxes.append(np.stack([cx - dist[:, 0] * stride, cy - dist[:, 1] * stride,
                                   cx + dist[:, 2] * stride, cy + dist[:, 3] * stride], axis=1))
        all_scores.append(best)
        all_classes.append(class_idx)
        all_argmax.append(scores[anchor_idx].argmax(axis=1))
    if not all_boxes:
        empty = (np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int))
        return (*empty, np.zeros(0, dtype=int)) if with_argmax else empty
    boxes, scores_, classes, keep = _class_nms(np.concatenate(all_boxes), np.concatenate(all_scores),
                                               np.concatenate(all_classes), nms_iou, return_index=True)
    if with_argmax:
        return boxes, scores_, classes, np.concatenate(all_argmax)[keep]
    return boxes, scores_, classes


def _class_nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, nms_iou: float,
               return_index: bool = False):
    keep: list[int] = []
    for cls_id in np.unique(classes):
        sel = np.nonzero(classes == cls_id)[0]
        keep.extend(sel[_nms(boxes[sel], scores[sel], nms_iou)])
    keep = sorted(keep, key=lambda i: -scores[i])[:MAX_DETS]
    if return_index:
        return boxes[keep], scores[keep], classes[keep], np.asarray(keep, dtype=int)
    return boxes[keep], scores[keep], classes[keep]


def _to_normalized(boxes, scores, classes, sx, sy, left, top, w, h, extra=None):
    """Input-pixel boxes -> [(label, score, x1, y1, x2, y2)] normalized to the image, plus aligned extras."""
    out, kept_extra = [], []
    for n, ((x1, y1, x2, y2), s, c) in enumerate(zip(boxes, scores, classes, strict=True)):
        label = COCO_TO_OURS.get(int(c))
        if label is None:
            continue
        nx1 = min(max((x1 - left) / sx, 0.0), w) / w
        ny1 = min(max((y1 - top) / sy, 0.0), h) / h
        nx2 = min(max((x2 - left) / sx, 0.0), w) / w
        ny2 = min(max((y2 - top) / sy, 0.0), h) / h
        if nx2 > nx1 and ny2 > ny1:
            out.append((label, float(s), nx1, ny1, nx2, ny2))
            kept_extra.append(None if extra is None else extra[n])
    return out, kept_extra


def device_decode(heads, decoder: str, conf: float, nms_iou: float, size: int):
    """Decode with the PRODUCT's code (core/accel.py HailoYOLO._decode_raw), in input-pixel space."""
    from core.accel import HailoYOLO

    device = HailoYOLO.__new__(HailoYOLO)
    device._nms_iou = nms_iou
    if decoder != "legacy":
        if "reference" not in getattr(HailoYOLO, "DECODERS", ()):
            raise RuntimeError("this core/accel.py has no reference decoder (overlay the candidate branch)")
        device.decoder = decoder
    raw, device._scales = {}, []
    for box, cls, stride in heads:
        raw[f"b{stride}"], raw[f"c{stride}"] = box[None], cls[None]
        device._scales.append({"box": f"b{stride}", "cls": f"c{stride}", "grid": box.shape[0], "stride": stride})
    (result,) = device._decode_raw(raw, 1.0, 0, 0, size, size, None, conf)
    ranked = sorted(result.boxes, key=lambda b: -float(b.conf[0]))[:MAX_DETS]
    return (np.array([b.xyxy[0] for b in ranked]).reshape(-1, 4), np.array([float(b.conf[0]) for b in ranked]),
            np.array([int(b.cls[0]) for b in ranked], dtype=int))


def _session(path: str, provider: str):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if provider == "cuda":
        import torch  # noqa: F401  # pyright: ignore[reportMissingImports] - loads the CUDA/cuDNN libraries ORT needs

        providers: list[Any] = [("CUDAExecutionProvider", {"use_tf32": 0}), "CPUExecutionProvider"]
    elif provider == "dml":
        opts.enable_mem_pattern = False
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        providers = [("DmlExecutionProvider", {"device_id": int(os.environ.get("SD_DML_DEVICE", "1"))}),
                     "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(path, opts, providers=providers)
    want = {"cuda": "CUDAExecutionProvider", "dml": "DmlExecutionProvider", "cpu": "CPUExecutionProvider"}[provider]
    if sess.get_providers()[0] != want:  # ORT silently falls back to CPU on a bad device
        raise RuntimeError(f"{path}: wanted {want}, got {sess.get_providers()}")
    return sess


class DamoYolo:
    """DAMO-YOLO from the Hailo Model Zoo pretrained ONNX (the float source of the shipping .hef)."""

    family = "damo-yolo"

    #: The end nodes the Hailo Model Zoo compiles for damoyolo_tinynasL20_T (its .yaml `parser.nodes`).
    HEAD_NODES = ("Mul_224", "Sigmoid_244", "Mul_264", "Sigmoid_284", "Mul_304", "Sigmoid_324")

    def __init__(self, onnx_path: str, provider: str, *, variant: str, head_nodes: tuple[str, ...] = HEAD_NODES,
                 size: int = 640, decoder: str = "research", preprocess: str = "letterbox114",
                 nms_iou: float = 0.50) -> None:
        """``decoder``: "research" (decode_dfl, per ``variant``) or the product's own
        "device_legacy" / "device_reference" (core/accel.py)."""
        self.variant = variant
        self.size = size
        self.offset = 0.5 if variant == "production" else 0.0
        self.multilabel = variant == "offset0_multilabel"
        self.decoder, self.preprocess, self.nms_iou = decoder, preprocess, nms_iou
        self.last_argmax: list = []  # per returned detection: the class that won its anchor (research decoder)
        self.timing = Timing()
        self.session = _session(_with_outputs(onnx_path, list(head_nodes)), provider)
        self.input = self.session.get_inputs()[0].name
        self.outputs = [o.name for o in self.session.get_outputs()]

    def __call__(self, image, conf: float = SCORE_FLOOR):
        t0 = time.perf_counter()
        canvas, sx, sy, left, top = preprocess(image, self.size, self.preprocess)
        tensor = canvas.transpose(2, 0, 1)[None].astype(np.float32)  # RGB 0..255; normalisation is in-net
        t1 = time.perf_counter()
        outputs = self.session.run(self.outputs, {self.input: tensor})
        t2 = time.perf_counter()
        heads = split_heads(outputs, self.size)
        argmax = None
        if self.decoder == "research":
            boxes, scores, classes, argmax = decode_dfl(heads, conf=conf, offset=self.offset,
                                                        multilabel=self.multilabel, nms_iou=self.nms_iou,
                                                        with_argmax=True)
        else:
            boxes, scores, classes = device_decode(heads, self.decoder.removeprefix("device_"), conf,
                                                   self.nms_iou, self.size)
        dets, self.last_argmax = _to_normalized(boxes, scores, classes, sx, sy, left, top, *image.size,
                                                extra=argmax)
        t3 = time.perf_counter()
        self.timing.pre_ms.append((t1 - t0) * 1e3)
        self.timing.infer_ms.append((t2 - t1) * 1e3)
        self.timing.post_ms.append((t3 - t2) * 1e3)
        return dets


class Yolox:
    """YOLOX official release ONNX (Megvii, Apache-2.0): BGR 0..255, top-left letterbox pad 114."""

    family = "yolox"

    def __init__(self, onnx_path: str, provider: str, *, size: int) -> None:
        self.size = size
        self.variant = "official"
        self.timing = Timing()
        self.session = _session(onnx_path, provider)
        self.input = self.session.get_inputs()[0].name
        grids, strides = [], []
        for stride in (8, 16, 32):
            g = size // stride
            ys, xs = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
            grids.append(np.stack([xs, ys], axis=-1).reshape(-1, 2))
            strides.append(np.full((g * g, 1), stride))
        self.grids, self.strides = np.concatenate(grids), np.concatenate(strides)

    def __call__(self, image, conf: float = SCORE_FLOOR):
        t0 = time.perf_counter()
        canvas, scale, left, top = letterbox(image, self.size, centered=False, pad=114)
        tensor = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)  # BGR, as YOLOX trains
        t1 = time.perf_counter()
        (raw,) = self.session.run(None, {self.input: tensor})
        t2 = time.perf_counter()
        out = raw[0]
        xy = (out[:, :2] + self.grids) * self.strides
        wh = np.exp(out[:, 2:4]) * self.strides
        scores_all = out[:, 4:5] * out[:, 5:]
        anchor_idx, class_idx = np.nonzero(scores_all >= conf)  # multilabel, as YOLOX's own eval does
        boxes = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1)[anchor_idx]
        boxes, scores, classes = _class_nms(boxes, scores_all[anchor_idx, class_idx], class_idx, 0.50)
        dets, _ = _to_normalized(boxes, scores, classes, scale, scale, left, top, *image.size)
        t3 = time.perf_counter()
        self.timing.pre_ms.append((t1 - t0) * 1e3)
        self.timing.infer_ms.append((t2 - t1) * 1e3)
        self.timing.post_ms.append((t3 - t2) * 1e3)
        return dets


def split_heads(outputs: list[np.ndarray], size: int) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Pair box (4*17 channels) and class heads by grid, as core/accel.py does, from NCHW or NHWC tensors."""
    by_grid: dict[int, dict] = {}
    box_channels = 4 * DFL_BINS
    for t in outputs:
        hwc = t[0] if t.shape[1] == t.shape[2] else t[0].transpose(1, 2, 0)  # NHWC has square dims 1,2
        entry = by_grid.setdefault(hwc.shape[0], {})
        entry["box" if hwc.shape[-1] == box_channels else "cls"] = hwc
    return [(e["box"], e["cls"], size // grid) for grid, e in sorted(by_grid.items(), reverse=True)]


def _with_outputs(onnx_path: str, node_names: list[str]) -> str:
    """A copy of the model whose graph outputs are the named nodes' outputs (the heads Hailo compiles)."""
    import onnx  # pyright: ignore[reportMissingImports] - worker-only dependency

    target = Path(onnx_path).with_suffix(".heads.onnx")
    if target.exists():
        return str(target)
    model = onnx.load(onnx_path)
    by_name = {n.name: n for n in model.graph.node}
    missing = [n for n in node_names if n not in by_name]
    if missing:
        raise RuntimeError(f"{onnx_path}: nodes {missing} not found; cannot extract the compiled heads")
    del model.graph.output[:]
    for name in node_names:
        model.graph.output.append(onnx.helper.make_empty_tensor_value_info(by_name[name].output[0]))
    onnx.save(model, str(target))
    return str(target)


def percentile(xs: list[float], q: float) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * (len(s) - 1)))]
