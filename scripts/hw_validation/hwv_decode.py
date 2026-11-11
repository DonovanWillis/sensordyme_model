"""Raw-tensor decoders for damoyolo_tinynasL20_T: the production ("legacy") one and
the DAMO-YOLO reference, plus the box matching used to compare them.

Pure numpy. No hailo_platform, no cv2, no `core` import: the SAME file decodes on
the Pi (capture_hailo_outputs.py, for timing) and on a worker
(compare_float_vs_int8.py, for agreement), so a difference between the two
places can never be a difference between two copies of a decoder.

The two decoders differ in exactly three ways, each a DecoderSpec field so the
variants below can isolate them one at a time:

  anchor_offset  production: (cell + 0.5) * stride
                 reference:  cell * stride  (DAMO-YOLO ZeroHead
                 get_single_level_center_priors: arange(w) * stride; the Hailo
                 Model Zoo nanodet_split postprocess also uses offset 0)
  multilabel     production: argmax class per anchor only
                 reference:  every (anchor, class) above threshold
                 (DAMO-YOLO multiclass_nms)
  nms / cap      production: class-wise greedy NMS at IoU 0.50, no cap
                 reference:  class-wise NMS at IoU 0.70, top 100
                 (DAMO-YOLO default nms_iou_thre / max detections; both are
                 CLI-overridable in the comparison script because the candidate
                 decoder being implemented separately may choose differently)

`legacy` reproduces core.accel.HailoYOLO._decode_raw exactly (pinned by
tests/test_m5_hw_validation.py against the production method). Preprocessing
(letterbox vs stretch) is NOT a decoder property: it is recorded per sample in
the bundle and undone by `unmap`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

import numpy as np

DFL_BINS = 17
BOX_CHANNELS = 4 * DFL_BINS
NUM_CLASSES = 80  # COCO; channels past 80 in the class head are padding


@dataclass(frozen=True)
class DecoderSpec:
    name: str
    anchor_offset: float
    multilabel: bool
    nms_iou: float
    max_det: Optional[int] = None
    #: True: keep score > conf (DAMO-YOLO multiclass_nms). False: score >= conf
    #: (production _decode_raw).
    strict_threshold: bool = False


LEGACY = DecoderSpec("legacy", anchor_offset=0.5, multilabel=False, nms_iou=0.50)
REFERENCE = DecoderSpec("reference", anchor_offset=0.0, multilabel=True, nms_iou=0.70,
                        max_det=100, strict_threshold=True)

#: Factorial variants: legacy and reference plus one change at a time, so a
#: difference between the end points can be attributed to offset or to labels.
VARIANTS: dict[str, DecoderSpec] = {
    spec.name: spec
    for spec in (
        LEGACY,
        DecoderSpec("legacy_offset0", anchor_offset=0.0, multilabel=False, nms_iou=0.50),
        DecoderSpec("legacy_multilabel", anchor_offset=0.5, multilabel=True, nms_iou=0.50),
        REFERENCE,
    )
}


def with_overrides(spec: DecoderSpec, *, nms_iou: Optional[float] = None,
                   max_det: Optional[int] = None) -> DecoderSpec:
    changes = {}
    if nms_iou is not None:
        changes["nms_iou"] = float(nms_iou)
    if max_det is not None:
        changes["max_det"] = int(max_det) if max_det > 0 else None
    return dataclasses.replace(spec, **changes) if changes else spec


# ------------------------------------------------------------------ head layout
@dataclass(frozen=True)
class Scale:
    grid: int
    stride: int
    box: str
    cls: str
    num_classes: int


def group_heads(shapes: dict[str, tuple[int, ...]], input_h: int) -> list[Scale]:
    """Pair each grid's 68-channel box head with its class head, BY SHAPE.

    Same rule as HailoYOLO._group_raw_outputs (stride = input_h // grid, finest
    grid first). `shapes` maps tensor name -> (H, W, C) or (1, H, W, C), NHWC.
    """
    by_grid: dict[int, dict] = {}
    for name, shape in shapes.items():
        h, w, c = tuple(shape)[-3:]
        if h != w:
            raise ValueError(f"non-square detection grid {tuple(shape)} on {name}")
        entry = by_grid.setdefault(h, {"grid": h, "stride": input_h // h})
        if c == BOX_CHANNELS:
            entry["box"] = name
        else:
            entry["cls"] = name
            entry["num_classes"] = c
    scales = []
    for grid in sorted(by_grid, reverse=True):
        entry = by_grid[grid]
        if "box" not in entry or "cls" not in entry:
            raise ValueError(f"grid {grid} is missing a box or class head: {entry}")
        scales.append(Scale(**entry))
    return scales


# ------------------------------------------------------------------ detections
@dataclass
class Detections:
    """Decoded boxes, xyxy. Coordinates are in model-input pixels until unmap()."""

    boxes: np.ndarray    # (N, 4) float64
    scores: np.ndarray   # (N,) float64
    classes: np.ndarray  # (N,) int64
    strides: np.ndarray  # (N,) int64, the FPN level each box came from

    @classmethod
    def empty(cls) -> Detections:
        return cls(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int64),
                   np.zeros(0, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.scores)

    def take(self, idx) -> Detections:
        idx = np.asarray(idx, dtype=np.int64)
        return Detections(self.boxes[idx], self.scores[idx], self.classes[idx], self.strides[idx])

    def with_boxes(self, boxes: np.ndarray) -> Detections:
        return Detections(np.asarray(boxes, dtype=np.float64), self.scores, self.classes, self.strides)

    def to_list(self) -> list[dict]:
        return [
            {"xyxy": [round(float(v), 2) for v in b], "score": round(float(s), 5),
             "cls": int(c), "stride": int(st)}
            for b, s, c, st in zip(self.boxes, self.scores, self.classes, self.strides, strict=True)
        ]


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy NMS, the same algorithm as core.accel._nms (pinned by a test)."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[iou <= iou_threshold]
    return keep


def _hwc(array) -> np.ndarray:
    a = np.asarray(array)
    return a[0] if a.ndim == 4 else a


def dfl_distances(box_logits: np.ndarray) -> np.ndarray:
    """(N, 4*17) or (N, 4, 17) logits -> (N, 4) expected distances in stride units."""
    dfl = np.asarray(box_logits).reshape(-1, 4, DFL_BINS)
    dfl = dfl - dfl.max(axis=2, keepdims=True)
    prob = np.exp(dfl)
    prob /= prob.sum(axis=2, keepdims=True)
    return (prob * np.arange(DFL_BINS, dtype=np.float32)).sum(axis=2)


def decode(outputs: dict, scales: list[Scale], spec: DecoderSpec, conf: float, *,
           num_classes: int = NUM_CLASSES, classes=None) -> Detections:
    """Decode one image's raw head tensors into NMS'd detections (input pixels)."""
    wanted = None if classes is None else np.asarray(sorted({int(c) for c in classes}))
    all_boxes, all_scores, all_classes, all_strides = [], [], [], []
    for s in scales:
        box_raw = _hwc(outputs[s.box])
        cls_raw = _hwc(outputs[s.cls])
        scores = cls_raw.reshape(-1, cls_raw.shape[-1])[:, :num_classes]
        if spec.multilabel:
            mask = scores > conf if spec.strict_threshold else scores >= conf
            if wanted is not None:
                mask &= np.isin(np.arange(scores.shape[1]), wanted)[None, :]
            anchor_idx, cls_idx = np.nonzero(mask)
            cls_scores = scores[anchor_idx, cls_idx]
        else:
            best_cls = scores.argmax(axis=1)
            best_score = scores.max(axis=1)
            keep = best_score > conf if spec.strict_threshold else best_score >= conf
            if wanted is not None:
                keep &= np.isin(best_cls, wanted)
            anchor_idx = np.nonzero(keep)[0]
            cls_idx = best_cls[anchor_idx]
            cls_scores = best_score[anchor_idx]
        if anchor_idx.size == 0:
            continue
        dist = dfl_distances(box_raw.reshape(-1, box_raw.shape[-1])[anchor_idx])
        ys, xs = np.divmod(anchor_idx, s.grid)
        cx = (xs + spec.anchor_offset) * s.stride
        cy = (ys + spec.anchor_offset) * s.stride
        all_boxes.append(np.stack([
            cx - dist[:, 0] * s.stride, cy - dist[:, 1] * s.stride,
            cx + dist[:, 2] * s.stride, cy + dist[:, 3] * s.stride,
        ], axis=1))
        all_scores.append(cls_scores)
        all_classes.append(cls_idx)
        all_strides.append(np.full(anchor_idx.size, s.stride, dtype=np.int64))
    if not all_boxes:
        return Detections.empty()
    boxes = np.concatenate(all_boxes).astype(np.float64)
    scores = np.concatenate(all_scores).astype(np.float64)
    class_ids = np.concatenate(all_classes).astype(np.int64)
    strides = np.concatenate(all_strides)

    keep_idx: list[int] = []
    for cls_id in np.unique(class_ids):
        sel = np.nonzero(class_ids == cls_id)[0]
        keep_idx.extend(int(i) for i in sel[nms(boxes[sel], scores[sel], spec.nms_iou)])
    keep_idx.sort(key=lambda j: -scores[j])
    if spec.max_det is not None:
        keep_idx = keep_idx[: spec.max_det]
    return Detections(boxes, scores, class_ids, strides).take(keep_idx)


def unmap(dets: Detections, geometry: dict) -> Detections:
    """Model-input pixels -> original-frame pixels, clipped to the frame.

    geometry (recorded per sample by the capture script):
      letterbox: {"mode": "letterbox", "scale", "left", "top", "frame_w", "frame_h"}
      stretch:   {"mode": "stretch", "input_w", "input_h", "frame_w", "frame_h"}
    """
    if len(dets) == 0:
        return dets
    w, h = float(geometry["frame_w"]), float(geometry["frame_h"])
    b = dets.boxes.copy()
    if geometry["mode"] == "letterbox":
        scale = float(geometry["scale"])
        b[:, [0, 2]] = (b[:, [0, 2]] - float(geometry["left"])) / scale
        b[:, [1, 3]] = (b[:, [1, 3]] - float(geometry["top"])) / scale
    elif geometry["mode"] == "stretch":
        b[:, [0, 2]] = b[:, [0, 2]] / (float(geometry["input_w"]) / w)
        b[:, [1, 3]] = b[:, [1, 3]] / (float(geometry["input_h"]) / h)
    else:
        raise ValueError(f"unknown preprocessing mode {geometry['mode']!r}")
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0.0, w)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0.0, h)
    return dets.with_boxes(b)


# ------------------------------------------------------------------ comparison
def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,4) x (M,4) xyxy -> (N, M) IoU."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def match(a: Detections, b: Detections, iou_threshold: float = 0.5) -> list[tuple[int, int, float]]:
    """Greedy same-class matching, A in descending score order; each B used once."""
    pairs: list[tuple[int, int, float]] = []
    if len(a) == 0 or len(b) == 0:
        return pairs
    ious = box_iou(a.boxes, b.boxes)
    ious[a.classes[:, None] != b.classes[None, :]] = -1.0
    used = np.zeros(len(b), dtype=bool)
    for i in np.argsort(-a.scores, kind="stable"):
        row = np.where(used, -1.0, ious[i])
        j = int(row.argmax())
        if row[j] >= iou_threshold:
            used[j] = True
            pairs.append((int(i), j, float(row[j])))
    return pairs


def percentiles(values, qs=(0.05, 0.5, 0.95)) -> Optional[dict]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return None
    out = {"n": int(arr.size), "mean": float(arr.mean()), "min": float(arr.min()),
           "max": float(arr.max())}
    for q in qs:
        out[f"p{round(q * 100):d}"] = float(np.quantile(arr, q))
    return out


class Agreement:
    """Accumulates A-vs-B detection agreement over many images.

    Score delta is B - A. Centre shift is A - B, per FPN stride of the A box:
    for legacy (A) vs an offset-0 decoder (B) on the SAME tensors, a pure anchor
    offset shows up as dx = dy = +stride/2 exactly.
    """

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = iou_threshold
        self.n_a = self.n_b = self.images = 0
        self.ious: list[float] = []
        self.best_ious: list[float] = []
        self.score_deltas: list[float] = []
        self.shift: dict[int, list[tuple[float, float]]] = {}

    def add(self, a: Detections, b: Detections) -> list[tuple[int, int, float]]:
        self.images += 1
        self.n_a += len(a)
        self.n_b += len(b)
        if len(a) and len(b):
            ious = box_iou(a.boxes, b.boxes)
            ious[a.classes[:, None] != b.classes[None, :]] = 0.0
            self.best_ious.extend(ious.max(axis=1).tolist())
        else:
            self.best_ious.extend([0.0] * len(a))
        pairs = match(a, b, self.iou_threshold)
        for i, j, iou in pairs:
            self.ious.append(iou)
            self.score_deltas.append(float(b.scores[j] - a.scores[i]))
            ca = (a.boxes[i, :2] + a.boxes[i, 2:]) / 2
            cb = (b.boxes[j, :2] + b.boxes[j, 2:]) / 2
            self.shift.setdefault(int(a.strides[i]), []).append(
                (float(ca[0] - cb[0]), float(ca[1] - cb[1])))
        return pairs

    def summary(self) -> dict:
        matched = len(self.ious)
        deltas = np.asarray(self.score_deltas)
        return {
            "images": self.images,
            "iou_threshold": self.iou_threshold,
            "n_a": self.n_a,
            "n_b": self.n_b,
            "matched": matched,
            "unmatched_a": self.n_a - matched,
            "unmatched_b": self.n_b - matched,
            "match_rate_a": None if self.n_a == 0 else matched / self.n_a,
            "match_rate_b": None if self.n_b == 0 else matched / self.n_b,
            "matched_iou": percentiles(self.ious),
            "best_iou_any_a": percentiles(self.best_ious),
            "score_delta_b_minus_a": None if deltas.size == 0 else {
                "mean": float(deltas.mean()), "mean_abs": float(np.abs(deltas).mean()),
                "p95_abs": float(np.quantile(np.abs(deltas), 0.95)),
                "max_abs": float(np.abs(deltas).max()),
            },
            "centre_shift_a_minus_b_by_stride": {
                str(stride): {
                    "n": len(v),
                    "dx_mean": float(np.mean([d[0] for d in v])),
                    "dy_mean": float(np.mean([d[1] for d in v])),
                }
                for stride, v in sorted(self.shift.items())
            },
        }
