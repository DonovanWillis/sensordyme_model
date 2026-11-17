"""M13.1: the device-drawn face mesh is pixel-identical to mediapipe's, and cheaper.

M13 measured mediapipe's draw_landmarks of FACEMESH_TESSELATION at ~3.4 ms
per streamed frame (docs/research/M13_PERFORMANCE.md §4).
modules/driver_attention.draw_face_tessellation draws the same pixels in one
cv2.polylines call.
"""

from __future__ import annotations

import time

import mediapipe as mp
import numpy as np
import pytest
from mediapipe.framework.formats import landmark_pb2

from modules.driver_attention import draw_face_tessellation


def _face(rng, n=478, *, off_frame=0, low_visibility=0, nan=0):
    lms = landmark_pb2.NormalizedLandmarkList()
    centre = rng.uniform(0.3, 0.7, size=2)
    for i in range(n):
        x, y = centre + rng.normal(0, 0.12, size=2)
        lm = lms.landmark.add(x=float(x), y=float(y), z=0.0)
        if i < off_frame:
            lm.x = 1.2 if i % 2 else -0.1
        elif i < off_frame + low_visibility:
            lm.visibility = 0.2
        elif i < off_frame + low_visibility + nan:
            lm.y = float("nan")
    return lms


def _mediapipe(frame, lms):
    mp.solutions.drawing_utils.draw_landmarks(
        image=frame, landmark_list=lms, connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())


@pytest.mark.parametrize("shape", [(480, 640, 3), (720, 1280, 3), (240, 320, 3)])
@pytest.mark.parametrize("kind", ["plain", "edges_of_frame", "visibility", "nan"])
def test_the_tessellation_matches_mediapipe_pixel_for_pixel(shape, kind):
    rng = np.random.default_rng(hash((shape, kind)) % 2**32)
    for _ in range(5):
        lms = _face(rng, off_frame=40 if kind == "edges_of_frame" else 0,
                    low_visibility=30 if kind == "visibility" else 0, nan=10 if kind == "nan" else 0)
        base = rng.integers(0, 255, size=shape, dtype=np.uint8)
        a, b = base.copy(), base.copy()
        _mediapipe(a, lms)
        draw_face_tessellation(b, lms)
        assert np.array_equal(a, b)


def test_the_exact_corners_map_like_mediapipe():
    lms = landmark_pb2.NormalizedLandmarkList()
    for i in range(468):
        lms.landmark.add(x=float(i % 2), y=float((i // 2) % 2), z=0.0)  # 0 and 1 exactly
    a = np.zeros((48, 64, 3), np.uint8)
    b = a.copy()
    _mediapipe(a, lms)
    draw_face_tessellation(b, lms)
    assert np.array_equal(a, b) and a.any()


def test_nothing_to_draw_is_a_no_op():
    frame = np.zeros((48, 64, 3), np.uint8)
    draw_face_tessellation(frame, landmark_pb2.NormalizedLandmarkList())
    draw_face_tessellation(frame, None)
    assert not frame.any()


def test_it_is_clearly_cheaper_than_mediapipe():
    """Interleaved best-of runs, so a busy host slows both arms alike.

    Measured on a Mac M3: 2.31 -> 0.90 ms p50 when quiet; 2.37 -> 1.21 ms best-of
    inside the full suite. The bound (1.5x) is loose on purpose: this pins the
    regression, the docs carry the measurement.
    """
    rng = np.random.default_rng(3)
    lms = _face(rng)
    frame = np.zeros((480, 640, 3), np.uint8)
    old, new = [], []
    for _ in range(25):
        for fn, out in ((_mediapipe, old), (draw_face_tessellation, new)):
            t = time.perf_counter()
            fn(frame, lms)
            out.append(time.perf_counter() - t)
    assert min(new) * 1.5 < min(old), (min(old), min(new))
