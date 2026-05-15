from types import SimpleNamespace

import numpy as np
import pytest

from app.services.face_analysis.detector import InsightFaceDetector
from app.services.face_analysis.models import VideoFrame


class _FakeFace:
    _SENTINEL = object()

    def __init__(
        self,
        bbox: tuple[float, float, float, float],
        det_score: float,
        embedding=_SENTINEL,
    ):
        self.bbox = bbox
        self.det_score = det_score
        if embedding is _FakeFace._SENTINEL:
            self.embedding = [0.1] * 512
        else:
            self.embedding = embedding


class _FakeFaceApp:
    def __init__(self, faces: list[_FakeFace]):
        self.faces = faces

    def get(self, _frame):
        return self.faces


def _frame() -> VideoFrame:
    return VideoFrame(timestamp_seconds=0.5, frame_index=5, frame=np.zeros((4, 4, 3)))


def _make_detector(
    faces: list[_FakeFace],
    *,
    min_confidence: float = 0.5,
    min_bbox_side_pixels: float = 40.0,
) -> InsightFaceDetector:
    detector = InsightFaceDetector(
        min_confidence=min_confidence,
        min_bbox_side_pixels=min_bbox_side_pixels,
    )
    detector._app = _FakeFaceApp(faces)
    return detector


def test_detect_drops_faces_below_confidence_threshold():
    faces = [
        _FakeFace((10, 10, 90, 90), det_score=0.95),
        _FakeFace((10, 10, 90, 90), det_score=0.3),
    ]
    detector = _make_detector(faces, min_confidence=0.5)

    results = detector.detect(_frame())

    assert len(results) == 1
    assert results[0].confidence == pytest.approx(0.95)


def test_detect_drops_faces_below_min_bbox_side():
    faces = [
        _FakeFace((10, 10, 20, 20), det_score=0.9),
        _FakeFace((10, 10, 100, 110), det_score=0.9),
    ]
    detector = _make_detector(faces, min_bbox_side_pixels=40.0)

    results = detector.detect(_frame())

    assert len(results) == 1
    assert results[0].bbox == (10.0, 10.0, 100.0, 110.0)


def test_detect_skips_faces_without_embedding():
    faces = [_FakeFace((10, 10, 90, 90), det_score=0.9, embedding=None)]
    detector = _make_detector(faces)

    results = detector.detect(_frame())
    assert results == []


def test_detect_returns_faces_passing_all_filters():
    faces = [
        _FakeFace((0, 0, 80, 80), det_score=0.7),
        _FakeFace((100, 100, 200, 200), det_score=0.85),
    ]
    detector = _make_detector(faces, min_confidence=0.5, min_bbox_side_pixels=40.0)

    results = detector.detect(_frame())

    assert len(results) == 2
    assert all(d.timestamp_seconds == 0.5 for d in results)
    assert all(d.frame_index == 5 for d in results)


def test_detect_rejects_invalid_filter_thresholds():
    with pytest.raises(ValueError):
        InsightFaceDetector(min_confidence=1.5)
    with pytest.raises(ValueError):
        InsightFaceDetector(min_bbox_side_pixels=-1)
