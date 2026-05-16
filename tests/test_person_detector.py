from types import SimpleNamespace

import numpy as np
import pytest

from app.services.face_analysis.models import VideoFrame
from app.services.face_analysis.person_detector import PersonDetector


class _FakeBoxes:
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray):
        self.xyxy = _FakeTensor(xyxy)
        self.conf = _FakeTensor(conf)

    def __len__(self) -> int:
        return self.xyxy.array.shape[0]


class _FakeTensor:
    def __init__(self, array: np.ndarray):
        self.array = array

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.array


class _FakeYolo:
    def __init__(self, boxes: _FakeBoxes | None):
        self._boxes = boxes
        self.predict_kwargs: dict | None = None

    def predict(self, frame, **kwargs):
        self.predict_kwargs = kwargs
        return [SimpleNamespace(boxes=self._boxes)]


def _frame() -> VideoFrame:
    return VideoFrame(timestamp_seconds=1.5, frame_index=15, frame=np.zeros((4, 4, 3)))


def test_detect_returns_person_detections_from_yolo_output():
    boxes = _FakeBoxes(
        xyxy=np.array([[10, 20, 30, 60], [50, 50, 200, 300]], dtype=np.float32),
        conf=np.array([0.91, 0.55], dtype=np.float32),
    )
    detector = PersonDetector(model_name="dummy.pt")
    detector._model = _FakeYolo(boxes)

    detections = detector.detect(_frame())

    assert [d.bbox for d in detections] == [(10.0, 20.0, 30.0, 60.0), (50.0, 50.0, 200.0, 300.0)]
    assert [round(d.confidence, 4) for d in detections] == [0.91, 0.55]
    assert all(d.timestamp_seconds == 1.5 for d in detections)
    assert all(d.frame_index == 15 for d in detections)


def test_detect_filters_out_bboxes_below_minimum_side_length():
    boxes = _FakeBoxes(
        xyxy=np.array([[0, 0, 5, 5], [0, 0, 100, 100]], dtype=np.float32),
        conf=np.array([0.9, 0.9], dtype=np.float32),
    )
    detector = PersonDetector(model_name="dummy.pt", min_bbox_side_pixels=20)
    detector._model = _FakeYolo(boxes)

    detections = detector.detect(_frame())

    assert len(detections) == 1
    assert detections[0].bbox == (0.0, 0.0, 100.0, 100.0)


def test_detect_returns_empty_when_no_boxes():
    detector = PersonDetector(model_name="dummy.pt")
    detector._model = _FakeYolo(boxes=None)

    assert detector.detect(_frame()) == []


def test_detect_forwards_class_and_confidence_filter_to_yolo():
    boxes = _FakeBoxes(
        xyxy=np.zeros((0, 4), dtype=np.float32),
        conf=np.zeros((0,), dtype=np.float32),
    )
    fake = _FakeYolo(boxes)
    detector = PersonDetector(model_name="dummy.pt", confidence_threshold=0.7, imgsz=320)
    detector._model = fake

    detector.detect(_frame())

    assert fake.predict_kwargs is not None
    assert fake.predict_kwargs["classes"] == [0]
    assert fake.predict_kwargs["conf"] == 0.7
    assert fake.predict_kwargs["imgsz"] == 320
    assert fake.predict_kwargs["verbose"] is False


def test_constructor_rejects_invalid_thresholds():
    with pytest.raises(ValueError):
        PersonDetector(confidence_threshold=1.1)
    with pytest.raises(ValueError):
        PersonDetector(min_bbox_side_pixels=-1)
