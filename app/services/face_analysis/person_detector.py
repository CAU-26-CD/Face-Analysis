from typing import Any

from app.services.face_analysis.models import PersonDetection, VideoFrame

PERSON_CLASS_INDEX = 0  # COCO 'person'


class PersonDetector:
    """Detects people in a video frame using an Ultralytics YOLO model.

    Only the COCO ``person`` class (index 0) is kept. The model and its weights
    are loaded lazily on first call to avoid paying the import / download cost
    when the detector is constructed but never used (e.g. during tests).
    """

    def __init__(
        self,
        model_name: str = "yolo11s.pt",
        confidence_threshold: float = 0.4,
        min_bbox_side_pixels: float = 0.0,
        device: str | None = None,
        imgsz: int = 640,
    ):
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if min_bbox_side_pixels < 0:
            raise ValueError("min_bbox_side_pixels must be >= 0")

        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.min_bbox_side_pixels = min_bbox_side_pixels
        self.device = device
        self.imgsz = imgsz
        self._model: Any = None

    def detect(self, video_frame: VideoFrame) -> list[PersonDetection]:
        model = self._get_model()
        results = model.predict(
            video_frame.frame,
            classes=[PERSON_CLASS_INDEX],
            conf=self.confidence_threshold,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
        )
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()

        detections: list[PersonDetection] = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            if min(x2 - x1, y2 - y1) < self.min_bbox_side_pixels:
                continue
            detections.append(
                PersonDetection(
                    timestamp_seconds=float(video_frame.timestamp_seconds),
                    frame_index=int(video_frame.frame_index),
                    bbox=(x1, y1, x2, y2),
                    confidence=float(conf[i]),
                )
            )
        return detections

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required for person detection. "
                "Install it with `poetry add ultralytics`."
            ) from exc
        self._model = YOLO(self.model_name)
        return self._model
