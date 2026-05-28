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
        model_name: str = "yolo11n.pt",
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
        return self.detect_batch([video_frame])[0]

    def detect_batch(
        self, video_frames: list[VideoFrame]
    ) -> list[list[PersonDetection]]:
        """Run YOLO on a batch of frames at once.

        Single-frame inference leaves the GPU mostly idle between kernel
        launches; batching amortizes that overhead so utilization stays high
        on long videos. Per-frame order is preserved so the tracker downstream
        still sees frames in the order they were read.
        """
        if not video_frames:
            return []

        model = self._get_model()
        results = model.predict(
            [frame.frame for frame in video_frames],
            classes=[PERSON_CLASS_INDEX],
            conf=self.confidence_threshold,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
        )

        per_frame: list[list[PersonDetection]] = []
        for video_frame, result in zip(video_frames, results):
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                per_frame.append([])
                continue

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
            per_frame.append(detections)
        return per_frame

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
