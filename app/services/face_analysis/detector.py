from app.services.face_analysis.models import FaceDetection, VideoFrame


class InsightFaceDetector:
    def __init__(
        self,
        model_name: str = "buffalo_l",
        providers: list[str] | None = None,
        det_size: tuple[int, int] = (640, 640),
        allowed_modules: list[str] | None = None,
        min_confidence: float = 0.5,
        min_bbox_side_pixels: float = 40.0,
    ):
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if min_bbox_side_pixels < 0:
            raise ValueError("min_bbox_side_pixels must be >= 0")

        self.model_name = model_name
        self.providers = providers or ["CPUExecutionProvider"]
        self.det_size = det_size
        self.allowed_modules = allowed_modules or ["detection", "recognition"]
        self.min_confidence = min_confidence
        self.min_bbox_side_pixels = min_bbox_side_pixels
        self._app = None

    def detect(self, video_frame: VideoFrame) -> list[FaceDetection]:
        app = self._get_app()
        faces = app.get(video_frame.frame)

        detections: list[FaceDetection] = []
        for face in faces:
            embedding = getattr(face, "embedding", None)
            if embedding is None:
                continue

            confidence = float(face.det_score)
            if confidence < self.min_confidence:
                continue

            x1, y1, x2, y2 = (float(value) for value in face.bbox)
            if min(x2 - x1, y2 - y1) < self.min_bbox_side_pixels:
                continue

            detections.append(
                FaceDetection(
                    timestamp_seconds=float(video_frame.timestamp_seconds),
                    frame_index=int(video_frame.frame_index),
                    bbox=(x1, y1, x2, y2),
                    embedding=[float(value) for value in embedding],
                    confidence=confidence,
                )
            )

        return detections

    def _get_app(self):
        if self._app is not None:
            return self._app

        import os
        import tempfile

        os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
        os.environ.setdefault(
            "MPLCONFIGDIR",
            str(tempfile.gettempdir()),
        )

        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError(
                "insightface is required to detect faces. "
                "Install it with `poetry add insightface onnxruntime`."
            ) from exc

        app = FaceAnalysis(
            name=self.model_name,
            providers=self.providers,
            allowed_modules=self.allowed_modules,
        )
        app.prepare(ctx_id=0, det_size=self.det_size)
        self._app = app
        return app
