from app.services.face_analysis.models import FaceDetection, VideoFrame


class InsightFaceDetector:
    def __init__(
        self,
        model_name: str = "buffalo_l",
        providers: list[str] | None = None,
        det_size: tuple[int, int] = (640, 640),
        allowed_modules: list[str] | None = None,
    ):
        self.model_name = model_name
        self.providers = providers or ["CPUExecutionProvider"]
        self.det_size = det_size
        self.allowed_modules = allowed_modules or ["detection", "recognition"]
        self._app = None

    def detect(self, video_frame: VideoFrame) -> list[FaceDetection]:
        app = self._get_app()
        faces = app.get(video_frame.frame)

        detections: list[FaceDetection] = []
        for face in faces:
            embedding = getattr(face, "embedding", None)
            if embedding is None:
                continue

            detections.append(
                FaceDetection(
                    timestamp_seconds=float(video_frame.timestamp_seconds),
                    frame_index=int(video_frame.frame_index),
                    bbox=tuple(float(value) for value in face.bbox),
                    embedding=[float(value) for value in embedding],
                    confidence=float(face.det_score),
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
