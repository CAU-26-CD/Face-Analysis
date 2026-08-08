import logging

from app.services.face_analysis.models import FaceDetection, VideoFrame

logger = logging.getLogger(__name__)


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
        self.providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
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
            providers=self._providers_with_fast_cold_start(self.providers),
            allowed_modules=self.allowed_modules,
        )
        app.prepare(ctx_id=0, det_size=self.det_size)
        self._check_providers_applied(app)
        self._app = app
        return app

    def _check_providers_applied(self, app) -> None:
        """Log loudly when ONNX Runtime silently dropped an accelerator EP.

        ORT does not raise when a provider's shared library fails to dlopen —
        it prints one stderr line and quietly builds the session on the next
        provider in the list, which is always CPU. That failure mode cost us a
        7x slower face pass in production without a single error-level log,
        because the only symptom was a timing number nobody was diffing. This
        turns it into something greppable.
        """
        applied: set[str] = set()
        for model in getattr(app, "models", {}).values():
            session = getattr(model, "session", None)
            if session is not None:
                applied.update(session.get_providers())

        missing = [
            p for p in self.providers
            if p != "CPUExecutionProvider" and p not in applied
        ]
        if missing:
            logger.error(
                "InsightFace fell back to %s — ONNX Runtime could not load %s. "
                "Face detection + embedding will run on CPU. Look for the "
                "'provider_bridge_ort.cc ... Failed to load library' line above "
                "for the missing CUDA library and its expected version.",
                sorted(applied),
                missing,
            )
        else:
            logger.info("InsightFace providers active: %s", sorted(applied))

    @staticmethod
    def _providers_with_fast_cold_start(
        providers: list[str],
    ) -> list[str | tuple[str, dict]]:
        """Override CUDAExecutionProvider's default EXHAUSTIVE algo search.

        EXHAUSTIVE benchmarks every cuDNN conv kernel against the live model
        on first use — great if the worker stays warm for thousands of
        inferences, brutal on RunPod serverless where each cold worker eats
        2-3 minutes of autotune before the first frame. HEURISTIC picks a
        good kernel from input shapes in <1s, costing ~5% per-inference vs
        EXHAUSTIVE. Net win for our access pattern (one job per cold start).
        """
        cuda_opts = {
            "cudnn_conv_algo_search": "HEURISTIC",
            "do_copy_in_default_stream": "1",
        }
        return [
            (p, cuda_opts) if p == "CUDAExecutionProvider" else p
            for p in providers
        ]
