import logging
import os
from collections.abc import Iterator
from pathlib import Path

from app.services.face_analysis.models import VideoFrame

logger = logging.getLogger(__name__)


class VideoFrameReader:
    """Sample frames from a video at ``frame_interval_seconds``.

    Prefers decord with GPU context when ``FACE_ANALYZER_DEVICE=cuda`` because
    it decodes on NVDEC and lets us request only the indices we want — OpenCV
    decodes *every* frame even when we keep 1 in 10, which is the main CPU
    bottleneck on long videos. Falls back to OpenCV when decord isn't
    installed or the GPU path fails for this file.
    """

    def __init__(self, frame_interval_seconds: float = 0.3):
        if frame_interval_seconds <= 0:
            raise ValueError("frame_interval_seconds must be greater than 0")

        self.frame_interval_seconds = frame_interval_seconds

    def read_frames(self, video_path: str | Path) -> Iterator[VideoFrame]:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        use_gpu = os.getenv("FACE_ANALYZER_DEVICE", "").lower() == "cuda"
        try:
            import decord  # noqa: F401
        except ImportError:
            return self._read_with_opencv(path)

        try:
            yield from self._read_with_decord(path, use_gpu=use_gpu)
        except Exception as exc:
            logger.warning(
                "decord read failed (%s); falling back to OpenCV", exc
            )
            yield from self._read_with_opencv(path)

    def _read_with_decord(self, path: Path, use_gpu: bool) -> Iterator[VideoFrame]:
        from decord import VideoReader, cpu, gpu

        ctx = gpu(0) if use_gpu else cpu(0)
        vr = VideoReader(str(path), ctx=ctx)
        fps = vr.get_avg_fps()
        if fps <= 0:
            raise ValueError(f"Could not read FPS from video file: {path}")

        total_frames = len(vr)
        frame_step = max(1, int(round(fps * self.frame_interval_seconds)))
        target_indices = list(range(0, total_frames, frame_step))
        if not target_indices:
            return

        # Batch-decode only the frames we sample. NVDEC throughput is much
        # higher than OpenCV's per-frame software decode, and skipping the
        # unused 9-of-10 frames saves the bulk of the work.
        batch = vr.get_batch(target_indices).asnumpy()

        for batch_pos, frame_index in enumerate(target_indices):
            # decord returns RGB; downstream (InsightFace, OpenCV writes) expects BGR.
            frame_bgr = batch[batch_pos, :, :, ::-1]
            yield VideoFrame(
                timestamp_seconds=float(frame_index / fps),
                frame_index=int(frame_index),
                frame=frame_bgr,
            )

    def _read_with_opencv(self, path: Path) -> Iterator[VideoFrame]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "opencv-python is required to read video frames. "
                "Install it with `poetry add opencv-python`."
            ) from exc

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video file: {path}")

        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                raise ValueError(f"Could not read FPS from video file: {path}")

            frame_step = max(1, int(round(fps * self.frame_interval_seconds)))

            frame_index = 0
            while True:
                success, frame = capture.read()
                if not success:
                    break

                if frame_index % frame_step == 0:
                    yield VideoFrame(
                        timestamp_seconds=float(frame_index / fps),
                        frame_index=int(frame_index),
                        frame=frame,
                    )

                frame_index += 1
        finally:
            capture.release()
