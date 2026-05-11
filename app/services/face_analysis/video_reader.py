from collections.abc import Iterator
from pathlib import Path

from app.services.face_analysis.models import VideoFrame


class VideoFrameReader:
    def __init__(self, frame_interval_seconds: float = 1.0):
        if frame_interval_seconds <= 0:
            raise ValueError("frame_interval_seconds must be greater than 0")

        self.frame_interval_seconds = frame_interval_seconds

    def read_frames(self, video_path: str | Path) -> Iterator[VideoFrame]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "opencv-python is required to read video frames. "
                "Install it with `poetry add opencv-python`."
            ) from exc

        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

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
