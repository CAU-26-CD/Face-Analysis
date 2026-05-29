import logging
import os
import subprocess
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
            yield from self._read_with_opencv(path)
            return

        # Try decord on the original, then on each ffmpeg-fixed variant in
        # cost order. We keep advancing past the previous failure instead of
        # bailing out, because MediaRecorder webm tends to fail on the
        # original *and* the cheap MKV remux — only the re-encode produces a
        # file with clean keyframes / EOF markers.
        for attempt, source in self._decord_attempts(path):
            if source is None:
                continue
            try:
                yield from self._read_with_decord(source, use_gpu=use_gpu)
                return
            except Exception as exc:
                logger.warning(
                    "decord failed on %s (%s); trying next strategy",
                    attempt,
                    exc,
                )

        # Last resort: software decode every frame on CPU.
        logger.warning("all decord strategies failed; falling back to OpenCV")
        yield from self._read_with_opencv(path)

    def _decord_attempts(self, path: Path) -> Iterator[tuple[str, Path | None]]:
        """Yield (label, path) pairs for each decord attempt, in cost order."""
        yield "original", path
        yield "MKV stream-copy", self._remux_mkv(path)
        yield "H.264 re-encode", self._reencode_h264(path)

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

    def _remux_mkv(self, path: Path) -> Path | None:
        """Stream-copy video into MKV with no audio. ~1-2s for 5-min 1080p.

        MKV accepts VP8/VP9 directly (MP4 rejects VP8), and dropping audio
        sidesteps the Opus-in-MP4 compatibility headache. Works on most
        MediaRecorder webm — but the resulting MKV sometimes still trips
        decord at EOF (broken trailer), in which case the caller advances
        to the re-encode path.
        """
        mkv_out = path.with_suffix(".video.mkv")
        if self._run_ffmpeg(
            [
                "ffmpeg", "-y",
                "-err_detect", "ignore_err",
                "-fflags", "+genpts",
                "-i", str(path),
                "-map", "0:v:0",
                "-c:v", "copy",
                "-an",
                str(mkv_out),
            ],
            timeout=60,
            description="MKV stream-copy remux",
        ):
            return mkv_out
        return None

    def _reencode_h264(self, path: Path) -> Path | None:
        """Re-encode video to H.264 MP4 with libx264 ultrafast, no audio.

        Slower than stream-copy (~5-10x realtime), but produces a file with
        clean keyframes and a valid moov atom, so decord can always seek
        and probe it. CRF 30 ultrafast loses visual fidelity, but face
        landmarks survive well — better than the OpenCV CPU fallback.
        """
        mp4_out = path.with_suffix(".reenc.mp4")
        if self._run_ffmpeg(
            [
                "ffmpeg", "-y",
                "-i", str(path),
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "30",
                "-an",
                "-movflags", "+faststart",
                str(mp4_out),
            ],
            timeout=300,
            description="H.264 re-encode",
        ):
            return mp4_out
        return None

    @staticmethod
    def _run_ffmpeg(cmd: list[str], timeout: int, description: str) -> bool:
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
            return True
        except FileNotFoundError:
            logger.warning("ffmpeg not available; %s skipped", description)
            return False
        except subprocess.TimeoutExpired:
            logger.warning("%s timed out", description)
            return False
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
            tail = "\n".join(stderr.strip().splitlines()[-3:]) if stderr else str(exc)
            logger.warning("%s failed:\n%s", description, tail)
            return False

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
