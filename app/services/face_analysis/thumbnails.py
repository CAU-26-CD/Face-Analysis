from pathlib import Path

from app.services.face_analysis.exemplars import select_top_k_diverse
from app.services.face_analysis.models import FaceDetection, WithinVideoCluster


class ClusterThumbnailExtractor:
    """Saves a few representative face crops per cluster for human verification.

    Picks ``samples_per_cluster`` faces using
    :func:`app.services.face_analysis.exemplars.select_top_k_diverse`, the
    same quality-and-time-diversity selection that identity matching uses,
    so what we *see* lines up with what the matcher *judges by*.
    """

    def __init__(
        self,
        padding_ratio: float = 0.25,
        jpeg_quality: int = 90,
        samples_per_cluster: int = 3,
    ):
        if padding_ratio < 0:
            raise ValueError("padding_ratio must be >= 0")
        if samples_per_cluster < 1:
            raise ValueError("samples_per_cluster must be >= 1")
        self.padding_ratio = padding_ratio
        self.jpeg_quality = jpeg_quality
        self.samples_per_cluster = samples_per_cluster

    def extract(
        self,
        video_path: str | Path,
        clusters: list[WithinVideoCluster],
        output_dir: str | Path,
    ) -> dict[str, list[Path]]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "opencv-python is required to crop face thumbnails."
            ) from exc

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        picks_by_frame: dict[int, list[tuple[str, int, FaceDetection]]] = {}
        for cluster in clusters:
            faces = [
                detection
                for tracklet in cluster.tracklets
                for detection in tracklet.face_detections
            ]
            picks = select_top_k_diverse(faces, self.samples_per_cluster)
            for sample_index, detection in enumerate(picks, start=1):
                picks_by_frame.setdefault(detection.frame_index, []).append(
                    (cluster.cluster_id, sample_index, detection)
                )

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        saved: dict[str, list[Path]] = {}
        try:
            for frame_index in sorted(picks_by_frame.keys()):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = capture.read()
                if not success:
                    continue
                for cluster_id, sample_index, detection in picks_by_frame[frame_index]:
                    crop = self._crop(frame, detection.bbox)
                    if crop is None:
                        continue
                    target = output_path / f"{cluster_id}_{sample_index}.jpg"
                    cv2.imwrite(
                        str(target),
                        crop,
                        [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                    )
                    saved.setdefault(cluster_id, []).append(target)
        finally:
            capture.release()
        return saved

    def _crop(self, frame, bbox: tuple[float, float, float, float]):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        box_width = x2 - x1
        box_height = y2 - y1
        pad_x = box_width * self.padding_ratio
        pad_y = box_height * self.padding_ratio
        x1p = max(0, int(round(x1 - pad_x)))
        y1p = max(0, int(round(y1 - pad_y)))
        x2p = min(width, int(round(x2 + pad_x)))
        y2p = min(height, int(round(y2 + pad_y)))
        if x2p <= x1p or y2p <= y1p:
            return None
        return frame[y1p:y2p, x1p:x2p]
