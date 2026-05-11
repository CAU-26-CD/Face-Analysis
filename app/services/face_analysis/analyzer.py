from collections.abc import Callable, Iterator
from pathlib import Path

from app.services.face_analysis.clustering import FaceClusterer
from app.services.face_analysis.detector import InsightFaceDetector
from app.services.face_analysis.models import (
    FaceAnalysisResult,
    FaceAppearance,
    ClusteredFaceDetection,
    FaceDetection,
    VideoFrame,
)
from app.services.face_analysis.timeline import AppearanceTimelineBuilder
from app.services.face_analysis.video_reader import VideoFrameReader


class FaceVideoAnalyzer:
    def __init__(
        self,
        frame_reader: VideoFrameReader | None = None,
        detector: InsightFaceDetector | None = None,
        clusterer: FaceClusterer | None = None,
        timeline_builder: AppearanceTimelineBuilder | None = None,
    ):
        self.frame_reader = frame_reader or VideoFrameReader()
        self.detector = detector or InsightFaceDetector()
        self.clusterer = clusterer or FaceClusterer()
        self.timeline_builder = timeline_builder or AppearanceTimelineBuilder()

    def read_sampled_frames(self, video_path: str | Path) -> Iterator[VideoFrame]:
        return self.frame_reader.read_frames(video_path)

    def detect_faces(self, video_path: str | Path) -> Iterator[FaceDetection]:
        for video_frame in self.frame_reader.read_frames(video_path):
            yield from self.detector.detect(video_frame)

    def cluster_faces(self, video_path: str | Path) -> Iterator[ClusteredFaceDetection]:
        for detection in self.detect_faces(video_path):
            yield self.clusterer.assign(detection)

    def build_timeline(
        self,
        video_path: str | Path,
        progress_callback: Callable[[int, float, int], None] | None = None,
    ) -> list[FaceAppearance]:
        timeline_builder = AppearanceTimelineBuilder(
            max_gap_seconds=self.timeline_builder.max_gap_seconds
        )
        clusterer = FaceClusterer(
            similarity_threshold=self.clusterer.similarity_threshold
        )

        for sampled_frame_count, video_frame in enumerate(
            self.read_sampled_frames(video_path),
            start=1,
        ):
            detections = self.detector.detect(video_frame)
            for detection in detections:
                clustered = clusterer.assign(detection)
                timeline_builder.add(clustered)

            if progress_callback is not None:
                progress_callback(
                    sampled_frame_count,
                    video_frame.timestamp_seconds,
                    len(detections),
                )

        return timeline_builder.build()

    def analyze(
        self,
        video_path: str | Path,
        progress_callback: Callable[[int, float, int], None] | None = None,
    ) -> FaceAnalysisResult:
        path = Path(video_path)
        appearances = self.build_timeline(
            path,
            progress_callback=progress_callback,
        )
        return FaceAnalysisResult(
            video_path=str(path),
            appearances=appearances,
        )
