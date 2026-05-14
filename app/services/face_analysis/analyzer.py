from collections.abc import Callable, Iterator
from pathlib import Path

from app.services.face_analysis.actor_matcher import ActorMatcher
from app.services.face_analysis.detector import InsightFaceDetector
from app.services.face_analysis.models import (
    FaceAnalysisResult,
    FaceAppearance,
    FaceDetection,
    KnownActor,
    MatchResult,
    TrackedDetection,
    VideoFrame,
    WithinVideoCluster,
)
from app.services.face_analysis.timeline import AppearanceTimelineBuilder
from app.services.face_analysis.tracker import FaceTracker
from app.services.face_analysis.tracklet_clusterer import TrackletClusterer
from app.services.face_analysis.video_reader import VideoFrameReader


class FaceVideoAnalyzer:
    def __init__(
        self,
        frame_reader: VideoFrameReader | None = None,
        detector: InsightFaceDetector | None = None,
        tracker: FaceTracker | None = None,
        tracklet_clusterer: TrackletClusterer | None = None,
        actor_matcher: ActorMatcher | None = None,
        timeline_builder: AppearanceTimelineBuilder | None = None,
    ):
        self.frame_reader = frame_reader or VideoFrameReader()
        self.detector = detector or InsightFaceDetector()
        self.tracker = tracker or FaceTracker()
        self.tracklet_clusterer = tracklet_clusterer or TrackletClusterer()
        self.actor_matcher = actor_matcher or ActorMatcher()
        self.timeline_builder = timeline_builder or AppearanceTimelineBuilder()

    def read_sampled_frames(self, video_path: str | Path) -> Iterator[VideoFrame]:
        return self.frame_reader.read_frames(video_path)

    def detect_faces(self, video_path: str | Path) -> Iterator[FaceDetection]:
        for video_frame in self.frame_reader.read_frames(video_path):
            yield from self.detector.detect(video_frame)

    def track_faces(self, video_path: str | Path) -> Iterator[TrackedDetection]:
        tracker = self._fresh_tracker()
        for video_frame in self.frame_reader.read_frames(video_path):
            detections = self.detector.detect(video_frame)
            yield from tracker.update(video_frame, detections)

    def analyze(
        self,
        video_path: str | Path,
        known_actors: list[KnownActor] | None = None,
        progress_callback: Callable[[int, float, int], None] | None = None,
    ) -> FaceAnalysisResult:
        path = Path(video_path)
        known_actors = known_actors or []

        tracker = self._fresh_tracker()

        for sampled_frame_count, video_frame in enumerate(
            self.frame_reader.read_frames(path),
            start=1,
        ):
            detections = self.detector.detect(video_frame)
            tracker.update(video_frame, detections)

            if progress_callback is not None:
                progress_callback(
                    sampled_frame_count,
                    video_frame.timestamp_seconds,
                    len(detections),
                )

        tracklets = tracker.finalize()
        clusters = self.tracklet_clusterer.cluster(tracklets)
        match_result = self.actor_matcher.match(clusters, known_actors)
        appearances = self._build_appearances(clusters, match_result)

        return FaceAnalysisResult(
            video_path=str(path),
            appearances=appearances,
            new_candidates=match_result.new_candidates,
        )

    def _fresh_tracker(self) -> FaceTracker:
        return FaceTracker(
            similarity_threshold=self.tracker.similarity_threshold,
            max_gap_seconds=self.tracker.max_gap_seconds,
        )

    def _build_appearances(
        self,
        clusters: list[WithinVideoCluster],
        match_result: MatchResult,
    ) -> list[FaceAppearance]:
        person_id_by_cluster: dict[str, str] = {
            matched.cluster_id: matched.actor_id for matched in match_result.matched
        }
        for candidate in match_result.new_candidates:
            person_id_by_cluster[candidate.cluster_id] = candidate.cluster_id

        labeled_timestamps: list[tuple[str, float]] = []
        for cluster in clusters:
            person_id = person_id_by_cluster[cluster.cluster_id]
            for tracklet in cluster.tracklets:
                for detection in tracklet.detections:
                    labeled_timestamps.append((person_id, detection.timestamp_seconds))

        labeled_timestamps.sort(key=lambda pair: pair[1])

        timeline_builder = AppearanceTimelineBuilder(
            max_gap_seconds=self.timeline_builder.max_gap_seconds,
        )
        for person_id, timestamp in labeled_timestamps:
            timeline_builder.add(person_id, timestamp)
        return timeline_builder.build()
