from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from app.services.face_analysis.actor_matcher import ActorMatcher
from app.services.face_analysis.detector import InsightFaceDetector
from app.services.face_analysis.face_person_associator import FacePersonAssociator
from app.services.face_analysis.models import (
    FaceAnalysisResult,
    FaceAppearance,
    FaceDetection,
    KnownActor,
    MatchResult,
    PersonDetection,
    PersonTracklet,
    TrackedFaceDetection,
    TrackedPerson,
    VideoFrame,
    WithinVideoCluster,
)
from app.services.face_analysis.person_detector import PersonDetector
from app.services.face_analysis.person_tracker import PersonTracker
from app.services.face_analysis.thumbnails import ClusterThumbnailExtractor
from app.services.face_analysis.timeline import AppearanceTimelineBuilder
from app.services.face_analysis.tracklet_clusterer import TrackletClusterer
from app.services.face_analysis.video_reader import VideoFrameReader


@dataclass
class _TrackBuffer:
    track_id: str
    person_detections: list[PersonDetection] = field(default_factory=list)
    face_detections: list[FaceDetection] = field(default_factory=list)


class FaceVideoAnalyzer:
    """Per-video orchestrator.

    Pipeline (per sampled frame):
        person_detector  → person bboxes
        person_tracker   → stable person_track_id (IoU + Kalman, ByteTrack)
        face_detector    → face bbox + 512-d embedding
        face_associator  → bind each face to a person_track_id

    Per-track face embeddings are aggregated into a PersonTracklet, then:
        tracklet_clusterer → merge tracks split by long occlusions
        actor_matcher      → resolve clusters against the known actor gallery
        timeline_builder   → flatten observations into appearance intervals
    """

    def __init__(
        self,
        frame_reader: VideoFrameReader | None = None,
        person_detector: PersonDetector | None = None,
        person_tracker: PersonTracker | None = None,
        face_detector: InsightFaceDetector | None = None,
        face_associator: FacePersonAssociator | None = None,
        tracklet_clusterer: TrackletClusterer | None = None,
        actor_matcher: ActorMatcher | None = None,
        timeline_builder: AppearanceTimelineBuilder | None = None,
        thumbnail_extractor: ClusterThumbnailExtractor | None = None,
    ):
        self.frame_reader = frame_reader or VideoFrameReader()
        self.person_detector = person_detector or PersonDetector()
        self.person_tracker = person_tracker or PersonTracker()
        self.face_detector = face_detector or InsightFaceDetector()
        self.face_associator = face_associator or FacePersonAssociator()
        self.tracklet_clusterer = tracklet_clusterer or TrackletClusterer()
        self.actor_matcher = actor_matcher or ActorMatcher()
        self.timeline_builder = timeline_builder or AppearanceTimelineBuilder()
        self.thumbnail_extractor = thumbnail_extractor or ClusterThumbnailExtractor()

    def read_sampled_frames(self, video_path: str | Path) -> Iterator[VideoFrame]:
        return self.frame_reader.read_frames(video_path)

    def detect_persons(self, video_frame: VideoFrame) -> list[PersonDetection]:
        return self.person_detector.detect(video_frame)

    def detect_faces(self, video_frame: VideoFrame) -> list[FaceDetection]:
        return self.face_detector.detect(video_frame)

    def analyze(
        self,
        video_path: str | Path,
        known_actors: list[KnownActor] | None = None,
        progress_callback: Callable[[int, float, int, int], None] | None = None,
        thumbnail_dir: str | Path | None = None,
    ) -> FaceAnalysisResult:
        path = Path(video_path)
        known_actors = known_actors or []

        self.person_tracker.reset()
        track_buffers: dict[str, _TrackBuffer] = {}

        for sampled_frame_count, video_frame in enumerate(
            self.frame_reader.read_frames(path),
            start=1,
        ):
            person_detections = self.person_detector.detect(video_frame)
            tracked_persons = self.person_tracker.update(video_frame, person_detections)
            face_detections = self.face_detector.detect(video_frame)
            tracked_faces = self.face_associator.associate(face_detections, tracked_persons)

            self._record_persons(tracked_persons, track_buffers)
            self._record_faces(tracked_faces, track_buffers)

            if progress_callback is not None:
                progress_callback(
                    sampled_frame_count,
                    video_frame.timestamp_seconds,
                    len(tracked_persons),
                    len(face_detections),
                )

        tracklets = self._finalize_tracklets(track_buffers)
        clusters = self.tracklet_clusterer.cluster(tracklets)
        if thumbnail_dir is not None:
            self.thumbnail_extractor.extract(path, clusters, thumbnail_dir)
        match_result = self.actor_matcher.match(clusters, known_actors)
        appearances = self._build_appearances(clusters, match_result)

        return FaceAnalysisResult(
            video_path=str(path),
            appearances=appearances,
            new_candidates=match_result.new_candidates,
        )

    @staticmethod
    def _record_persons(
        tracked_persons: list[TrackedPerson],
        track_buffers: dict[str, _TrackBuffer],
    ) -> None:
        for tracked in tracked_persons:
            buffer = track_buffers.setdefault(
                tracked.track_id,
                _TrackBuffer(track_id=tracked.track_id),
            )
            buffer.person_detections.append(tracked.detection)

    @staticmethod
    def _record_faces(
        tracked_faces: list[TrackedFaceDetection],
        track_buffers: dict[str, _TrackBuffer],
    ) -> None:
        for tracked in tracked_faces:
            if tracked.person_track_id is None:
                continue
            buffer = track_buffers.get(tracked.person_track_id)
            if buffer is None:
                continue
            buffer.face_detections.append(tracked.detection)

    @staticmethod
    def _finalize_tracklets(
        track_buffers: dict[str, _TrackBuffer],
    ) -> list[PersonTracklet]:
        tracklets: list[PersonTracklet] = []
        for buffer in track_buffers.values():
            if not buffer.person_detections:
                continue
            aggregated = _mean_embedding(
                [face.embedding for face in buffer.face_detections]
            )
            tracklets.append(
                PersonTracklet(
                    track_id=buffer.track_id,
                    person_detections=list(buffer.person_detections),
                    face_detections=list(buffer.face_detections),
                    aggregated_embedding=aggregated,
                    start_seconds=buffer.person_detections[0].timestamp_seconds,
                    end_seconds=buffer.person_detections[-1].timestamp_seconds,
                )
            )
        return tracklets

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
                for person_detection in tracklet.person_detections:
                    labeled_timestamps.append((person_id, person_detection.timestamp_seconds))

        labeled_timestamps.sort(key=lambda pair: pair[1])

        timeline_builder = AppearanceTimelineBuilder(
            max_gap_seconds=self.timeline_builder.max_gap_seconds,
        )
        for person_id, timestamp in labeled_timestamps:
            timeline_builder.add(person_id, timestamp)
        return timeline_builder.build()


def _mean_embedding(embeddings: list[list[float]]) -> list[float]:
    if not embeddings:
        return []
    dim = len(embeddings[0])
    if any(len(e) != dim for e in embeddings):
        raise ValueError("all embeddings must have the same dimension")
    summed = [0.0] * dim
    for embedding in embeddings:
        for i in range(dim):
            summed[i] += embedding[i]
    return [value / len(embeddings) for value in summed]
