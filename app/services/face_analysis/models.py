from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VideoFrame:
    timestamp_seconds: float
    frame_index: int
    frame: Any


@dataclass(frozen=True)
class FaceDetection:
    timestamp_seconds: float
    frame_index: int
    bbox: tuple[float, float, float, float]
    embedding: list[float]
    confidence: float


@dataclass(frozen=True)
class TrackedDetection:
    track_id: str
    detection: FaceDetection


@dataclass(frozen=True)
class Tracklet:
    track_id: str
    detections: list[FaceDetection]
    aggregated_embedding: list[float]
    start_seconds: float
    end_seconds: float

    @property
    def detection_count(self) -> int:
        return len(self.detections)


@dataclass(frozen=True)
class WithinVideoCluster:
    cluster_id: str
    tracklets: list[Tracklet]
    aggregated_embedding: list[float]
    start_seconds: float
    end_seconds: float

    @property
    def detection_count(self) -> int:
        return sum(tracklet.detection_count for tracklet in self.tracklets)


@dataclass(frozen=True)
class KnownActor:
    actor_id: str
    face_template: list[float]


@dataclass(frozen=True)
class MatchedActor:
    cluster_id: str
    actor_id: str
    similarity: float


@dataclass(frozen=True)
class NewActorCandidate:
    cluster_id: str
    embedding: list[float]
    detection_count: int
    start_seconds: float
    end_seconds: float
    suggested_actor_id: str | None = None
    suggested_similarity: float | None = None


@dataclass(frozen=True)
class MatchResult:
    matched: list[MatchedActor]
    new_candidates: list[NewActorCandidate]


@dataclass(frozen=True)
class FaceAppearance:
    person_id: str
    start_seconds: float
    end_seconds: float
    detection_count: int


@dataclass(frozen=True)
class FaceAnalysisResult:
    video_path: str
    appearances: list[FaceAppearance]
    new_candidates: list[NewActorCandidate] = field(default_factory=list)
