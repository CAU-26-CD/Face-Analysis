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
class PersonDetection:
    timestamp_seconds: float
    frame_index: int
    bbox: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True)
class TrackedPerson:
    track_id: str
    detection: PersonDetection


@dataclass(frozen=True)
class TrackedFaceDetection:
    detection: FaceDetection
    person_track_id: str | None


@dataclass(frozen=True)
class PersonTracklet:
    """All observations of one person (by MOT track_id) across one video.

    ``person_detections`` is when the person bbox was on screen.
    ``face_detections`` is the subset of frames where a face inside that bbox
    was also recognized — this is what gives the track its identity.
    """

    track_id: str
    person_detections: list[PersonDetection]
    face_detections: list[FaceDetection]
    aggregated_embedding: list[float]
    start_seconds: float
    end_seconds: float

    @property
    def face_count(self) -> int:
        return len(self.face_detections)

    @property
    def person_observation_count(self) -> int:
        return len(self.person_detections)

    @property
    def has_identity(self) -> bool:
        return bool(self.face_detections) and bool(self.aggregated_embedding)


@dataclass(frozen=True)
class WithinVideoCluster:
    cluster_id: str
    tracklets: list[PersonTracklet]
    aggregated_embedding: list[float]
    start_seconds: float
    end_seconds: float

    @property
    def detection_count(self) -> int:
        return sum(tracklet.person_observation_count for tracklet in self.tracklets)


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
