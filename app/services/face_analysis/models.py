from dataclasses import dataclass
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
class ClusteredFaceDetection:
    person_id: str
    detection: FaceDetection


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
