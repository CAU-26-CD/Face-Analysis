from dataclasses import dataclass

from app.services.face_analysis.models import ClusteredFaceDetection, FaceAppearance


@dataclass
class _OpenAppearance:
    person_id: str
    start_seconds: float
    end_seconds: float
    detection_count: int


class AppearanceTimelineBuilder:
    def __init__(self, max_gap_seconds: float = 2.0):
        if max_gap_seconds < 0:
            raise ValueError("max_gap_seconds must be greater than or equal to 0")

        self.max_gap_seconds = max_gap_seconds
        self._open_appearances: dict[str, _OpenAppearance] = {}
        self._closed_appearances: list[FaceAppearance] = []

    def add(self, clustered_detection: ClusteredFaceDetection) -> None:
        person_id = clustered_detection.person_id
        timestamp = clustered_detection.detection.timestamp_seconds
        current = self._open_appearances.get(person_id)

        if current is None:
            self._open_appearances[person_id] = _OpenAppearance(
                person_id=person_id,
                start_seconds=timestamp,
                end_seconds=timestamp,
                detection_count=1,
            )
            return

        if timestamp - current.end_seconds <= self.max_gap_seconds:
            current.end_seconds = timestamp
            current.detection_count += 1
            return

        self._closed_appearances.append(_to_face_appearance(current))
        self._open_appearances[person_id] = _OpenAppearance(
            person_id=person_id,
            start_seconds=timestamp,
            end_seconds=timestamp,
            detection_count=1,
        )

    def build(self) -> list[FaceAppearance]:
        appearances = [
            *self._closed_appearances,
            *(_to_face_appearance(item) for item in self._open_appearances.values()),
        ]
        return sorted(
            appearances,
            key=lambda item: (item.start_seconds, item.person_id),
        )


def _to_face_appearance(appearance: _OpenAppearance) -> FaceAppearance:
    return FaceAppearance(
        person_id=appearance.person_id,
        start_seconds=float(appearance.start_seconds),
        end_seconds=float(appearance.end_seconds),
        detection_count=int(appearance.detection_count),
    )
