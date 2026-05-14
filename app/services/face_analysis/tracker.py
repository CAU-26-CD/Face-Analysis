from dataclasses import dataclass

from app.services.face_analysis.embeddings import cosine_similarity, running_average
from app.services.face_analysis.models import (
    FaceDetection,
    TrackedDetection,
    Tracklet,
    VideoFrame,
)


@dataclass
class _ActiveTrack:
    track_id: str
    representative_embedding: list[float]
    detections: list[FaceDetection]
    last_seen_seconds: float


class FaceTracker:
    """Assigns a video-scoped track_id to each detection.

    At the current sampling rate (~1 fps) the only reliable continuity signal is
    the face embedding, so this tracker is embedding-only. The interface is
    designed to be swapped for an IoU/motion tracker (BoT-SORT/ByteTrack) once
    we sample at 5+ fps.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.50,
        max_gap_seconds: float = 2.0,
    ):
        if not -1.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between -1.0 and 1.0")
        if max_gap_seconds < 0:
            raise ValueError("max_gap_seconds must be greater than or equal to 0")

        self.similarity_threshold = similarity_threshold
        self.max_gap_seconds = max_gap_seconds
        self._active_tracks: list[_ActiveTrack] = []
        self._closed_tracklets: list[Tracklet] = []
        self._next_track_id = 1

    def update(
        self,
        video_frame: VideoFrame,
        detections: list[FaceDetection],
    ) -> list[TrackedDetection]:
        self._expire_stale_tracks(video_frame.timestamp_seconds)

        ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
        bound_track_ids: set[str] = set()
        track_id_by_detection: dict[int, str] = {}

        for detection in ordered:
            best_track = self._best_match(detection, exclude=bound_track_ids)

            if best_track is None:
                new_track = self._open_track(detection)
                track_id_by_detection[id(detection)] = new_track.track_id
                bound_track_ids.add(new_track.track_id)
            else:
                self._extend_track(best_track, detection)
                track_id_by_detection[id(detection)] = best_track.track_id
                bound_track_ids.add(best_track.track_id)

        return [
            TrackedDetection(
                track_id=track_id_by_detection[id(detection)],
                detection=detection,
            )
            for detection in detections
        ]

    def finalize(self) -> list[Tracklet]:
        for track in self._active_tracks:
            self._closed_tracklets.append(_to_tracklet(track))
        self._active_tracks = []
        return list(self._closed_tracklets)

    def _best_match(
        self,
        detection: FaceDetection,
        exclude: set[str],
    ) -> _ActiveTrack | None:
        best_track: _ActiveTrack | None = None
        best_similarity = -1.0
        for track in self._active_tracks:
            if track.track_id in exclude:
                continue
            similarity = cosine_similarity(
                detection.embedding,
                track.representative_embedding,
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_track = track

        if best_track is None or best_similarity < self.similarity_threshold:
            return None
        return best_track

    def _expire_stale_tracks(self, current_timestamp: float) -> None:
        retained: list[_ActiveTrack] = []
        for track in self._active_tracks:
            if current_timestamp - track.last_seen_seconds > self.max_gap_seconds:
                self._closed_tracklets.append(_to_tracklet(track))
            else:
                retained.append(track)
        self._active_tracks = retained

    def _open_track(self, detection: FaceDetection) -> _ActiveTrack:
        track = _ActiveTrack(
            track_id=f"track_{self._next_track_id}",
            representative_embedding=list(detection.embedding),
            detections=[detection],
            last_seen_seconds=detection.timestamp_seconds,
        )
        self._next_track_id += 1
        self._active_tracks.append(track)
        return track

    def _extend_track(self, track: _ActiveTrack, detection: FaceDetection) -> None:
        track.representative_embedding = running_average(
            track.representative_embedding,
            detection.embedding,
            current_count=len(track.detections),
        )
        track.detections.append(detection)
        track.last_seen_seconds = detection.timestamp_seconds


def _to_tracklet(track: _ActiveTrack) -> Tracklet:
    return Tracklet(
        track_id=track.track_id,
        detections=list(track.detections),
        aggregated_embedding=list(track.representative_embedding),
        start_seconds=track.detections[0].timestamp_seconds,
        end_seconds=track.detections[-1].timestamp_seconds,
    )
