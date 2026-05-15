from dataclasses import dataclass, field

from app.services.face_analysis.embeddings import cosine_similarity
from app.services.face_analysis.models import PersonTracklet, WithinVideoCluster


@dataclass
class _PendingCluster:
    cluster_id: str
    tracklets: list[PersonTracklet] = field(default_factory=list)
    aggregated_embedding: list[float] = field(default_factory=list)
    face_count: int = 0
    start_seconds: float = 0.0
    end_seconds: float = 0.0

    @classmethod
    def from_tracklet(cls, cluster_id: str, tracklet: PersonTracklet) -> "_PendingCluster":
        return cls(
            cluster_id=cluster_id,
            tracklets=[tracklet],
            aggregated_embedding=list(tracklet.aggregated_embedding),
            face_count=tracklet.face_count,
            start_seconds=tracklet.start_seconds,
            end_seconds=tracklet.end_seconds,
        )

    def add(self, tracklet: PersonTracklet) -> None:
        current_count = self.face_count
        incoming_count = tracklet.face_count
        total = current_count + incoming_count
        if total > 0:
            self.aggregated_embedding = [
                ((value * current_count) + (new_value * incoming_count)) / total
                for value, new_value in zip(self.aggregated_embedding, tracklet.aggregated_embedding)
            ]
        self.tracklets.append(tracklet)
        self.face_count = total
        self.start_seconds = min(self.start_seconds, tracklet.start_seconds)
        self.end_seconds = max(self.end_seconds, tracklet.end_seconds)

    def finalize(self) -> WithinVideoCluster:
        return WithinVideoCluster(
            cluster_id=self.cluster_id,
            tracklets=list(self.tracklets),
            aggregated_embedding=list(self.aggregated_embedding),
            start_seconds=self.start_seconds,
            end_seconds=self.end_seconds,
        )


class TrackletClusterer:
    """Merges person tracklets that belong to the same identity inside one video.

    ByteTrack already keeps a stable track_id while the person bbox is visible,
    but a person who leaves the frame longer than ``track_buffer`` gets a new
    track_id. This stage compares per-track face embeddings to stitch those
    back into a single within-video identity.

    Tracklets with no face_count (the person was tracked but their face was
    never recognized — e.g. they stayed turned away) cannot be matched to any
    known actor and are dropped here.
    """

    def __init__(self, similarity_threshold: float = 0.40):
        if not -1.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between -1.0 and 1.0")

        self.similarity_threshold = similarity_threshold

    def cluster(self, tracklets: list[PersonTracklet]) -> list[WithinVideoCluster]:
        identified = [t for t in tracklets if t.has_identity]
        ordered = sorted(identified, key=lambda tracklet: tracklet.start_seconds)

        pending: list[_PendingCluster] = []
        for tracklet in ordered:
            best_cluster = None
            best_similarity = -1.0
            for cluster in pending:
                similarity = cosine_similarity(
                    tracklet.aggregated_embedding,
                    cluster.aggregated_embedding,
                )
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = cluster

            if best_cluster is None or best_similarity < self.similarity_threshold:
                pending.append(
                    _PendingCluster.from_tracklet(
                        cluster_id=f"cluster_{len(pending) + 1}",
                        tracklet=tracklet,
                    )
                )
            else:
                best_cluster.add(tracklet)

        return [cluster.finalize() for cluster in pending]
