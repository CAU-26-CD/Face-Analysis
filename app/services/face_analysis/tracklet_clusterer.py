from dataclasses import dataclass, field

from app.services.face_analysis.embeddings import cosine_similarity
from app.services.face_analysis.models import Tracklet, WithinVideoCluster


@dataclass
class _PendingCluster:
    cluster_id: str
    tracklets: list[Tracklet] = field(default_factory=list)
    aggregated_embedding: list[float] = field(default_factory=list)
    detection_count: int = 0
    start_seconds: float = 0.0
    end_seconds: float = 0.0

    @classmethod
    def from_tracklet(cls, cluster_id: str, tracklet: Tracklet) -> "_PendingCluster":
        return cls(
            cluster_id=cluster_id,
            tracklets=[tracklet],
            aggregated_embedding=list(tracklet.aggregated_embedding),
            detection_count=tracklet.detection_count,
            start_seconds=tracklet.start_seconds,
            end_seconds=tracklet.end_seconds,
        )

    def add(self, tracklet: Tracklet) -> None:
        current_count = self.detection_count
        incoming_count = tracklet.detection_count
        total = current_count + incoming_count
        self.aggregated_embedding = [
            ((value * current_count) + (new_value * incoming_count)) / total
            for value, new_value in zip(self.aggregated_embedding, tracklet.aggregated_embedding)
        ]
        self.tracklets.append(tracklet)
        self.detection_count = total
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
    """Merges tracklets that belong to the same person inside one video.

    The tracker is intentionally strict so that a person who leaves the frame
    and re-enters gets a new track_id. This stage compares aggregated
    (less-noisy) tracklet embeddings with a more lenient threshold to stitch
    those tracklets back into a single within-video identity.
    """

    def __init__(self, similarity_threshold: float = 0.40):
        if not -1.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between -1.0 and 1.0")

        self.similarity_threshold = similarity_threshold

    def cluster(self, tracklets: list[Tracklet]) -> list[WithinVideoCluster]:
        pending: list[_PendingCluster] = []

        ordered = sorted(tracklets, key=lambda tracklet: tracklet.start_seconds)
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
