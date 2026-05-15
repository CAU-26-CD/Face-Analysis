from dataclasses import dataclass, field

from app.services.face_analysis.embeddings import cosine_similarity
from app.services.face_analysis.models import PersonTracklet, WithinVideoCluster


@dataclass
class _PendingCluster:
    cluster_id: str
    tracklets: list[PersonTracklet] = field(default_factory=list)
    aggregated_embedding: list[float] = field(default_factory=list)
    exemplar_embeddings: list[list[float]] = field(default_factory=list)
    face_count: int = 0
    start_seconds: float = 0.0
    end_seconds: float = 0.0

    @classmethod
    def from_tracklet(cls, cluster_id: str, tracklet: PersonTracklet) -> "_PendingCluster":
        return cls(
            cluster_id=cluster_id,
            tracklets=[tracklet],
            aggregated_embedding=list(tracklet.aggregated_embedding),
            exemplar_embeddings=[list(e) for e in tracklet.exemplar_embeddings],
            face_count=tracklet.face_count,
            start_seconds=tracklet.start_seconds,
            end_seconds=tracklet.end_seconds,
        )

    def add(self, tracklet: PersonTracklet) -> None:
        current_count = self.face_count
        incoming_count = tracklet.face_count
        total = current_count + incoming_count
        if total > 0 and self.aggregated_embedding and tracklet.aggregated_embedding:
            self.aggregated_embedding = [
                ((value * current_count) + (new_value * incoming_count)) / total
                for value, new_value in zip(self.aggregated_embedding, tracklet.aggregated_embedding)
            ]
        elif tracklet.aggregated_embedding and not self.aggregated_embedding:
            self.aggregated_embedding = list(tracklet.aggregated_embedding)
        self.exemplar_embeddings.extend(list(e) for e in tracklet.exemplar_embeddings)
        self.tracklets.append(tracklet)
        self.face_count = total
        self.start_seconds = min(self.start_seconds, tracklet.start_seconds)
        self.end_seconds = max(self.end_seconds, tracklet.end_seconds)

    def finalize(self) -> WithinVideoCluster:
        return WithinVideoCluster(
            cluster_id=self.cluster_id,
            tracklets=list(self.tracklets),
            aggregated_embedding=list(self.aggregated_embedding),
            exemplar_embeddings=[list(e) for e in self.exemplar_embeddings],
            start_seconds=self.start_seconds,
            end_seconds=self.end_seconds,
        )


class TrackletClusterer:
    """Merges person tracklets that belong to the same identity inside one video.

    ByteTrack keeps a stable track_id while the person bbox is visible, but a
    person who leaves the frame longer than ``track_buffer`` gets a new
    track_id. This stage stitches those back together by comparing tracklets'
    *exemplar sets* using **max** cosine similarity — so a tracklet whose
    centroid is profile-heavy can still merge with a cluster whose centroid
    is frontal, as long as at least one exemplar pair matches well.

    Tracklets that never observed a face (no identity available) are dropped.
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
                similarity = _max_pair_similarity(
                    tracklet.exemplar_embeddings,
                    cluster.exemplar_embeddings,
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


def _max_pair_similarity(
    left: list[list[float]], right: list[list[float]]
) -> float:
    if not left or not right:
        return -1.0
    best = -1.0
    for a in left:
        for b in right:
            similarity = cosine_similarity(a, b)
            if similarity > best:
                best = similarity
    return best
