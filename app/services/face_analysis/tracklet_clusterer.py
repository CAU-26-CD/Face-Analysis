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
    *exemplar sets* using the **median** of all cross-pair cosine similarities.

    Why median, not max or mean:
      - Max merges on a single outlier pair. Different people occasionally
        produce one high-similarity pair (profile/lighting/mask coincidence),
        which Phase 5 saw as cross-identity over-merges in real videos.
      - Mean is brittle when one tracklet has highly self-similar exemplars
        (same person should!) — a single outlier pair gets multiplied across
        all paired exemplars from the other set and skews the average.
      - Median requires the *majority* of cross-pair similarities to be high.
        Same-person sets produce many high pairs; different-person sets
        produce a few outliers at most, which the median ignores.

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
                similarity = _median_pair_similarity(
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


def _median_pair_similarity(
    left: list[list[float]], right: list[list[float]]
) -> float:
    if not left or not right:
        return -1.0
    similarities = sorted(
        cosine_similarity(a, b) for a in left for b in right
    )
    n = len(similarities)
    if n == 0:
        return -1.0
    middle = n // 2
    if n % 2 == 1:
        return similarities[middle]
    return (similarities[middle - 1] + similarities[middle]) / 2.0
