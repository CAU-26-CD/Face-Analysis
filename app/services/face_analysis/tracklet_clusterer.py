from dataclasses import dataclass, field

import numpy as np

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
    *exemplar sets* using the **median** of all cross-pair cosine similarities,
    backed up by an aggregate-centroid rule and gated by a temporal-overlap
    veto.

    Three rules govern a merge of tracklet ``T`` into pending cluster ``C``:

    1. **Temporal-overlap veto.** If ``T``'s on-screen interval intersects any
       tracklet already in ``C``, they co-occur and cannot be the same person.
       Hard block — short-circuits all similarity scoring.

    2. **Median pair similarity** (primary). ``median(cosine(T_exemplars,
       C_exemplars)) >= similarity_threshold``. Same-person pairs are mostly
       high; different-people pairs are mostly low with a few outliers, which
       the median ignores. Why not max/mean: max merges on a single outlier;
       mean amplifies one stray pair across the whole product.

    3. **Centroid backup** (secondary). ``cosine(T.aggregated, C.aggregated) >=
       centroid_merge_threshold``. Catches the case where the same person's
       exemplar set is split across very different poses (frontal/profile),
       so cross-pair medians dip below the primary threshold even though the
       per-set means are tightly aligned. The threshold is set well above the
       median bar so it only fires on confident centroid agreement.

    Either #2 OR #3 clears the bar; #1 vetoes both. Tracklets that never
    observed a face (no identity available) are dropped.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.40,
        centroid_merge_threshold: float = 0.60,
    ):
        if not -1.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between -1.0 and 1.0")
        if not -1.0 <= centroid_merge_threshold <= 1.0:
            raise ValueError(
                "centroid_merge_threshold must be between -1.0 and 1.0"
            )

        self.similarity_threshold = similarity_threshold
        self.centroid_merge_threshold = centroid_merge_threshold

    def cluster(self, tracklets: list[PersonTracklet]) -> list[WithinVideoCluster]:
        identified = [t for t in tracklets if t.has_identity]
        ordered = sorted(identified, key=lambda tracklet: tracklet.start_seconds)

        pending: list[_PendingCluster] = []
        for tracklet in ordered:
            best_cluster = None
            best_score = -1.0
            for cluster in pending:
                if _temporal_overlap(tracklet, cluster):
                    continue
                median_sim = _median_pair_similarity(
                    tracklet.exemplar_embeddings,
                    cluster.exemplar_embeddings,
                )
                centroid_sim = _centroid_cosine(
                    tracklet.aggregated_embedding,
                    cluster.aggregated_embedding,
                )
                passes = (
                    median_sim >= self.similarity_threshold
                    or centroid_sim >= self.centroid_merge_threshold
                )
                if not passes:
                    continue
                score = max(median_sim, centroid_sim)
                if score > best_score:
                    best_score = score
                    best_cluster = cluster

            if best_cluster is None:
                pending.append(
                    _PendingCluster.from_tracklet(
                        cluster_id=f"cluster_{len(pending) + 1}",
                        tracklet=tracklet,
                    )
                )
            else:
                best_cluster.add(tracklet)

        return [cluster.finalize() for cluster in pending]


def _temporal_overlap(
    tracklet: PersonTracklet, cluster: _PendingCluster
) -> bool:
    """True iff the tracklet's interval intersects any tracklet already in
    the cluster. A person can't share a frame with themselves, so any overlap
    means different identities — even if their embeddings happen to agree.
    """
    for existing in cluster.tracklets:
        if (
            tracklet.start_seconds <= existing.end_seconds
            and existing.start_seconds <= tracklet.end_seconds
        ):
            return True
    return False


def _centroid_cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return -1.0
    left_vec = np.asarray(left, dtype=np.float32)
    right_vec = np.asarray(right, dtype=np.float32)
    left_norm = float(np.linalg.norm(left_vec))
    right_norm = float(np.linalg.norm(right_vec))
    if left_norm == 0.0 or right_norm == 0.0:
        return -1.0
    return float(np.dot(left_vec, right_vec) / (left_norm * right_norm))


def _median_pair_similarity(
    left: list[list[float]], right: list[list[float]]
) -> float:
    if not left or not right:
        return -1.0

    left_matrix = np.asarray(left, dtype=np.float32)
    right_matrix = np.asarray(right, dtype=np.float32)

    left_norms = np.linalg.norm(left_matrix, axis=1, keepdims=True)
    right_norms = np.linalg.norm(right_matrix, axis=1, keepdims=True)
    # Treat zero-norm rows as cosine=0 against everything else.
    left_norms[left_norms == 0.0] = 1.0
    right_norms[right_norms == 0.0] = 1.0

    left_normalized = left_matrix / left_norms
    right_normalized = right_matrix / right_norms
    similarities = left_normalized @ right_normalized.T
    return float(np.median(similarities))
