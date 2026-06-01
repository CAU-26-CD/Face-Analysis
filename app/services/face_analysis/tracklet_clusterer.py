import math
from dataclasses import dataclass, field

import numpy as np

from app.services.face_analysis.models import PersonTracklet, WithinVideoCluster


# A short on-screen overlap between two tracklets is treated as a likely
# ByteTrack ID-flip rather than two simultaneous people: a person briefly
# turns or gets occluded, the tracker spawns a new track_id, and for a few
# frames both ids exist. Anything longer is almost certainly two different
# people sharing the frame and gets vetoed regardless of similarity.
DEFAULT_MIN_OVERLAP_SECONDS = 0.5

# Spatio-temporal bridge for face-less tracklets. When an actor turns their
# back (or is otherwise occluded such that no face fires above InsightFace's
# confidence floor) ByteTrack still keeps a person bbox running, but at the
# next ID flip the new track has no face embeddings — ``has_identity`` is
# False and the tracklet would be dropped. We rescue it by stitching it into
# whichever identified cluster has a bbox that's both temporally and
# spatially adjacent. The veto still fires for sustained co-occurrence, so
# two distinct people standing in the same spot in turn cannot bleed into
# each other.
DEFAULT_ORPHAN_MAX_GAP_SECONDS = 2.0
DEFAULT_ORPHAN_MIN_BBOX_IOU = 0.3


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

    def absorb(self, other: "_PendingCluster") -> None:
        """Fold another pending cluster into this one (2-pass merge path).

        Replays the tracklets through ``add`` so the running averages and
        the exemplar pool stay consistent with the 1-pass append semantics
        — same arithmetic, no special case for "this came from a cluster".
        """
        for tracklet in other.tracklets:
            self.add(tracklet)

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

    Two stages, same similarity bar in both:

    1. **Streaming 1-pass.** Tracklets are processed in start-time order;
       each is matched against the best non-vetoed pending cluster. This
       handles the common case where a person leaves and re-enters frame
       cleanly.

    2. **Pairwise 2-pass.** The 1-pass output is re-examined: every pair of
       surviving clusters is scored, and the best mergeable pair is folded
       together. Repeats until no pair clears the bar. Catches the split
       that 1-pass can't fix — when a person's *frontal* run lands in one
       cluster and their *profile* run lands in another, neither would
       merge into the other on first sight (cross-pair median is still
       low), but once both clusters have accumulated their full exemplar
       sets the second-pass scoring sees the alignment.

    Three rules govern any merge:

    - **Temporal-overlap veto.** Overlap longer than ``min_overlap_seconds``
       between any two tracklets in the candidate cluster pair means two
       people, not one. Short overlaps (default 0.5s) are tolerated since
       ByteTrack briefly assigns two ids during a re-identification flip.

    - **Median pair similarity** (primary). ``median(cosine(A_exemplars,
       B_exemplars)) >= similarity_threshold``. Same-person pairs are mostly
       high; different-people pairs are mostly low with a few outliers,
       which the median ignores.

    - **Centroid backup** (secondary). ``cosine(A.aggregated, B.aggregated)
       >= centroid_merge_threshold``. Catches the case where the same
       person's exemplar set is split across very different poses, so
       cross-pair medians dip below the primary threshold even though the
       per-set means are tightly aligned.

    Either similarity rule clears the bar; the veto blocks both. Tracklets
    that never observed a face (no identity available) are dropped.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.40,
        centroid_merge_threshold: float = 0.60,
        min_overlap_seconds: float = DEFAULT_MIN_OVERLAP_SECONDS,
        orphan_max_gap_seconds: float = DEFAULT_ORPHAN_MAX_GAP_SECONDS,
        orphan_min_bbox_iou: float = DEFAULT_ORPHAN_MIN_BBOX_IOU,
    ):
        if not -1.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between -1.0 and 1.0")
        if not -1.0 <= centroid_merge_threshold <= 1.0:
            raise ValueError(
                "centroid_merge_threshold must be between -1.0 and 1.0"
            )
        if min_overlap_seconds < 0:
            raise ValueError("min_overlap_seconds must be >= 0")
        if orphan_max_gap_seconds < 0:
            raise ValueError("orphan_max_gap_seconds must be >= 0")
        if not 0.0 <= orphan_min_bbox_iou <= 1.0:
            raise ValueError("orphan_min_bbox_iou must be between 0.0 and 1.0")

        self.similarity_threshold = similarity_threshold
        self.centroid_merge_threshold = centroid_merge_threshold
        self.min_overlap_seconds = min_overlap_seconds
        self.orphan_max_gap_seconds = orphan_max_gap_seconds
        self.orphan_min_bbox_iou = orphan_min_bbox_iou

    def cluster(self, tracklets: list[PersonTracklet]) -> list[WithinVideoCluster]:
        identified = [t for t in tracklets if t.has_identity]
        orphans = [t for t in tracklets if not t.has_identity]
        ordered = sorted(identified, key=lambda tracklet: tracklet.start_seconds)

        pending: list[_PendingCluster] = []
        for tracklet in ordered:
            best_cluster = None
            best_score = -1.0
            for cluster in pending:
                if self._tracklets_overlap_excessively(
                    [tracklet], cluster.tracklets
                ):
                    continue
                median_sim, centroid_sim = _pair_similarities(
                    tracklet.exemplar_embeddings,
                    tracklet.aggregated_embedding,
                    cluster.exemplar_embeddings,
                    cluster.aggregated_embedding,
                )
                if not self._passes_similarity(median_sim, centroid_sim):
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

        pending = self._merge_clusters_pairwise(pending)
        pending = self._attach_orphan_tracklets(pending, orphans)
        return [cluster.finalize() for cluster in pending]

    def _merge_clusters_pairwise(
        self, pending: list[_PendingCluster]
    ) -> list[_PendingCluster]:
        """Greedy agglomerative pass: repeatedly fold the best mergeable
        pair until no pair clears the bar. Same vetoes and thresholds as
        the streaming pass; the only thing that changes is the exemplar
        pool size on both sides of the comparison, which is what unlocks
        the frontal/profile case that 1-pass can't reach.
        """
        while True:
            best: tuple[int, int, float] | None = None
            for i in range(len(pending)):
                for j in range(i + 1, len(pending)):
                    if self._tracklets_overlap_excessively(
                        pending[i].tracklets, pending[j].tracklets
                    ):
                        continue
                    median_sim, centroid_sim = _pair_similarities(
                        pending[i].exemplar_embeddings,
                        pending[i].aggregated_embedding,
                        pending[j].exemplar_embeddings,
                        pending[j].aggregated_embedding,
                    )
                    if not self._passes_similarity(median_sim, centroid_sim):
                        continue
                    score = max(median_sim, centroid_sim)
                    if best is None or score > best[2]:
                        best = (i, j, score)

            if best is None:
                return pending

            i, j, _ = best
            pending[i].absorb(pending[j])
            del pending[j]

    def _attach_orphan_tracklets(
        self,
        pending: list[_PendingCluster],
        orphans: list[PersonTracklet],
    ) -> list[_PendingCluster]:
        """Stitch face-less tracklets into the closest identified cluster.

        Targets the "actor turns their back" failure mode: ByteTrack ID
        flips while the face is hidden, spawning a tracklet with no
        embedding. Without this pass that tracklet — and the entire
        back-turned span of the appearance — would be dropped by the
        ``has_identity`` filter. We attach it to whichever existing cluster
        has a bbox that's temporally near (gap ≤ ``orphan_max_gap_seconds``)
        AND spatially overlapping (IoU ≥ ``orphan_min_bbox_iou``) at the
        boundary. The temporal-overlap veto is preserved so two distinct
        actors taking turns in the same spot can't be glued together.
        """
        if not pending or not orphans:
            return pending

        ordered = sorted(orphans, key=lambda tracklet: tracklet.start_seconds)
        for orphan in ordered:
            best_cluster = None
            best_iou = -1.0
            for cluster in pending:
                if self._tracklets_overlap_excessively(
                    [orphan], cluster.tracklets
                ):
                    continue
                iou = self._orphan_attachment_iou(orphan, cluster)
                if iou is None:
                    continue
                if iou > best_iou:
                    best_iou = iou
                    best_cluster = cluster
            if best_cluster is not None:
                best_cluster.add(orphan)
        return pending

    def _orphan_attachment_iou(
        self,
        orphan: PersonTracklet,
        cluster: _PendingCluster,
    ) -> float | None:
        """Best bbox IoU between ``orphan`` and any tracklet in ``cluster``
        whose time gap is within ``orphan_max_gap_seconds``. The bbox pair
        is chosen so each side contributes the box closest to the other in
        time — i.e. the box just before the gap and just after.

        Returns ``None`` when nothing in the cluster lands inside the
        temporal window OR the best IoU falls below ``orphan_min_bbox_iou``.
        """
        best_iou = -1.0
        within_window = False
        for tracklet in cluster.tracklets:
            gap = max(
                orphan.start_seconds - tracklet.end_seconds,
                tracklet.start_seconds - orphan.end_seconds,
            )
            if gap > self.orphan_max_gap_seconds:
                continue
            within_window = True
            if orphan.start_seconds >= tracklet.end_seconds:
                orphan_box = orphan.person_detections[0].bbox
                tracklet_box = tracklet.person_detections[-1].bbox
            elif tracklet.start_seconds >= orphan.end_seconds:
                orphan_box = orphan.person_detections[-1].bbox
                tracklet_box = tracklet.person_detections[0].bbox
            else:
                orphan_box = orphan.person_detections[0].bbox
                tracklet_box = tracklet.person_detections[-1].bbox
            iou = _bbox_iou(orphan_box, tracklet_box)
            if iou > best_iou:
                best_iou = iou
        if not within_window or best_iou < self.orphan_min_bbox_iou:
            return None
        return best_iou

    def _passes_similarity(self, median_sim: float, centroid_sim: float) -> bool:
        return (
            median_sim >= self.similarity_threshold
            or centroid_sim >= self.centroid_merge_threshold
        )

    def _tracklets_overlap_excessively(
        self,
        left: list[PersonTracklet],
        right: list[PersonTracklet],
    ) -> bool:
        """True iff *any* pair of tracklets (one from each side) overlaps
        on screen for longer than ``min_overlap_seconds``. A person can't
        share a long frame window with themselves, so a sustained overlap
        means different identities — even if their embeddings happen to
        agree. Short overlaps are tolerated as ID-flip noise.
        """
        max_overlap = _max_overlap_seconds(left, right)
        return max_overlap > self.min_overlap_seconds


def _max_overlap_seconds(
    left: list[PersonTracklet], right: list[PersonTracklet]
) -> float:
    max_overlap = -math.inf
    for a in left:
        for b in right:
            overlap = min(a.end_seconds, b.end_seconds) - max(
                a.start_seconds, b.start_seconds
            )
            if overlap > max_overlap:
                max_overlap = overlap
    return max_overlap


def _pair_similarities(
    left_exemplars: list[list[float]],
    left_centroid: list[float],
    right_exemplars: list[list[float]],
    right_centroid: list[float],
) -> tuple[float, float]:
    median_sim = _median_pair_similarity(left_exemplars, right_exemplars)
    centroid_sim = _centroid_cosine(left_centroid, right_centroid)
    return median_sim, centroid_sim


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


def _bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


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
