from dataclasses import dataclass
from math import sqrt

from app.services.face_analysis.models import ClusteredFaceDetection, FaceDetection


@dataclass
class _FaceCluster:
    person_id: str
    representative_embedding: list[float]
    detection_count: int = 1


class FaceClusterer:
    def __init__(self, similarity_threshold: float = 0.45):
        if not -1.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between -1.0 and 1.0")

        self.similarity_threshold = similarity_threshold
        self._clusters: list[_FaceCluster] = []

    def assign(self, detection: FaceDetection) -> ClusteredFaceDetection:
        best_cluster = None
        best_similarity = -1.0

        for cluster in self._clusters:
            similarity = cosine_similarity(
                detection.embedding,
                cluster.representative_embedding,
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster

        if best_cluster is None or best_similarity < self.similarity_threshold:
            best_cluster = self._create_cluster(detection.embedding)
        else:
            self._update_representative_embedding(best_cluster, detection.embedding)

        return ClusteredFaceDetection(
            person_id=best_cluster.person_id,
            detection=detection,
        )

    def get_cluster_count(self) -> int:
        return len(self._clusters)

    def _create_cluster(self, embedding: list[float]) -> _FaceCluster:
        cluster = _FaceCluster(
            person_id=f"person_{len(self._clusters) + 1}",
            representative_embedding=list(embedding),
        )
        self._clusters.append(cluster)
        return cluster

    def _update_representative_embedding(
        self,
        cluster: _FaceCluster,
        embedding: list[float],
    ) -> None:
        count = cluster.detection_count
        next_count = count + 1
        cluster.representative_embedding = [
            ((value * count) + next_value) / next_count
            for value, next_value in zip(cluster.representative_embedding, embedding)
        ]
        cluster.detection_count = next_count


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embeddings must have the same dimension")

    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot_product / (left_norm * right_norm)
