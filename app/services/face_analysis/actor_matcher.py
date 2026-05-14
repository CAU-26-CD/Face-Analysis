from app.services.face_analysis.embeddings import cosine_similarity
from app.services.face_analysis.models import (
    KnownActor,
    MatchResult,
    MatchedActor,
    NewActorCandidate,
    WithinVideoCluster,
)


class ActorMatcher:
    """Resolves within-video clusters against the project actor gallery.

    A cluster lands in one of three buckets:
    - matched: similarity ≥ match_threshold → confirmed as that known actor
    - new candidate with suggestion: suggest_threshold ≤ similarity < match_threshold
      → surfaced to the user as "is this <actor>?"
    - new candidate without suggestion: similarity < suggest_threshold → fresh person
    """

    def __init__(
        self,
        match_threshold: float = 0.50,
        suggest_threshold: float = 0.40,
    ):
        if not -1.0 <= suggest_threshold <= 1.0:
            raise ValueError("suggest_threshold must be between -1.0 and 1.0")
        if not -1.0 <= match_threshold <= 1.0:
            raise ValueError("match_threshold must be between -1.0 and 1.0")
        if suggest_threshold > match_threshold:
            raise ValueError("suggest_threshold must be <= match_threshold")

        self.match_threshold = match_threshold
        self.suggest_threshold = suggest_threshold

    def match(
        self,
        clusters: list[WithinVideoCluster],
        known_actors: list[KnownActor],
    ) -> MatchResult:
        matched: list[MatchedActor] = []
        new_candidates: list[NewActorCandidate] = []

        for cluster in clusters:
            best_actor, best_similarity = self._best_actor(cluster, known_actors)

            if best_actor is not None and best_similarity >= self.match_threshold:
                matched.append(
                    MatchedActor(
                        cluster_id=cluster.cluster_id,
                        actor_id=best_actor.actor_id,
                        similarity=best_similarity,
                    )
                )
                continue

            suggested_actor_id: str | None = None
            suggested_similarity: float | None = None
            if best_actor is not None and best_similarity >= self.suggest_threshold:
                suggested_actor_id = best_actor.actor_id
                suggested_similarity = best_similarity

            new_candidates.append(
                NewActorCandidate(
                    cluster_id=cluster.cluster_id,
                    embedding=list(cluster.aggregated_embedding),
                    detection_count=cluster.detection_count,
                    start_seconds=cluster.start_seconds,
                    end_seconds=cluster.end_seconds,
                    suggested_actor_id=suggested_actor_id,
                    suggested_similarity=suggested_similarity,
                )
            )

        return MatchResult(matched=matched, new_candidates=new_candidates)

    def _best_actor(
        self,
        cluster: WithinVideoCluster,
        known_actors: list[KnownActor],
    ) -> tuple[KnownActor | None, float]:
        best_actor: KnownActor | None = None
        best_similarity = -1.0
        for actor in known_actors:
            similarity = cosine_similarity(
                cluster.aggregated_embedding,
                actor.face_template,
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_actor = actor
        return best_actor, best_similarity
