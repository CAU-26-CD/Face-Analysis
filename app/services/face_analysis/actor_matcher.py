from app.services.face_analysis.embeddings import cosine_similarity
from app.services.face_analysis.models import (
    KnownActor,
    MatchResult,
    MatchedActor,
    NewActorCandidate,
    WithinVideoCluster,
)


# Per-matched-cluster cap on how many exemplars we ship back as "new angles".
# Larger values dilute the gallery (and BE's per-actor cap evicts faster);
# smaller values starve the gallery from growing.
DEFAULT_MAX_NEW_EXEMPLARS_PER_MATCH = 5

# A cluster exemplar is treated as a "new angle" only when its best cosine
# vs the actor's existing templates is BELOW this bar. Tuned so frontal
# duplicates don't pile up but profile/lighting variants do get appended.
DEFAULT_NEW_EXEMPLAR_NOVELTY_THRESHOLD = 0.90


class ActorMatcher:
    """Resolves within-video clusters against the project actor gallery.

    Each known actor now carries a *list* of templates (frontal/profile/
    makeup variants accumulated across rehearsals), and the cluster carries
    a list of exemplars. Match score is the **max cosine over the full
    NxM pairing** — picking the closest single (cluster-exemplar,
    actor-template) pair. Centroid averaging across poses tends to drag
    same-person scores down; max-over-pairs keeps them up.

    A cluster lands in one of three buckets:
    - matched: similarity ≥ match_threshold → confirmed as that known actor.
      We also pick out which of this cluster's exemplars look like a *new*
      angle (not already represented in the actor's gallery) so BE can
      append them — that is how the gallery actually grows over time.
    - new candidate with suggestion: suggest_threshold ≤ similarity <
      match_threshold → surfaced to the user as "is this <actor>?"
    - new candidate without suggestion: similarity < suggest_threshold →
      fresh person, BE seeds a new actor with the full exemplar set.
    """

    def __init__(
        self,
        match_threshold: float = 0.50,
        suggest_threshold: float = 0.40,
        max_new_exemplars_per_match: int = DEFAULT_MAX_NEW_EXEMPLARS_PER_MATCH,
        new_exemplar_novelty_threshold: float = DEFAULT_NEW_EXEMPLAR_NOVELTY_THRESHOLD,
    ):
        if not -1.0 <= suggest_threshold <= 1.0:
            raise ValueError("suggest_threshold must be between -1.0 and 1.0")
        if not -1.0 <= match_threshold <= 1.0:
            raise ValueError("match_threshold must be between -1.0 and 1.0")
        if suggest_threshold > match_threshold:
            raise ValueError("suggest_threshold must be <= match_threshold")
        if not -1.0 <= new_exemplar_novelty_threshold <= 1.0:
            raise ValueError(
                "new_exemplar_novelty_threshold must be between -1.0 and 1.0"
            )
        if max_new_exemplars_per_match < 0:
            raise ValueError("max_new_exemplars_per_match must be >= 0")

        self.match_threshold = match_threshold
        self.suggest_threshold = suggest_threshold
        self.max_new_exemplars_per_match = max_new_exemplars_per_match
        self.new_exemplar_novelty_threshold = new_exemplar_novelty_threshold

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
                new_exemplars = self._select_novel_exemplars(
                    cluster_exemplars=_cluster_candidate_embeddings(cluster),
                    actor_templates=best_actor.face_templates,
                )
                matched.append(
                    MatchedActor(
                        cluster_id=cluster.cluster_id,
                        actor_id=best_actor.actor_id,
                        similarity=best_similarity,
                        new_exemplars=new_exemplars,
                    )
                )
                continue

            suggested_actor_id: int | None = None
            suggested_similarity: float | None = None
            if best_actor is not None and best_similarity >= self.suggest_threshold:
                suggested_actor_id = best_actor.actor_id
                suggested_similarity = best_similarity

            new_candidates.append(
                NewActorCandidate(
                    cluster_id=cluster.cluster_id,
                    embeddings=_cluster_candidate_embeddings(cluster),
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
        candidate_embeddings = _cluster_candidate_embeddings(cluster)
        if not candidate_embeddings:
            return None, -1.0

        best_actor: KnownActor | None = None
        best_similarity = -1.0
        for actor in known_actors:
            if not actor.face_templates:
                continue
            similarity = max(
                cosine_similarity(embedding, template)
                for embedding in candidate_embeddings
                for template in actor.face_templates
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_actor = actor
        return best_actor, best_similarity

    def _select_novel_exemplars(
        self,
        cluster_exemplars: list[list[float]],
        actor_templates: list[list[float]],
    ) -> list[list[float]]:
        """Keep only the exemplars that are *not* already covered by the
        actor's gallery — i.e. max cosine vs every existing template stays
        below ``new_exemplar_novelty_threshold``.

        BE's per-actor cap is dumb (oldest-drop only), so without this filter
        a single rehearsal full of frontal frames would evict every profile
        template the actor had accumulated. Returning only novel angles lets
        the cap stay small without losing pose coverage.
        """
        if self.max_new_exemplars_per_match == 0:
            return []
        if not cluster_exemplars:
            return []
        if not actor_templates:
            # No baseline to compare against — happens if BE sent an actor
            # row with an empty gallery (shouldn't, but be defensive).
            return [list(e) for e in cluster_exemplars[: self.max_new_exemplars_per_match]]

        novel: list[list[float]] = []
        for exemplar in cluster_exemplars:
            max_sim = max(
                cosine_similarity(exemplar, template) for template in actor_templates
            )
            if max_sim < self.new_exemplar_novelty_threshold:
                novel.append(list(exemplar))
                if len(novel) >= self.max_new_exemplars_per_match:
                    break
        return novel


def _cluster_candidate_embeddings(cluster: WithinVideoCluster) -> list[list[float]]:
    """Prefer the exemplar set (pose-diverse). Fall back to the centroid
    when a cluster never collected exemplars (face_count too low, etc.) so
    the cluster still has *something* to match against rather than being
    silently dropped.
    """
    if cluster.exemplar_embeddings:
        return [list(e) for e in cluster.exemplar_embeddings]
    if cluster.aggregated_embedding:
        return [list(cluster.aggregated_embedding)]
    return []
