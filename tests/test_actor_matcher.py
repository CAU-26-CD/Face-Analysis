from app.services.face_analysis.actor_matcher import ActorMatcher
from app.services.face_analysis.models import KnownActor, PersonTracklet, WithinVideoCluster


def _cluster(
    cluster_id: str,
    aggregated: list[float],
    exemplars: list[list[float]],
) -> WithinVideoCluster:
    return WithinVideoCluster(
        cluster_id=cluster_id,
        tracklets=[
            PersonTracklet(
                track_id=f"{cluster_id}_track",
                person_detections=[],
                face_detections=[],
                aggregated_embedding=aggregated,
                exemplar_embeddings=exemplars,
                start_seconds=0.0,
                end_seconds=10.0,
            )
        ],
        aggregated_embedding=aggregated,
        exemplar_embeddings=exemplars,
        start_seconds=0.0,
        end_seconds=10.0,
    )


def test_max_exemplar_similarity_matches_when_centroid_would_miss():
    """The classic over-split case: cluster centroid is profile-dominated and
    looks dissimilar to a frontal actor template, but the cluster has one
    frontal exemplar that matches strongly."""
    profile_embedding = [0.0, 1.0, 0.0]
    frontal_embedding = [1.0, 0.0, 0.0]
    aggregated = [0.2, 0.8, 0.0]
    cluster = _cluster("cluster_1", aggregated, [profile_embedding, frontal_embedding])

    actor = KnownActor(actor_id=1, face_templates=[[1.0, 0.0, 0.0]])

    matcher = ActorMatcher(match_threshold=0.5, suggest_threshold=0.4)
    result = matcher.match([cluster], [actor])

    assert len(result.matched) == 1
    assert result.matched[0].actor_id == 1
    assert result.matched[0].similarity >= 0.5


def test_cluster_without_exemplars_falls_back_to_aggregated_embedding():
    cluster = _cluster("cluster_1", [1.0, 0.0, 0.0], [])
    actor = KnownActor(actor_id=1, face_templates=[[1.0, 0.0, 0.0]])

    matcher = ActorMatcher(match_threshold=0.5, suggest_threshold=0.4)
    result = matcher.match([cluster], [actor])

    assert len(result.matched) == 1


def test_unrelated_cluster_lands_as_new_candidate():
    cluster = _cluster("cluster_1", [1.0, 0.0, 0.0], [[1.0, 0.0, 0.0]])
    actor = KnownActor(actor_id=1, face_templates=[[0.0, 1.0, 0.0]])

    matcher = ActorMatcher(match_threshold=0.5, suggest_threshold=0.4)
    result = matcher.match([cluster], [actor])

    assert result.matched == []
    assert len(result.new_candidates) == 1
    assert result.new_candidates[0].suggested_actor_id is None
    # Multi-exemplar seed: the new-candidate carries the cluster's exemplar
    # set, not just the centroid, so BE gets a populated gallery.
    assert result.new_candidates[0].embeddings == [[1.0, 0.0, 0.0]]


def test_suggested_when_similarity_between_thresholds():
    cluster = _cluster(
        "cluster_1",
        [0.7, 0.7, 0.0],
        [[0.7, 0.7, 0.0]],
    )
    actor = KnownActor(actor_id=7, face_templates=[[1.0, 0.0, 0.0]])

    matcher = ActorMatcher(match_threshold=0.8, suggest_threshold=0.5)
    result = matcher.match([cluster], [actor])

    assert result.matched == []
    assert len(result.new_candidates) == 1
    assert result.new_candidates[0].suggested_actor_id == 7
    assert result.new_candidates[0].suggested_similarity is not None


def test_match_uses_max_over_actor_templates_when_one_pose_matches():
    """An actor whose gallery has only a frontal template should still match
    a profile-only cluster once we add the profile angle into the gallery.
    Verifies multi-template lookup uses *max* across templates, not a
    centroid that would dilute both poses."""
    profile_emb = [0.0, 1.0, 0.0]
    cluster = _cluster("cluster_1", profile_emb, [profile_emb])

    # Two galleries: frontal-only fails, frontal+profile succeeds.
    frontal_only = KnownActor(actor_id=1, face_templates=[[1.0, 0.0, 0.0]])
    matcher = ActorMatcher(match_threshold=0.5, suggest_threshold=0.4)
    miss = matcher.match([cluster], [frontal_only])
    assert miss.matched == []

    with_profile = KnownActor(
        actor_id=1,
        face_templates=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    hit = matcher.match([cluster], [with_profile])
    assert len(hit.matched) == 1
    assert hit.matched[0].actor_id == 1


def test_matched_includes_novel_exemplars_only():
    """Matched cluster ships back exemplars BE should append — but only the
    ones that are *new* angles. Anything already covered by an existing
    template (cosine ≥ novelty bar) is skipped so the gallery doesn't fill
    up with duplicates that would evict useful older entries on cap."""
    near_dup_of_existing = [1.0, 0.0, 0.0]      # cos ~ 1.0 vs template
    novel_angle = [0.0, 1.0, 0.0]               # cos ~ 0.0 vs template
    cluster = _cluster(
        "cluster_1",
        aggregated=[1.0, 0.0, 0.0],
        exemplars=[near_dup_of_existing, novel_angle],
    )
    actor = KnownActor(actor_id=1, face_templates=[[1.0, 0.0, 0.0]])

    matcher = ActorMatcher(
        match_threshold=0.5,
        suggest_threshold=0.4,
        new_exemplar_novelty_threshold=0.9,
    )
    result = matcher.match([cluster], [actor])

    assert len(result.matched) == 1
    assert result.matched[0].new_exemplars == [[0.0, 1.0, 0.0]]


def test_matched_caps_new_exemplars_per_match():
    # First exemplar matches the existing template → guarantees the cluster
    # gets matched. The other three are all novel angles (cos < novelty bar)
    # so without a cap all three would be appended; with cap=2 only two
    # should come back.
    cluster = _cluster(
        "cluster_1",
        aggregated=[1.0, 0.0, 0.0],
        exemplars=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
        ],
    )
    actor = KnownActor(actor_id=1, face_templates=[[1.0, 0.0, 0.0]])

    matcher = ActorMatcher(
        match_threshold=0.5,
        suggest_threshold=0.4,
        max_new_exemplars_per_match=2,
        new_exemplar_novelty_threshold=0.9,
    )
    result = matcher.match([cluster], [actor])

    assert len(result.matched) == 1
    assert len(result.matched[0].new_exemplars) == 2


def test_new_candidate_carries_full_exemplar_seed():
    """For a brand-new actor (no match), BE should receive the cluster's
    *multi-exemplar* seed so the gallery starts pose-diverse from day one,
    not a single centroid that would collapse on the first match."""
    cluster = _cluster(
        "cluster_1",
        aggregated=[0.5, 0.5, 0.0],
        exemplars=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )

    matcher = ActorMatcher(match_threshold=0.5, suggest_threshold=0.4)
    result = matcher.match([cluster], [])  # no known actors

    assert len(result.new_candidates) == 1
    assert result.new_candidates[0].embeddings == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
