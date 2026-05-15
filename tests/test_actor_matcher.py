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

    actor = KnownActor(actor_id="actor_kim", face_template=[1.0, 0.0, 0.0])

    matcher = ActorMatcher(match_threshold=0.5, suggest_threshold=0.4)
    result = matcher.match([cluster], [actor])

    assert len(result.matched) == 1
    assert result.matched[0].actor_id == "actor_kim"
    assert result.matched[0].similarity >= 0.5


def test_cluster_without_exemplars_falls_back_to_aggregated_embedding():
    cluster = _cluster("cluster_1", [1.0, 0.0, 0.0], [])
    actor = KnownActor(actor_id="actor_kim", face_template=[1.0, 0.0, 0.0])

    matcher = ActorMatcher(match_threshold=0.5, suggest_threshold=0.4)
    result = matcher.match([cluster], [actor])

    assert len(result.matched) == 1


def test_unrelated_cluster_lands_as_new_candidate():
    cluster = _cluster("cluster_1", [1.0, 0.0, 0.0], [[1.0, 0.0, 0.0]])
    actor = KnownActor(actor_id="actor_kim", face_template=[0.0, 1.0, 0.0])

    matcher = ActorMatcher(match_threshold=0.5, suggest_threshold=0.4)
    result = matcher.match([cluster], [actor])

    assert result.matched == []
    assert len(result.new_candidates) == 1
    assert result.new_candidates[0].suggested_actor_id is None


def test_suggested_when_similarity_between_thresholds():
    cluster = _cluster(
        "cluster_1",
        [0.7, 0.7, 0.0],
        [[0.7, 0.7, 0.0]],
    )
    actor = KnownActor(actor_id="actor_kim", face_template=[1.0, 0.0, 0.0])

    matcher = ActorMatcher(match_threshold=0.8, suggest_threshold=0.5)
    result = matcher.match([cluster], [actor])

    assert result.matched == []
    assert len(result.new_candidates) == 1
    assert result.new_candidates[0].suggested_actor_id == "actor_kim"
    assert result.new_candidates[0].suggested_similarity is not None
