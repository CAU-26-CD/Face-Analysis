from app.services.face_analysis.models import (
    FaceDetection,
    PersonDetection,
    PersonTracklet,
)
from app.services.face_analysis.tracklet_clusterer import TrackletClusterer


def _person_detection(t: float, frame_index: int = 0) -> PersonDetection:
    return PersonDetection(
        timestamp_seconds=t,
        frame_index=frame_index,
        bbox=(0, 0, 100, 200),
        confidence=0.9,
    )


def _face_detection(
    t: float,
    embedding: list[float],
    frame_index: int = 0,
) -> FaceDetection:
    return FaceDetection(
        timestamp_seconds=t,
        frame_index=frame_index,
        bbox=(10, 10, 90, 90),
        embedding=embedding,
        confidence=0.95,
    )


def _tracklet(
    track_id: str,
    start: float,
    end: float,
    embedding: list[float],
    face_count: int = 3,
    person_count: int = 5,
) -> PersonTracklet:
    person_detections = [
        _person_detection(start + (end - start) * (i / max(1, person_count - 1)))
        for i in range(person_count)
    ]
    face_detections = [
        _face_detection(
            t=person_detections[i % len(person_detections)].timestamp_seconds,
            embedding=embedding,
        )
        for i in range(face_count)
    ]
    return PersonTracklet(
        track_id=track_id,
        person_detections=person_detections,
        face_detections=face_detections,
        aggregated_embedding=embedding,
        exemplar_embeddings=[embedding],
        start_seconds=start,
        end_seconds=end,
    )


def test_tracklets_with_similar_embeddings_merge_into_one_cluster():
    clusterer = TrackletClusterer(similarity_threshold=0.5)
    t1 = _tracklet("person_1", 0.0, 5.0, [1.0, 0.0, 0.0])
    t2 = _tracklet("person_2", 10.0, 15.0, [1.0, 0.0, 0.0])

    clusters = clusterer.cluster([t1, t2])

    assert len(clusters) == 1
    assert len(clusters[0].tracklets) == 2
    assert clusters[0].start_seconds == 0.0
    assert clusters[0].end_seconds == 15.0


def test_tracklets_with_dissimilar_embeddings_form_separate_clusters():
    clusterer = TrackletClusterer(similarity_threshold=0.5)
    t1 = _tracklet("person_1", 0.0, 5.0, [1.0, 0.0, 0.0])
    t2 = _tracklet("person_2", 10.0, 15.0, [0.0, 1.0, 0.0])

    clusters = clusterer.cluster([t1, t2])

    assert len(clusters) == 2
    ids = {c.cluster_id for c in clusters}
    assert ids == {"cluster_1", "cluster_2"}


def test_tracklets_without_face_identity_are_dropped():
    clusterer = TrackletClusterer()
    identified = _tracklet("person_1", 0.0, 5.0, [1.0, 0.0, 0.0])
    unidentified = PersonTracklet(
        track_id="person_2",
        person_detections=[_person_detection(6.0), _person_detection(8.0)],
        face_detections=[],
        aggregated_embedding=[],
        exemplar_embeddings=[],
        start_seconds=6.0,
        end_seconds=8.0,
    )

    clusters = clusterer.cluster([identified, unidentified])

    assert len(clusters) == 1
    assert clusters[0].tracklets[0].track_id == "person_1"


def test_cluster_detection_count_sums_person_observations():
    clusterer = TrackletClusterer(similarity_threshold=0.5)
    t1 = _tracklet("person_1", 0.0, 5.0, [1.0, 0.0, 0.0], person_count=6)
    t2 = _tracklet("person_2", 10.0, 15.0, [1.0, 0.0, 0.0], person_count=4)

    clusters = clusterer.cluster([t1, t2])

    assert clusters[0].detection_count == 10


def test_empty_input_returns_empty_clusters():
    clusterer = TrackletClusterer()
    assert clusterer.cluster([]) == []


def test_same_person_with_multiple_agreeing_exemplars_merges():
    """Same person across two tracklets: every exemplar pair has high
    similarity. Top-K mean should easily clear the threshold."""
    embedding_a = [1.0, 0.0, 0.0]
    embedding_b = [0.97, 0.05, 0.0]
    embedding_c = [0.95, 0.0, 0.05]

    tracklet_1 = PersonTracklet(
        track_id="person_1",
        person_detections=[_person_detection(0.0)],
        face_detections=[_face_detection(0.0, embedding_a)],
        aggregated_embedding=embedding_a,
        exemplar_embeddings=[embedding_a, embedding_b, embedding_c],
        start_seconds=0.0,
        end_seconds=5.0,
    )
    tracklet_2 = PersonTracklet(
        track_id="person_2",
        person_detections=[_person_detection(10.0)],
        face_detections=[_face_detection(10.0, embedding_b)],
        aggregated_embedding=embedding_b,
        exemplar_embeddings=[embedding_a, embedding_b, embedding_c],
        start_seconds=10.0,
        end_seconds=15.0,
    )

    clusters = TrackletClusterer().cluster([tracklet_1, tracklet_2])
    assert len(clusters) == 1


def test_different_people_with_single_outlier_pair_does_not_merge():
    """Over-merge guard. Two unrelated identities, but one exemplar pair
    happens to be similar (the classic Phase-5 false-merge cause). Top-K mean
    should dilute this single outlier with the other low-similarity pairs and
    keep them in separate clusters."""
    person_a_embeddings = [
        [1.0, 0.0, 0.0],
        [0.95, 0.1, 0.0],
        [0.9, 0.2, 0.0],
    ]
    person_b_embeddings = [
        [0.6, 0.8, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.9, 0.1],
    ]

    tracklet_a = PersonTracklet(
        track_id="person_a",
        person_detections=[_person_detection(0.0)],
        face_detections=[_face_detection(0.0, person_a_embeddings[0])],
        aggregated_embedding=person_a_embeddings[0],
        exemplar_embeddings=person_a_embeddings,
        start_seconds=0.0,
        end_seconds=5.0,
    )
    tracklet_b = PersonTracklet(
        track_id="person_b",
        person_detections=[_person_detection(10.0)],
        face_detections=[_face_detection(10.0, person_b_embeddings[1])],
        aggregated_embedding=person_b_embeddings[1],
        exemplar_embeddings=person_b_embeddings,
        start_seconds=10.0,
        end_seconds=15.0,
    )

    clusters = TrackletClusterer().cluster([tracklet_a, tracklet_b])
    assert len(clusters) == 2, "single high-similarity outlier pair should not trigger merge"


def test_temporally_overlapping_tracklets_do_not_merge_even_if_embeddings_match():
    """Two different people on screen at the same time can have surprisingly
    high embedding similarity (similar features, similar lighting). The
    temporal-overlap veto blocks the merge regardless of similarity, since
    one person can't occupy two screen positions simultaneously."""
    same_embedding = [1.0, 0.0, 0.0]
    tracklet_1 = _tracklet("person_1", 0.0, 40.0, same_embedding)
    tracklet_2 = _tracklet("person_2", 10.0, 50.0, same_embedding)  # overlaps 10-40

    clusters = TrackletClusterer().cluster([tracklet_1, tracklet_2])
    assert len(clusters) == 2, "co-occurring tracklets must stay separate"


def test_centroid_backup_merges_when_median_dips_below_threshold():
    """Same person, but a wide pose mix makes most cross-pairs land at ~0.3,
    pulling the median below the primary threshold. The centroid-backup rule
    catches it because the per-set mean embeddings are still tightly aligned.
    Non-overlapping time spans, so the veto doesn't fire."""
    frontal = [1.0, 0.0, 0.0]
    profile_left = [0.3, 0.95, 0.0]
    profile_right = [0.3, -0.95, 0.0]

    tracklet_1 = PersonTracklet(
        track_id="t1",
        person_detections=[_person_detection(0.0)],
        face_detections=[_face_detection(0.0, frontal)],
        aggregated_embedding=frontal,
        exemplar_embeddings=[frontal, profile_left, profile_right],
        start_seconds=0.0,
        end_seconds=5.0,
    )
    tracklet_2 = PersonTracklet(
        track_id="t2",
        person_detections=[_person_detection(20.0)],
        face_detections=[_face_detection(20.0, frontal)],
        aggregated_embedding=frontal,
        exemplar_embeddings=[frontal, profile_left, profile_right],
        start_seconds=20.0,
        end_seconds=25.0,
    )

    clusterer = TrackletClusterer(
        similarity_threshold=0.50,
        centroid_merge_threshold=0.60,
    )
    clusters = clusterer.cluster([tracklet_1, tracklet_2])
    assert len(clusters) == 1, (
        "centroid backup should merge when per-set means agree even if the "
        "cross-pair median dips below the primary threshold"
    )


def test_short_overlap_is_treated_as_id_flip_and_allowed_to_merge():
    """ByteTrack briefly assigns two ids when a person re-emerges from
    occlusion: for ~a few frames the old and new tracks coexist. The veto
    must tolerate that short window so the same-person tracklets can
    stitch back together; a hard 'any overlap' check would split them
    forever even though their embeddings agree."""
    same_embedding = [1.0, 0.0, 0.0]
    # Overlap = 0.3s, well under the default 0.5s ID-flip tolerance.
    t1 = _tracklet("person_1", 0.0, 5.0, same_embedding)
    t2 = _tracklet("person_2", 4.7, 9.0, same_embedding)

    clusters = TrackletClusterer().cluster([t1, t2])
    assert len(clusters) == 1, "short ID-flip overlap should not block merge"


def test_long_overlap_still_vetoes_even_with_matching_embeddings():
    """Sanity guard against the soft-veto: a sustained co-occurrence is
    still two different people on screen at once, no matter how similar
    their embeddings happen to be. The hard part of the veto is the part
    we keep."""
    same_embedding = [1.0, 0.0, 0.0]
    t1 = _tracklet("person_1", 0.0, 10.0, same_embedding)
    t2 = _tracklet("person_2", 2.0, 12.0, same_embedding)  # overlap = 8s

    clusters = TrackletClusterer(min_overlap_seconds=0.5).cluster([t1, t2])
    assert len(clusters) == 2, "sustained overlap must keep tracklets separate"


def test_second_pass_keeps_distinct_identities_separate():
    """The 2-pass is meant to be an *extra* merging chance, not a relaxed
    one. Distinct identities that 1-pass correctly kept apart must still
    be kept apart after the pairwise re-evaluation — otherwise we'd
    happily merge two strangers just because nobody else stops us.

    Three person-A tracklets interleaved with one person-B tracklet.
    Expected: one cluster of size 3 (A) + one cluster of size 1 (B).
    """
    embedding_a = [1.0, 0.0, 0.0]
    embedding_b = [0.0, 1.0, 0.0]  # orthogonal — clearly different person

    tracklets = [
        _tracklet("a1", 0.0, 5.0, embedding_a),
        _tracklet("a2", 6.0, 11.0, embedding_a),
        _tracklet("b1", 12.0, 17.0, embedding_b),
        _tracklet("a3", 18.0, 23.0, embedding_a),
    ]
    clusters = TrackletClusterer().cluster(tracklets)

    assert len(clusters) == 2
    sizes = sorted(len(c.tracklets) for c in clusters)
    assert sizes == [1, 3]


def test_second_pass_merge_pair_directly_when_invoked():
    """Direct invocation of the 2-pass path proves it actually folds
    cluster pairs (not just a no-op wrapper). Construct two pending
    clusters that would clearly merge — same embedding, non-overlapping
    time spans — and feed them through ``_merge_clusters_pairwise``.
    """
    from app.services.face_analysis.tracklet_clusterer import _PendingCluster

    embedding = [1.0, 0.0, 0.0]
    t_left = _tracklet("left", 0.0, 5.0, embedding)
    t_right = _tracklet("right", 10.0, 15.0, embedding)

    left = _PendingCluster.from_tracklet("cluster_1", t_left)
    right = _PendingCluster.from_tracklet("cluster_2", t_right)

    merged = TrackletClusterer()._merge_clusters_pairwise([left, right])
    assert len(merged) == 1
    assert len(merged[0].tracklets) == 2


def test_second_pass_respects_overlap_veto():
    """2-pass shouldn't override the temporal veto: even if two cluster's
    exemplars line up perfectly, sustained on-screen co-occurrence still
    means two different people."""
    same_embedding = [1.0, 0.0, 0.0]
    # Two long-overlapping tracklets, identical embeddings → 1-pass keeps
    # them separate via the veto. 2-pass should respect the same veto and
    # not glue them.
    t1 = _tracklet("person_1", 0.0, 30.0, same_embedding)
    t2 = _tracklet("person_2", 5.0, 35.0, same_embedding)  # overlap 25s

    clusters = TrackletClusterer(min_overlap_seconds=0.5).cluster([t1, t2])
    assert len(clusters) == 2, "2-pass must not bypass the temporal veto"


def test_profile_and_frontal_same_person_still_merges_under_median():
    """Phase 5's original target case must keep working. Two tracklets of the
    same person, each carrying both a profile and a frontal exemplar. The
    cross pairs are: (frontal, frontal) high, (profile, profile) high,
    (frontal, profile) lower. Median stays well above threshold because half
    the pairs are same-pose matches."""
    frontal = [1.0, 0.0, 0.0]
    profile = [0.3, 0.95, 0.0]

    tracklet_1 = PersonTracklet(
        track_id="t1",
        person_detections=[_person_detection(0.0)],
        face_detections=[_face_detection(0.0, frontal)],
        aggregated_embedding=frontal,
        exemplar_embeddings=[frontal, profile],
        start_seconds=0.0,
        end_seconds=5.0,
    )
    tracklet_2 = PersonTracklet(
        track_id="t2",
        person_detections=[_person_detection(10.0)],
        face_detections=[_face_detection(10.0, profile)],
        aggregated_embedding=profile,
        exemplar_embeddings=[frontal, profile],
        start_seconds=10.0,
        end_seconds=15.0,
    )

    clusters = TrackletClusterer().cluster([tracklet_1, tracklet_2])
    assert len(clusters) == 1


