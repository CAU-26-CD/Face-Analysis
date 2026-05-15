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


def test_max_exemplar_similarity_merges_profile_and_frontal_tracklets():
    """Reproduces the over-split case from real videos: a profile-only
    tracklet's centroid is far from a frontal tracklet's centroid, but each
    tracklet retains a high-quality exemplar that matches the other strongly.
    Multi-exemplar (max-pair) similarity should merge them."""
    frontal_embedding = [1.0, 0.0, 0.0]
    profile_embedding = [0.2, 0.98, 0.0]

    frontal_tracklet = PersonTracklet(
        track_id="frontal",
        person_detections=[_person_detection(0.0)],
        face_detections=[_face_detection(0.0, frontal_embedding)],
        aggregated_embedding=frontal_embedding,
        exemplar_embeddings=[frontal_embedding, profile_embedding],
        start_seconds=0.0,
        end_seconds=5.0,
    )
    profile_tracklet = PersonTracklet(
        track_id="profile",
        person_detections=[_person_detection(10.0)],
        face_detections=[_face_detection(10.0, profile_embedding)],
        aggregated_embedding=profile_embedding,
        exemplar_embeddings=[profile_embedding, frontal_embedding],
        start_seconds=10.0,
        end_seconds=15.0,
    )

    clusterer = TrackletClusterer(similarity_threshold=0.6)
    clusters = clusterer.cluster([frontal_tracklet, profile_tracklet])

    assert len(clusters) == 1, "profile + frontal exemplars should bridge the merge"
