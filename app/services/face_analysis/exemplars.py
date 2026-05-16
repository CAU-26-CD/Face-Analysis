"""Quality scoring + top-K selection for face detections.

Shared between identity matching (analyzer → tracklet_clusterer → actor_matcher)
and thumbnail rendering. Both want a handful of *representative* face crops
per cluster instead of a single centroid, since the centroid frame is often
hair / profile / occlusion when most observations are.
"""

from math import sqrt

from app.services.face_analysis.models import FaceDetection


def quality_score(detection: FaceDetection) -> float:
    """Higher = more identity-reliable. ``confidence × sqrt(bbox_area)``.

    Combines detector confidence (rejects soft hits) with face size
    (small faces have noisy embeddings).
    """
    x1, y1, x2, y2 = detection.bbox
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return detection.confidence * sqrt(area)


def select_top_k_diverse(
    detections: list[FaceDetection], k: int
) -> list[FaceDetection]:
    """Pick up to ``k`` detections, splitting the time range into ``k`` even
    buckets and taking the highest-quality detection from each.

    This guarantees temporal spread (profile, frontal, side all represented
    in a long-running tracklet) instead of picking ``k`` near-duplicate frames
    that happen to all have high quality.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if not detections:
        return []

    ordered = sorted(detections, key=lambda d: d.timestamp_seconds)
    buckets = _bucket_by_time(ordered, k)
    picks: list[FaceDetection] = []
    for bucket in buckets:
        if not bucket:
            continue
        picks.append(max(bucket, key=quality_score))
    picks.sort(key=lambda d: d.timestamp_seconds)
    return picks


def _bucket_by_time(
    detections: list[FaceDetection], bucket_count: int
) -> list[list[FaceDetection]]:
    if not detections:
        return []
    start = detections[0].timestamp_seconds
    end = detections[-1].timestamp_seconds
    span = end - start
    if span <= 0 or bucket_count == 1:
        return [list(detections)]
    bucket_size = span / bucket_count
    buckets: list[list[FaceDetection]] = [[] for _ in range(bucket_count)]
    for detection in detections:
        offset = detection.timestamp_seconds - start
        index = min(bucket_count - 1, int(offset / bucket_size))
        buckets[index].append(detection)
    return buckets
