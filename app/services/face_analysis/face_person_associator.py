from app.services.face_analysis.models import (
    FaceDetection,
    TrackedFaceDetection,
    TrackedPerson,
)


class FacePersonAssociator:
    """Binds each face detection to a tracked person in the same frame.

    A face is attached to a person when:
      1. the face bbox center lies inside the person bbox, AND
      2. at least ``min_face_overlap`` of the face bbox area is inside
         the person bbox (``intersection / face_area``).

    When multiple persons qualify, the one with the **highest IoU** with the
    face wins — that picks the tightest person bbox around the face when two
    persons overlap. Faces that match nobody return with
    ``person_track_id=None`` so callers can either log them or drop them.

    Note we deliberately do *not* use ``IoU(face, person)`` as the gating
    metric: a face is far smaller than a person bbox, so even a perfectly
    contained face has IoU around 0.05–0.10, which would force a very low
    (and meaningless) threshold.
    """

    def __init__(self, min_face_overlap: float = 0.7):
        if not 0.0 <= min_face_overlap <= 1.0:
            raise ValueError("min_face_overlap must be in [0, 1]")
        self.min_face_overlap = min_face_overlap

    def associate(
        self,
        faces: list[FaceDetection],
        persons: list[TrackedPerson],
    ) -> list[TrackedFaceDetection]:
        return [
            TrackedFaceDetection(
                detection=face,
                person_track_id=self._best_person_for_face(face, persons),
            )
            for face in faces
        ]

    def _best_person_for_face(
        self, face: FaceDetection, persons: list[TrackedPerson]
    ) -> str | None:
        face_center = _bbox_center(face.bbox)
        face_area = _bbox_area(face.bbox)
        if face_area <= 0.0:
            return None

        best_track_id: str | None = None
        best_iou = -1.0
        for person in persons:
            if not _point_inside(face_center, person.detection.bbox):
                continue
            intersection = _intersection_area(face.bbox, person.detection.bbox)
            face_overlap = intersection / face_area
            if face_overlap < self.min_face_overlap:
                continue
            iou = _iou_from_intersection(face.bbox, person.detection.bbox, intersection)
            if iou > best_iou:
                best_iou = iou
                best_track_id = person.track_id
        return best_track_id


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _point_inside(
    point: tuple[float, float], bbox: tuple[float, float, float, float]
) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def _intersection_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    return inter_w * inter_h


def _iou_from_intersection(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    intersection: float,
) -> float:
    union = _bbox_area(a) + _bbox_area(b) - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union
