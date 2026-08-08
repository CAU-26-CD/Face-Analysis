import logging
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.services.face_analysis.actor_matcher import ActorMatcher
from app.services.face_analysis.detector import InsightFaceDetector
from app.services.face_analysis.exemplars import select_top_k_diverse
from app.services.face_analysis.face_person_associator import FacePersonAssociator
from app.services.face_analysis.models import (
    FaceAnalysisResult,
    FaceAppearance,
    FaceDetection,
    KnownActor,
    MatchResult,
    PersonDetection,
    PersonTracklet,
    TrackedFaceDetection,
    TrackedPerson,
    VideoFrame,
    WithinVideoCluster,
)
from app.services.face_analysis.exemplars import quality_score
from app.services.face_analysis.person_detector import PersonDetector
from app.services.face_analysis.person_tracker import PersonTracker
from app.services.face_analysis.thumbnails import ClusterThumbnailExtractor, crop_padded
from app.services.face_analysis.timeline import AppearanceTimelineBuilder
from app.services.face_analysis.tracklet_clusterer import TrackletClusterer
from app.services.face_analysis.video_reader import VideoFrameReader

logger = logging.getLogger(__name__)

# Per-track cap on cached BGR crops kept for the thumbnail step. Crops are
# evicted by lowest quality_score once the cap is hit, so a long tracklet
# only keeps the K most thumbnail-worthy frames. Cap × tracks × ~100 KB is
# the memory footprint, so this trades a small bound for skipping the
# end-of-job video reopen + keyframe seek.
MAX_CROPS_PER_TRACK = 20

# Per-track cap on face observations. Once every currently-visible track has
# this many face detections, we skip the InsightFace pass on that frame —
# detection runs at fixed 640×640 regardless of the input, so on long single-
# person tracks the model burns most of its frames recomputing the same face.
# 50 leaves comfortable headroom over the 10 exemplars we keep per cluster
# and over the ~30 samples it takes the mean embedding to stabilize.
MAX_FACE_OBS_PER_TRACK = 50

# How many sampled frames to push through YOLO at once. Single-frame predict
# leaves the GPU idle between kernel launches; batches of 8 keep utilization
# high without blowing GPU memory on 1080p/4K inputs. Tracker still consumes
# frames in order — only the YOLO call is batched.
YOLO_BATCH_SIZE = 8


@dataclass
class _TrackBuffer:
    track_id: str
    person_detections: list[PersonDetection] = field(default_factory=list)
    face_detections: list[FaceDetection] = field(default_factory=list)
    cropped_faces: list[tuple[FaceDetection, Any]] = field(default_factory=list)


class FaceVideoAnalyzer:
    """Per-video orchestrator.

    Pipeline (per sampled frame):
        person_detector  → person bboxes
        person_tracker   → stable person_track_id (IoU + Kalman, ByteTrack)
        face_detector    → face bbox + 512-d embedding
        face_associator  → bind each face to a person_track_id

    Per-track face embeddings are aggregated into a PersonTracklet, then:
        tracklet_clusterer → merge tracks split by long occlusions
        actor_matcher      → resolve clusters against the known actor gallery
        timeline_builder   → flatten observations into appearance intervals
    """

    def __init__(
        self,
        frame_reader: VideoFrameReader | None = None,
        person_detector: PersonDetector | None = None,
        person_tracker: PersonTracker | None = None,
        face_detector: InsightFaceDetector | None = None,
        face_associator: FacePersonAssociator | None = None,
        tracklet_clusterer: TrackletClusterer | None = None,
        actor_matcher: ActorMatcher | None = None,
        timeline_builder: AppearanceTimelineBuilder | None = None,
        thumbnail_extractor: ClusterThumbnailExtractor | None = None,
        # Default bumped from 5 → 10: a thicker exemplar pool per tracklet
        # gives both the 1-pass clusterer and the 2-pass merge more pose
        # coverage to score on, so frontal/profile splits stitch back
        # together more often without false merges.
        exemplars_per_tracklet: int = 10,
        device: str | None = None,
        onnx_providers: list[str] | None = None,
    ):
        if exemplars_per_tracklet < 1:
            raise ValueError("exemplars_per_tracklet must be >= 1")
        self.frame_reader = frame_reader or VideoFrameReader()
        self.person_detector = person_detector or PersonDetector(device=device)

        self.person_tracker = person_tracker or PersonTracker()
        self.face_detector = face_detector or InsightFaceDetector(
            providers=onnx_providers,
        )
        self.face_associator = face_associator or FacePersonAssociator()
        self.tracklet_clusterer = tracklet_clusterer or TrackletClusterer()
        self.actor_matcher = actor_matcher or ActorMatcher()
        self.timeline_builder = timeline_builder or AppearanceTimelineBuilder()
        self.thumbnail_extractor = thumbnail_extractor or ClusterThumbnailExtractor()
        self.exemplars_per_tracklet = exemplars_per_tracklet

    def warmup(self) -> None:
        """Force both detectors to load weights and build their CUDA context.

        Constructing this class loads nothing: YOLO and InsightFace both defer
        weight loading to their first ``detect`` call. So the RunPod warmup
        request used to return "ok" while the models were still cold, and the
        ~7s of session build + CUDA context creation landed inside the first
        real job instead. Pushing one synthetic frame through each moves that
        cost onto the warmup request, which is exactly what BE fires it for.
        """
        blank = np.zeros((640, 640, 3), dtype=np.uint8)
        frame = VideoFrame(timestamp_seconds=0.0, frame_index=0, frame=blank)
        start = time.monotonic()
        try:
            self.person_detector.detect_batch([frame])
            self.face_detector.detect(frame)
        except Exception:
            # A cold worker that can't warm up can still serve jobs (it just
            # pays the load cost inline), so never let this kill the process.
            logger.exception("Model warmup failed; models will load lazily")
            return
        logger.info("Model warmup complete in %.1fs", time.monotonic() - start)

    def read_sampled_frames(self, video_path: str | Path) -> Iterator[VideoFrame]:
        return self.frame_reader.read_frames(video_path)

    def detect_persons(self, video_frame: VideoFrame) -> list[PersonDetection]:
        return self.person_detector.detect(video_frame)

    def detect_faces(self, video_frame: VideoFrame) -> list[FaceDetection]:
        return self.face_detector.detect(video_frame)

    def analyze(
        self,
        video_path: str | Path,
        known_actors: list[KnownActor] | None = None,
        progress_callback: Callable[[int, float, int, int], None] | None = None,
        thumbnail_dir: str | Path | None = None,
    ) -> FaceAnalysisResult:
        path = Path(video_path)
        known_actors = known_actors or []

        self.person_tracker.reset()
        track_buffers: dict[str, _TrackBuffer] = {}
        face_obs_count: dict[str, int] = {}

        want_crops = thumbnail_dir is not None
        padding_ratio = self.thumbnail_extractor.padding_ratio if want_crops else 0.0

        # Decode vs. inference are interleaved (the reader is a generator
        # consumed by this loop), so wall-clock alone can't attribute time.
        # _timed_frames charges time spent pulling frames to "decode"; the
        # explicit start/stop pairs below charge the model calls.
        analysis_start = time.monotonic()
        stage_seconds = {"decode": 0.0, "person_detect": 0.0, "face_detect": 0.0}
        frame_shape: tuple[int, int] | None = None

        sampled_frame_count = 0
        for frame_batch in _batched(
            _timed_frames(self.frame_reader.read_frames(path), stage_seconds),
            YOLO_BATCH_SIZE,
        ):
            if frame_shape is None:
                height, width = frame_batch[0].frame.shape[:2]
                frame_shape = (width, height)
            stage_start = time.monotonic()
            person_detections_per_frame = self.person_detector.detect_batch(
                frame_batch
            )
            stage_seconds["person_detect"] += time.monotonic() - stage_start
            for video_frame, person_detections in zip(
                frame_batch, person_detections_per_frame
            ):
                sampled_frame_count += 1
                tracked_persons = self.person_tracker.update(
                    video_frame, person_detections
                )

                # Only run InsightFace when at least one visible track still
                # needs more face observations. On a stable 1-person track this
                # cuts ~90% of face detection calls without hurting embedding or
                # exemplar quality.
                needs_face_pass = any(
                    face_obs_count.get(p.track_id, 0) < MAX_FACE_OBS_PER_TRACK
                    for p in tracked_persons
                )
                if needs_face_pass:
                    stage_start = time.monotonic()
                    face_detections = self.face_detector.detect(video_frame)
                    stage_seconds["face_detect"] += time.monotonic() - stage_start
                    tracked_faces = self.face_associator.associate(
                        face_detections, tracked_persons
                    )
                else:
                    face_detections = []
                    tracked_faces = []

                self._record_persons(tracked_persons, track_buffers)
                self._record_faces(
                    tracked_faces,
                    track_buffers,
                    frame=video_frame.frame if want_crops else None,
                    padding_ratio=padding_ratio,
                )
                for tracked_face in tracked_faces:
                    if tracked_face.person_track_id is not None:
                        face_obs_count[tracked_face.person_track_id] = (
                            face_obs_count.get(tracked_face.person_track_id, 0) + 1
                        )

                if progress_callback is not None:
                    progress_callback(
                        sampled_frame_count,
                        video_frame.timestamp_seconds,
                        len(tracked_persons),
                        len(face_detections),
                    )

        post_start = time.monotonic()
        tracklets = self._finalize_tracklets(track_buffers)
        clusters = self.tracklet_clusterer.cluster(tracklets)
        cluster_thumbnail_paths: dict[str, list[str]] = {}
        if thumbnail_dir is not None:
            cropped_faces_by_track = {
                track_id: buf.cropped_faces
                for track_id, buf in track_buffers.items()
            }
            saved = self.thumbnail_extractor.extract(
                clusters, thumbnail_dir, cropped_faces_by_track
            )
            cluster_thumbnail_paths = {
                cluster_id: [str(path) for path in paths]
                for cluster_id, paths in saved.items()
            }
        match_result = self.actor_matcher.match(clusters, known_actors)
        appearances = self._build_appearances(clusters, match_result)

        total_seconds = time.monotonic() - analysis_start
        sample_interval = getattr(
            self.frame_reader, "frame_interval_seconds", None
        )
        width, height = frame_shape or (0, 0)
        logger.info(
            "Analysis timings for %s: decode=%.1fs person_detect=%.1fs "
            "face_detect+embed=%.1fs post=%.1fs total=%.1fs | "
            "sampled_frames=%d sample_interval=%ss (~%s fps) resolution=%dx%d",
            path.name,
            stage_seconds["decode"],
            stage_seconds["person_detect"],
            stage_seconds["face_detect"],
            time.monotonic() - post_start,
            total_seconds,
            sampled_frame_count,
            sample_interval if sample_interval is not None else "?",
            f"{1.0 / sample_interval:.1f}" if sample_interval else "?",
            width,
            height,
        )

        return FaceAnalysisResult(
            video_path=str(path),
            appearances=appearances,
            new_candidates=match_result.new_candidates,
            matched=match_result.matched,
            cluster_thumbnail_paths=cluster_thumbnail_paths,
        )

    @staticmethod
    def _record_persons(
        tracked_persons: list[TrackedPerson],
        track_buffers: dict[str, _TrackBuffer],
    ) -> None:
        for tracked in tracked_persons:
            buffer = track_buffers.setdefault(
                tracked.track_id,
                _TrackBuffer(track_id=tracked.track_id),
            )
            buffer.person_detections.append(tracked.detection)

    @staticmethod
    def _record_faces(
        tracked_faces: list[TrackedFaceDetection],
        track_buffers: dict[str, _TrackBuffer],
        frame: Any | None = None,
        padding_ratio: float = 0.0,
    ) -> None:
        for tracked in tracked_faces:
            if tracked.person_track_id is None:
                continue
            buffer = track_buffers.get(tracked.person_track_id)
            if buffer is None:
                continue
            buffer.face_detections.append(tracked.detection)

            if frame is None:
                continue
            crop = crop_padded(frame, tracked.detection.bbox, padding_ratio)
            if crop is None:
                continue
            buffer.cropped_faces.append((tracked.detection, crop))
            if len(buffer.cropped_faces) > MAX_CROPS_PER_TRACK:
                worst_index = min(
                    range(len(buffer.cropped_faces)),
                    key=lambda i: quality_score(buffer.cropped_faces[i][0]),
                )
                buffer.cropped_faces.pop(worst_index)

    def _finalize_tracklets(
        self,
        track_buffers: dict[str, _TrackBuffer],
    ) -> list[PersonTracklet]:
        tracklets: list[PersonTracklet] = []
        for buffer in track_buffers.values():
            if not buffer.person_detections:
                continue
            aggregated = _mean_embedding(
                [face.embedding for face in buffer.face_detections]
            )
            exemplars = [
                list(detection.embedding)
                for detection in select_top_k_diverse(
                    buffer.face_detections, self.exemplars_per_tracklet
                )
            ]
            tracklets.append(
                PersonTracklet(
                    track_id=buffer.track_id,
                    person_detections=list(buffer.person_detections),
                    face_detections=list(buffer.face_detections),
                    aggregated_embedding=aggregated,
                    exemplar_embeddings=exemplars,
                    start_seconds=buffer.person_detections[0].timestamp_seconds,
                    end_seconds=buffer.person_detections[-1].timestamp_seconds,
                )
            )
        return tracklets

    def _build_appearances(
        self,
        clusters: list[WithinVideoCluster],
        match_result: MatchResult,
    ) -> list[FaceAppearance]:
        # actor_id is int but cluster_id is str — coerce so the timeline
        # builder's sort key sees a uniform type. worker._build_callback_payload
        # uses the new_cluster→temp_index map to disambiguate "actor:" vs
        # "new:" anyway, so the stringified actor id here is just an opaque
        # key, not user-facing data.
        person_id_by_cluster: dict[str, str] = {
            matched.cluster_id: str(matched.actor_id)
            for matched in match_result.matched
        }
        for candidate in match_result.new_candidates:
            person_id_by_cluster[candidate.cluster_id] = candidate.cluster_id

        labeled_timestamps: list[tuple[str, float]] = []
        for cluster in clusters:
            person_id = person_id_by_cluster[cluster.cluster_id]
            for tracklet in cluster.tracklets:
                for person_detection in tracklet.person_detections:
                    labeled_timestamps.append((person_id, person_detection.timestamp_seconds))

        labeled_timestamps.sort(key=lambda pair: pair[1])

        timeline_builder = AppearanceTimelineBuilder(
            max_gap_seconds=self.timeline_builder.max_gap_seconds,
        )
        for person_id, timestamp in labeled_timestamps:
            timeline_builder.add(person_id, timestamp)
        return timeline_builder.build()


def _timed_frames(
    frames: Iterator[VideoFrame], stage_seconds: dict[str, float]
) -> Iterator[VideoFrame]:
    """Charge time spent pulling frames from the reader to ``decode``.

    The reader generator does its decord/OpenCV work lazily inside ``next()``,
    so this wrapper is the only place decode time is observable separately
    from the inference calls interleaved with it.
    """
    iterator = iter(frames)
    while True:
        start = time.monotonic()
        try:
            frame = next(iterator)
        except StopIteration:
            stage_seconds["decode"] += time.monotonic() - start
            return
        stage_seconds["decode"] += time.monotonic() - start
        yield frame


def _batched(
    iterable: Iterator[VideoFrame], size: int
) -> Iterator[list[VideoFrame]]:
    """Yield consecutive ``size``-element chunks from ``iterable``.

    Stdlib ``itertools.batched`` is 3.12+ only, and we want to keep working
    on 3.11 in dev. The last batch may be shorter.
    """
    if size < 1:
        raise ValueError("size must be >= 1")
    batch: list[VideoFrame] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _mean_embedding(embeddings: list[list[float]]) -> list[float]:
    if not embeddings:
        return []
    dim = len(embeddings[0])
    if any(len(e) != dim for e in embeddings):
        raise ValueError("all embeddings must have the same dimension")
    matrix = np.asarray(embeddings, dtype=np.float64)
    return matrix.mean(axis=0).tolist()
