from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.face_analysis.analyzer import FaceVideoAnalyzer
from app.services.face_analysis.tracker import FaceTracker
from app.services.face_analysis.tracklet_clusterer import TrackletClusterer
from app.services.face_analysis.video_reader import VideoFrameReader


def main() -> None:
    if len(sys.argv) > 1:
        video_path = Path(sys.argv[1])
    else:
        video_path = PROJECT_ROOT / "uploads" / "session_7.webm"

    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / video_path

    analyzer = FaceVideoAnalyzer(
        frame_reader=VideoFrameReader(frame_interval_seconds=5.0),
        tracker=FaceTracker(similarity_threshold=0.50),
        tracklet_clusterer=TrackletClusterer(similarity_threshold=0.40),
    )

    tracker = FaceTracker(similarity_threshold=0.50)
    for video_frame in analyzer.read_sampled_frames(video_path):
        detections = analyzer.detector.detect(video_frame)
        tracked = tracker.update(video_frame, detections)
        for item in tracked:
            print(
                item.detection.timestamp_seconds,
                item.track_id,
                f"{item.detection.confidence:.3f}",
                item.detection.bbox,
            )

    tracklets = tracker.finalize()
    clusters = analyzer.tracklet_clusterer.cluster(tracklets)

    print()
    print(f"tracklets={len(tracklets)} within_video_clusters={len(clusters)}")
    for cluster in clusters:
        print(
            cluster.cluster_id,
            f"detections={cluster.detection_count}",
            f"span=({cluster.start_seconds:.1f}s, {cluster.end_seconds:.1f}s)",
        )


if __name__ == "__main__":
    main()
