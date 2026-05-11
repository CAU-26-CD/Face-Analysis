from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.face_analysis.analyzer import FaceVideoAnalyzer
from app.services.face_analysis.clustering import FaceClusterer
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
        clusterer=FaceClusterer(similarity_threshold=0.45),
    )

    clustered_count = 0
    for clustered_detection in analyzer.cluster_faces(video_path):
        detection = clustered_detection.detection
        print(
            detection.timestamp_seconds,
            clustered_detection.person_id,
            detection.confidence,
            detection.bbox,
        )

        clustered_count += 1
        if clustered_count >= 20:
            break

    print("clusters", analyzer.clusterer.get_cluster_count())
    print("clustered_detections", clustered_count)


if __name__ == "__main__":
    main()
