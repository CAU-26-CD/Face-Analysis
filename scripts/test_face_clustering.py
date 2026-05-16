from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.face_analysis.analyzer import FaceVideoAnalyzer
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
        frame_reader=VideoFrameReader(frame_interval_seconds=0.1),
        tracklet_clusterer=TrackletClusterer(similarity_threshold=0.40),
    )

    def progress(frames: int, t: float, persons: int, faces: int) -> None:
        if frames == 1 or frames % 50 == 0:
            print(
                f"frame={frames} t={t:.2f}s persons={persons} faces={faces}",
                file=sys.stderr,
            )

    result = analyzer.analyze(video_path, progress_callback=progress)

    print()
    print(
        f"appearances={len(result.appearances)} "
        f"new_candidates={len(result.new_candidates)}"
    )
    for candidate in result.new_candidates:
        print(
            candidate.cluster_id,
            f"detections={candidate.detection_count}",
            f"span=({candidate.start_seconds:.1f}s, {candidate.end_seconds:.1f}s)",
        )


if __name__ == "__main__":
    main()
