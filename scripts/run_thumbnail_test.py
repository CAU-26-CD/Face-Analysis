"""One-shot analyzer run that writes cluster thumbnails to disk.

Used to eyeball whether the within-video clusterer is over-splitting or
under-splitting on a specific real video. Prints a one-line summary per
cluster plus the path of every thumbnail it dropped on disk.
"""

import sys
import time
from pathlib import Path

from app.services.face_analysis.analyzer import FaceVideoAnalyzer

VIDEO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("uploads/test_runthrough.mov")
THUMB_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("uploads/thumbs_runthrough")


def _progress(idx: int, ts: float, persons: int, faces: int) -> None:
    # Heartbeat every 50 sampled frames so a long video doesn't look hung.
    if idx == 1 or idx % 50 == 0:
        print(f"  frame {idx:>4} @ {ts:6.1f}s  persons={persons} faces={faces}", flush=True)


def main() -> int:
    if not VIDEO.exists():
        print(f"video not found: {VIDEO}", file=sys.stderr)
        return 2

    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    print(f"analyzing {VIDEO}")
    print(f"thumbnails → {THUMB_DIR}\n")

    started = time.monotonic()
    analyzer = FaceVideoAnalyzer()
    result = analyzer.analyze(
        VIDEO,
        known_actors=[],
        thumbnail_dir=THUMB_DIR,
        progress_callback=_progress,
    )
    elapsed = time.monotonic() - started

    print(f"\n=== done in {elapsed:.1f}s — {len(result.new_candidates)} cluster(s) ===")
    for c in result.new_candidates:
        print(
            f"  {c.cluster_id}: "
            f"time={c.start_seconds:6.2f}–{c.end_seconds:6.2f}s "
            f"observations={c.detection_count:>4} "
            f"exemplars={len(c.embeddings)}"
        )

    print("\nthumbnails:")
    for cid, paths in result.cluster_thumbnail_paths.items():
        for p in paths:
            print(f"  {cid}: {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
