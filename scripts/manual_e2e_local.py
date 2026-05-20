"""Local E2E sanity check for the multi-template contract.

Runs FaceVideoAnalyzer twice on a local video:
  1) known_actors=[]  → everything lands in new_candidates.
  2) Re-seed the most-detected candidate as a known actor, run again, and
     confirm it now shows up under matched[] with the new payload shape
     (face_embeddings, new_exemplars, integer actor_id).

No S3, no callback server. Bypasses worker._analyze_video and calls
_build_callback_payload directly so we test the same payload BE would
receive.
"""

import json
import sys
from pathlib import Path

from app.services.face_analysis.analyzer import FaceVideoAnalyzer
from app.services.face_analysis.models import KnownActor
from app.services.face_analysis.worker import _build_callback_payload


VIDEO_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("uploads/session_1.webm")


def _summarize(payload: dict) -> dict:
    """Replace bulky embedding arrays with <N vectors of D-d> labels so the
    printed payload is human-skimmable."""
    def walk(obj):
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k in ("face_templates", "face_embeddings", "new_exemplars") \
                   and isinstance(v, list) and v and isinstance(v[0], list):
                    out[k] = f"<{len(v)} vectors of {len(v[0])}-d>"
                else:
                    out[k] = walk(v)
            return out
        if isinstance(obj, list):
            return [walk(x) for x in obj]
        return obj

    return walk(payload)


def _check(name: str, condition: bool, detail: str = "") -> bool:
    mark = "OK " if condition else "FAIL"
    print(f"  [{mark}] {name}{(' — ' + detail) if detail else ''}")
    return condition


def main() -> int:
    if not VIDEO_PATH.exists():
        print(f"video not found: {VIDEO_PATH}", file=sys.stderr)
        return 2

    print(f"=== Pass 1: cold start (known_actors=[]) on {VIDEO_PATH} ===")
    analyzer = FaceVideoAnalyzer()
    result1 = analyzer.analyze(VIDEO_PATH, known_actors=[])
    payload1 = _build_callback_payload(
        video_id=1, analysis=result1, thumbnail_s3_keys={}
    )

    print(json.dumps(_summarize(payload1), ensure_ascii=False, indent=2))
    print()
    print("Contract checks (pass 1):")
    all_ok = True
    all_ok &= _check(
        "matched is empty",
        payload1["matched"] == [],
    )
    all_ok &= _check(
        "new_candidates carry face_embeddings (plural) only",
        all(
            "face_embeddings" in c and "face_embedding" not in c
            for c in payload1["new_candidates"]
        ),
    )
    all_ok &= _check(
        "every face_embeddings entry is list[list[float]]",
        all(
            isinstance(c["face_embeddings"], list)
            and all(isinstance(v, list) for v in c["face_embeddings"])
            for c in payload1["new_candidates"]
        ),
        f"{len(payload1['new_candidates'])} candidates",
    )

    if not result1.new_candidates:
        print("\nno new_candidates — can't test pass 2. Try a longer video.")
        return 0 if all_ok else 1

    top = max(result1.new_candidates, key=lambda c: c.detection_count)
    print(
        f"\n=== Pass 2: re-seed top cluster (detection_count={top.detection_count}, "
        f"{len(top.embeddings)} exemplars) as actor_id=1 ==="
    )

    analyzer2 = FaceVideoAnalyzer()
    result2 = analyzer2.analyze(
        VIDEO_PATH,
        known_actors=[KnownActor(actor_id=1, face_templates=top.embeddings)],
    )
    payload2 = _build_callback_payload(
        video_id=2, analysis=result2, thumbnail_s3_keys={}
    )

    print(json.dumps(_summarize(payload2), ensure_ascii=False, indent=2))
    print()
    print("Contract checks (pass 2):")
    all_ok &= _check(
        "at least one matched entry",
        len(payload2["matched"]) >= 1,
        f"matched={len(payload2['matched'])}",
    )
    if payload2["matched"]:
        m0 = payload2["matched"][0]
        all_ok &= _check(
            "matched[0].actor_id is int and == 1",
            isinstance(m0["actor_id"], int) and m0["actor_id"] == 1,
            f"got {m0['actor_id']!r}",
        )
        all_ok &= _check(
            "matched[0].new_exemplars key present and is a list",
            "new_exemplars" in m0 and isinstance(m0["new_exemplars"], list),
        )
    all_ok &= _check(
        "appearances use actor:1 prefix for the matched cluster",
        any(
            ap["person_id"] == "actor:1"
            for ap in payload2["analysis_result"]["appearances"]
        ),
    )

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
