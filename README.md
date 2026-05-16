# Face Analyzer

Standalone FastAPI service for analyzing faces in videos stored on S3.

This repository does not connect to the BE database. The BE server owns upload,
video metadata, `analysis_status`, persistence, and the **project actor gallery**.
This analyzer only receives an S3 video reference (plus the project's known
actors), runs OpenCV / insightFace / onnxruntime analysis, and POSTs the result
back to the BE callback URL.

## Pipeline

```
S3 download
  → VideoFrameReader            (10 fps sampling)
  → for each frame:
      PersonDetector            (YOLOv11s, COCO person class)
      PersonTracker             (Ultralytics ByteTrack — IoU + Kalman)
      InsightFaceDetector       (face bbox + 512-d embedding)
      FacePersonAssociator      (bind faces to person_track_id)
  → PersonTracklets             (per track: top-K exemplar embeddings)
  → TrackletClusterer           (median-pair similarity → cluster_id)
  → ActorMatcher                (cross-video match vs known_actors)
  → AppearanceTimelineBuilder
  → callback to BE
```

Tracking is on the **person bbox**, not the face. A person whose face turns
to profile or is briefly occluded by hair keeps the same track_id as long as
their body bbox is detected; face identity is layered on top of the body
track instead of being the tracking signal itself.

Cluster matching uses **median** of all cross-pair cosine similarities
between two exemplar sets — same-person pairs are mostly high (median high),
while different-people pairs are mostly low with a few outliers (median
stays low). Default thresholds: clusterer `0.40`, matcher `match 0.50 /
suggest 0.40`.

## API

### POST `/analyze`

Starts a background analysis job and returns immediately.

Request:

```json
{
  "video_id": 1,
  "session_id": 7,
  "s3_key": "videos/7/abc123.webm",
  "s3_url": "https://bucket.s3.ap-northeast-2.amazonaws.com/videos/7/abc123.webm",
  "callback_url": "https://BE_SERVER/api/v1/videos/analysis-callback",
  "known_actors": [
    { "actor_id": "actor_1", "face_template": [0.12, -0.03, ...] }
  ]
}
```

`known_actors` is optional (defaults to `[]`). Pass the project's labeled
actors and their stored face templates so the analyzer can resolve each
within-video cluster to an existing identity.

Immediate response:

```json
{
  "video_id": 1,
  "analysis_status": "accepted"
}
```

Success callback:

```json
{
  "video_id": 1,
  "analysis_status": "done",
  "analysis_result": {
    "video_path": "...",
    "appearances": [
      {
        "person_id": "actor_1",
        "start_seconds": 15.0,
        "end_seconds": 20.0,
        "detection_count": 3
      },
      {
        "person_id": "cluster_2",
        "start_seconds": 30.0,
        "end_seconds": 42.0,
        "detection_count": 8
      }
    ],
    "new_candidates": [
      {
        "cluster_id": "cluster_2",
        "embedding": [0.04, 0.21, ...],
        "detection_count": 8,
        "start_seconds": 30.0,
        "end_seconds": 42.0,
        "suggested_actor_id": null,
        "suggested_similarity": null
      }
    ]
  }
}
```

- `appearances[].person_id` is either a known `actor_id` (matched) or a
  `cluster_X` temp id for newly discovered people.
- `new_candidates` lists every unmatched cluster with its embedding so BE/UI
  can prompt the user to label or merge.
- `suggested_actor_id` is non-null when similarity ≥ suggest threshold but
  below the confident match threshold — surface as "is this <actor>?".

Failure callback:

```json
{
  "video_id": 1,
  "analysis_status": "failed",
  "error_message": "..."
}
```

Every callback includes:

```http
X-Analyzer-Secret: {ANALYZER_SECRET}
```

## Environment

```bash
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=ap-northeast-2
S3_BUCKET_NAME=your-video-bucket
ANALYZER_SECRET=change-me
```

`CALLBACK_SECRET` is also accepted as a fallback for `ANALYZER_SECRET`.

If `S3_BUCKET_NAME` is set, the analyzer downloads with boto3 using `s3_key`.
If it is not set, it falls back to downloading `s3_url` directly.

## Run Locally

```bash
poetry install
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Docs: `http://127.0.0.1:8000/docs`

## Docker

```bash
docker build -t face-analyzer .
docker run --rm -p 8000:8000 --env-file .env face-analyzer
```

## Testing

### Prerequisites
- Poetry installed
- AWS S3 access (GetObject on the target bucket)
- `.env` populated (see Environment above)

### 1. Unit tests (no external deps, ~1s)

```bash
poetry run pytest tests/ -v
```

### 2. End-to-end against real S3 + mock BE

Three terminals.

**Terminal 1 — Mock BE callback receiver**

```bash
poetry run python -c '
from fastapi import FastAPI, Request
import uvicorn, json, pathlib
app = FastAPI()
def trim(o):
    if isinstance(o, dict):
        return {k: ("<%d floats>" % len(v)) if k in ("embedding","face_template") and isinstance(v, list) else trim(v) for k, v in o.items()}
    if isinstance(o, list): return [trim(x) for x in o]
    return o
@app.post("/cb")
async def cb(r: Request):
    body = await r.json()
    pathlib.Path("/tmp/last_callback.json").write_text(json.dumps(body))
    ar = body.get("analysis_result", {})
    print("saved /tmp/last_callback.json | appearances:", len(ar.get("appearances",[])), "new_candidates:", len(ar.get("new_candidates",[])))
    print(json.dumps(trim(body), ensure_ascii=False, indent=2))
    return {"ok": True}
uvicorn.run(app, host="0.0.0.0", port=9000)
'
```

**Terminal 2 — Analyzer service**

```bash
set -a; source .env; set +a
echo $S3_BUCKET_NAME   # should print your bucket name
poetry run uvicorn app.main:app --reload --port 8000
```

**Terminal 3 — Trigger analysis**

```bash
cat > /tmp/req.json <<'EOF'
{
  "video_id": 1,
  "session_id": 2,
  "s3_key": "<S3 object key>",
  "s3_url": "<S3 object URL>",
  "callback_url": "http://localhost:9000/cb",
  "known_actors": []
}
EOF

curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d @/tmp/req.json
```

Expected:
- Immediate `{"video_id":1,"analysis_status":"accepted"}` response
- Terminal 2: insightface logs + frame processing
  - First run downloads the `buffalo_l` model (~300MB, cached afterward)
- Terminal 1: `saved /tmp/last_callback.json` + trimmed result summary

### 3. Two-pass labeling flow (gallery UX verification)

Verifies that labeling a discovered cluster makes the next analysis call
resolve it to the known `actor_id` instead of surfacing it as a new
candidate.

After step 2 completes:

```bash
poetry run python -c 'exec("""
import json, httpx

with open(\"/tmp/last_callback.json\") as f:
    data = json.load(f)

top = max(data[\"analysis_result\"][\"new_candidates\"], key=lambda c: c[\"detection_count\"])
print(\"labeling\", top[\"cluster_id\"], \"as actor_kim (n=\" + str(top[\"detection_count\"]) + \")\")

r = httpx.post(\"http://127.0.0.1:8000/analyze\", json={
    \"video_id\": 2, \"session_id\": 2,
    \"s3_key\": \"<S3 object key>\",
    \"s3_url\": \"<S3 object URL>\",
    \"callback_url\": \"http://localhost:9000/cb\",
    \"known_actors\": [{\"actor_id\": \"actor_kim\", \"face_template\": top[\"embedding\"]}],
})
print(r.status_code, r.json())
""")'
```

Pass criteria:
- The newly labeled cluster's intervals in `appearances` switch from
  `cluster_X` to `actor_kim`.
- That cluster is removed from `new_candidates`.
- Other clusters remain as candidates.

### Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `NoCredentialsError` | Forgot `set -a; source .env; set +a` in the analyzer shell |
| `ModuleNotFoundError: s3transfer` | Run `poetry install` to pull transitive deps |
| `422 Unprocessable Entity` | Malformed JSON in curl — use the `/tmp/req.json` file approach |
| Callback never arrives | Check analyzer logs for `ANALYZER_SECRET is required` (env not loaded) |
| S3 `403 / 404` | s3_key typo or missing GetObject permission |
| `Could not open video file` | Codec mismatch — most webm files work on macOS by default |

## BE Contract

The BE server should:

1. Upload the video to S3 and store `s3_key` / `s3_url`.
2. Call analyzer `POST /analyze` with the project's `known_actors`.
3. Receive the callback and validate `X-Analyzer-Secret`.
4. Update `videos.analysis_status` and `videos.analysis_result`.
5. Surface `new_candidates` to the user for labeling, then persist the
   resulting `(actor_id, face_template)` rows into the project gallery so
   they are passed back as `known_actors` on subsequent `/analyze` calls.

## Migration notes (MOT redesign)

The analyzer's tracking layer was rewritten from a face-only embedding
tracker (sampling at 1 fps) into a proper MOT pipeline: YOLO person
detection + ByteTrack on person bboxes + face-to-person association at
10 fps. Things BE-side teams should be aware of:

- **Callback JSON shape is unchanged.** `appearances`, `new_candidates`,
  `embedding`, `known_actors[].face_template` — all the same. No BE code
  changes required.
- **`detection_count` semantics shifted.** It now counts *person bbox
  observations*, not face detections, so values are typically 5–10× larger
  than before. If any BE/UI logic uses `detection_count` as a confidence
  threshold (e.g. "drop candidates with n < 10"), rescale accordingly.
- **First run downloads YOLO weights.** Ultralytics fetches `yolo11s.pt`
  (~18 MB) into the working directory the first time `PersonDetector`
  initializes. Deploys without internet should pre-bundle the file or set
  `model_name` to a local path.
- **CPU vs GPU.** 10 fps × YOLO + InsightFace is CPU-bound; expect roughly
  real-time (or slower) on Apple Silicon CPU, much faster with MPS/CUDA.
  `PersonDetector(device="mps")` / `device="cuda"` to enable.
