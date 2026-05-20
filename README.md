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
  "s3_key": "42/7/video.webm",
  "s3_url": "https://bucket.s3.ap-northeast-2.amazonaws.com/42/7/video.webm",
  "callback_url": "https://BE_SERVER/api/v1/videos/analysis-callback",
  "known_actors": [
    { "actor_id": "actor_1", "face_template": [0.12, -0.03, ...] }
  ],
  "thumbnail_dir": "42/7/"
}
```

`known_actors` is optional (defaults to `[]`). Pass the project's labeled
actors and their stored face templates so the analyzer can resolve each
within-video cluster to an existing identity.

`thumbnail_dir` is the S3 key prefix BE wants thumbnails written under. The
analyzer appends `thumb-{idx}.jpg` so video and thumbnails end up in the
same per-session folder (e.g. `42/7/video.webm` + `42/7/thumb-0.jpg`,
`42/7/thumb-1.jpg`, ...). Optional for back-compat: older callers that omit
the field fall back to the flat `thumbnails/{video_id}/{idx}.jpg` layout.

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
  "matched": [
    {
      "actor_id": "actor_1",
      "thumbnail_s3_key": "42/7/thumb-0.jpg",
      "similarity": 0.82
    }
  ],
  "new_candidates": [
    {
      "temp_index": 0,
      "thumbnail_s3_key": "42/7/thumb-1.jpg",
      "face_embedding": [0.04, 0.21, ...],
      "detection_count": 8,
      "start_seconds": 30.0,
      "end_seconds": 42.0,
      "suggested_actor_id": null,
      "suggested_similarity": null
    }
  ],
  "analysis_result": {
    "video_path": "...",
    "appearances": [
      {
        "person_id": "actor:actor_1",
        "start_seconds": 15.0,
        "end_seconds": 20.0,
        "detection_count": 3
      },
      {
        "person_id": "new:0",
        "start_seconds": 30.0,
        "end_seconds": 42.0,
        "detection_count": 8
      }
    ]
  }
}
```

- `matched[]` lists clusters resolved to a known actor. The analyzer uploads
  the cluster's best thumbnail to S3 (`thumbnails/{video_id}/{idx}.jpg`) and
  returns the key — BE should keep the existing actor's thumbnail by default
  (decision: thumbnail is fixed at first sighting), but is free to compare
  and swap.
- `new_candidates[]` lists clusters with no confident match. `temp_index`
  is the within-video ordinal (0, 1, 2, ...) used to refer to the candidate
  from `appearances`. BE creates a new actor per entry and stores
  `face_embedding` as the actor's face template (no later averaging — the
  embedding is fixed at first creation per the agreed policy).
- `analysis_result.appearances[].person_id` uses one of two prefixes:
  `"actor:{actor_id}"` for matched clusters or `"new:{temp_index}"` for new
  candidates. BE substitutes `new:*` with the actual actor_id minted from
  the corresponding new_candidates entry before persisting.
- `suggested_actor_id` is non-null when similarity ≥ suggest threshold but
  below the confident match threshold — surface as "is this <actor>?". The
  agreed MVP UX skips this chip and lets the user merge manually instead;
  the field stays in the payload so a future UX iteration can enable it
  without a contract change.
- `thumbnail_s3_key` may be `null` if S3 credentials/bucket are unset (local
  dev) or if a per-cluster upload fails. The rest of the callback still
  goes out so BE can persist embeddings and timeline data.

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

# Optional — inference accelerator (auto-detected if unset)
FACE_ANALYZER_DEVICE=mps
FACE_ANALYZER_ONNX_PROVIDERS=CoreMLExecutionProvider,CPUExecutionProvider

# Optional — S3 multipart download tuning
S3_DOWNLOAD_MAX_CONCURRENCY=16
S3_DOWNLOAD_CHUNKSIZE_MB=16

# Optional — thumbnail upload destination (defaults to S3_BUCKET_NAME)
S3_THUMBNAIL_BUCKET=your-video-bucket
S3_THUMBNAIL_PREFIX=thumbnails
```

`CALLBACK_SECRET` is also accepted as a fallback for `ANALYZER_SECRET`.

If `S3_BUCKET_NAME` is set, the analyzer downloads with boto3 using `s3_key`.
If it is not set, it falls back to downloading `s3_url` directly.

### Thumbnail upload

After clustering, the analyzer picks the top-quality face crop per cluster
and uploads it to S3 (JPEG, content-type `image/jpeg`). The returned key
shows up as `thumbnail_s3_key` on each `matched` / `new_candidates` entry in
the callback.

The S3 location is chosen by the request's `thumbnail_dir` field:

- **`thumbnail_dir` provided** (preferred, e.g. `"{project_id}/{session_id}/"`):
  thumbnails land at `{thumbnail_dir}thumb-{idx}.jpg`, sitting next to the
  video file BE uploaded to the same folder.
- **`thumbnail_dir` omitted** (legacy callers): falls back to the flat
  `{S3_THUMBNAIL_PREFIX or "thumbnails"}/{video_id}/{idx}.jpg` layout.

If neither `S3_BUCKET_NAME` nor `S3_THUMBNAIL_BUCKET` is set, upload is
skipped entirely and every `thumbnail_s3_key` is `null` — useful for local
development without S3 write credentials.

### Accelerator selection

When unset, the analyzer auto-detects: Apple Silicon resolves to
`device="mps"` with ONNX Runtime providers
`["CoreMLExecutionProvider", "CPUExecutionProvider"]`; other hosts fall back
to CPU. Override `FACE_ANALYZER_DEVICE` (`mps` / `cuda` / `cpu`) and
`FACE_ANALYZER_ONNX_PROVIDERS` (comma-separated) to deploy with CUDA or to
force CPU for benchmarking.

### S3 key normalization

`s3_key` is tried as-is first. On a 404 (`HeadObject` not found) the
analyzer retries under the alternate Unicode normalization (NFC ↔ NFD).
macOS Finder uploads often store Hangul filenames as NFD while server-side
keys arrive in NFC (or vice-versa); this fallback prevents phantom 404s
without requiring callers to canonicalize.

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

top = max(data[\"new_candidates\"], key=lambda c: c[\"detection_count\"])
print(\"labeling temp_index\", top[\"temp_index\"], \"as actor_kim (n=\" + str(top[\"detection_count\"]) + \")\")

r = httpx.post(\"http://127.0.0.1:8000/analyze\", json={
    \"video_id\": 2, \"session_id\": 2,
    \"s3_key\": \"<S3 object key>\",
    \"s3_url\": \"<S3 object URL>\",
    \"callback_url\": \"http://localhost:9000/cb\",
    \"known_actors\": [{\"actor_id\": \"actor_kim\", \"face_template\": top[\"face_embedding\"]}],
})
print(r.status_code, r.json())
""")'
```

Pass criteria:
- The newly labeled cluster now appears in the top-level `matched[]` array
  with `actor_id="actor_kim"`.
- That cluster is no longer present in `new_candidates[]`.
- The corresponding `appearances[]` entry's `person_id` is now
  `"actor:actor_kim"` instead of `"new:N"`.
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
2. Load every persisted actor for the video's project and call analyzer
   `POST /analyze` with that list as `known_actors`.
3. Receive the callback and validate `X-Analyzer-Secret`.
4. For each `matched[]` entry: insert a `video_actors(video_id, actor_id)`
   link with `is_new_in_video=False`. Do not update the actor's stored
   embedding or thumbnail (both are fixed at first sighting).
5. For each `new_candidates[]` entry: INSERT a new `actors` row using
   `face_embedding` + `thumbnail_s3_key`, name it `'배우 ' || actor_id` once
   the SERIAL id is known, and insert a `video_actors` link with
   `is_new_in_video=True`. Record the mapping `temp_index → actor_id` so the
   next step can rewrite `appearances`.
6. Substitute every `appearances[].person_id` of the form `"new:{idx}"`
   with `"actor:{actor_id}"` using the mapping from step 5 before storing
   `analysis_result`.
7. On re-analysis of an existing video: delete the video's existing
   `video_actors` rows first, then call `/analyze` as usual. After the new
   callback is processed, run a cleanup pass that deletes any project
   actors whose name still matches the auto-generated `'배우 \\d+'` pattern
   and have zero `video_actors` links — this removes the orphans from the
   replaced analysis without touching actors the user has named.

`suggested_actor_id` is informational and may be ignored by the MVP UI;
the analyzer keeps populating it for a future "is this <actor>?" chip.

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
