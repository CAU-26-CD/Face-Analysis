from app.services.face_analysis.models import (
    FaceAnalysisResult,
    FaceAppearance,
    MatchedActor,
    NewActorCandidate,
)
from app.services.face_analysis.worker import (
    _build_callback_payload,
    _upload_cluster_thumbnails,
)


def _make_result(
    appearances: list[FaceAppearance],
    matched: list[MatchedActor],
    new_candidates: list[NewActorCandidate],
) -> FaceAnalysisResult:
    return FaceAnalysisResult(
        video_path="/tmp/whatever.webm",
        appearances=appearances,
        new_candidates=new_candidates,
        matched=matched,
    )


def test_payload_emits_matched_array_with_actor_id_and_thumbnail():
    result = _make_result(
        appearances=[
            FaceAppearance(
                person_id="42",
                start_seconds=0.0,
                end_seconds=5.0,
                detection_count=12,
            )
        ],
        matched=[
            MatchedActor(
                cluster_id="cluster_0",
                actor_id=42,
                similarity=0.82,
                new_exemplars=[[0.1, 0.2, 0.3]],
            )
        ],
        new_candidates=[],
    )

    payload = _build_callback_payload(
        video_id=27,
        analysis=result,
        thumbnail_s3_keys={"cluster_0": "thumbnails/27/0.jpg"},
    )

    assert payload["video_id"] == 27
    assert payload["analysis_status"] == "done"
    assert payload["matched"] == [
        {
            "actor_id": 42,
            "thumbnail_s3_key": "thumbnails/27/0.jpg",
            "similarity": 0.82,
            "new_exemplars": [[0.1, 0.2, 0.3]],
        }
    ]
    assert payload["new_candidates"] == []


def test_payload_emits_new_candidates_with_temp_index_starting_at_zero():
    result = _make_result(
        appearances=[
            FaceAppearance(
                person_id="cluster_a",
                start_seconds=1.0,
                end_seconds=4.0,
                detection_count=8,
            ),
            FaceAppearance(
                person_id="cluster_b",
                start_seconds=10.0,
                end_seconds=15.0,
                detection_count=15,
            ),
        ],
        matched=[],
        new_candidates=[
            NewActorCandidate(
                cluster_id="cluster_a",
                embeddings=[[0.1, 0.2, 0.3]],
                detection_count=8,
                start_seconds=1.0,
                end_seconds=4.0,
            ),
            NewActorCandidate(
                cluster_id="cluster_b",
                embeddings=[[0.4, 0.5, 0.6], [0.45, 0.5, 0.6]],
                detection_count=15,
                start_seconds=10.0,
                end_seconds=15.0,
                suggested_actor_id=7,
                suggested_similarity=0.45,
            ),
        ],
    )

    payload = _build_callback_payload(
        video_id=99,
        analysis=result,
        thumbnail_s3_keys={
            "cluster_a": "thumbnails/99/0.jpg",
            "cluster_b": "thumbnails/99/1.jpg",
        },
    )

    assert payload["new_candidates"] == [
        {
            "temp_index": 0,
            "thumbnail_s3_key": "thumbnails/99/0.jpg",
            "face_embeddings": [[0.1, 0.2, 0.3]],
            "detection_count": 8,
            "start_seconds": 1.0,
            "end_seconds": 4.0,
            "suggested_actor_id": None,
            "suggested_similarity": None,
        },
        {
            "temp_index": 1,
            "thumbnail_s3_key": "thumbnails/99/1.jpg",
            "face_embeddings": [[0.4, 0.5, 0.6], [0.45, 0.5, 0.6]],
            "detection_count": 15,
            "start_seconds": 10.0,
            "end_seconds": 15.0,
            "suggested_actor_id": 7,
            "suggested_similarity": 0.45,
        },
    ]


def test_appearances_use_actor_and_new_prefixed_person_ids():
    result = _make_result(
        appearances=[
            FaceAppearance(
                person_id="42",
                start_seconds=0.0,
                end_seconds=5.0,
                detection_count=12,
            ),
            FaceAppearance(
                person_id="cluster_2",
                start_seconds=6.0,
                end_seconds=9.0,
                detection_count=7,
            ),
        ],
        matched=[
            MatchedActor(
                cluster_id="cluster_1",
                actor_id=42,
                similarity=0.81,
            )
        ],
        new_candidates=[
            NewActorCandidate(
                cluster_id="cluster_2",
                embeddings=[[0.0, 1.0, 0.0]],
                detection_count=7,
                start_seconds=6.0,
                end_seconds=9.0,
            )
        ],
    )

    payload = _build_callback_payload(
        video_id=1, analysis=result, thumbnail_s3_keys={}
    )

    appearances = payload["analysis_result"]["appearances"]
    assert [a["person_id"] for a in appearances] == ["actor:42", "new:0"]


def test_payload_keeps_thumbnail_null_when_upload_skipped():
    result = _make_result(
        appearances=[],
        matched=[],
        new_candidates=[
            NewActorCandidate(
                cluster_id="cluster_0",
                embeddings=[[0.0]],
                detection_count=1,
                start_seconds=0.0,
                end_seconds=0.0,
            )
        ],
    )

    payload = _build_callback_payload(
        video_id=1, analysis=result, thumbnail_s3_keys={}
    )

    assert payload["new_candidates"][0]["thumbnail_s3_key"] is None


def test_thumbnail_upload_no_op_without_bucket(monkeypatch):
    monkeypatch.delenv("S3_BUCKET_NAME", raising=False)
    monkeypatch.delenv("S3_THUMBNAIL_BUCKET", raising=False)

    result = _upload_cluster_thumbnails(
        {"cluster_0": ["/tmp/whatever.jpg"]}, video_id=1
    )
    assert result == {}


def test_thumbnail_upload_returns_keys_for_uploaded_clusters(monkeypatch, tmp_path):
    # Pretend we have two clusters with thumbnails on disk
    img_a = tmp_path / "a.jpg"
    img_b = tmp_path / "b.jpg"
    img_a.write_bytes(b"\xff\xd8\xff")
    img_b.write_bytes(b"\xff\xd8\xff")

    monkeypatch.setenv("S3_BUCKET_NAME", "test-bucket")
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("S3_THUMBNAIL_BUCKET", raising=False)
    monkeypatch.delenv("S3_THUMBNAIL_PREFIX", raising=False)

    calls = []

    class _FakeClient:
        def upload_file(self, source, bucket, key, ExtraArgs=None):
            calls.append((source, bucket, key, ExtraArgs))

    def _fake_boto_client(service, **kwargs):
        assert service == "s3"
        return _FakeClient()

    import boto3

    monkeypatch.setattr(boto3, "client", _fake_boto_client)

    result = _upload_cluster_thumbnails(
        {"cluster_a": [str(img_a)], "cluster_b": [str(img_b)]},
        video_id=42,
    )

    assert result == {
        "cluster_a": "thumbnails/42/0.jpg",
        "cluster_b": "thumbnails/42/1.jpg",
    }
    assert len(calls) == 2
    assert calls[0][1] == "test-bucket"
    assert calls[0][3] == {"ContentType": "image/jpeg"}


def test_thumbnail_upload_uses_dir_prefix_when_provided(monkeypatch, tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"\xff\xd8\xff")

    monkeypatch.setenv("S3_BUCKET_NAME", "test-bucket")
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("S3_THUMBNAIL_BUCKET", raising=False)
    monkeypatch.delenv("S3_THUMBNAIL_PREFIX", raising=False)

    calls = []

    class _FakeClient:
        def upload_file(self, source, bucket, key, ExtraArgs=None):
            calls.append(key)

    monkeypatch.setattr("boto3.client", lambda service, **kw: _FakeClient())

    # BE-supplied directory — analyzer appends "thumb-{idx}.jpg" so the
    # per-session folder ends up with video + thumbnails side by side.
    result = _upload_cluster_thumbnails(
        {"cluster_a": [str(img)]},
        video_id=999,                 # video_id is ignored when thumbnail_dir is given
        thumbnail_dir="42/27/",
    )

    assert result == {"cluster_a": "42/27/thumb-0.jpg"}
    assert calls == ["42/27/thumb-0.jpg"]


def test_thumbnail_upload_normalizes_trailing_slash(monkeypatch, tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"\xff\xd8\xff")

    monkeypatch.setenv("S3_BUCKET_NAME", "test-bucket")
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("S3_THUMBNAIL_BUCKET", raising=False)

    captured: list[str] = []
    monkeypatch.setattr(
        "boto3.client",
        lambda service, **kw: type(
            "F", (), {"upload_file": lambda self, src, b, k, ExtraArgs=None: captured.append(k)}
        )(),
    )

    # No trailing slash — analyzer adds one so we don't end up with "42/27thumb-0.jpg"
    _upload_cluster_thumbnails(
        {"c": [str(img)]}, video_id=1, thumbnail_dir="42/27"
    )
    assert captured[-1] == "42/27/thumb-0.jpg"

    # Double trailing slash — collapses to one
    _upload_cluster_thumbnails(
        {"c": [str(img)]}, video_id=1, thumbnail_dir="42/27//"
    )
    assert captured[-1] == "42/27/thumb-0.jpg"
