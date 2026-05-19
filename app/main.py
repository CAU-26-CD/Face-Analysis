from fastapi import BackgroundTasks, FastAPI, status
from pydantic import BaseModel, Field, HttpUrl

from app.services.face_analysis.worker import run_analysis_job


class KnownActorPayload(BaseModel):
    actor_id: str = Field(..., min_length=1)
    face_template: list[float] = Field(..., min_length=1)


class AnalyzeRequest(BaseModel):
    video_id: int = Field(..., ge=1)
    session_id: int = Field(..., ge=1)
    s3_key: str = Field(..., min_length=1)
    s3_url: HttpUrl
    callback_url: HttpUrl
    known_actors: list[KnownActorPayload] = Field(default_factory=list)
    # S3 key prefix BE wants thumbnails uploaded under, e.g. "42/27/" — the
    # analyzer appends "thumb-{idx}.jpg" so video + thumbnails end up in the
    # same per-session folder. Optional for back-compat with older callers
    # that have not yet been updated; in that case the legacy
    # "thumbnails/{video_id}/" layout is used instead.
    thumbnail_dir: str | None = None


class AnalyzeAcceptedResponse(BaseModel):
    video_id: int
    analysis_status: str


app = FastAPI(
    title="Face Analyzer API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/analyze",
    response_model=AnalyzeAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def analyze(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
) -> AnalyzeAcceptedResponse:
    background_tasks.add_task(run_analysis_job, request.model_dump(mode="json"))
    return AnalyzeAcceptedResponse(
        video_id=request.video_id,
        analysis_status="accepted",
    )
