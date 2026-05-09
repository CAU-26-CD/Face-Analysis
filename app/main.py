from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.database import engine, Base
from app.api.v1.endpoints import auth, projects, sessions, feedbacks, actors, video, report
from app.api.v1.endpoints import camera_session          # ← 추가

from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="Rehearsal Feedback Platform API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ─────────────────────────────────────────────
API_PREFIX = "/api/v1"

app.include_router(auth.router,           prefix=API_PREFIX)
app.include_router(projects.router,       prefix=API_PREFIX)
app.include_router(sessions.router,       prefix=API_PREFIX)
app.include_router(feedbacks.router,      prefix=API_PREFIX)
app.include_router(actors.router,         prefix=API_PREFIX)
app.include_router(video.router,          prefix=API_PREFIX)
app.include_router(report.router,         prefix=API_PREFIX)
app.include_router(camera_session.router, prefix=API_PREFIX)  # ← 추가


# ─── DB 초기화 (개발용 — 운영은 alembic 마이그레이션 사용) ──
@app.on_event("startup")
async def on_startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@app.get("/health") 
def health():
    return {"status": "ok"}


app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads") # 영상 파일 서빙을 위한 정적 파일 경로 설정
