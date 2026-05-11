from collections import defaultdict
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.models import User, Feedback, Actor
from app.schemas.schemas import SessionReportOut, FeedbackReportItem

router = APIRouter(
    prefix="/projects/{project_id}/sessions/{session_id}/report",
    tags=["report"],
)


@router.get("", response_model=SessionReportOut)
async def get_session_report(
    project_id: int,
    session_id: int,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """세션 피드백 레포트 — 배우별 / 카테고리별 통계"""
    fb_result = await db.execute(
        select(Feedback)
        .options(selectinload(Feedback.categories))
        .where(Feedback.session_id == session_id, Feedback.project_id == project_id)
    )
    feedbacks = fb_result.scalars().all()

    actor_result = await db.execute(
        select(Actor).where(Actor.session_id == session_id, Actor.project_id == project_id)
    )
    actor_map = {a.actor_id: a.a_name for a in actor_result.scalars().all()}

    # actor_id → {category → count, timestamps}
    actor_data: dict[str | None, dict] = defaultdict(
        lambda: {"count": 0, "categories": defaultdict(int), "timestamps": []}
    )

    for fb in feedbacks:
        key = fb.actor_id
        actor_data[key]["count"] += 1
        if fb.timestamp_sec is not None:
            actor_data[key]["timestamps"].append(fb.timestamp_sec)
        for cat in fb.categories:
            actor_data[key]["categories"][cat.f_name] += 1

    actor_reports = [
        FeedbackReportItem(
            actor_id=actor_id,
            actor_name=actor_map.get(actor_id) if actor_id else None,
            total_count=data["count"],
            categories=dict(data["categories"]),
            top_timestamps=sorted(data["timestamps"])[:10],
        )
        for actor_id, data in actor_data.items()
    ]

    return SessionReportOut(
        session_id=session_id,
        total_feedbacks=len(feedbacks),
        actor_reports=actor_reports,
    )
