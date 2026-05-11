from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.models import User, Feedback, FCategory, Session
from app.schemas.schemas import FeedbackCreate, FeedbackOut

router = APIRouter(
    prefix="/projects/{project_id}/sessions/{session_id}/feedbacks",
    tags=["feedbacks"],
)


def _to_out(fb: Feedback) -> FeedbackOut:
    return FeedbackOut(
        feedback_id=fb.feedback_id,
        session_id=fb.session_id,
        project_id=fb.project_id,
        user_id=fb.user_id,
        actor_id=fb.actor_id,
        content=fb.content,
        timestamp_sec=fb.timestamp_sec,
        created_at=fb.created_at,
        categories=[c.f_name for c in fb.categories],
    )


# ─── 피드백 목록 조회 (actor / category 필터 지원) ─────────
@router.get("", response_model=list[FeedbackOut])
async def list_feedbacks(
    project_id: int,
    session_id: int,
    actor_id: str | None = Query(default=None),
    category: str | None = Query(default=None),
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(Feedback)
        .options(selectinload(Feedback.categories))
        .where(
            Feedback.session_id == session_id,
            Feedback.project_id == project_id,
        )
        .order_by(Feedback.timestamp_sec.asc().nulls_last(), Feedback.created_at.asc())
    )

    if actor_id:
        stmt = stmt.where(Feedback.actor_id == actor_id)

    if category:
        stmt = stmt.join(FCategory).where(FCategory.f_name == category)

    result = await db.execute(stmt)
    return [_to_out(fb) for fb in result.scalars().all()]


# ─── 피드백 작성 ─────────────────────────────────────────
@router.post("", response_model=FeedbackOut, status_code=201)
async def create_feedback(
    project_id: int,
    session_id: int,
    body: FeedbackCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # 세션이 in_progress 상태인지 확인
    sess_result = await db.execute(
        select(Session).where(Session.session_id == session_id)
    )
    session = sess_result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    fb = Feedback(
        session_id=session_id,
        project_id=project_id,
        user_id=current_user.user_id,
        actor_id=body.actor_id,
        content=body.content,
        timestamp_sec=body.timestamp_sec,
    )
    db.add(fb)
    await db.flush()  # feedback_id 확보

    # 카테고리 태그 저장
    for cat_name in body.categories:
        cat = FCategory(
            feedback_id=fb.feedback_id,
            actor_id=body.actor_id,
            session_id=session_id,
            project_id=project_id,
            f_name=cat_name,
        )
        db.add(cat)

    await db.flush()
    await db.refresh(fb, ["categories"])
    return _to_out(fb)


# ─── 피드백 단건 조회 ────────────────────────────────────
@router.get("/{feedback_id}", response_model=FeedbackOut)
async def get_feedback(
    project_id: int,
    session_id: int,
    feedback_id: int,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Feedback)
        .options(selectinload(Feedback.categories))
        .where(
            Feedback.feedback_id == feedback_id,
            Feedback.session_id == session_id,
            Feedback.project_id == project_id,
        )
    )
    fb = result.scalar_one_or_none()
    if not fb:
        raise HTTPException(status_code=404, detail="Feedback not found")
    return _to_out(fb)


# ─── 피드백 삭제 ─────────────────────────────────────────
@router.delete("/{feedback_id}", status_code=204)
async def delete_feedback(
    project_id: int,
    session_id: int,
    feedback_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Feedback).where(
            Feedback.feedback_id == feedback_id,
            Feedback.session_id == session_id,
            Feedback.project_id == project_id,
            Feedback.user_id == current_user.user_id,  # 본인 피드백만 삭제
        )
    )
    fb = result.scalar_one_or_none()
    if not fb:
        raise HTTPException(status_code=404, detail="Feedback not found or not authorized")
    await db.delete(fb)
