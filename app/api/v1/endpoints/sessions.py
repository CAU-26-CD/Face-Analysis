from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.models import User, Session
from app.schemas.schemas import SessionCreate, SessionUpdate, SessionOut

router = APIRouter(prefix="/projects/{project_id}/sessions", tags=["sessions"])


@router.get("", response_model=list[SessionOut])
async def list_sessions(
    project_id: int,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Session).where(Session.project_id == project_id)
        .order_by(Session.s_created_at.desc())
    )
    return result.scalars().all()


@router.post("", response_model=SessionOut, status_code=201)
async def create_session(
    project_id: int,
    body: SessionCreate,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    session = Session(project_id=project_id, **body.model_dump())
    db.add(session)
    await db.flush()
    return session


@router.get("/{session_id}", response_model=SessionOut)
async def get_session(
    project_id: int,
    session_id: int,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Session).where(
            Session.session_id == session_id,
            Session.project_id == project_id,
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.patch("/{session_id}", response_model=SessionOut)
async def update_session(
    project_id: int,
    session_id: int,
    body: SessionUpdate,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Session).where(
            Session.session_id == session_id,
            Session.project_id == project_id,
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    for field, value in body.model_dump(exclude_none=True).items():
        setattr(session, field, value)

    return session
