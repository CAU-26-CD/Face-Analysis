from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.models import User, Actor
from app.schemas.schemas import ActorCreate, ActorOut

router = APIRouter(
    prefix="/projects/{project_id}/sessions/{session_id}/actors",
    tags=["actors"],
)

MAX_ACTORS = 8


@router.get("", response_model=list[ActorOut])
async def list_actors(
    project_id: int,
    session_id: int,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Actor).where(
            Actor.session_id == session_id,
            Actor.project_id == project_id,
        )
    )
    return result.scalars().all()


@router.post("", response_model=ActorOut, status_code=201)
async def create_actor(
    project_id: int,
    session_id: int,
    body: ActorCreate,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    count_result = await db.execute(
        select(func.count()).where(
            Actor.session_id == session_id,
            Actor.project_id == project_id,
        )
    )
    count = count_result.scalar()
    if count >= MAX_ACTORS:
        raise HTTPException(status_code=400, detail=f"최대 {MAX_ACTORS}명까지 등록 가능합니다")

    actor = Actor(
        actor_id=body.actor_id,
        session_id=session_id,
        project_id=project_id,
        a_name=body.a_name,
    )
    db.add(actor)
    await db.flush()
    return actor


@router.delete("/{actor_id}", status_code=204)
async def delete_actor(
    project_id: int,
    session_id: int,
    actor_id: str,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Actor).where(
            Actor.actor_id == actor_id,
            Actor.session_id == session_id,
            Actor.project_id == project_id,
        )
    )
    actor = result.scalar_one_or_none()
    if not actor:
        raise HTTPException(status_code=404, detail="Actor not found")
    await db.delete(actor)
