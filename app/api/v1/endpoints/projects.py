import random, string
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.models import User, Project
from app.schemas.schemas import ProjectCreate, ProjectJoin, ProjectOut

router = APIRouter(prefix="/projects", tags=["projects"])


def _generate_code() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=4))


# ─── 내 프로젝트 목록 ────────────────────────────────────
@router.get("", response_model=list[ProjectOut])
async def list_my_projects(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Project).where(Project.project_id == current_user.project_id)
    )
    # TODO: user-project N:M 관계로 확장 시 조인 테이블 추가 필요
    projects = result.scalars().all()
    return projects


# ─── 프로젝트 생성 ───────────────────────────────────────
@router.post("", response_model=ProjectOut, status_code=201)
async def create_project(
    body: ProjectCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # 유니크한 4자 코드 생성
    for _ in range(10):
        code = _generate_code()
        exists = await db.execute(select(Project).where(Project.code == code))
        if not exists.scalar_one_or_none():
            break

    project = Project(code=code, p_name=body.p_name, p_detail=body.p_detail)
    db.add(project)
    await db.flush()

    current_user.project_id = project.project_id
    return project


# ─── 코드로 프로젝트 조인 ────────────────────────────────
@router.post("/join", response_model=ProjectOut)
async def join_project(
    body: ProjectJoin,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Project).where(Project.code == body.code.upper())
    )
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Invalid join code")

    current_user.project_id = project.project_id
    return project


# ─── 단일 프로젝트 조회 ──────────────────────────────────
@router.get("/{project_id}", response_model=ProjectOut)
async def get_project(
    project_id: int,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Project).where(Project.project_id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project
