from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.deps import get_db
from app.models import User
from app.schemas import UserCreate, UserResponse

router = APIRouter(
    prefix="/users",
    tags=["users"]
)


@router.post("", response_model=UserResponse)
def create_user(payload: UserCreate, db: Session = Depends(get_db)):
    user = User(name=payload.name)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.get("", response_model=list[UserResponse])
def read_users(db: Session = Depends(get_db)):
    return db.query(User).all()
