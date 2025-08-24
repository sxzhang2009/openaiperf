from __future__ import annotations

import datetime as dt
import enum
from contextlib import contextmanager
from typing import Any, Dict, List
from passlib.context import CryptContext
from sqlmodel import Field, SQLModel, create_engine, Session
from sqlalchemy import Column, JSON


DB_PATH = "sqlite:///openaiperf.db"
engine = create_engine(DB_PATH, echo=False)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class UserRole(str, enum.Enum):
    ADMIN = "admin"
    AUDITOR = "auditor"
    USER = "user"


class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(unique=True, index=True)
    hashed_password: str
    role: UserRole = Field(default=UserRole.USER, index=True)
    email: str = Field(unique=True, index=True)
    organization: str | None = Field(default=None)
    phone: str | None = Field(default=None)
    approved: bool = Field(default=False, index=True)  # New field for approval status
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.utcnow())



    def verify_password(self, plain_password: str) -> bool:
        return pwd_context.verify(plain_password, self.hashed_password)


class Submission(SQLModel, table=True):
    model_config = {"protected_namespaces": ()}
    id: str = Field(primary_key=True)
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.utcnow())
    user_id: int | None = Field(default=None, foreign_key="user.id")


    # Indexed summary fields for quick filtering
    task: str | None = Field(default=None, index=True)
    scenario: str | None = Field(default=None, index=True)
    model_name: str | None = Field(default=None, index=True)
    backend: str | None = Field(default=None, index=True)

    quality: float | None = Field(default=None, index=True)
    throughput: float | None = Field(default=None, index=True)
    latency_p99_ms: float | None = Field(default=None, index=True)
    energy_j_per_request: float | None = Field(default=None, index=True)

    # Moderation state
    approved: bool = Field(default=False, index=True)

    # Raw cards as JSON column
    cards: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    notes: str | None = None


class Blog(SQLModel, table=True):
    id: int = Field(primary_key=True)
    title: str
    content: str
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.utcnow())


class Comment(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    content: str
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.utcnow())
    user_id: int | None = Field(default=None, foreign_key="user.id")
    blog_id: int | None = Field(default=None, foreign_key="blog.id")


def init_db() -> None:
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session

