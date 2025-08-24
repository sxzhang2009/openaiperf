from __future__ import annotations

import datetime as dt
from contextlib import contextmanager
from typing import Any, Dict

from sqlmodel import Field, SQLModel, create_engine, Session
from sqlalchemy import Column, JSON


DB_PATH = "sqlite:///openaiperf.db"
engine = create_engine(DB_PATH, echo=False)


class Submission(SQLModel, table=True):
    model_config = {"protected_namespaces": ()}
    id: str = Field(primary_key=True)
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.utcnow())

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


def init_db() -> None:
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


