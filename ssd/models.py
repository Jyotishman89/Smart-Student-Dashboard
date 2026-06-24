"""SQLAlchemy 2.0 ORM models.

Generic column types (JSON rather than Postgres JSONB) are used deliberately so
the same models run on Postgres in production and on SQLite for tests/dev.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    full_name: Mapped[str] = mapped_column(String(255), default="")
    # Roll number is the login identifier — unique + indexed.
    roll_no: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    semesters: Mapped[list[Semester]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    snapshots: Mapped[list[Snapshot]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Semester(Base):
    __tablename__ = "semesters"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    label: Mapped[str] = mapped_column(String(64), default="Sem-1")
    is_active: Mapped[bool] = mapped_column(default=False)
    perf_threshold: Mapped[float] = mapped_column(Float, default=50.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    user: Mapped[User] = relationship(back_populates="semesters")
    subjects: Mapped[list[Subject]] = relationship(
        back_populates="semester", cascade="all, delete-orphan", order_by="Subject.order_index"
    )
    components: Mapped[list[Component]] = relationship(
        back_populates="semester", cascade="all, delete-orphan", order_by="Component.order_index"
    )


class Subject(Base):
    __tablename__ = "subjects"

    id: Mapped[int] = mapped_column(primary_key=True)
    semester_id: Mapped[int] = mapped_column(
        ForeignKey("semesters.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(128))
    credits: Mapped[float] = mapped_column(Float, default=0.0)
    order_index: Mapped[int] = mapped_column(Integer, default=0)

    semester: Mapped[Semester] = relationship(back_populates="subjects")
    scores: Mapped[list[Score]] = relationship(
        back_populates="subject", cascade="all, delete-orphan"
    )
    attendance: Mapped[Attendance | None] = relationship(
        back_populates="subject", cascade="all, delete-orphan", uselist=False
    )


class Component(Base):
    __tablename__ = "components"

    id: Mapped[int] = mapped_column(primary_key=True)
    semester_id: Mapped[int] = mapped_column(
        ForeignKey("semesters.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(64))
    max_marks: Mapped[float] = mapped_column(Float, default=0.0)
    order_index: Mapped[int] = mapped_column(Integer, default=0)

    semester: Mapped[Semester] = relationship(back_populates="components")


class Score(Base):
    __tablename__ = "scores"
    __table_args__ = (UniqueConstraint("subject_id", "component_id", name="uq_score"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    subject_id: Mapped[int] = mapped_column(
        ForeignKey("subjects.id", ondelete="CASCADE"), index=True
    )
    component_id: Mapped[int] = mapped_column(
        ForeignKey("components.id", ondelete="CASCADE"), index=True
    )
    value: Mapped[float] = mapped_column(Float, default=0.0)

    subject: Mapped[Subject] = relationship(back_populates="scores")


class Attendance(Base):
    __tablename__ = "attendance"

    id: Mapped[int] = mapped_column(primary_key=True)
    subject_id: Mapped[int] = mapped_column(
        ForeignKey("subjects.id", ondelete="CASCADE"), unique=True, index=True
    )
    classes_held: Mapped[int] = mapped_column(Integer, default=0)
    classes_attended: Mapped[int] = mapped_column(Integer, default=0)

    subject: Mapped[Subject] = relationship(back_populates="attendance")


class Snapshot(Base):
    __tablename__ = "snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    semester_id: Mapped[int | None] = mapped_column(
        ForeignKey("semesters.id", ondelete="SET NULL"), nullable=True
    )
    semester_label: Mapped[str] = mapped_column(String(64), default="")
    taken_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    sgpa: Mapped[float] = mapped_column(Float, default=0.0)
    total_credits: Mapped[float] = mapped_column(Float, default=0.0)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)

    user: Mapped[User] = relationship(back_populates="snapshots")
