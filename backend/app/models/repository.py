import enum

from core.config import EMBEDDING_DIMENSIONS
from database.database import Base
from pgvector.sqlalchemy import Vector
from pydantic import BaseModel
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

class RepositoryStatus(str, enum.Enum):
    NOT_INDEXED = "not_indexed"
    INDEXING = "indexing"
    READY = "ready"
    ERROR = "error"


class Repository(Base):
    """A GitHub repo that has been (or is being) indexed for RAG."""

    __tablename__ = "repositories"

    id = Column(Integer, primary_key=True)

    full_name = Column(String, nullable=False, unique=True, index=True)  # "owner/name"
    owner = Column(String, nullable=False)
    name = Column(String, nullable=False)
    default_branch = Column(String, nullable=True)

    status = Column(
        # values_callable: persist the lowercase .value strings ("ready", ...)
        # instead of SQLAlchemy's default of persisting the member .name
        # ("READY", ...) - keeps the DB, the Python enum, and the API
        # response (which also serializes to .value) all in sync.
        SAEnum(
            RepositoryStatus,
            name="repository_status",
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
        default=RepositoryStatus.NOT_INDEXED,
        server_default=RepositoryStatus.NOT_INDEXED.value,
    )
    error_message = Column(Text, nullable=True)

    file_count = Column(Integer, nullable=False, default=0)
    chunk_count = Column(Integer, nullable=False, default=0)
    indexed_at = Column(DateTime(timezone=True), nullable=True)

    requested_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    chunks = relationship(
        "DocumentChunk",
        back_populates="repository",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class DocumentChunk(Base):
    """A single chunk of a file's text plus its embedding vector."""

    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True)
    repository_id = Column(
        Integer,
        ForeignKey("repositories.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    file_path = Column(String, nullable=False)
    chunk_index = Column(Integer, nullable=False)

    content = Column(Text, nullable=False)
    start_line = Column(Integer, nullable=True)
    end_line = Column(Integer, nullable=True)

    embedding = Column(Vector(EMBEDDING_DIMENSIONS), nullable=False)

    repository = relationship("Repository", back_populates="chunks")


class RepositoryAccess(Base):
    """Nyilvántartja, melyik felhasználó melyik repóhoz fér hozzá az
    alkalmazásban - a tényleges jogosultság forrása mindig a GitHub marad,
    ez a tábla csak azt jegyzi, mikor ellenőriztük ezt utoljára élőben."""

    __tablename__ = "repository_access"

    id = Column(Integer, primary_key=True)

    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    repository_id = Column(
        Integer,
        ForeignKey("repositories.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    last_verified_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("user_id", "repository_id", name="uq_repository_access_user_repo"),
    )


# --- Pydantic schemas ---------------------------------------------------------------


class IngestRequest(BaseModel):
    repo_full_name: str


class RepositoryStatusResponse(BaseModel):
    repo_full_name: str
    status: RepositoryStatus
    file_count: int
    chunk_count: int
    indexed_at: str | None = None
    error_message: str | None = None
