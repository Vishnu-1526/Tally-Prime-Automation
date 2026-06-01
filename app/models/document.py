from __future__ import annotations

"""
SQLAlchemy model for ingested documents.
Includes a SHA-256 hash for duplicate detection before pipeline execution.
"""

import uuid as uuid_lib
from datetime import datetime

from sqlalchemy import DateTime, Index, Integer, String, Text, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    company_id: Mapped[int | None] = mapped_column(ForeignKey("companies.id"), nullable=True)
    document_uuid: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        unique=True,
        default=lambda: str(uuid_lib.uuid4()),
    )

    # File info
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=True)

    # SHA-256 deduplication hash (unique constraint prevents double-processing)
    sha256_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)

    # Pipeline outcome
    overall_status: Mapped[str] = mapped_column(String(32), nullable=True)
    pipeline_result_json: Mapped[str] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    processed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_documents_sha256", "sha256_hash", unique=True),
        Index("ix_documents_uuid", "document_uuid", unique=True),
    )

    def __repr__(self) -> str:
        return (
            f"<Document id={self.id} uuid={self.document_uuid!r} "
            f"file={self.original_filename!r} status={self.overall_status!r}>"
        )
