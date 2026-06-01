from __future__ import annotations

"""
SQLAlchemy model for ledger mapping rules.
Rules drive the automatic vendor → TallyPrime ledger matching.
"""

from typing import Literal

from sqlalchemy import Boolean, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class LedgerRule(Base):
    __tablename__ = "ledger_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Classification
    category: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    provider_pattern: Mapped[str] = mapped_column(String(255), nullable=False)
    provider_name_display: Mapped[str] = mapped_column(String(255), nullable=False)

    # TallyPrime ledger configuration
    tally_ledger_name: Mapped[str] = mapped_column(String(255), nullable=False)
    tally_parent_ledger: Mapped[str] = mapped_column(String(255), nullable=False)
    cgst_ledger: Mapped[str] = mapped_column(String(128), nullable=False)
    sgst_ledger: Mapped[str] = mapped_column(String(128), nullable=False)
    igst_ledger: Mapped[str] = mapped_column(String(128), nullable=False)
    credit_ledger: Mapped[str] = mapped_column(String(255), nullable=False)

    # Matching priority (higher = checked first; learned rules get +100)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=10, index=True)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_by: Mapped[str] = mapped_column(
        String(16), nullable=False, default="seed"
    )  # "seed" | "manual" | "learned"

    __table_args__ = (
        Index("ix_ledger_rules_category_priority", "category", "priority"),
    )

    def __repr__(self) -> str:
        return (
            f"<LedgerRule id={self.id} category={self.category!r} "
            f"provider={self.provider_name_display!r} priority={self.priority}>"
        )
