from __future__ import annotations

"""
SQLAlchemy model for companies.
Allows managing ledger mappings, configurations, and document scoping per company.
"""

from datetime import datetime
from sqlalchemy import DateTime, Integer, String, Boolean, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    gstin: Mapped[str] = mapped_column(String(15), nullable=False, unique=True)
    state_code: Mapped[str] = mapped_column(String(2), nullable=False)
    tally_host: Mapped[str] = mapped_column(String(255), nullable=False, default="localhost")
    tally_port: Mapped[int] = mapped_column(Integer, nullable=False, default=9000)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<Company id={self.id} name={self.name!r} gstin={self.gstin!r}>"
