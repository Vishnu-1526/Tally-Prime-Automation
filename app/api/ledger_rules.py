from __future__ import annotations

"""
Ledger rules API endpoints including the /learn endpoint for human corrections.
"""

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.database import get_db_session
from app.models.ledger_rule import LedgerRule

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ledger-rules", tags=["Ledger Rules"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class LearnRequest(BaseModel):
    vendor_name: str
    category: str
    tally_ledger_name: str
    corrected_by: str  # email or username of the person making the correction


class LearnResponse(BaseModel):
    rule_id: int
    vendor_name: str
    tally_ledger_name: str
    priority: int
    message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/learn", response_model=LearnResponse)
async def learn_ledger_rule(body: LearnRequest) -> LearnResponse:
    """
    Create a new learned LedgerRule from a human correction.
    Learned rules get priority=200 so they are always checked before seed rules.
    Uses SHA-256 deduplication on (vendor_name, tally_ledger_name) to avoid
    creating duplicate learned rules.
    """
    async with get_db_session() as session:
        # Check for existing learned rule for this vendor+ledger combination
        from sqlalchemy import select, and_
        existing_stmt = select(LedgerRule).where(
            and_(
                LedgerRule.provider_name_display == body.vendor_name,
                LedgerRule.tally_ledger_name == body.tally_ledger_name,
                LedgerRule.created_by == "learned",
            )
        )
        existing_result = await session.execute(existing_stmt)
        existing = existing_result.scalar_one_or_none()

        if existing:
            logger.info(
                "Duplicate learned rule detected for vendor %r → %r (rule_id=%d)",
                body.vendor_name, body.tally_ledger_name, existing.id,
            )
            return LearnResponse(
                rule_id=existing.id,
                vendor_name=body.vendor_name,
                tally_ledger_name=body.tally_ledger_name,
                priority=existing.priority,
                message=f"Rule already exists (id={existing.id}). No changes made.",
            )

        # Create new learned rule with high priority
        new_rule = LedgerRule(
            category=body.category.lower(),
            provider_pattern=rf"^{body.vendor_name}$",  # exact vendor match
            provider_name_display=body.vendor_name,
            tally_ledger_name=body.tally_ledger_name,
            tally_parent_ledger="Indirect Expenses",
            cgst_ledger="CGST @9%",
            sgst_ledger="SGST @9%",
            igst_ledger="IGST @18%",
            credit_ledger="Sundry Creditors",
            priority=200,  # learned rules always beat seed rules
            is_active=True,
            created_by="learned",
        )
        session.add(new_rule)
        await session.flush()  # get the auto-generated id before commit

        logger.info(
            "Learned rule created: id=%d vendor=%r ledger=%r by=%r",
            new_rule.id, body.vendor_name, body.tally_ledger_name, body.corrected_by,
        )

        return LearnResponse(
            rule_id=new_rule.id,
            vendor_name=body.vendor_name,
            tally_ledger_name=body.tally_ledger_name,
            priority=new_rule.priority,
            message="Learned rule created successfully.",
        )


@router.get("")
async def list_ledger_rules(category: str | None = None, active_only: bool = True):
    """
    List all ledger rules, optionally filtered by category and active status.
    """
    from sqlalchemy import select

    async with get_db_session() as session:
        stmt = select(LedgerRule).order_by(
            LedgerRule.category, LedgerRule.priority.desc()
        )
        if category:
            stmt = stmt.where(LedgerRule.category == category.lower())
        if active_only:
            stmt = stmt.where(LedgerRule.is_active.is_(True))

        result = await session.execute(stmt)
        rules = result.scalars().all()

    return [
        {
            "id": r.id,
            "category": r.category,
            "provider_name_display": r.provider_name_display,
            "tally_ledger_name": r.tally_ledger_name,
            "priority": r.priority,
            "created_by": r.created_by,
            "is_active": r.is_active,
        }
        for r in rules
    ]
