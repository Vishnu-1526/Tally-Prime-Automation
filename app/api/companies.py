from __future__ import annotations

"""
Companies API Endpoints for managing multi-company support.
"""

import logging
import json
from typing import Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from app.database import get_db_session
from app.models.company import Company
from app.models.document import Document

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/companies", tags=["Companies"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class CompanyCreate(BaseModel):
    name: str = Field(..., description="The name of the company, e.g. KODRYX AI PRIVATE LIMITED")
    gstin: str = Field(..., description="The company's 15-character GSTIN")
    tally_host: str = Field("localhost", description="Tally Prime server hostname or IP")
    tally_port: int = Field(9000, description="XML Port configured in Tally Prime")


class CompanyResponse(BaseModel):
    id: int
    name: str
    gstin: str
    state_code: str
    tally_host: str
    tally_port: int
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class CompanyStatsResponse(BaseModel):
    total_invoices: int
    auto_post_count: int
    pending_count: int
    total_spend: float
    spend_by_category: dict[str, float]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("", response_model=CompanyResponse, status_code=201)
async def create_company(body: CompanyCreate) -> CompanyResponse:
    """
    Create a new company. State code is automatically derived from the first 2 digits of the GSTIN.
    """
    gstin_clean = body.gstin.strip().upper()
    if len(gstin_clean) != 15:
        raise HTTPException(
            status_code=422,
            detail="GSTIN must be exactly 15 characters long."
        )

    state_code = gstin_clean[:2]
    if not state_code.isdigit():
        raise HTTPException(
            status_code=422,
            detail="The first 2 characters of GSTIN must represent a numeric state code."
        )

    async with get_db_session() as session:
        # Check if GSTIN already exists
        from sqlalchemy import select
        existing_stmt = select(Company).where(Company.gstin == gstin_clean)
        existing_res = await session.execute(existing_stmt)
        existing = existing_res.scalar_one_or_none()

        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"Company with GSTIN '{gstin_clean}' already exists."
            )

        # Safety Guard Block: Prevent creating companies for billing vendors in ledger_rules
        from app.models.ledger_rule import LedgerRule
        import re
        rules_stmt = select(LedgerRule).where(LedgerRule.is_active.is_(True))
        rules_res = await session.execute(rules_stmt)
        rules = rules_res.scalars().all()
        
        for rule in rules:
            name_lower = body.name.lower().strip()
            rule_name_lower = rule.provider_name_display.lower().strip()
            
            if name_lower == rule_name_lower:
                raise HTTPException(
                    status_code=422,
                    detail=f"Safety Block: '{body.name}' appears to be a billing vendor registered in ledger_rules (Rule ID {rule.id}). Companies must be client companies/tenants, not billing vendors."
                )
            
            if rule.provider_pattern and rule.provider_pattern != ".*":
                try:
                    if re.search(rule.provider_pattern, gstin_clean, re.IGNORECASE):
                        raise HTTPException(
                            status_code=422,
                            detail=f"Safety Block: GSTIN matches a known vendor pattern in ledger_rules (Rule ID {rule.id}). Companies must be client companies/tenants, not billing vendors."
                        )
                    if re.search(rule.provider_pattern, body.name, re.IGNORECASE):
                        raise HTTPException(
                            status_code=422,
                            detail=f"Safety Block: The name '{body.name}' matches a known vendor pattern in ledger_rules (Rule ID {rule.id}). Companies must be client companies/tenants, not billing vendors."
                        )
                except re.error:
                    pass

        new_company = Company(
            name=body.name.strip(),
            gstin=gstin_clean,
            state_code=state_code,
            tally_host=body.tally_host.strip(),
            tally_port=body.tally_port,
            is_active=True
        )
        session.add(new_company)
        await session.flush()  # Obtain the autoincrement ID before commit

        logger.info("Company registered successfully: %r (ID: %d)", new_company.name, new_company.id)
        return CompanyResponse.model_validate(new_company)


@router.get("", response_model=list[CompanyResponse])
async def list_companies() -> list[CompanyResponse]:
    """
    List all registered and active companies.
    """
    async with get_db_session() as session:
        from sqlalchemy import select
        stmt = select(Company).where(Company.is_active.is_(True)).order_by(Company.id)
        res = await session.execute(stmt)
        companies = res.scalars().all()

    return [CompanyResponse.model_validate(c) for c in companies]


@router.get("/{company_id}", response_model=CompanyResponse)
async def get_company_details(company_id: int) -> CompanyResponse:
    """
    Get detailed company profile.
    """
    async with get_db_session() as session:
        from sqlalchemy import select
        stmt = select(Company).where(Company.id == company_id)
        res = await session.execute(stmt)
        company = res.scalars().first()

    if not company:
        raise HTTPException(
            status_code=404,
            detail=f"Company with ID {company_id} not found."
        )

    return CompanyResponse.model_validate(company)


@router.get("/{company_id}/documents")
async def get_company_documents(company_id: int) -> list[dict[str, Any]]:
    """
    Retrieve all processed document records belonging specifically to a company.
    """
    async with get_db_session() as session:
        from sqlalchemy import select
        stmt = select(Document).where(Document.company_id == company_id).order_by(Document.created_at.desc())
        db_res = await session.execute(stmt)
        documents = db_res.scalars().all()

    results = []
    for doc in documents:
        avg_confidence = 0.0
        if doc.pipeline_result_json:
            try:
                data = json.loads(doc.pipeline_result_json)
                segments = data.get("segments", [])
                if segments:
                    valid_scores = [
                        s.get("confidence_score", 0.0)
                        for s in segments
                        if s.get("status") != "failed" and s.get("confidence_score", 0.0) > 0
                    ]
                    if valid_scores:
                        avg_confidence = sum(valid_scores) / len(valid_scores)
            except Exception:
                pass
        results.append({
            "document_uuid": doc.document_uuid,
            "original_filename": doc.original_filename,
            "overall_status": doc.overall_status,
            "confidence_score": round(avg_confidence, 4),
            "created_at": doc.created_at,
            "processed_at": doc.processed_at
        })
    return results


@router.get("/{company_id}/stats", response_model=CompanyStatsResponse)
async def get_company_stats(company_id: int) -> CompanyStatsResponse:
    """
    Retrieve total spend analytics and audit trail counts scoped to a specific company.
    """
    async with get_db_session() as session:
        from sqlalchemy import select
        stmt = select(Document).where(Document.company_id == company_id)
        res = await session.execute(stmt)
        documents = res.scalars().all()

    total_invoices = len(documents)
    auto_post_count = 0
    pending_count = 0
    total_spend = 0.0
    spend_by_category: dict[str, float] = {}

    for doc in documents:
        if doc.overall_status == "auto_post":
            auto_post_count += 1
        elif doc.overall_status in ("full_review", "partial_review", "pending_review", "failed"):
            pending_count += 1

        if doc.pipeline_result_json:
            try:
                data = json.loads(doc.pipeline_result_json)
                segments = data.get("segments", [])
                for seg in segments:
                    if seg.get("status") != "failed":
                        inv = seg.get("extracted_invoice")
                        if inv:
                            grand_total = float(inv.get("grand_total") or 0.0)
                            category = seg.get("category") or "unknown"
                            spend_by_category[category] = spend_by_category.get(category, 0.0) + grand_total
                            total_spend += grand_total
            except Exception:
                pass

    # Round all calculated float values to 2 decimal places
    total_spend = round(total_spend, 2)
    for cat in spend_by_category:
        spend_by_category[cat] = round(spend_by_category[cat], 2)

    return CompanyStatsResponse(
        total_invoices=total_invoices,
        auto_post_count=auto_post_count,
        pending_count=pending_count,
        total_spend=total_spend,
        spend_by_category=spend_by_category
    )


@router.delete("/{company_id}", status_code=204)
async def delete_company(company_id: int):
    """
    Delete a company and all of its associated documents (cascade delete).
    """
    if company_id == 1:
        raise HTTPException(
            status_code=400,
            detail="The default company (KODRYX AI PRIVATE LIMITED) cannot be deleted."
        )

    async with get_db_session() as session:
        from sqlalchemy import select, delete
        # Check if company exists
        company_stmt = select(Company).where(Company.id == company_id)
        company_res = await session.execute(company_stmt)
        company = company_res.scalar_one_or_none()

        if not company:
            raise HTTPException(
                status_code=404,
                detail=f"Company with ID {company_id} not found."
            )

        # Delete all documents associated with the company
        docs_delete_stmt = delete(Document).where(Document.company_id == company_id)
        await session.execute(docs_delete_stmt)

        # Delete the company itself
        await session.execute(delete(Company).where(Company.id == company_id))
        
        logger.info("Company ID %d (%s) and its documents deleted successfully.", company_id, company.name)

    return None
