from __future__ import annotations

"""
Pipeline runner service — orchestrates the full document processing pipeline:
  1. SHA-256 deduplication check (fast, before any expensive work)
  2. OCR via Nemotron / Tesseract fallback
  3. Classification + Splitting
  4. Per-segment Enrichment (concurrent for bundles)
  5. Ledger mapping
  6. Persist result to Document record
"""

import asyncio
import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select

from app.agents.classifier_agent import classifier_node
from app.agents.enrichment_agent import ExtractionError, enrichment_node
from app.agents.ledger_agent import ledger_node
from app.agents.splitter_agent import splitter_node
from app.database import get_db_session
from app.integrations.ocr import nemotron_client
from app.models.document import Document
from app.schemas.ledger import LedgerMapping
from app.schemas.pipeline import (
    PipelineResult,
    SegmentResult,
    ValidationReport,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SHA-256 helpers
# ---------------------------------------------------------------------------


def compute_sha256(file_path: str) -> str:
    """Compute the SHA-256 hex digest of a file's binary content."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


async def _find_duplicate(sha256_hash: str) -> Optional[Document]:
    """Return an existing Document with the same hash, or None."""
    async with get_db_session() as session:
        stmt = select(Document).where(Document.sha256_hash == sha256_hash)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()


async def _save_document(
    document_uuid: str,
    file_path: str,
    sha256_hash: str,
    pipeline_result: PipelineResult,
    company_id: int = 1,
) -> None:
    """Persist the Document record after a successful pipeline run, or update it if a duplicate exists."""
    async with get_db_session() as session:
        # Check if a document with this sha256_hash already exists
        stmt = select(Document).where(Document.sha256_hash == sha256_hash)
        result = await session.execute(stmt)
        doc = result.scalar_one_or_none()

        if doc:
            # Update existing document and make sure the result json references the original doc UUID
            pipeline_result.document_uuid = doc.document_uuid
            doc.overall_status = pipeline_result.overall_status
            doc.pipeline_result_json = pipeline_result.model_dump_json()
            doc.processed_at = datetime.now(timezone.utc)
            logger.info("Existing document updated: uuid=%s status=%s", doc.document_uuid, doc.overall_status)
        else:
            # Create new document record
            doc = Document(
                document_uuid=document_uuid,
                original_filename=os.path.basename(file_path),
                file_path=file_path,
                file_size_bytes=os.path.getsize(file_path),
                sha256_hash=sha256_hash,
                overall_status=pipeline_result.overall_status,
                pipeline_result_json=pipeline_result.model_dump_json(),
                processed_at=datetime.now(timezone.utc),
                company_id=company_id,
            )
            session.add(doc)
            logger.info("New document saved: uuid=%s status=%s", document_uuid, pipeline_result.overall_status)


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


async def run_pipeline(file_path: str, document_uuid: str | None = None, company_id: int = 1) -> PipelineResult:
    """
    Full document processing pipeline with SHA-256 deduplication.

    Returns the cached PipelineResult immediately if the document was already
    processed. Otherwise runs the full pipeline and saves the result.
    """
    import time
    start_time = time.time()

    # ------------------------------------------------------------------
    # Step 0 — SHA-256 deduplication
    # ------------------------------------------------------------------
    sha256_hash = await asyncio.to_thread(compute_sha256, file_path)
    duplicate = await _find_duplicate(sha256_hash)

    if duplicate:
        logger.warning(
            "Duplicate document detected, skipping: %s (sha256=%s)",
            duplicate.document_uuid,
            sha256_hash,
        )
        if duplicate.pipeline_result_json:
            return PipelineResult.model_validate_json(duplicate.pipeline_result_json)
        # Fallback if result JSON is missing for some reason
        return PipelineResult(
            document_uuid=duplicate.document_uuid,
            file_path=file_path,
            overall_status="auto_post",
            segments=[],
            auto_post_count=0,
            review_count=0,
            processing_time_ms=0,
        )

    if not document_uuid:
        document_uuid = str(uuid.uuid4())

    # ------------------------------------------------------------------
    # Step 1 — OCR
    # ------------------------------------------------------------------
    logger.info("[%s] Starting OCR for %s", document_uuid, file_path)
    ocr_result = await nemotron_client.extract_text(file_path)

    # ------------------------------------------------------------------
    # Step 2 — Classification
    # ------------------------------------------------------------------
    logger.info("[%s] Classifying document", document_uuid)
    cls_state = await classifier_node({"ocr_result": ocr_result})
    classification = cls_state["classification"]

    # ------------------------------------------------------------------
    # Step 3 — Splitting
    # ------------------------------------------------------------------
    logger.info("[%s] Splitting into segments (doc_type=%s)", document_uuid, classification.doc_type)
    split_state = splitter_node({"ocr_result": ocr_result, "classification": classification})
    segments = split_state["segments"]

    # ------------------------------------------------------------------
    # Step 4 — Enrichment (concurrent for all segments)
    # ------------------------------------------------------------------
    logger.info("[%s] Enriching %d segment(s) concurrently", document_uuid, len(segments))
    enrichment_tasks = [
        enrichment_node({"segment": seg}) for seg in segments
    ]
    enrichment_results = await asyncio.gather(*enrichment_tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # Step 5 — Ledger mapping + build SegmentResults
    # ------------------------------------------------------------------
    segment_results: list[SegmentResult] = []
    auto_post_count = 0
    review_count = 0

    from app.config import settings

    for seg, enrich_outcome in zip(segments, enrichment_results):
        if isinstance(enrich_outcome, ExtractionError):
            logger.error(
                "[%s] Segment %s failed enrichment: %s",
                document_uuid, seg.segment_id, enrich_outcome,
            )
            segment_results.append(SegmentResult(
                segment_id=seg.segment_id,
                category=seg.category,
                extracted_invoice=None,
                ledger_mapping=None,
                validation_report=None,
                confidence_score=0.0,
                status="failed",
                failure_reason=str(enrich_outcome),
            ))
            review_count += 1
            continue

        if isinstance(enrich_outcome, BaseException):
            logger.error(
                "[%s] Segment %s unexpected error: %s",
                document_uuid, seg.segment_id, enrich_outcome,
            )
            segment_results.append(SegmentResult(
                segment_id=seg.segment_id,
                category=seg.category,
                extracted_invoice=None,
                ledger_mapping=None,
                validation_report=None,
                confidence_score=0.0,
                status="failed",
                failure_reason=str(enrich_outcome),
            ))
            review_count += 1
            continue

        extracted_invoice = enrich_outcome["extracted_invoice"]

        # Ledger mapping
        try:
            ledger_state = await ledger_node({"invoice": extracted_invoice, "company_id": company_id})
            ledger_mapping = ledger_state["ledger_mapping"]
        except Exception as e:
            logger.error(
                "[%s] Ledger mapping failed for segment %s: %s",
                document_uuid, seg.segment_id, e,
            )
            ledger_mapping = None

        # GST Validation
        from app.agents.gst_validator_agent import validate
        from app.agents.confidence_agent import compute_score
        
        validation_report = None
        if ledger_mapping:
            try:
                validation_report = await validate(extracted_invoice, ledger_mapping)
            except Exception as e:
                logger.error("[%s] GST validation failed for segment %s: %s",
                             document_uuid, seg.segment_id, e)

        # Full confidence score using all signals
        confidence = compute_score(
            ocr_result,
            extracted_invoice,
            ledger_mapping or LedgerMapping(
                debit_ledger="Miscellaneous Expenses",
                credit_ledger="Sundry Creditors",
                tax_type="cgst_sgst",
                match_type="fallback",
                confidence_penalty=0.20
            ),
            validation_report or ValidationReport(
                all_passed=False,
                checks=[],
                total_penalty=0.20,
                blocking_errors=["validation_skipped"]
            )
        )

        # Determine status
        if confidence >= settings.AUTO_POST_THRESHOLD:
            status = "auto_post"
            auto_post_count += 1
        else:
            status = "pending_review"
            review_count += 1

        segment_results.append(SegmentResult(
            segment_id=seg.segment_id,
            category=seg.category,
            extracted_invoice=extracted_invoice,
            ledger_mapping=ledger_mapping,
            validation_report=validation_report,  # was None before
            confidence_score=confidence,
            status=status,
            failure_reason=None,
        ))

    # ------------------------------------------------------------------
    # Step 6 — Compute overall status
    # ------------------------------------------------------------------
    total = len(segment_results)
    failed = sum(1 for s in segment_results if s.status == "failed")

    if failed == total:
        overall_status = "failed"
    elif auto_post_count == total:
        overall_status = "auto_post"
    elif review_count == total:
        overall_status = "full_review"
    else:
        overall_status = "partial_review"

    processing_time_ms = int((time.time() - start_time) * 1000)

    pipeline_result = PipelineResult(
        document_uuid=document_uuid,
        file_path=file_path,
        overall_status=overall_status,
        segments=segment_results,
        auto_post_count=auto_post_count,
        review_count=review_count,
        processing_time_ms=processing_time_ms,
    )

    # ------------------------------------------------------------------
    # Step 7 — Persist to DB
    # ------------------------------------------------------------------
    await _save_document(document_uuid, file_path, sha256_hash, pipeline_result, company_id)

    logger.info(
        "[%s] Pipeline complete in %dms — status=%s auto=%d review=%d",
        document_uuid, processing_time_ms, overall_status, auto_post_count, review_count,
    )

    return pipeline_result
