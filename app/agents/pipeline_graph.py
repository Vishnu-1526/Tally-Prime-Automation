from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.database import get_db_session
from app.models.document import Document
from app.schemas.invoice import ExtractedInvoice
from app.schemas.ledger import LedgerMapping
from app.schemas.ocr import OCRResult
from app.schemas.pipeline import (
    DocumentClassification,
    DocumentSegment,
    PipelineResult,
    SegmentResult,
    ValidationReport,
)

# Integrations and Agents
from app.integrations.ocr.nemotron_client import extract_text
from app.integrations.tally.client import TallyClient, TallyConnectionError
from app.agents.classifier_agent import classifier_node
from app.agents.splitter_agent import splitter_node
from app.agents.enrichment_agent import enrichment_node
from app.agents.ledger_agent import ledger_node
from app.agents.gst_validator_agent import validate
from app.agents.confidence_agent import compute_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline State
# ---------------------------------------------------------------------------


class PipelineState(TypedDict):
    document_uuid: str
    file_path: str
    ocr_result: OCRResult | None
    classification: DocumentClassification | None
    segments: list[DocumentSegment]
    segment_results: list[SegmentResult]
    errors: list[str]
    start_time: float
    pipeline_result: PipelineResult | None


# ---------------------------------------------------------------------------
# Node 1 — ocr
# ---------------------------------------------------------------------------


async def ocr_node(state: PipelineState) -> dict[str, Any]:
    """
    Node 1 — Calls OCR engine to extract text from file_path.
    """
    logger.info("Starting OCR extraction for file: %s", state["file_path"])
    try:
        ocr_res = await extract_text(state["file_path"])
        return {"ocr_result": ocr_res}
    except Exception as e:
        logger.error("OCR extraction failed: %s", e)
        return {
            "ocr_result": None,
            "errors": state["errors"] + [f"OCR extraction failed: {str(e)}"],
        }


def check_ocr_routing(state: PipelineState) -> str:
    """
    Routes after OCR completion. Directs to 'classify' on success, or exits early.
    """
    if state.get("ocr_result") is None:
        logger.error("OCR result is None, routing to END.")
        return "end"
    return "classify"


# ---------------------------------------------------------------------------
# Node 2 — classify
# ---------------------------------------------------------------------------


async def classify_node_wrapper(state: PipelineState) -> dict[str, Any]:
    """
    Node 2 — Runs document classifier agent.
    """
    logger.info("Running Document Classification.")
    res = await classifier_node({"ocr_result": state["ocr_result"]})
    return {"classification": res["classification"]}


# ---------------------------------------------------------------------------
# Node 3 — split
# ---------------------------------------------------------------------------


async def split_node_wrapper(state: PipelineState) -> dict[str, Any]:
    """
    Node 3 — Splits documents into logical segments based on classification results.
    """
    logger.info("Running Document Splitter.")
    res = splitter_node({
        "ocr_result": state["ocr_result"],
        "classification": state["classification"],
    })
    return {"segments": res["segments"]}


# ---------------------------------------------------------------------------
# Node 4 — process_segments
# ---------------------------------------------------------------------------


async def process_single_segment(
    segment: DocumentSegment, ocr_result: OCRResult
) -> SegmentResult:
    """
    Enriches, maps, validates, and scores a single document segment.
    """
    try:
        # a. Enrichment
        enrich_res = await enrichment_node({"segment": segment})
        invoice: ExtractedInvoice = enrich_res["extracted_invoice"]

        # b. Ledger Mapping
        ledger_res = await ledger_node({"invoice": invoice})
        ledger: LedgerMapping = ledger_res["ledger_mapping"]

        # c. Validation
        validation: ValidationReport = await validate(invoice, ledger)

        # d. Confidence Scoring
        score = compute_score(ocr_result, invoice, ledger, validation)

        # Determine auto_post status (score >= threshold & no blocking errors)
        has_blocking = len(validation.blocking_errors) > 0
        if score >= settings.AUTO_POST_THRESHOLD and not has_blocking:
            status = "auto_post"
        else:
            status = "pending_review"

        return SegmentResult(
            segment_id=segment.segment_id,
            category=segment.category,
            extracted_invoice=invoice,
            ledger_mapping=ledger,
            validation_report=validation,
            confidence_score=score,
            status=status,
            failure_reason=None,
        )

    except Exception as e:
        logger.error(
            "Failed to process segment %s: %s", segment.segment_id, e, exc_info=True
        )
        return SegmentResult(
            segment_id=segment.segment_id,
            category=segment.category,
            extracted_invoice=None,
            ledger_mapping=None,
            validation_report=None,
            confidence_score=0.0,
            status="failed",
            failure_reason=str(e),
        )


async def process_segments_node(state: PipelineState) -> dict[str, Any]:
    """
    Node 4 — Concurrently processes all document segments.
    """
    logger.info("Processing %d segment(s) concurrently.", len(state["segments"]))
    tasks = [
        process_single_segment(seg, state["ocr_result"])
        for seg in state["segments"]
    ]
    results = await asyncio.gather(*tasks)
    return {"segment_results": list(results)}


# ---------------------------------------------------------------------------
# Node 5 — assemble
# ---------------------------------------------------------------------------


async def assemble_node(state: PipelineState) -> dict[str, Any]:
    """
    Node 5 — Assembles the final PipelineResult and saves status to the database.
    """
    logger.info("Assembling pipeline results.")
    segment_results = state["segment_results"]
    total_segs = len(segment_results)
    
    auto_posts = sum(1 for r in segment_results if r.status == "auto_post")
    reviews = sum(
        1 for r in segment_results if r.status in {"pending_review", "failed"}
    )

    if total_segs == 0:
        overall_status = "failed"
    elif auto_posts == total_segs:
        overall_status = "auto_post"
    elif auto_posts == 0:
        overall_status = "full_review"
    else:
        overall_status = "partial_review"

    processing_time_ms = int((time.time() - state["start_time"]) * 1000)

    pipeline_result = PipelineResult(
        document_uuid=state["document_uuid"],
        file_path=state["file_path"],
        overall_status=overall_status,
        segments=segment_results,
        auto_post_count=auto_posts,
        review_count=reviews,
        processing_time_ms=processing_time_ms,
    )

    # Save to the database
    try:
        async with get_db_session() as session:
            from sqlalchemy import select
            stmt = select(Document).where(Document.document_uuid == state["document_uuid"])
            db_res = await session.execute(stmt)
            doc = db_res.scalar_one_or_none()
            if doc:
                from sqlalchemy import func
                doc.overall_status = overall_status
                doc.pipeline_result_json = pipeline_result.model_dump_json()
                doc.processed_at = func.now()
                logger.info("Saved pipeline results to DB for document %s", state["document_uuid"])
            else:
                logger.warning(
                    "No document found in DB for UUID %s (likely running mock / manual test)",
                    state["document_uuid"],
                )
    except Exception as db_err:
        logger.error("Failed to save pipeline result to DB: %s", db_err, exc_info=True)

    return {"pipeline_result": pipeline_result}


def check_assemble_routing(state: PipelineState) -> str:
    """
    Checks if there are auto-post vouchers that need posting to TallyPrime.
    """
    segment_results = state.get("segment_results", [])
    if any(r.status == "auto_post" for r in segment_results):
        logger.info("Found auto-post vouchers, routing to 'post_tally'.")
        return "post_tally"
    logger.info("No auto-post vouchers, routing to END.")
    return "end"


# ---------------------------------------------------------------------------
# Node 6 — post_tally
# ---------------------------------------------------------------------------


async def post_tally_node(state: PipelineState) -> dict[str, Any]:
    """
    Node 6 — Conditionally calls TallyClient to post all auto_post vouchers.
    """
    segment_results = list(state["segment_results"])
    pipeline_result = state["pipeline_result"]
    if not pipeline_result:
        return {}

    # Query custom Tally host and port for the company associated with this document
    tally_host = None
    tally_port = None
    try:
        async with get_db_session() as session:
            from sqlalchemy import select
            from app.models.company import Company
            stmt_comp = select(Company).join(Document, Document.company_id == Company.id).where(Document.document_uuid == state["document_uuid"])
            comp_res = await session.execute(stmt_comp)
            company = comp_res.scalar_one_or_none()
            if company:
                tally_host = company.tally_host
                tally_port = company.tally_port
    except Exception as e:
        logger.warning("Could not fetch company Tally configurations: %s", e)

    modified = False
    for res in segment_results:
        if res.status == "auto_post" and res.extracted_invoice and res.ledger_mapping:
            try:
                success = await TallyClient.post_voucher(
                    res.extracted_invoice,
                    res.ledger_mapping,
                    host=tally_host,
                    port=tally_port
                )
                if not success:
                    res.status = "pending_review"
                    res.failure_reason = "TallyPrime server rejected voucher."
                    modified = True
            except TallyConnectionError as conn_err:
                logger.warning(
                    "Tally connection failed for segment %s: %s. Moving to pending_review.",
                    res.segment_id,
                    conn_err,
                )
                res.status = "pending_review"
                res.failure_reason = f"Tally connection error: {str(conn_err)}"
                modified = True
            except Exception as e:
                logger.error("Unexpected error posting to Tally: %s", e, exc_info=True)
                res.status = "pending_review"
                res.failure_reason = f"Unexpected Tally posting error: {str(e)}"
                modified = True

    if modified:
        # Re-compute counts and overall status
        auto_posts = sum(1 for r in segment_results if r.status == "auto_post")
        reviews = sum(
            1 for r in segment_results if r.status in {"pending_review", "failed"}
        )

        if len(segment_results) == 0:
            overall_status = "failed"
        elif auto_posts == len(segment_results):
            overall_status = "auto_post"
        elif auto_posts == 0:
            overall_status = "full_review"
        else:
            overall_status = "partial_review"

        pipeline_result.overall_status = overall_status
        pipeline_result.auto_post_count = auto_posts
        pipeline_result.review_count = reviews

        # Sync update back to DB
        try:
            async with get_db_session() as session:
                from sqlalchemy import select
                stmt = select(Document).where(Document.document_uuid == state["document_uuid"])
                db_res = await session.execute(stmt)
                doc = db_res.scalar_one_or_none()
                if doc:
                    doc.overall_status = overall_status
                    doc.pipeline_result_json = pipeline_result.model_dump_json()
                    logger.info("Updated Tally-adjusted results in DB for document %s", state["document_uuid"])
        except Exception as db_err:
            logger.error("Failed to update database in post_tally_node: %s", db_err, exc_info=True)

    return {
        "segment_results": segment_results,
        "pipeline_result": pipeline_result,
    }


# ---------------------------------------------------------------------------
# Construct Workflow Graph
# ---------------------------------------------------------------------------

workflow = StateGraph(PipelineState)

# Add Nodes
workflow.add_node("ocr", ocr_node)
workflow.add_node("classify", classify_node_wrapper)
workflow.add_node("split", split_node_wrapper)
workflow.add_node("process_segments", process_segments_node)
workflow.add_node("assemble", assemble_node)
workflow.add_node("post_tally", post_tally_node)

# Add Edges
workflow.add_edge(START, "ocr")

workflow.add_conditional_edges(
    "ocr",
    check_ocr_routing,
    {
        "classify": "classify",
        "end": END,
    },
)

workflow.add_edge("classify", "split")
workflow.add_edge("split", "process_segments")
workflow.add_edge("process_segments", "assemble")

workflow.add_conditional_edges(
    "assemble",
    check_assemble_routing,
    {
        "post_tally": "post_tally",
        "end": END,
    },
)

workflow.add_edge("post_tally", END)

# Compile
pipeline = workflow.compile()


# ---------------------------------------------------------------------------
# Public Execution Interface
# ---------------------------------------------------------------------------


async def run_pipeline(document_uuid: str, file_path: str) -> PipelineResult:
    """
    Public async entry point to execute the complete LangGraph pipeline.
    """
    state = await pipeline.ainvoke({
        "document_uuid": document_uuid,
        "file_path": file_path,
        "ocr_result": None,
        "classification": None,
        "segments": [],
        "segment_results": [],
        "errors": [],
        "start_time": time.time(),
        "pipeline_result": None,
    })
    
    if state.get("pipeline_result") is None:
        raise RuntimeError(
            f"Pipeline failed to assemble results. Errors: {state.get('errors', [])}"
        )
        
    return state["pipeline_result"]
