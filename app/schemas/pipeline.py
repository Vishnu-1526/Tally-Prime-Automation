from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel

from app.schemas.invoice import ExtractedInvoice
from app.schemas.ledger import LedgerMapping


class DocumentSegment(BaseModel):
    segment_id: str
    category: str
    suspected_category: str
    start_page: int
    end_page: int
    raw_text: str


class DocumentClassification(BaseModel):
    doc_type: Literal["single_bill", "bundle", "expense_report", "unknown"]
    segments: list[DocumentSegment]
    classifier_confidence: float


class ValidationCheck(BaseModel):
    check_name: str
    passed: bool
    message: str
    severity: Literal["error", "warning"]
    confidence_penalty: float


class ValidationReport(BaseModel):
    all_passed: bool
    checks: list[ValidationCheck]
    total_penalty: float
    blocking_errors: list[str]


class SegmentResult(BaseModel):
    segment_id: str
    category: str
    extracted_invoice: Optional[ExtractedInvoice] = None
    ledger_mapping: Optional[LedgerMapping] = None
    validation_report: Optional[ValidationReport] = None
    confidence_score: float
    status: Literal["auto_post", "pending_review", "failed"]
    failure_reason: Optional[str] = None


class PipelineResult(BaseModel):
    document_uuid: str
    file_path: str
    overall_status: Literal["auto_post", "partial_review", "full_review", "failed"]
    segments: list[SegmentResult]
    auto_post_count: int
    review_count: int
    processing_time_ms: int
