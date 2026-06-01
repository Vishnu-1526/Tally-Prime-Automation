from __future__ import annotations

from app.schemas.ocr import OCRResult
from app.schemas.invoice import ExtractedInvoice
from app.schemas.ledger import LedgerMapping
from app.schemas.pipeline import ValidationReport


def compute_score(
    ocr: OCRResult,
    invoice: ExtractedInvoice,
    ledger: LedgerMapping,
    validation: ValidationReport
) -> float:
    """
    Computes a final composite confidence score (0.0 to 1.0) for a processed document segment
    by penalizing for low OCR quality, missing fields, weak ledger matches, and validation failures.
    """
    score = 1.0

    # OCR quality penalty: if overall quality < 80%, deduct half of the shortfall
    if ocr.overall_quality < 0.8:
        score -= (0.8 - ocr.overall_quality) * 0.5

    # Mandatory fields penalty: deduct 0.15 for each missing key field
    for field in ["vendor_name", "grand_total", "invoice_date"]:
        if getattr(invoice, field) is None:
            score -= 0.15

    # GSTIN missing: deduct 0.10 if vendor GSTIN is missing
    if invoice.vendor_gstin is None:
        score -= 0.10

    # Ledger match quality penalty
    score -= ledger.confidence_penalty

    # Validation penalties
    score -= validation.total_penalty

    # Ensure result is bounded between 0.0 and 1.0, rounded to 4 decimal places
    return round(max(0.0, min(1.0, score)), 4)
