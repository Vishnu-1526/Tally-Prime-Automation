from __future__ import annotations

"""
Splitter Agent — LangGraph node that uses a DocumentClassification to split
the OCR text into concrete DocumentSegments with UUIDs.

  - single_bill    → one segment, full text
  - bundle         → one segment per classification segment, pages concatenated
  - expense_report → split on ALL-CAPS section headers / "Category:" markers
  - unknown        → one segment, category "unknown"
"""

import logging
import re
import uuid
from typing import Any

from app.agents.classifier_agent import _detect_category
from app.schemas.ocr import OCRResult
from app.schemas.pipeline import DocumentClassification, DocumentSegment

logger = logging.getLogger(__name__)

_RE_SECTION_HEADER = re.compile(
    r"^([A-Z][A-Z\s]{3,}|Category\s*:|Expense\s*:)", re.MULTILINE
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _concat_pages(ocr: OCRResult, start_page: int, end_page: int) -> str:
    """Concatenate text from pages in [start_page, end_page] (1-indexed, inclusive)."""
    return "\n".join(
        p.text
        for p in ocr.pages
        if start_page <= p.page_number <= end_page
    )


def _make_segment(
    text: str,
    category: str,
    start_page: int,
    end_page: int,
) -> DocumentSegment:
    return DocumentSegment(
        segment_id=str(uuid.uuid4()),
        category=category,
        suspected_category=category,
        start_page=start_page,
        end_page=end_page,
        raw_text=text,
    )


# ---------------------------------------------------------------------------
# Split strategies
# ---------------------------------------------------------------------------


def _split_single_bill(ocr: OCRResult, cls: DocumentClassification) -> list[DocumentSegment]:
    """One segment covering all pages."""
    full_text = _concat_pages(ocr, 1, ocr.page_count)
    # Reuse category from classifier if available
    category = cls.segments[0].category if cls.segments else "unknown"
    return [_make_segment(full_text, category, 1, ocr.page_count)]


def _split_bundle(ocr: OCRResult, cls: DocumentClassification) -> list[DocumentSegment]:
    """
    One DocumentSegment per segment the classifier identified.
    Concatenates all pages that fall within [start_page, end_page].
    """
    segments: list[DocumentSegment] = []
    for seg in cls.segments:
        text = _concat_pages(ocr, seg.start_page, seg.end_page)
        segments.append(
            _make_segment(text, seg.category, seg.start_page, seg.end_page)
        )
    # Fallback: if classifier produced no segments, return one big segment
    if not segments:
        full = _concat_pages(ocr, 1, ocr.page_count)
        segments.append(_make_segment(full, "unknown", 1, ocr.page_count))
    return segments


def _split_expense_report(ocr: OCRResult, cls: DocumentClassification) -> list[DocumentSegment]:
    """
    Split on ALL-CAPS lines or "Category:" / "Expense:" section markers.
    Each chunk gets its own UUID segment.
    """
    full_text = _concat_pages(ocr, 1, ocr.page_count)
    lines = full_text.splitlines()

    # Identify split points — lines that look like section headers
    groups: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and _RE_SECTION_HEADER.match(stripped) and current:
            groups.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        groups.append(current)

    segments: list[DocumentSegment] = []
    for group in groups:
        text = "\n".join(group).strip()
        if not text:
            continue
        # Score each chunk's text directly against the keyword registry
        category = _detect_category(text)
        segments.append(_make_segment(text, category, 1, ocr.page_count))

    if not segments:
        segments.append(_make_segment(full_text, "unknown", 1, ocr.page_count))
    return segments


def _split_unknown(ocr: OCRResult) -> list[DocumentSegment]:
    """Return a single segment flagged as unknown."""
    full_text = _concat_pages(ocr, 1, ocr.page_count)
    return [_make_segment(full_text, "unknown", 1, ocr.page_count)]


# ---------------------------------------------------------------------------
# LangGraph node entry point
# ---------------------------------------------------------------------------


def splitter_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node — synchronous (no LLM calls, pure logic).
    Input state keys:  ocr_result (OCRResult), classification (DocumentClassification)
    Output state keys: segments (list[DocumentSegment])
    """
    ocr: OCRResult = state["ocr_result"]
    cls: DocumentClassification = state["classification"]

    doc_type = cls.doc_type
    logger.info("Splitter — doc_type: %s, classifier segments: %d", doc_type, len(cls.segments))

    if doc_type == "single_bill":
        segments = _split_single_bill(ocr, cls)
    elif doc_type == "bundle":
        segments = _split_bundle(ocr, cls)
    elif doc_type == "expense_report":
        segments = _split_expense_report(ocr, cls)
    else:
        segments = _split_unknown(ocr)

    logger.info("Splitter produced %d segment(s)", len(segments))
    return {"segments": segments}
