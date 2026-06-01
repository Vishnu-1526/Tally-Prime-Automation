from __future__ import annotations

"""
Classifier Agent — LangGraph node that inspects an OCRResult and produces a
DocumentClassification.  It runs three steps in order, stopping at the first
conclusive match:

  1. Heuristic classifier  (regex / pattern counting — no LLM)
  2. Local Ollama LLM call (llama3) when heuristics are ambiguous
  3. Segment detector      (invoice-boundary detection + keyword matching)

Key design choices:
  - For bundle PDFs from the SAME vendor (e.g. Jio multi-month bill pack),
    each individual invoice is detected by "Invoice Date" / "Document Number"
    markers appearing on a page, not by GSTIN (which is the same on all pages).
  - Cover/summary/history pages that contain no invoice-start marker and no
    meaningful financial data are discarded — they don't become segments.
  - Segments are sorted newest-first: the most recent invoice date is Segment 1.
"""

import logging
import re
import uuid
from datetime import date, datetime
from typing import Any, Optional

import httpx

from app.config import settings
from app.schemas.ocr import OCRResult
from app.schemas.pipeline import DocumentClassification, DocumentSegment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Category keyword registry
# ---------------------------------------------------------------------------

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "electricity": [
        "TSSPDCL", "APEPDCL", "electricity board", "EB bill",
        "units consumed", "meter reading",
        "BESCOM", "MSEDCL", "CESC", "WBSEDCL", "TANGEDCO",
        "UHBVN", "UPPCL", "units", "kWh", "meter no",
    ],
    "internet": [
        "broadband", "internet", "wifi", "Airtel", "JioFiber",
        "ACT Fibernet", "data charges", "connectivity services",
        "platform services", "reliance jio", "jio digital",
        "BSNL", "Hathway", "YOU Broadband", "Excitel",
        "monthly rental", "data plan", "broadband charges",
    ],
    "rent": [
        "rent", "lease", "premises", "landlord", "tenancy",
        "rental agreement", "lease deed", "deposit", "sq ft",
        "square feet", "commercial premises",
    ],
    "travel": [
        "cab", "flight", "hotel", "boarding pass", "Ola", "Uber",
        "IRCTC", "IndiGo",
        "PNR", "seat no", "fare breakup",
        "toll charges", "fuel reimbursement",
    ],
    "pantry": [
        "water", "canteen", "pantry", "beverages", "snacks", "refreshments",
        "drinking water", "aquaguard", "20L jar", "tea",
        "coffee", "biscuits", "stationery",
    ],
    "infrastructure": [
        "maintenance", "AMC", "annual maintenance",
        "security", "housekeeping",
        "generator", "UPS", "CCTV", "pest control",
        "lift maintenance", "elevator",
        "laptop", "switch", "router", "HMI", "panel",
        "server", "computer", "hardware", "printer", "sensor",
    ],
    "professional_services": [
        "consultancy", "professional fees", "retainer", "legal fees",
        "audit fees", "CA fees", "chartered accountant",
    ],
}

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_RE_INVOICE_NO = re.compile(
    r"\b(invoice\s*no|invoice\s*#|bill\s*no|bill\s*#|inv\s*no|document\s*number)[.:\s]*[A-Z0-9/-]+",
    re.IGNORECASE,
)
_RE_GSTIN = re.compile(
    r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b"
)
_RE_GRAND_TOTAL = re.compile(
    r"\b(grand\s*total|total\s*amount|amount\s*payable|current\s*charges)\b",
    re.IGNORECASE,
)
# Strong indicator that a page is the START of a new invoice
# (as opposed to a continuation page or cover/summary page)
_RE_INVOICE_START = re.compile(
    r"""
    (?:
        \b(?:invoice|bill|tax\s+invoice|gst\s+invoice|original\s+for\s+recipient)\b
        |
        \b(?:invoice\s*date|bill\s*date|document\s*date)\s*[:\-]?\s*\d
        |
        \b(?:invoice\s*no|document\s*number|bill\s*no)\s*[:\-]?\s*[A-Z0-9]
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
# Extract the most prominent date from a page for sorting
_RE_DATE = re.compile(
    r"""
    (?:
        (?:invoice\s*date|bill\s*date|document\s*date)\s*[:\-]?\s*
    )?
    (\d{1,2}[-/\s]\w{3,9}[-/\s]\d{4}   # 01-Dec-2025 / 1 December 2025
    |\d{4}[-/]\d{2}[-/]\d{2}             # 2025-12-01
    |\d{1,2}[-/]\d{1,2}[-/]\d{4})        # 01/12/2025
    """,
    re.IGNORECASE | re.VERBOSE,
)
_DATE_FORMATS = [
    "%d-%b-%Y", "%d %b %Y", "%d %B %Y",
    "%Y-%m-%d",
    "%d/%m/%Y", "%d-%m-%Y",
    "%d-%b-%y", "%d %b %y",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _full_text(ocr: OCRResult) -> str:
    return "\n".join(p.text for p in ocr.pages)


def _detect_category(text: str) -> str:
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = 0
        for kw in keywords:
            pattern = r"\b" + re.escape(kw.lower()) + r"\b"
            if re.search(pattern, text_lower):
                score += 1
        if score:
            scores[category] = score
    if scores:
        return max(scores, key=lambda k: scores[k])
    return "unknown"


def _build_segment(
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


def _parse_date(text: str) -> Optional[date]:
    """Extract and parse the first recognisable date from a block of text."""
    for m in _RE_DATE.finditer(text):
        raw = m.group(1).strip()
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(raw, fmt).date()
            except ValueError:
                continue
    return None


def _is_invoice_start_page(text: str) -> bool:
    """
    Return True if this page is the beginning of a new GST invoice.
    We require at least 2 strong signals to avoid false positives on
    continuation / cover pages.
    """
    signals = 0
    if _RE_INVOICE_START.search(text):
        signals += 1
    if _RE_GSTIN.search(text):
        signals += 1
    if _RE_INVOICE_NO.search(text):
        signals += 1
    # A page must have ≥2 signals to be treated as an invoice start
    return signals >= 2


def _page_has_financial_data(text: str) -> bool:
    """
    Return True if the page carries meaningful invoice financial data.
    Used to discard pure cover / T&C / payment-history pages.
    
    For a page group to qualify as a real invoice segment it must have:
    - A GSTIN (every legit GST invoice must have one), OR
    - An explicit tax total line (grand total / amount payable) with amounts
    
    Simple amount numbers alone are NOT enough — summary pages have amounts too.
    """
    has_gstin = bool(_RE_GSTIN.search(text))
    has_total = bool(_RE_GRAND_TOTAL.search(text))
    has_tax_line = bool(re.search(
        r"(cgst|sgst|igst)\s*[\(\[]?\s*\d+\s*%?\s*[\)\]]?\s*[:\-]?\s*[\d,]+",
        text, re.IGNORECASE
    ))
    return has_gstin or (has_total and has_tax_line)


# ---------------------------------------------------------------------------
# Step 1 — Heuristic classifier
# ---------------------------------------------------------------------------


def _heuristic_classify(ocr: OCRResult) -> Optional[str]:
    """
    Returns "single_bill", "bundle", "expense_report", or None (ambiguous).
    """
    full = _full_text(ocr)

    invoice_matches = set(_RE_INVOICE_NO.findall(full))
    gstin_matches   = set(_RE_GSTIN.findall(full))
    total_matches   = _RE_GRAND_TOTAL.findall(full)

    invoice_count = len(invoice_matches)
    gstin_count   = len(gstin_matches)
    total_count   = len(total_matches)

    logger.debug(
        "Heuristic counts — invoices: %d, GSTINs: %d, totals: %d",
        invoice_count, gstin_count, total_count,
    )

    # Multiple distinct GSTINs → clearly a bundle from different vendors
    if gstin_count > 1:
        return "bundle"

    # Multiple invoice numbers or totals — could still be same vendor, mark bundle
    if invoice_count > 1 or total_count > 1:
        return "bundle"

    # Clear single-bill signal
    if ocr.page_count == 1 and max(invoice_count, gstin_count, total_count) <= 1:
        return "single_bill"

    # Multi-page but only one set of identifiers — still single_bill
    if invoice_count <= 1 and gstin_count <= 1 and total_count <= 1 and total_count >= 1:
        return "single_bill"

    return None  # ambiguous


# ---------------------------------------------------------------------------
# Step 2 — LLM fallback (Ollama / llama3)
# ---------------------------------------------------------------------------


async def _llm_classify(first_500: str) -> str:
    prompt = (
        "You are analyzing a financial document. Based on this text, "
        "determine if it contains: (A) a single invoice from one vendor, "
        "(B) multiple invoices bundled together, or (C) an expense report. "
        "Reply with only: SINGLE, BUNDLE, or EXPENSE_REPORT.\n"
        f"Text: {first_500}"
    )
    payload = {"model": "llama3", "prompt": prompt, "stream": False}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate", json=payload
            )
            if resp.status_code == 200:
                raw = resp.json().get("response", "").strip().upper()
                if "SINGLE" in raw:
                    return "single_bill"
                if "BUNDLE" in raw:
                    return "bundle"
                if "EXPENSE" in raw:
                    return "expense_report"
    except Exception as e:
        logger.warning("LLM classifier call failed: %s", e)
    return "unknown"


# ---------------------------------------------------------------------------
# Step 3 — Segment detector
# ---------------------------------------------------------------------------


def _detect_segments(ocr: OCRResult, doc_type: str) -> list[DocumentSegment]:
    """Build DocumentSegment list appropriate for the detected doc_type."""
    pages = ocr.pages

    # -----------------------------------------------------------------------
    # SINGLE BILL — all pages belong to one invoice
    # -----------------------------------------------------------------------
    if doc_type == "single_bill":
        full = _full_text(ocr)
        category = _detect_category(full)
        return [_build_segment(full, category, 1, ocr.page_count)]

    # -----------------------------------------------------------------------
    # BUNDLE — detect invoice boundaries page-by-page
    #
    # Strategy:
    #   1. Walk each page and check if it is the START of a new invoice.
    #   2. Group consecutive pages into one invoice block.
    #   3. Skip page groups that have no financial data at all (cover pages).
    #   4. Sort segments newest-first by detected invoice date.
    # -----------------------------------------------------------------------
    if doc_type == "bundle":
        # Group pages into invoice blocks
        # A new block starts when a page has ≥2 invoice-start signals.
        blocks: list[list] = []   # list of page groups
        current_block: list = []

        for page in pages:
            if _is_invoice_start_page(page.text):
                if current_block:
                    blocks.append(current_block)
                current_block = [page]
            else:
                current_block.append(page)

        if current_block:
            blocks.append(current_block)

        # If we got only one block (couldn't split), fall back to GSTIN grouping
        if len(blocks) <= 1:
            logger.info("Invoice-start boundary detection found only 1 block; falling back to GSTIN grouping.")
            blocks = _gstin_group_pages(pages)

        # Build segments, filtering out pure cover/summary pages
        segments: list[DocumentSegment] = []
        for grp in blocks:
            combined = "\n".join(p.text for p in grp)
            # Skip pages with no financial data whatsoever (cover/T&C pages)
            if not _page_has_financial_data(combined):
                logger.info(
                    "Skipping non-invoice block (pages %d-%d) — no financial data.",
                    grp[0].page_number, grp[-1].page_number,
                )
                continue
            category = _detect_category(combined)
            start = grp[0].page_number
            end   = grp[-1].page_number
            segments.append(_build_segment(combined, category, start, end))

        # Sort segments newest-first by invoice date
        segments = _sort_newest_first(segments)

        if not segments:
            # All pages were discarded as cover pages — return full document
            full = _full_text(ocr)
            segments = [_build_segment(full, _detect_category(full), 1, ocr.page_count)]

        return segments

    # -----------------------------------------------------------------------
    # EXPENSE REPORT — split on section headers
    # -----------------------------------------------------------------------
    if doc_type == "expense_report":
        _RE_SECTION = re.compile(
            r"^([A-Z][A-Z\s]{3,}|Category\s*:|Expense\s*:)", re.MULTILINE
        )
        full = _full_text(ocr)
        splits = _RE_SECTION.split(full)
        segments = []
        for chunk in splits:
            chunk = chunk.strip()
            if chunk and _page_has_financial_data(chunk):
                category = _detect_category(chunk)
                segments.append(_build_segment(chunk, category, 1, ocr.page_count))
        return segments or [_build_segment(full, "unknown", 1, ocr.page_count)]

    # UNKNOWN — single segment
    full = _full_text(ocr)
    return [_build_segment(full, "unknown", 1, ocr.page_count)]


def _gstin_group_pages(pages: list) -> list[list]:
    """
    Fallback: group pages by the first GSTIN found on each page.
    Pages with no GSTIN are appended to the previous group (continuation pages).
    """
    groups: dict[str, list] = {}
    last_key = None
    for page in pages:
        gstins = _RE_GSTIN.findall(page.text)
        if gstins:
            key = gstins[0]
            last_key = key
        else:
            key = last_key or f"page_{page.page_number}"
        groups.setdefault(key, []).append(page)
    return list(groups.values())


def _sort_newest_first(segments: list[DocumentSegment]) -> list[DocumentSegment]:
    """
    Sort segments so the most recent invoice date appears first (Segment 1).
    Segments with no detectable date are pushed to the end.
    """
    def _key(seg: DocumentSegment):
        d = _parse_date(seg.raw_text)
        if d is None:
            return date.min          # no date → goes to the end
        return d

    return sorted(segments, key=_key, reverse=True)


# ---------------------------------------------------------------------------
# LangGraph node entry point
# ---------------------------------------------------------------------------


async def classifier_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node.
    Input state keys:  ocr_result (OCRResult)
    Output state keys: classification (DocumentClassification)
    """
    ocr: OCRResult = state["ocr_result"]
    full_text = _full_text(ocr)

    # Step 1 — Heuristic
    doc_type = _heuristic_classify(ocr)
    heuristic_succeeded = doc_type is not None
    logger.info("Heuristic doc_type: %s", doc_type)

    # Step 2 — LLM fallback if ambiguous
    if doc_type is None:
        doc_type = await _llm_classify(full_text[:500])
        logger.info("LLM doc_type: %s", doc_type)

    if doc_type not in {"single_bill", "bundle", "expense_report", "unknown"}:
        doc_type = "unknown"

    # Step 3 — Segment detection
    segments = _detect_segments(ocr, doc_type)
    logger.info(
        "Classifier produced %d segment(s) for doc_type=%s", len(segments), doc_type
    )

    classifier_confidence = (
        0.9 if heuristic_succeeded
        else 0.75 if doc_type != "unknown"
        else 0.4
    )

    classification = DocumentClassification(
        doc_type=doc_type,
        segments=segments,
        classifier_confidence=classifier_confidence,
    )

    return {"classification": classification}
