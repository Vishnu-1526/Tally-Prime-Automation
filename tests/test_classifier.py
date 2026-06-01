"""
Unit tests for classifier_agent and splitter_agent.
No real files or network calls — all tests use synthetic OCR text strings.
"""
from __future__ import annotations

import asyncio

import pytest

from app.agents.classifier_agent import (
    CATEGORY_KEYWORDS,
    _detect_category,
    _detect_segments,
    _heuristic_classify,
    classifier_node,
)
from app.agents.splitter_agent import splitter_node
from app.schemas.ocr import OCRResult, PageResult
from app.schemas.pipeline import DocumentClassification, DocumentSegment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_page(page_number: int, text: str) -> PageResult:
    return PageResult(
        page_number=page_number,
        text=text,
        quality_score=1.0,
        bounding_boxes=[],
        ocr_engine="nemotron",
    )


def _make_ocr(pages: list[PageResult]) -> OCRResult:
    return OCRResult(
        file_path="/tmp/test.pdf",
        page_count=len(pages),
        pages=pages,
        overall_quality=1.0,
        ocr_engine="nemotron",
        processing_time_ms=100,
    )


# ---------------------------------------------------------------------------
# Category keyword tests
# ---------------------------------------------------------------------------


class TestCategoryDetection:
    def test_electricity_detected(self):
        text = "BESCOM electricity bill, units consumed: 120 kWh, meter no: 12345"
        assert _detect_category(text) == "electricity"

    def test_internet_detected(self):
        text = "Airtel broadband monthly rental - data plan invoice"
        assert _detect_category(text) == "internet"

    def test_rent_detected(self):
        text = "Rent agreement for commercial premises 800 sq ft, deposit paid"
        assert _detect_category(text) == "rent"

    def test_travel_detected(self):
        text = "IndiGo boarding pass PNR: ABC123, fare breakup included"
        assert _detect_category(text) == "travel"

    def test_pantry_detected(self):
        text = "Aquaguard 20L jar drinking water, tea and coffee for canteen"
        assert _detect_category(text) == "pantry"

    def test_infrastructure_detected(self):
        text = "Annual maintenance AMC for CCTV, generator service and UPS"
        assert _detect_category(text) == "infrastructure"

    def test_professional_services_detected(self):
        text = "Professional fees for chartered accountant, CA fees for audit"
        assert _detect_category(text) == "professional_services"

    def test_unknown_when_no_match(self):
        text = "Some random text without any category keywords"
        assert _detect_category(text) == "unknown"

    def test_all_categories_have_keywords(self):
        """Every category must have at least one keyword."""
        for cat, kws in CATEGORY_KEYWORDS.items():
            assert len(kws) > 0, f"Category '{cat}' has no keywords"


# ---------------------------------------------------------------------------
# Heuristic classifier tests
# ---------------------------------------------------------------------------


class TestHeuristicClassifier:
    def test_single_page_single_gstin_is_single_bill(self):
        text = (
            "Invoice No: INV-001\n"
            "GSTIN: 27AABCS1429B1Z5\n"
            "Grand Total: ₹5000"
        )
        ocr = _make_ocr([_make_page(1, text)])
        result = _heuristic_classify(ocr)
        assert result == "single_bill"

    def test_two_distinct_gstins_is_bundle(self):
        text = (
            "Invoice No: INV-001\n"
            "GSTIN: 27AABCS1429B1Z5\n"
            "Grand Total: ₹5000\n\n"
            "Invoice No: INV-002\n"
            "GSTIN: 29AABCS1429B1Z5\n"
            "Grand Total: ₹3000"
        )
        ocr = _make_ocr([_make_page(1, text)])
        result = _heuristic_classify(ocr)
        assert result == "bundle"

    def test_two_invoice_numbers_is_bundle(self):
        page1 = "Invoice No: INV-2024-001\nVendor A\nGrand Total: 1000"
        page2 = "Invoice No: INV-2024-002\nVendor B\nGrand Total: 2000"
        ocr = _make_ocr([_make_page(1, page1), _make_page(2, page2)])
        result = _heuristic_classify(ocr)
        assert result == "bundle"

    def test_multipage_single_invoice_is_single_bill(self):
        page1 = "Invoice No: INV-001\nGSTIN: 27AABCS1429B1Z5\nItem details..."
        page2 = "Continued items...\nGrand Total: ₹50000"
        ocr = _make_ocr([_make_page(1, page1), _make_page(2, page2)])
        result = _heuristic_classify(ocr)
        assert result == "single_bill"


# ---------------------------------------------------------------------------
# Segment detection tests
# ---------------------------------------------------------------------------


class TestSegmentDetection:
    def test_single_bill_one_segment(self):
        ocr = _make_ocr([_make_page(1, "Invoice No: INV-001\nBESCOM electricity bill\nGrand Total: 1500")])
        segs = _detect_segments(ocr, "single_bill")
        assert len(segs) == 1
        assert segs[0].category == "electricity"

    def test_unknown_segment_has_unknown_category(self):
        ocr = _make_ocr([_make_page(1, "Random unrelated text")])
        segs = _detect_segments(ocr, "unknown")
        assert len(segs) == 1
        assert segs[0].category == "unknown"

    def test_segment_has_uuid(self):
        ocr = _make_ocr([_make_page(1, "Invoice No: INV-001\nGrand Total: 1000")])
        segs = _detect_segments(ocr, "single_bill")
        assert len(segs[0].segment_id) == 36  # UUID4 string length


# ---------------------------------------------------------------------------
# Full classifier_node tests (async)
# ---------------------------------------------------------------------------


class TestClassifierNode:
    def test_single_bill_classification(self):
        text = (
            "Invoice No: INV-2024-001\n"
            "GSTIN: 27AABCS1429B1Z5\n"
            "Airtel broadband monthly rental\n"
            "Grand Total: ₹1180"
        )
        ocr = _make_ocr([_make_page(1, text)])
        result = asyncio.run(classifier_node({"ocr_result": ocr}))
        cls: DocumentClassification = result["classification"]
        assert cls.doc_type == "single_bill"
        assert cls.classifier_confidence > 0
        assert len(cls.segments) == 1
        assert cls.segments[0].category == "internet"

    def test_bundle_classification(self):
        text = (
            "Invoice No: INV-001\nGSTIN: 27AABCS1429B1Z5\nGrand Total: 1000\n\n"
            "Invoice No: INV-002\nGSTIN: 29AABCS1429B1Z5\nGrand Total: 2000"
        )
        ocr = _make_ocr([_make_page(1, text)])
        result = asyncio.run(classifier_node({"ocr_result": ocr}))
        cls: DocumentClassification = result["classification"]
        assert cls.doc_type == "bundle"
        assert len(cls.segments) >= 1


# ---------------------------------------------------------------------------
# Splitter node tests
# ---------------------------------------------------------------------------


class TestSplitterNode:
    def _run_splitter(self, ocr: OCRResult, doc_type: str, segments: list) -> list[DocumentSegment]:
        cls = DocumentClassification(
            doc_type=doc_type,
            segments=segments,
            classifier_confidence=0.9,
        )
        result = splitter_node({"ocr_result": ocr, "classification": cls})
        return result["segments"]

    def test_single_bill_one_segment(self):
        ocr = _make_ocr([_make_page(1, "Invoice No: INV-001\nGrand Total: 500")])
        segs = self._run_splitter(ocr, "single_bill", [
            DocumentSegment(
                segment_id="test-id",
                category="internet",
                suspected_category="internet",
                start_page=1,
                end_page=1,
                raw_text="Invoice No: INV-001\nGrand Total: 500",
            )
        ])
        assert len(segs) == 1
        assert "Invoice No" in segs[0].raw_text

    def test_bundle_two_segments(self):
        ocr = _make_ocr([
            _make_page(1, "Invoice No: INV-001\nVendor A\nGSTIN: 27AABCS1429B1Z5"),
            _make_page(2, "Invoice No: INV-002\nVendor B\nGSTIN: 29AABCS1429B1Z5"),
        ])
        cls_segs = [
            DocumentSegment(
                segment_id="seg-1",
                category="electricity",
                suspected_category="electricity",
                start_page=1, end_page=1,
                raw_text="Invoice No: INV-001\nVendor A",
            ),
            DocumentSegment(
                segment_id="seg-2",
                category="internet",
                suspected_category="internet",
                start_page=2, end_page=2,
                raw_text="Invoice No: INV-002\nVendor B",
            ),
        ]
        segs = self._run_splitter(ocr, "bundle", cls_segs)
        assert len(segs) == 2
        assert segs[0].start_page == 1
        assert segs[1].start_page == 2

    def test_unknown_single_segment(self):
        ocr = _make_ocr([_make_page(1, "Unknown document")])
        segs = self._run_splitter(ocr, "unknown", [])
        assert len(segs) == 1
        assert segs[0].category == "unknown"

    def test_all_segments_have_uuid(self):
        ocr = _make_ocr([_make_page(1, "Invoice No: INV-001\nGrand Total: 1000")])
        segs = self._run_splitter(ocr, "single_bill", [
            DocumentSegment(
                segment_id="dummy",
                category="unknown",
                suspected_category="unknown",
                start_page=1, end_page=1,
                raw_text="Invoice No: INV-001",
            )
        ])
        for seg in segs:
            assert len(seg.segment_id) == 36, f"Segment ID '{seg.segment_id}' is not a UUID"
