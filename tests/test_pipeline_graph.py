from __future__ import annotations

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from app.agents.pipeline_graph import pipeline, run_pipeline
from app.schemas.ocr import OCRResult, PageResult
from app.schemas.pipeline import DocumentClassification, DocumentSegment, SegmentResult, PipelineResult
from app.schemas.invoice import ExtractedInvoice, LineItem
from app.schemas.ledger import LedgerMapping
from app.schemas.pipeline import ValidationReport, ValidationCheck


@pytest.mark.asyncio
async def test_run_pipeline_success():
    # Setup mocks for all individual agents and clients
    mock_ocr = OCRResult(
        file_path="/tmp/mock.pdf",
        page_count=1,
        pages=[PageResult(page_number=1, text="Invoice No: 123", quality_score=0.9, ocr_engine="nemotron")],
        overall_quality=0.9,
        ocr_engine="nemotron",
        processing_time_ms=10
    )

    mock_classification = DocumentClassification(
        doc_type="single_bill",
        segments=[
            DocumentSegment(
                segment_id="mock-seg-id",
                category="internet",
                suspected_category="internet",
                start_page=1,
                end_page=1,
                raw_text="Invoice No: 123"
            )
        ],
        classifier_confidence=0.9
    )

    mock_extracted_invoice = ExtractedInvoice(
        segment_id="mock-seg-id",
        vendor_name="Airtel",
        vendor_gstin="27AAAAA1111A1Z2",
        invoice_number="INV-123",
        invoice_date="2026-05-28",
        line_items=[
            LineItem(
                description="Broadband",
                hsn_code="9984",
                taxable_value=1000.0,
                igst_rate=18.0,
                igst_amount=180.0
            )
        ],
        subtotal=1000.0,
        total_igst=180.0,
        grand_total=1180.0,
        raw_text="Invoice No: 123"
    )

    mock_ledger = LedgerMapping(
        debit_ledger="Internet Expenses",
        credit_ledger="Sundry Creditors",
        igst_ledger="IGST @18%",
        tax_type="igst",
        match_type="exact",
        confidence_penalty=0.0
    )

    # All checks pass
    mock_validation = ValidationReport(
        all_passed=True,
        checks=[],
        total_penalty=0.0,
        blocking_errors=[]
    )

    # Patching dependencies
    with patch("app.agents.pipeline_graph.extract_text", new_callable=AsyncMock) as mock_extract_text_fn, \
         patch("app.agents.pipeline_graph.classifier_node", new_callable=AsyncMock) as mock_classifier_node_fn, \
         patch("app.agents.pipeline_graph.splitter_node", new_callable=MagicMock) as mock_splitter_node_fn, \
         patch("app.agents.pipeline_graph.enrichment_node", new_callable=AsyncMock) as mock_enrichment_node_fn, \
         patch("app.agents.pipeline_graph.ledger_node", new_callable=AsyncMock) as mock_ledger_node_fn, \
         patch("app.agents.pipeline_graph.validate", new_callable=AsyncMock) as mock_validate_fn, \
         patch("app.agents.pipeline_graph.compute_score", return_value=0.98) as mock_compute_score_fn, \
         patch("app.agents.pipeline_graph.TallyClient.post_voucher", new_callable=AsyncMock) as mock_post_voucher_fn, \
         patch("app.agents.pipeline_graph.get_db_session") as mock_db_session:

        # Configure mock return values
        mock_extract_text_fn.return_value = mock_ocr
        mock_classifier_node_fn.return_value = {"classification": mock_classification}
        mock_splitter_node_fn.return_value = {"segments": mock_classification.segments}
        mock_enrichment_node_fn.return_value = {"extracted_invoice": mock_extracted_invoice}
        mock_ledger_node_fn.return_value = {"ledger_mapping": mock_ledger}
        mock_validate_fn.return_value = mock_validation
        mock_post_voucher_fn.return_value = True

        # Mock database session
        mock_session = AsyncMock()
        mock_db_session.return_value.__aenter__.return_value = mock_session
        mock_execute_res = MagicMock()
        mock_execute_res.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_execute_res)

        # Run pipeline
        res = await run_pipeline("mock-doc-uuid", "/tmp/mock.pdf")

        # Assertions
        assert isinstance(res, PipelineResult)
        assert res.document_uuid == "mock-doc-uuid"
        assert res.overall_status == "auto_post"
        assert res.auto_post_count == 1
        assert res.review_count == 0
        assert len(res.segments) == 1
        assert res.segments[0].status == "auto_post"

        # Verify Tally client was invoked for auto-post voucher
        mock_post_voucher_fn.assert_called_once_with(mock_extracted_invoice, mock_ledger, host=None, port=None)
