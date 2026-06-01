from __future__ import annotations

import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.pipeline import PipelineResult, SegmentResult
from app.schemas.invoice import ExtractedInvoice
from app.schemas.ledger import LedgerMapping


client = TestClient(app)


def test_upload_invalid_file_type_fails():
    # Attempting to upload a TXT file instead of PDF/Images
    response = client.post(
        "/api/documents/upload",
        files={"file": ("test.txt", b"some txt text content", "text/plain")}
    )
    assert response.status_code == 422
    assert "Unsupported file format" in response.json()["detail"]


@pytest.mark.asyncio
async def test_upload_valid_document_success():
    # Setup mock pipeline result
    mock_pipeline_res = PipelineResult(
        document_uuid="test-api-uuid",
        file_path="storage/test-api-uuid/invoice.pdf",
        overall_status="auto_post",
        segments=[],
        auto_post_count=0,
        review_count=0,
        processing_time_ms=50
    )

    with patch("app.api.documents.run_pipeline", new_callable=AsyncMock) as mock_run_pipeline, \
         patch("app.api.documents._find_duplicate", new_callable=AsyncMock) as mock_find_dup, \
         patch("app.api.documents.compute_sha256", return_value="mock-sha256"), \
         patch("os.makedirs"), \
         patch("builtins.open", MagicMock()):

        mock_find_dup.return_value = None
        mock_run_pipeline.return_value = mock_pipeline_res

        response = client.post(
            "/api/documents/upload",
            files={"file": ("invoice.pdf", b"%PDF-1.4 mock content", "application/pdf")}
        )

        assert response.status_code == 200
        assert response.json()["document_uuid"] == "test-api-uuid"
        assert response.headers.get("X-Duplicate") is None


@pytest.mark.asyncio
async def test_upload_duplicate_document_success():
    mock_pipeline_res = PipelineResult(
        document_uuid="existing-uuid",
        file_path="storage/existing-uuid/invoice.pdf",
        overall_status="auto_post",
        segments=[],
        auto_post_count=0,
        review_count=0,
        processing_time_ms=10
    )

    # Mock DB record representing a duplicate
    mock_duplicate_db_doc = MagicMock()
    mock_duplicate_db_doc.document_uuid = "existing-uuid"
    mock_duplicate_db_doc.file_path = "storage/existing-uuid/invoice.pdf"

    with patch("app.api.documents.run_pipeline", new_callable=AsyncMock) as mock_run_pipeline, \
         patch("app.api.documents._find_duplicate", new_callable=AsyncMock) as mock_find_dup, \
         patch("app.api.documents.compute_sha256", return_value="existing-sha256"), \
         patch("os.makedirs"), \
         patch("os.remove"), \
         patch("os.rmdir"), \
         patch("builtins.open", MagicMock()):

        mock_find_dup.return_value = mock_duplicate_db_doc
        mock_run_pipeline.return_value = mock_pipeline_res

        response = client.post(
            "/api/documents/upload",
            files={"file": ("invoice.pdf", b"%PDF-1.4 mock content", "application/pdf")}
        )

        assert response.status_code == 200
        assert response.json()["document_uuid"] == "existing-uuid"
        assert response.headers.get("X-Duplicate") == "true"
