from __future__ import annotations

import asyncio
import base64
import io
import os
import tempfile
from unittest.mock import patch, AsyncMock, MagicMock
import pytest
from PIL import Image
import numpy as np

from app.integrations.ocr.nemotron_client import (
    calculate_page_quality,
    pil_to_base64,
    check_ollama_health,
    process_single_page_image,
    extract_text
)
from app.integrations.ocr.fallback_tesseract import preprocess_image_data, run_tesseract
from app.schemas.ocr import OCRResult, PageResult
from app.config import settings


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def _create_synthetic_image() -> Image.Image:
    """Creates a simple 100x100 white PIL image."""
    return Image.new("RGB", (100, 100), color="white")


# ---------------------------------------------------------------------------
# Test Core Functions
# ---------------------------------------------------------------------------

class TestOcrCoreUtility:
    def test_calculate_page_quality(self):
        # Empty string -> 0.0
        assert calculate_page_quality("") == 0.0
        assert calculate_page_quality("   ") == 0.0

        # Short text < 50 chars -> 0.3
        assert calculate_page_quality("Hello world") == 0.3

        # Medium text 50-200 chars -> 0.7
        medium_text = "This is a medium text designed to check if the quality scorer returns 0.7 correctly as expected for these characters."
        assert len(medium_text) >= 50 and len(medium_text) <= 200
        assert calculate_page_quality(medium_text) == 0.7

        # Long text > 200 chars but no digits -> 0.7
        long_text_no_digits = "a" * 250
        assert calculate_page_quality(long_text_no_digits) == 0.7

        # Long text > 200 chars and has digit -> 1.0
        long_text_with_digits = ("a" * 250) + " 123"
        assert calculate_page_quality(long_text_with_digits) == 1.0

    def test_pil_to_base64(self):
        img = _create_synthetic_image()
        b64_str = pil_to_base64(img)
        assert isinstance(b64_str, str)
        assert len(b64_str) > 0
        
        # Verify it can be decoded back
        decoded_bytes = base64.b64decode(b64_str)
        assert len(decoded_bytes) > 0
        img_decoded = Image.open(io.BytesIO(decoded_bytes))
        assert img_decoded.size == (100, 100)


# ---------------------------------------------------------------------------
# Test Health Check
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestOllamaHealthCheck:
    async def test_ollama_health_success(self):
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            status = await check_ollama_health()
            assert status is True

    async def test_ollama_health_unexpected_status(self):
        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            status = await check_ollama_health()
            assert status is False

    async def test_ollama_health_exception(self):
        with patch("httpx.AsyncClient.get", side_effect=Exception("Connection refused")):
            status = await check_ollama_health()
            assert status is False


# ---------------------------------------------------------------------------
# Test Preprocessing and Local Tesseract Run
# ---------------------------------------------------------------------------

class TestTesseractPreprocessing:
    def test_preprocess_image_data(self):
        # Create a synthetic 100x100 grayscale image as numpy array
        img_np = np.ones((100, 100, 3), dtype=np.uint8) * 255
        processed = preprocess_image_data(img_np)
        
        # Assert it returned a 2D binary image (since grayscale binarization)
        assert len(processed.shape) == 2
        # Assert it resized it to have the longest side be at least 2000px
        assert max(processed.shape) == 2000

    def test_run_tesseract_mock(self):
        with patch("pytesseract.image_to_string", return_value="Extracted text") as mock_tess:
            res = run_tesseract("dummy_path")
            assert res == "Extracted text"
            mock_tess.assert_called_once()


# ---------------------------------------------------------------------------
# Test Multi-Engine Fallback Pipeline
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestOcrPipelineFallbacks:
    async def test_process_single_page_gemini_success(self):
        img = _create_synthetic_image()
        mock_response = MagicMock()
        mock_response.content = ("Invoice No: 12345 " + ("a" * 200)) # Long text with digits -> 1.0

        with patch("langchain_google_genai.ChatGoogleGenerativeAI.ainvoke", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = mock_response
            res = await process_single_page_image(img, page_number=1)
            
            assert isinstance(res, PageResult)
            assert res.page_number == 1
            assert "Invoice No: 12345" in res.text
            assert res.ocr_engine == "gemini"
            assert res.quality_score == 1.0

    async def test_process_single_page_gemini_fails_ollama_succeeds(self):
        img = _create_synthetic_image()
        
        mock_ollama_resp = MagicMock()
        mock_ollama_resp.status_code = 200
        mock_ollama_resp.json.return_value = {"response": "Ollama extracted text from Jio wifi bill"}

        with patch("langchain_google_genai.ChatGoogleGenerativeAI.ainvoke", side_effect=Exception("Gemini quota exceeded")), \
             patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            
            mock_post.return_value = mock_ollama_resp
            res = await process_single_page_image(img, page_number=1)
            
            assert isinstance(res, PageResult)
            assert res.page_number == 1
            assert res.ocr_engine == "tesseract" # using tesseract slot for local model
            assert res.text == "Ollama extracted text from Jio wifi bill"
            assert res.quality_score == 0.3  # Short text

    async def test_process_single_page_all_fail_returns_empty(self):
        img = _create_synthetic_image()

        with patch("langchain_google_genai.ChatGoogleGenerativeAI.ainvoke", side_effect=Exception("Gemini failed")), \
             patch("httpx.AsyncClient.post", side_effect=Exception("Ollama down")), \
             patch("app.integrations.ocr.fallback_tesseract.extract_text_fallback", side_effect=Exception("Tesseract failed")):
            
            res = await process_single_page_image(img, page_number=1)
            assert isinstance(res, PageResult)
            assert res.text == ""
            assert res.quality_score == 0.0

    async def test_extract_text_pdf_flow(self):
        # Create a mock PDF conversion and process page
        mock_ocr_res = PageResult(
            page_number=1,
            text="Extracted Invoice Data 12345",
            quality_score=0.9,
            ocr_engine="gemini",
            bounding_boxes=[]
        )

        with patch("app.integrations.ocr.nemotron_client.convert_from_path", return_value=[_create_synthetic_image()]), \
             patch("app.integrations.ocr.nemotron_client.process_single_page_image", new_callable=AsyncMock, return_value=mock_ocr_res), \
             patch("os.path.exists", return_value=True):
            
            res = await extract_text("/tmp/test_invoice.pdf")
            assert isinstance(res, OCRResult)
            assert res.file_path == "/tmp/test_invoice.pdf"
            assert res.page_count == 1
            assert len(res.pages) == 1
            assert res.overall_quality == 0.9
            assert res.ocr_engine == "gemini"
