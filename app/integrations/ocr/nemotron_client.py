from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import time
from typing import Literal

import httpx
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from PIL import Image
from pdf2image import convert_from_path

from app.config import settings
from app.schemas.ocr import OCRResult, PageResult

logger = logging.getLogger(__name__)


async def check_ollama_health() -> bool:
    """
    Checks if Ollama is running and accessible at startup with a health ping.
    Logs a warning if unavailable but does not crash.
    """
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(settings.OLLAMA_BASE_URL)
            if response.status_code == 200:
                logger.info("Ollama service health check passed.")
                return True
            else:
                logger.warning(
                    f"Ollama health check returned unexpected status code: {response.status_code}"
                )
    except Exception as e:
        logger.warning(
            f"Ollama service check failed at {settings.OLLAMA_BASE_URL}. "
            f"Ensure Ollama is running locally if you want local OCR fallback. Error: {e}"
        )
    return False


def pil_to_base64(image: Image.Image) -> str:
    """Converts a PIL Image to a base64-encoded PNG string in memory."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def calculate_page_quality(text: str) -> float:
    """
    Computes quality_score per page:
    - 1.0 if text length > 200 chars and contains at least one number
    - 0.7 if text length 50-200 chars
    - 0.3 if text length < 50 chars
    - 0.0 if empty or error
    """
    if not text:
        return 0.0
    
    cleaned = text.strip()
    if not cleaned:
        return 0.0
        
    length = len(cleaned)
    if length > 200 and any(c.isdigit() for c in cleaned):
        return 1.0
    elif 50 <= length <= 200:
        return 0.7
    elif length < 50:
        return 0.3
    else:
        return 0.7


async def process_single_page_image(image: Image.Image, page_number: int) -> PageResult:
    """
    Processes a single page PIL Image using Gemini 2.5 Flash for state-of-the-art vision OCR.
    If the Gemini call fails, automatically falls back to local llama3.2-vision via Ollama.
    """
    start_time = time.time()
    base64_image = None
    try:
        # Base64 encode in-memory (highly fast, zero temp files on disk)
        base64_image = await asyncio.to_thread(pil_to_base64, image)

        prompt = (
            "Extract all text from this financial document image exactly as it appears. "
            "Preserve all numbers, amounts, GSTIN numbers, dates, and table structures. "
            "Output raw text only, no commentary."
        )

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ]
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0,
            max_retries=0,
        )

        import random
        # Custom exponential backoff retry loop for Rate Limit (429 / RESOURCE_EXHAUSTED)
        max_attempts = 8
        backoff = 2.0
        extracted_text = ""
        for attempt in range(max_attempts):
            try:
                response = await llm.ainvoke([message])
                extracted_text = response.content.strip()
                break
            except Exception as e:
                err_msg = str(e)
                is_rate_limit = any(
                    x in err_msg or x in err_msg.lower()
                    for x in ["429", "resource_exhausted", "quota", "rate limit", "limit exceeded"]
                )
                if is_rate_limit and attempt < max_attempts - 1:
                    sleep_time = (backoff * (1.5 ** attempt)) + random.uniform(1.0, 3.0)
                    logger.warning(
                        "Gemini OCR page %d hit rate limit (429/RESOURCE_EXHAUSTED). Retrying in %.1fs... (Attempt %d/%d)",
                        page_number, sleep_time, attempt + 1, max_attempts
                    )
                    await asyncio.sleep(sleep_time)
                else:
                    raise e

        quality_score = calculate_page_quality(extracted_text)

        logger.info(
            "Gemini OCR successfully extracted page %d in %.2fs",
            page_number,
            time.time() - start_time,
        )

        return PageResult(
            page_number=page_number,
            text=extracted_text,
            quality_score=quality_score,
            bounding_boxes=[],
            ocr_engine="gemini",
        )

    except Exception as e:
        logger.warning(
            "Gemini OCR failed for page %d: %s. Initiating Llama 3.2 Vision local fallback.",
            page_number,
            e,
        )
        try:
            if base64_image is None:
                base64_image = await asyncio.to_thread(pil_to_base64, image)

            prompt = (
                "Extract all text from this financial document image exactly as it appears. "
                "Preserve all numbers, amounts, GSTIN numbers, dates, and table structures. "
                "Output raw text only, no commentary."
            )

            payload = {
                "model": "llama3.2-vision",
                "prompt": prompt,
                "images": [base64_image],
                "stream": False,
            }

            async with httpx.AsyncClient(timeout=90.0) as client:
                response = await client.post(
                    f"{settings.OLLAMA_BASE_URL}/api/generate", json=payload
                )
                if response.status_code == 200:
                    result_data = response.json()
                    extracted_text = result_data.get("response", "").strip()
                    if extracted_text:
                        quality_score = calculate_page_quality(extracted_text)
                        logger.info(
                            "Llama 3.2 Vision local fallback successfully extracted page %d in %.2fs",
                            page_number,
                            time.time() - start_time,
                        )
                        return PageResult(
                            page_number=page_number,
                            text=extracted_text,
                            quality_score=quality_score,
                            bounding_boxes=[],
                            ocr_engine="tesseract",  # Using tesseract slot for fallback representation
                        )

            raise RuntimeError("Ollama llama3.2-vision fallback returned empty response or error.")
        except Exception as fallback_err:
            logger.warning(
                "Llama 3.2 Vision fallback also failed or timed out for page %d: %s. Attempting fast local Tesseract OCR fallback...",
                page_number,
                fallback_err,
            )
            try:
                import tempfile
                from app.integrations.ocr.fallback_tesseract import extract_text_fallback
                
                # Save PIL Image to a temporary PNG file to pass to Tesseract
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                    temp_path = tmp_file.name
                
                await asyncio.to_thread(image.save, temp_path, format="PNG")
                
                tess_result = await extract_text_fallback(temp_path, page_number)
                
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
                
                logger.info(
                    "Fast local Tesseract OCR fallback successfully extracted page %d in %.2fs",
                    page_number,
                    time.time() - start_time,
                )
                return tess_result
            except Exception as tess_err:
                logger.error(
                    "All OCR engines (Gemini, Llama 3.2 Vision, Tesseract) failed for page %d: %s",
                    page_number,
                    tess_err,
                    exc_info=True,
                )
                return PageResult(
                    page_number=page_number,
                    text="",
                    quality_score=0.0,
                    bounding_boxes=[],
                    ocr_engine="tesseract",
                )


async def extract_text(file_path: str) -> OCRResult:
    """
    Main OCR entry point. Uses Gemini 2.5 Flash multimodal vision support.
    Reads PDFs (converting pages in-memory) or raw image files.
    Falls back to Llama 3.2 Vision locally on failure.
    """
    start_time = time.time()
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path.lower())[1]
    is_pdf = ext == ".pdf"
    
    pages_results: list[PageResult] = []

    try:
        if is_pdf:
            # Convert PDF pages to PIL images
            pdf_pages = await asyncio.to_thread(
                convert_from_path, file_path, dpi=300
            )
            page_count = len(pdf_pages)
            
            # Use a semaphore of 1 to process pages sequentially, preventing rate-limit spikes on free tier
            sem = asyncio.Semaphore(1)
            
            async def sem_process(page, idx):
                # Add a staggered delay to naturally space out requests on the free-tier API
                if idx > 1:
                    await asyncio.sleep(1.5 * (idx - 1))
                async with sem:
                    return await process_single_page_image(page, idx)
            
            # Process all pages sequentially with spacing using Gemini OCR
            tasks = [
                sem_process(page, idx + 1)
                for idx, page in enumerate(pdf_pages)
            ]
            pages_results = await asyncio.gather(*tasks)
        else:
            # Open single image file using PIL
            image = await asyncio.to_thread(Image.open, file_path)
            page_result = await process_single_page_image(image, page_number=1)
            pages_results = [page_result]
            page_count = 1

    except Exception as e:
        logger.error("extract_text failed for file %s: %s", file_path, e, exc_info=True)
        raise

    # Compute overall quality score
    if pages_results:
        overall_quality = sum(p.quality_score for p in pages_results) / len(pages_results)
    else:
        overall_quality = 0.0

    processing_time_ms = int((time.time() - start_time) * 1000)

    # If any page used the fallback engine ("tesseract"), tag the overall OCR engine as "tesseract" (representing the local model)
    any_fallback = any(p.ocr_engine == "tesseract" for p in pages_results)
    overall_engine: Literal["gemini", "tesseract", "nemotron"] = "tesseract" if any_fallback else "gemini"

    return OCRResult(
        file_path=file_path,
        page_count=page_count,
        pages=pages_results,
        overall_quality=overall_quality,
        ocr_engine=overall_engine,
        processing_time_ms=processing_time_ms,
    )
