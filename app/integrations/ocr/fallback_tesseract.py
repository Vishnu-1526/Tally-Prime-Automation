from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import cv2
import numpy as np
import pytesseract

from app.schemas.ocr import PageResult

logger = logging.getLogger(__name__)


def preprocess_image_data(img: np.ndarray) -> np.ndarray:
    """
    Applies the OpenCV preprocessing pipeline to the image:
    a. Convert to grayscale
    b. Deskew (detect and correct rotation using Hough transform)
    c. Denoise (fastNlMeansDenoising)
    d. Adaptive threshold for binarization
    e. Upscale to minimum 2000px on longest side if smaller
    """
    # a. Convert to grayscale if multiple channels
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # b. Deskew using Hough Lines transform
    try:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
        
        angles = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                if -45 < angle < 45:
                    angles.append(angle)
                elif angle > 45:
                    angles.append(angle - 90)
                elif angle < -45:
                    angles.append(angle + 90)

        if angles:
            median_angle = np.median(angles)
            if abs(median_angle) > 0.5:
                h, w = gray.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                # Fill with white background (255) to keep it clean
                gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    except Exception as e:
        logger.warning(f"Deskew failed (falling back to original orientation): {e}")

    # c. Denoise using fastNlMeansDenoising
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # d. Adaptive threshold for binarization
    binarized = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # e. Upscale to minimum 2000px on longest side if smaller
    h, w = binarized.shape[:2]
    longest_side = max(h, w)
    if longest_side < 2000:
        scale = 2000.0 / longest_side
        new_w = int(w * scale)
        new_h = int(h * scale)
        binarized = cv2.resize(binarized, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    return binarized


def preprocess_image_file(input_path: str, output_path: str) -> None:
    """Loads image from path, runs preprocessing, and saves to output path."""
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not read image from {input_path}")
    processed = preprocess_image_data(img)
    cv2.imwrite(output_path, processed)


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
        # length > 200 but does not contain a digit
        return 0.7


def run_tesseract(image_path: str) -> str:
    """Blocking PyTesseract text extraction."""
    config = "--oem 3 --psm 6 -l eng"
    return pytesseract.image_to_string(image_path, config=config)


async def extract_text_fallback(image_path: str, page_number: int) -> PageResult:
    """
    Async Tesseract OCR fallback runner.
    Applies the preprocessing pipeline, runs Tesseract, and formats as PageResult.
    """
    temp_processed_path = None
    try:
        # Create a temporary file to store the preprocessed image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_processed_path = temp_file.name

        # Preprocess blocking call run in thread pool
        await asyncio.to_thread(preprocess_image_file, image_path, temp_processed_path)

        # Run Tesseract blocking call in thread pool
        extracted_text = await asyncio.to_thread(run_tesseract, temp_processed_path)
        
        quality_score = calculate_page_quality(extracted_text)

        return PageResult(
            page_number=page_number,
            text=extracted_text,
            quality_score=quality_score,
            bounding_boxes=[],
            ocr_engine="tesseract",
        )
    except Exception as e:
        logger.error(f"Tesseract fallback failed for page {page_number}: {e}")
        return PageResult(
            page_number=page_number,
            text="",
            quality_score=0.0,
            bounding_boxes=[],
            ocr_engine="tesseract",
        )
    finally:
        # Cleanup processed temp file
        if temp_processed_path and os.path.exists(temp_processed_path):
            try:
                os.remove(temp_processed_path)
            except Exception as e:
                logger.warning(f"Could not delete temp preprocessed file {temp_processed_path}: {e}")
