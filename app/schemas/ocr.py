from __future__ import annotations

from typing import Literal
from pydantic import BaseModel


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class PageResult(BaseModel):
    page_number: int
    text: str
    quality_score: float  # 0.0-1.0, how readable the OCR output is
    bounding_boxes: list[BoundingBox] = []
    ocr_engine: Literal["nemotron", "tesseract", "gemini"] = "gemini"  # which engine produced this page


class OCRResult(BaseModel):
    file_path: str
    page_count: int
    pages: list[PageResult]
    overall_quality: float  # average of page quality scores
    ocr_engine: Literal["nemotron", "tesseract", "gemini"]
    processing_time_ms: int
