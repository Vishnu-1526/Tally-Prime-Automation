from __future__ import annotations

import logging
import os
import uuid
import json
from typing import Any, Optional

from fastapi import APIRouter, File, HTTPException, Response, UploadFile

from app.database import get_db_session
from app.models.document import Document
from app.schemas.pipeline import PipelineResult
from app.services.pipeline_runner import run_pipeline, _find_duplicate, compute_sha256

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif"}
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB hard limit


def _safe_filename(raw: str) -> str:
    """
    Strip any path separators or dangerous characters from an uploaded filename
    to prevent path-traversal attacks (e.g. '../../etc/passwd.pdf').
    Only the base name is kept; the rest is discarded.
    """
    # os.path.basename handles both / and \\ separators
    name = os.path.basename(raw.replace("\\", "/"))
    # Replace any remaining whitespace or special chars that aren't alphanumeric / . - _
    import re as _re
    name = _re.sub(r"[^A-Za-z0-9._\-() ]", "_", name)
    return name or "upload"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/upload", response_model=PipelineResult)
async def upload_document(
    response: Response,
    file: UploadFile = File(...),
    company_id: int = 1
) -> PipelineResult:
    """
    Upload a document (PDF, JPEG, PNG, WEBP, TIFF) to ingest into the automation pipeline.
    Returns 200 with PipelineResult JSON.
    If the document has already been processed (SHA-256 match), returns the cached result
    and adds the 'X-Duplicate: true' response header.
    """
    # Validate that the company_id exists in the database before processing
    from app.models.company import Company
    async with get_db_session() as session:
        from sqlalchemy import select
        company_exists_stmt = select(Company.id).where(Company.id == company_id)
        company_exists_res = await session.execute(company_exists_stmt)
        if company_exists_res.scalar_one_or_none() is None:
            raise HTTPException(
                status_code=422,
                detail=f"Company with ID {company_id} does not exist."
            )

    # 1. Validate and sanitise filename (prevent path traversal)
    raw_filename = file.filename or "upload"
    filename = _safe_filename(raw_filename)
    _, ext = os.path.splitext(filename.lower())
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported file format '{ext}'. "
                "Supported formats: PDF, JPEG, PNG, WEBP, TIFF."
            )
        )

    # 2. Generate folders & save file temporarily to storage/{uuid}/{safe_filename}
    doc_uuid = str(uuid.uuid4())
    save_dir = os.path.join("storage", doc_uuid)
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, filename)

    # 2a. Read content with size check before writing (DoS protection)
    try:
        content = await file.read(MAX_FILE_SIZE_BYTES + 1)
    except Exception as e:
        logger.error("Failed to read uploaded file: %s", e)
        raise HTTPException(status_code=500, detail="Failed to read uploaded file.")

    if len(content) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is 50 MB."
        )

    try:
        with open(file_path, "wb") as buffer:
            buffer.write(content)
    except Exception as e:
        logger.error("Failed to write uploaded file to disk: %s", e)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file to disk.")


    # 3. Check for duplicates using SHA-256 before running the full pipeline
    try:
        file_hash = compute_sha256(file_path)
        duplicate = await _find_duplicate(file_hash)
        if duplicate:
            logger.info("Duplicate document detected on upload: %s", duplicate.document_uuid)
            response.headers["X-Duplicate"] = "true"
            # Cleanup the newly saved duplicate file since we don't need to keep it
            try:
                os.remove(file_path)
                os.rmdir(save_dir)
            except Exception as cleanup_err:
                logger.warning("Could not clean up temp duplicate directory: %s", cleanup_err)
            
            # Retrieve duplicate file path and call run_pipeline to return validation result
            # Using the original file path allows duplicate detection cache hit in run_pipeline
            return await run_pipeline(duplicate.file_path, company_id=company_id)
    except Exception as hash_err:
        logger.error("Error during duplicate verification check: %s", hash_err)

    # 4. Process new document through the pipeline
    try:
        result = await run_pipeline(file_path, document_uuid=doc_uuid, company_id=company_id)
        return result
    except Exception as err:
        logger.error("Pipeline run failed for document %s: %s", doc_uuid, err, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline execution failed: {str(err)}"
        )


@router.get("")
async def list_documents(company_id: Optional[int] = None) -> list[dict[str, Any]]:
    """
    Retrieve all processed document records from the database ordered by created_at desc.
    Includes average confidence score computed dynamically across all voucher segments.
    """
    async with get_db_session() as session:
        from sqlalchemy import select
        stmt = select(Document).order_by(Document.created_at.desc())
        if company_id is not None:
            stmt = stmt.where(Document.company_id == company_id)
        db_res = await session.execute(stmt)
        documents = db_res.scalars().all()

    results = []
    for doc in documents:
        avg_confidence = 0.0
        if doc.pipeline_result_json:
            try:
                data = json.loads(doc.pipeline_result_json)
                segments = data.get("segments", [])
                if segments:
                    # Only average segments that actually processed successfully.
                    # Failed / zero-score segments (e.g. unrecognised cover pages) would
                    # unfairly drag the overall score down.
                    valid_scores = [
                        s.get("confidence_score", 0.0)
                        for s in segments
                        if s.get("status") != "failed" and s.get("confidence_score", 0.0) > 0
                    ]
                    if valid_scores:
                        avg_confidence = sum(valid_scores) / len(valid_scores)
            except Exception:
                pass
        results.append({
            "document_uuid": doc.document_uuid,
            "original_filename": doc.original_filename,
            "overall_status": doc.overall_status,
            "confidence_score": round(avg_confidence, 4),
            "created_at": doc.created_at,
            "processed_at": doc.processed_at
        })
    return results


@router.get("/{document_uuid}")
async def get_document_details(document_uuid: str) -> dict[str, Any]:
    """
    Retrieve the full serialized PipelineResult JSON for a specific document.
    """
    async with get_db_session() as session:
        from sqlalchemy import select
        stmt = select(Document).where(Document.document_uuid == document_uuid)
        db_res = await session.execute(stmt)
        doc = db_res.scalar_one_or_none()

    if not doc:
        raise HTTPException(
            status_code=404,
            detail=f"Document with UUID {document_uuid} not found."
        )

    if doc.pipeline_result_json:
        try:
            return json.loads(doc.pipeline_result_json)
        except Exception as e:
            logger.error("Failed to parse DB pipeline_result_json for %s: %s", document_uuid, e)

    # Fallback structure
    return {
        "document_uuid": doc.document_uuid,
        "file_path": doc.file_path,
        "overall_status": doc.overall_status,
        "segments": [],
        "auto_post_count": 0,
        "review_count": 0,
        "processing_time_ms": 0
    }


@router.delete("/{document_uuid}", status_code=204)
async def delete_document(document_uuid: str) -> None:
    """
    Remove a document from the database, and permanently delete its raw file from storage disk.
    """
    async with get_db_session() as session:
        from sqlalchemy import select
        stmt = select(Document).where(Document.document_uuid == document_uuid)
        db_res = await session.execute(stmt)
        doc = db_res.scalar_one_or_none()

        if not doc:
            raise HTTPException(
                status_code=404,
                detail=f"Document with UUID {document_uuid} not found."
            )

        # Remove containing directory and its contents
        file_path = doc.file_path
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                parent_dir = os.path.dirname(file_path)
                if os.path.isdir(parent_dir) and not os.listdir(parent_dir):
                    os.rmdir(parent_dir)
                logger.info("Successfully deleted disk files for document %s", document_uuid)
            except Exception as disk_err:
                logger.error("Failed to delete document files on disk: %s", disk_err)

        await session.delete(doc)
        logger.info("Successfully deleted database record for document %s", document_uuid)
