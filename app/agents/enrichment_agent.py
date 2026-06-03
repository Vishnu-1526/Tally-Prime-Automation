from __future__ import annotations

"""
Enrichment Agent — LangGraph node that extracts structured invoice data from
a DocumentSegment's raw OCR text using GPT-4o-mini.

Input state:  { "segment": DocumentSegment }
Output state: { "extracted_invoice": ExtractedInvoice }

Failure modes:
  - JSON parse error  → retry once with stricter prefix
  - Validation error  → partial ExtractedInvoice, log missing fields
  - API timeout       → raise ExtractionError (pipeline marks segment failed)
"""

import asyncio
import json
import logging
import re
from typing import Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.schemas.invoice import ExtractedInvoice, LineItem
from app.schemas.pipeline import DocumentSegment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class ExtractionError(Exception):
    """Raised when GPT-4o-mini call fails unrecoverably (timeout / API error)."""


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a financial data extraction specialist for Indian GST invoices. "
    "Extract all invoice fields from the provided text with 100% accuracy. "
    "For amounts, always extract the numeric value only (no currency symbols). "
    "For GSTIN, extract the full 15-character code exactly as it appears. "
    "For dates, convert to YYYY-MM-DD format. "
    "For state codes, extract the first 2 digits of the GSTIN. "
    "If a field is not present, return null — never guess or hallucinate. "
    "Return a valid JSON object matching the schema exactly.\n\n"
    "CRITICAL VENDOR IDENTIFICATION RULES:\n"
    "- The VENDOR is the company ISSUING the invoice (seller/service provider)\n"
    "- The BUYER is the company RECEIVING the invoice (customer/recipient)\n"
    "- On telecom bills: vendor details are in the FOOTER or under \n"
    "  'Registered Office' / 'Corporate Office' — NOT under \n"
    "  'Original for Recipient' or 'Bill To' or 'Customer Details'\n"
    "- Never set vendor_name to the buyer/recipient company\n"
    "- If you see 'Original for Recipient' — that entity is the BUYER\n"
    "- The vendor is whoever has the GST registration that issued this invoice\n"
    "- For Jio bills: vendor is 'Reliance Jio Infocomm Limited' or \n"
    "  'Jio Platforms Limited', never the customer"
)

_SIMPLIFIED_SCHEMA_JSON = """{
  "vendor_name": "string (the vendor's company name) or null",
  "vendor_gstin": "string (15-character vendor GSTIN) or null",
  "vendor_address": "string (vendor's address) or null",
  "vendor_state_code": "string (2-digit vendor state code) or null",
  "invoice_number": "string (invoice serial number) or null",
  "invoice_date": "string (invoice date in YYYY-MM-DD format) or null",
  "buyer_gstin": "string (15-character buyer GSTIN) or null",
  "buyer_state_code": "string (2-digit buyer state code) or null",
  "line_items": [
    {
      "description": "string (item name / description)",
      "hsn_code": "string (HSN / SAC code) or null",
      "quantity": "number (quantity of items) or null",
      "unit": "string (unit of measurement e.g. NOS, PCS) or null",
      "rate": "number (unit rate) or null",
      "taxable_value": "number (taxable value / line item amount)",
      "cgst_rate": "number (CGST rate percentage e.g. 9.0) or 0.0",
      "cgst_amount": "number (CGST tax amount) or 0.0",
      "sgst_rate": "number (SGST rate percentage e.g. 9.0) or 0.0",
      "sgst_amount": "number (SGST tax amount) or 0.0",
      "igst_rate": "number (IGST rate percentage e.g. 18.0) or 0.0",
      "igst_amount": "number (IGST tax amount) or 0.0"
    }
  ],
  "subtotal": "number (sum of taxable values) or null",
  "total_cgst": "number (sum of CGST amounts) or 0.0",
  "total_sgst": "number (sum of SGST amounts) or 0.0",
  "total_igst": "number (sum of IGST amounts) or 0.0",
  "grand_total": "number (invoice grand total amount) or null",
  "payment_terms": "string (payment terms description) or null",
  "category": "string (one of: electricity, internet, rent, travel, pantry, infrastructure, professional_services, unknown) or null"
}"""


def _build_user_prompt(raw_text: str, retry: bool = False) -> str:
    prefix = "Return valid JSON only:\n\n" if retry else ""
    return (
        f"{prefix}Extract all invoice data from the TEXT TO EXTRACT FROM below and return a JSON object matching the required structure.\n\n"
        f"REQUIRED JSON FORMAT:\n{_SIMPLIFIED_SCHEMA_JSON}\n\n"
        f"TEXT TO EXTRACT FROM:\n{raw_text}\n\n"
        "Return a valid JSON object matching the keys above exactly, with no conversational text."
    )


# ---------------------------------------------------------------------------
# LLM client factory (lazy singleton per-call to stay stateless)
# ---------------------------------------------------------------------------


_llm_instance = None


def _get_llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOpenAI(
            model="gpt-5-mini",
            api_key=settings.OPENAI_API_KEY,
            timeout=60,
            max_retries=0,
        )
    return _llm_instance


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

_MD_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _strip_fences(text: str) -> str:
    """Strip markdown code fences from LLM response."""
    match = _MD_FENCE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _parse_json(raw: str) -> Optional[dict]:
    """Try to parse JSON, returning None on failure."""
    try:
        return json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


def _post_process(data: dict, segment: DocumentSegment) -> ExtractedInvoice:
    """
    Apply derivations and defaults after parsing the JSON:
    1. Derive vendor_state_code from vendor_gstin[:2]
    2. Derive buyer_state_code from buyer_gstin[:2]
    3. Compute grand_total from line items if missing
    4. Fill category from segment.suspected_category if missing
    """
    # Force direct metadata assignment to prevent LLM null overrides
    data["segment_id"] = segment.segment_id
    data["raw_text"] = segment.raw_text

    # Sanitize and validate line items individually to prevent sub-element validation failures
    cleaned_items = []
    for item in data.get("line_items") or []:
        if not isinstance(item, dict):
            continue
        desc = item.get("description")
        if not desc:
            desc = "Item"
        
        val = item.get("taxable_value")
        try:
            val = float(val) if val is not None else 0.0
        except (ValueError, TypeError):
            val = 0.0
            
        try:
            qty = float(item.get("quantity")) if item.get("quantity") is not None else None
        except (ValueError, TypeError):
            qty = None

        try:
            rate = float(item.get("rate")) if item.get("rate") is not None else None
        except (ValueError, TypeError):
            rate = None

        try:
            cleaned_item = {
                "description": str(desc),
                "hsn_code": str(item.get("hsn_code")) if item.get("hsn_code") is not None else None,
                "quantity": qty,
                "unit": str(item.get("unit")) if item.get("unit") is not None else None,
                "rate": rate,
                "taxable_value": val,
                "cgst_rate": float(item.get("cgst_rate") or 0.0) if item.get("cgst_rate") is not None else 0.0,
                "cgst_amount": float(item.get("cgst_amount") or 0.0) if item.get("cgst_amount") is not None else 0.0,
                "sgst_rate": float(item.get("sgst_rate") or 0.0) if item.get("sgst_rate") is not None else 0.0,
                "sgst_amount": float(item.get("sgst_amount") or 0.0) if item.get("sgst_amount") is not None else 0.0,
                "igst_rate": float(item.get("igst_rate") or 0.0) if item.get("igst_rate") is not None else 0.0,
                "igst_amount": float(item.get("igst_amount") or 0.0) if item.get("igst_amount") is not None else 0.0,
            }
            cleaned_items.append(cleaned_item)
        except Exception:
            # Skip any item that fails completely to form a basic dict
            continue
    data["line_items"] = cleaned_items

    # 3. Sanitise GSTINs — fix common OCR artifacts before state code derivation
    # A valid GSTIN is exactly 15 characters: 2 digits + 5 letters + 4 digits + 1 letter + 1 alphanumeric + Z + 1 alphanumeric
    _GSTIN_RE = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")

    def _sanitise_gstin(raw: Optional[str]) -> Optional[str]:
        """
        Remove non-alphanumeric chars and fix case.
        Handles two common OCR artifacts:
        1. Extra digit inserted before the Z separator (16 chars → remove the extra char at pos 13)
           e.g. '36AALCK7998P12Z6' → '36AALCK7998P1Z6'
        2. Extra letter at the start of PAN section (17 chars)
        """
        if not raw:
            return raw
        cleaned = re.sub(r"[^A-Z0-9]", "", raw.upper().strip())

        if len(cleaned) == 15 and _GSTIN_RE.match(cleaned):
            return cleaned  # Already perfect

        if len(cleaned) == 16:
            # Try removing each char one at a time and check if it gives a valid GSTIN
            for i in range(len(cleaned)):
                candidate = cleaned[:i] + cleaned[i+1:]
                if _GSTIN_RE.match(candidate):
                    return candidate  # Found the extra character, removed it

        return cleaned  # Return best-effort even if still invalid

    vendor_gstin: Optional[str] = _sanitise_gstin(data.get("vendor_gstin"))
    data["vendor_gstin"] = vendor_gstin

    buyer_gstin: Optional[str] = _sanitise_gstin(data.get("buyer_gstin"))
    data["buyer_gstin"] = buyer_gstin

    # Derive and CLEAN vendor_state_code
    vendor_gstin = data.get("vendor_gstin")
    if vendor_gstin and len(vendor_gstin) >= 2:
        # Always derive from GSTIN — most reliable source
        data["vendor_state_code"] = vendor_gstin[:2]
    elif data.get("vendor_state_code"):
        # Clean whatever was extracted — keep only digits, take first 2
        raw_state = str(data["vendor_state_code"])
        digits_only = "".join(filter(str.isdigit, raw_state))
        if len(digits_only) >= 2:
            data["vendor_state_code"] = digits_only[:2]
        else:
            data["vendor_state_code"] = None

    # Same for buyer_state_code
    buyer_gstin = data.get("buyer_gstin")
    if buyer_gstin and len(buyer_gstin) >= 2:
        data["buyer_state_code"] = buyer_gstin[:2]
    elif data.get("buyer_state_code"):
        raw_state = str(data["buyer_state_code"])
        digits_only = "".join(filter(str.isdigit, raw_state))
        if len(digits_only) >= 2:
            data["buyer_state_code"] = digits_only[:2]
        else:
            data["buyer_state_code"] = None




    # 4. Always recompute subtotal and grand_total from verified components.
    # Never trust the OCR-extracted grand_total directly — it can be corrupted
    # (e.g., "217,700" instead of "17,700" due to OCR artifacts).
    subtotal = data.get("subtotal")
    if not subtotal or float(subtotal) == 0.0:
        subtotal = sum(item.get("taxable_value") or 0.0 for item in cleaned_items)
        data["subtotal"] = subtotal
    else:
        subtotal = float(subtotal)

    total_cgst = data.get("total_cgst") or 0.0
    total_sgst = data.get("total_sgst") or 0.0
    total_igst = data.get("total_igst") or 0.0
    computed_total = subtotal + total_cgst + total_sgst + total_igst
    data["grand_total"] = computed_total

    # Auto-compute subtotal if missing but line items exist
    if (data.get("subtotal") is None or data.get("subtotal") == 0.0):
        line_items = data.get("line_items") or []
        if line_items:
            computed_subtotal = sum(
                item.get("taxable_value", 0.0) or 0.0
                for item in line_items
            )
            if computed_subtotal > 0:
                data["subtotal"] = round(computed_subtotal, 2)
                logger.info(
                    "Auto-computed subtotal from line items: %s",
                    data["subtotal"]
                )

    # 5. Zero out round_off — the "Amount Due" field on Wave/FreshBooks
    # invoices is NOT a GST round-off; setting it non-zero breaks validation.
    data["round_off"] = 0.0

    # 6. Fill category from segment
    if not data.get("category"):
        data["category"] = segment.suspected_category

    return ExtractedInvoice.model_validate(data)


def _partial_invoice(
    data: dict, segment: DocumentSegment, error: Exception
) -> ExtractedInvoice:
    """
    Construct a best-effort partial ExtractedInvoice when model_validate fails.
    Logs which fields were missing or invalid.
    """
    logger.warning(
        "Validation error for segment %s — building partial invoice. Error: %s",
        segment.segment_id,
        error,
    )
    
    # We must sanitize the line items here too to guarantee safe validation!
    cleaned_items = []
    for item in data.get("line_items") or []:
        if not isinstance(item, dict):
            continue
        desc = item.get("description") or "Item"
        val = item.get("taxable_value")
        try:
            val = float(val) if val is not None else 0.0
        except (ValueError, TypeError):
            val = 0.0
        cleaned_items.append({
            "description": str(desc),
            "taxable_value": val,
            "hsn_code": str(item.get("hsn_code")) if item.get("hsn_code") is not None else None,
            "quantity": float(item.get("quantity")) if item.get("quantity") is not None else None,
            "unit": str(item.get("unit")) if item.get("unit") is not None else None,
            "rate": float(item.get("rate")) if item.get("rate") is not None else None,
        })

    # Minimal required fields for a valid ExtractedInvoice (preserving extracted line items if present)
    safe: dict[str, Any] = {
        "segment_id": segment.segment_id,
        "line_items": cleaned_items,
        "raw_text": segment.raw_text,
        "category": segment.suspected_category,
        "round_off": 0.0,  # Always zero — Wave/FreshBooks "Amount Due" is not a GST round-off
    }
    # Copy over any scalar fields that are present and of the right type
    scalar_fields = [
        "vendor_name", "vendor_gstin", "vendor_address", "vendor_state_code",
        "invoice_number", "invoice_date", "buyer_gstin", "buyer_state_code",
        "subtotal", "total_cgst", "total_sgst", "total_igst",
        "payment_terms",
    ]
    for field in scalar_fields:
        value = data.get(field)
        if value is not None:
            safe[field] = value

    # Always recompute subtotal and grand_total from components (OCR values can be corrupted)
    _sub = safe.get("subtotal")
    if not _sub or float(_sub) == 0.0:
        _sub = sum(item.get("taxable_value") or 0.0 for item in cleaned_items)
        safe["subtotal"] = _sub
    else:
        _sub = float(_sub)

    _cgst = safe.get("total_cgst") or 0.0
    _sgst = safe.get("total_sgst") or 0.0
    _igst = safe.get("total_igst") or 0.0
    safe["grand_total"] = _sub + _cgst + _sgst + _igst

    # Log genuinely missing fields
    missing = [f for f in scalar_fields if data.get(f) is None]
    logger.info("Partial invoice for %s — missing fields: %s", segment.segment_id, missing)

    return ExtractedInvoice.model_validate(safe)


# ---------------------------------------------------------------------------
# Core extraction function
# ---------------------------------------------------------------------------


async def _extract_once(
    llm: ChatGoogleGenerativeAI, raw_text: str, segment: DocumentSegment, retry: bool = False
) -> ExtractedInvoice:
    """
    Call Gemini once and attempt to return a validated ExtractedInvoice.
    Falls back to local models via Ollama if Gemini fails (e.g. out of quota).
    """
    if not raw_text.strip():
        logger.info("Raw text is empty for segment %s, skipping enrichment LLM calls.", segment.segment_id)
        return ExtractedInvoice(
            segment_id=segment.segment_id,
            line_items=[],
            raw_text="",
            category=segment.suspected_category,
        )

    raw_response = ""
    try:
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=_build_user_prompt(raw_text, retry=retry)),
        ]
        
        # Custom exponential backoff retry loop for Rate Limit (429 / RESOURCE_EXHAUSTED)
        max_attempts = 5
        backoff = 2.0
        for attempt in range(max_attempts):
            try:
                response = await llm.ainvoke(messages)
                raw_response = response.content
                break
            except Exception as e:
                err_msg = str(e)
                if ("429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower()) and attempt < max_attempts - 1:
                    sleep_time = backoff * (2 ** attempt)
                    logger.warning(
                        "Gemini enrichment hit rate limit (429/RESOURCE_EXHAUSTED) for segment %s. Retrying in %.1fs... (Attempt %d/%d)",
                        segment.segment_id, sleep_time, attempt + 1, max_attempts
                    )
                    await asyncio.sleep(sleep_time)
                else:
                    raise e
    except Exception as e:
        logger.warning(
            "Gemini enrichment failed for segment %s: %s. Initiating local model fallback.",
            segment.segment_id,
            e,
        )
        import httpx
        # Strategy: Try Llama 3 first (8B, highly accurate), then Nemotron-Mini (2.7B) if it fails
        for model in ["llama3", "nemotron-mini"]:
            try:
                logger.info("Attempting local enrichment fallback using Ollama model: %s", model)
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": _build_user_prompt(raw_text, retry=retry)}
                    ],
                    "format": "json",
                    "stream": False,
                }
                timeout_val = 90.0 if model == "llama3" else 60.0
                async with httpx.AsyncClient(timeout=timeout_val) as client:
                    response = await client.post(
                        f"{settings.OLLAMA_BASE_URL}/api/chat", json=payload
                    )
                    if response.status_code == 200:
                        result_data = response.json()
                        raw_response = result_data.get("message", {}).get("content", "").strip()
                        logger.info("Local %s enrichment fallback successfully processed segment %s", model, segment.segment_id)
                        break
                    else:
                        raise RuntimeError(f"Ollama returned status code {response.status_code} for {model}")
            except Exception as model_err:
                logger.warning("Local fallback with model %s failed: %s", model, model_err)
                continue
        else:
            raise ExtractionError("All local enrichment fallbacks (llama3, nemotron-mini) failed to process segment.")

    data = _parse_json(raw_response)
    if data is None:
        raise json.JSONDecodeError("No valid JSON in LLM response", raw_response, 0)

    try:
        return _post_process(data, segment)
    except Exception as validation_err:
        return _partial_invoice(data, segment, validation_err)


# ---------------------------------------------------------------------------
# LangGraph node entry point
# ---------------------------------------------------------------------------


async def enrichment_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node — processes one segment per invocation.

    Input state keys:  segment (DocumentSegment)
    Output state keys: extracted_invoice (ExtractedInvoice)

    For bundle documents the pipeline orchestrator should call this node
    concurrently via asyncio.gather([enrichment_node(s) for s in segments]).
    """
    segment: DocumentSegment = state["segment"]
    raw_text = segment.raw_text

    llm = _get_llm()

    try:
        # First attempt
        invoice = await _extract_once(llm, raw_text, segment, retry=False)
        logger.info(
            "Enrichment OK for segment %s — vendor: %s, total: %s",
            segment.segment_id,
            invoice.vendor_name,
            invoice.grand_total,
        )
        return {"extracted_invoice": invoice}

    except json.JSONDecodeError:
        logger.warning(
            "JSON parse error for segment %s — retrying with strict prefix.",
            segment.segment_id,
        )
        try:
            invoice = await _extract_once(llm, raw_text, segment, retry=True)
            logger.info("Retry succeeded for segment %s", segment.segment_id)
            return {"extracted_invoice": invoice}

        except Exception as final_err:
            logger.error(
                "Retry failed to produce valid JSON or validate for segment %s: %s",
                segment.segment_id,
                final_err,
            )
            # Return minimal partial invoice
            partial = _partial_invoice({}, segment, final_err)
            return {"extracted_invoice": partial}

    except Exception as e:
        logger.error(
            "Enrichment node encountered an unexpected validation or parsing error for segment %s: %s. Returning best-effort partial invoice.",
            segment.segment_id,
            e,
            exc_info=True,
        )
        # Attempt to extract raw dictionary from raw_response if possible to salvage fields
        try:
            # We can scrape data from the state if needed, but since it's unhandled, return best-effort
            partial = _partial_invoice({}, segment, e)
        except Exception as fallback_err:
            logger.critical("Critical: even partial invoice creation failed: %s", fallback_err)
            from app.schemas.invoice import ExtractedInvoice
            partial = ExtractedInvoice(
                segment_id=segment.segment_id,
                line_items=[],
                raw_text=segment.raw_text,
                category=segment.suspected_category
            )
        return {"extracted_invoice": partial}