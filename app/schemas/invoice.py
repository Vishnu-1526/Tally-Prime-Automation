from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class LineItem(BaseModel):
    description: str
    hsn_code: Optional[str] = None
    quantity: Optional[float] = None
    unit: Optional[str] = None
    rate: Optional[float] = None
    taxable_value: float
    cgst_rate: float = 0.0
    cgst_amount: float = 0.0
    sgst_rate: float = 0.0
    sgst_amount: float = 0.0
    igst_rate: float = 0.0
    igst_amount: float = 0.0


class ExtractedInvoice(BaseModel):
    segment_id: str
    vendor_name: Optional[str] = None
    vendor_gstin: Optional[str] = None
    vendor_address: Optional[str] = None
    vendor_state_code: Optional[str] = None  # first 2 digits of GSTIN
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    buyer_gstin: Optional[str] = None
    buyer_state_code: Optional[str] = None
    line_items: list[LineItem]
    subtotal: Optional[float] = None
    total_cgst: float = 0.0
    total_sgst: float = 0.0
    total_igst: float = 0.0
    round_off: float = 0.0
    grand_total: Optional[float] = None
    payment_terms: Optional[str] = None
    category: Optional[str] = None  # electricity, internet, rent, travel, pantry
    raw_text: str  # original OCR chunk for this segment
