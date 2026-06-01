from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel


class LedgerMapping(BaseModel):
    debit_ledger: str
    credit_ledger: str
    cgst_ledger: Optional[str] = None
    sgst_ledger: Optional[str] = None
    igst_ledger: Optional[str] = None
    tax_type: Literal["cgst_sgst", "igst", "exempt"]
    match_type: Literal["exact", "regex", "category_default", "fallback"]
    confidence_penalty: float  # 0.0 = perfect match, 0.2 = fallback
