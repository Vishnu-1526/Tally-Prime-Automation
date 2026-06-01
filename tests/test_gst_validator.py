from __future__ import annotations

import pytest
import asyncio
from datetime import date, timedelta

from app.agents.gst_validator_agent import validate
from app.schemas.invoice import ExtractedInvoice, LineItem
from app.schemas.ledger import LedgerMapping


@pytest.mark.asyncio
async def test_valid_igst_invoice_all_passed():
    # 1. Valid IGST invoice where all checks pass
    invoice = ExtractedInvoice(
        segment_id="seg-123",
        vendor_name="Airtel India",
        vendor_gstin="27AAAAA1111A1Z2",  # Valid format
        vendor_address="Mumbai, Maharashtra",
        vendor_state_code="27",
        invoice_number="INV-2026-001",
        invoice_date=date.today().strftime("%Y-%m-%d"),  # Today (valid)
        buyer_gstin="27AAAAA0000A1Z5",
        buyer_state_code="27",
        line_items=[
            LineItem(
                description="Broadband Subscription",
                hsn_code="9984",  # Valid 4 digit HSN
                quantity=1.0,
                unit="months",
                rate=1000.0,
                taxable_value=1000.0,
                cgst_rate=0.0,
                cgst_amount=0.0,
                sgst_rate=0.0,
                sgst_amount=0.0,
                igst_rate=18.0,
                igst_amount=180.0
            )
        ],
        subtotal=1000.0,
        total_cgst=0.0,
        total_sgst=0.0,
        total_igst=180.0,
        round_off=0.0,
        grand_total=1180.0,
        payment_terms="Net 30",
        category="internet",
        raw_text="Airtel internet invoice for INR 1180. GSTIN 27AAAAA1111A1Z2"
    )

    ledger = LedgerMapping(
        debit_ledger="Internet Expenses",
        credit_ledger="Sundry Creditors",
        igst_ledger="IGST @18%",
        tax_type="igst",
        match_type="exact",
        confidence_penalty=0.0
    )

    report = await validate(invoice, ledger)
    assert report.all_passed is True
    assert len(report.blocking_errors) == 0
    assert report.total_penalty == 0.0


@pytest.mark.asyncio
async def test_cgst_sgst_invoice_wrong_state_consistency_fails():
    # 2. CGST/SGST invoice with wrong tax_type configuration inside LedgerMapping
    invoice = ExtractedInvoice(
        segment_id="seg-124",
        vendor_name="Local Pantry Supplier",
        vendor_gstin="27AAAAA1111A1Z2",  # Valid format
        vendor_state_code="27",
        invoice_number="INV-2026-002",
        invoice_date=date.today().strftime("%Y-%m-%d"),
        line_items=[
            LineItem(
                description="Biscuits and Tea",
                hsn_code="1905",  # Valid
                taxable_value=500.0,
                cgst_rate=2.5,
                cgst_amount=12.5,
                sgst_rate=2.5,
                sgst_amount=12.5,
                igst_rate=0.0,
                igst_amount=0.0
            )
        ],
        subtotal=500.0,
        total_cgst=12.5,
        total_sgst=12.5,
        total_igst=0.0,
        round_off=0.0,
        grand_total=525.0,
        category="pantry",
        raw_text="Tea invoice. CGST 12.50, SGST 12.50"
    )

    # Wrong matching ledger: tax_type configured as 'igst', but invoice has CGST/SGST charges
    ledger = LedgerMapping(
        debit_ledger="Pantry Expenses",
        credit_ledger="Sundry Creditors",
        igst_ledger="IGST @5%",
        tax_type="igst",  # Conflict!
        match_type="category_default",
        confidence_penalty=0.1
    )

    report = await validate(invoice, ledger)
    assert report.all_passed is False
    assert "tax_type_consistency" in report.blocking_errors


@pytest.mark.asyncio
async def test_grand_total_mismatch_fails():
    # 3. Grand total mismatch -> grand_total_balance fails
    invoice = ExtractedInvoice(
        segment_id="seg-125",
        vendor_name="Ola Cabs",
        vendor_gstin="27AAAAA1111A1Z2",
        invoice_number="OLA-9988",
        invoice_date=date.today().strftime("%Y-%m-%d"),
        line_items=[
            LineItem(
                description="Cab ride",
                hsn_code="9964",
                taxable_value=200.0,
                cgst_rate=2.5,
                cgst_amount=5.0,
                sgst_rate=2.5,
                sgst_amount=5.0
            )
        ],
        subtotal=200.0,
        total_cgst=5.0,
        total_sgst=5.0,
        total_igst=0.0,
        round_off=0.0,
        grand_total=300.0,  # Expected total is 210.0! (200 + 5 + 5). Difference is 90.0
        category="travel",
        raw_text="Cab receipt grand total: 300.0"
    )

    ledger = LedgerMapping(
        debit_ledger="Travel Expenses",
        credit_ledger="Sundry Creditors",
        cgst_ledger="CGST @2.5%",
        sgst_ledger="SGST @2.5%",
        tax_type="cgst_sgst",
        match_type="exact",
        confidence_penalty=0.0
    )

    report = await validate(invoice, ledger)
    assert report.all_passed is False
    assert "grand_total_balance" in report.blocking_errors


@pytest.mark.asyncio
async def test_missing_gstin_warning_only():
    # 4. Missing GSTIN -> warning only, not blocking, penalty added
    invoice = ExtractedInvoice(
        segment_id="seg-126",
        vendor_name="Taxi Receipt",
        vendor_gstin=None,  # Missing!
        invoice_number="TX-123",
        invoice_date=date.today().strftime("%Y-%m-%d"),
        line_items=[
            LineItem(
                description="Local taxi",
                hsn_code="9964",
                taxable_value=150.0,
                cgst_rate=0.0,
                sgst_rate=0.0
            )
        ],
        subtotal=150.0,
        total_cgst=0.0,
        total_sgst=0.0,
        total_igst=0.0,
        round_off=0.0,
        grand_total=150.0,
        category="travel",
        raw_text="Cash cab receipt 150"
    )

    ledger = LedgerMapping(
        debit_ledger="Travel Expenses",
        credit_ledger="Sundry Creditors",
        tax_type="exempt",
        match_type="fallback",
        confidence_penalty=0.2
    )

    report = await validate(invoice, ledger)
    
    # It fails overall validation due to the warning (or missing GSTIN format check not passing)
    assert report.all_passed is False
    
    # But because missing GSTIN is a warning, it should NOT be in blocking_errors
    assert "gstin_format" not in report.blocking_errors
    
    # Check that the 0.10 penalty is included in total_penalty
    assert report.total_penalty >= 0.10
