from __future__ import annotations

import re
from datetime import datetime, date
from typing import Literal

from app.schemas.invoice import ExtractedInvoice
from app.schemas.ledger import LedgerMapping
from app.schemas.pipeline import ValidationCheck, ValidationReport


async def validate(invoice: ExtractedInvoice, ledger: LedgerMapping) -> ValidationReport:
    """
    Validates the extracted invoice data and ledger mapping against standard compliance checks.
    Returns a complete ValidationReport including confidence penalties and blocking errors.
    """
    checks: list[ValidationCheck] = []

    # CHECK 1 — gstin_format
    gstin_pattern = r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$"
    if invoice.vendor_gstin is not None:
        if re.match(gstin_pattern, invoice.vendor_gstin.strip()):
            c1 = ValidationCheck(
                check_name="gstin_format",
                passed=True,
                message="Vendor GSTIN format is valid.",
                severity="error",
                confidence_penalty=0.0
            )
        else:
            c1 = ValidationCheck(
                check_name="gstin_format",
                passed=False,
                message=f"Vendor GSTIN '{invoice.vendor_gstin}' does not match the standard GSTIN pattern.",
                severity="error",
                confidence_penalty=0.15
            )
    else:
        c1 = ValidationCheck(
            check_name="gstin_format",
            passed=False,
            message="Vendor GSTIN is missing.",
            severity="warning",
            confidence_penalty=0.10
        )
    checks.append(c1)

    # CHECK 2 — tax_type_consistency
    has_cgst = (invoice.total_cgst > 0.0) or any(item.cgst_amount > 0.0 for item in invoice.line_items)
    has_igst = (invoice.total_igst > 0.0) or any(item.igst_amount > 0.0 for item in invoice.line_items)
    
    c2_passed = True
    c2_msg = "Tax type is consistent with invoice charges."
    if ledger.tax_type == "igst" and has_cgst:
        c2_passed = False
        c2_msg = "Ledger tax type is IGST, but CGST charges were found on the invoice."
    elif ledger.tax_type == "cgst_sgst" and has_igst:
        c2_passed = False
        c2_msg = "Ledger tax type is CGST/SGST, but IGST charges were found on the invoice."
        
    c2 = ValidationCheck(
        check_name="tax_type_consistency",
        passed=c2_passed,
        message=c2_msg,
        severity="error",
        confidence_penalty=0.0 if c2_passed else 0.40
    )
    checks.append(c2)

    # CHECK 3 — cgst_sgst_rate_parity
    c3_passed = True
    valid_rates = {0.0, 2.5, 6.0, 9.0, 14.0}
    c3_violations = []
    for idx, item in enumerate(invoice.line_items):
        rates_match = abs(item.cgst_rate - item.sgst_rate) < 1e-5
        cgst_ok = any(abs(item.cgst_rate - r) < 1e-5 for r in valid_rates)
        sgst_ok = any(abs(item.sgst_rate - r) < 1e-5 for r in valid_rates)
        if not rates_match or not cgst_ok or not sgst_ok:
            c3_passed = False
            c3_violations.append(f"Line item {idx+1} (cgst={item.cgst_rate}%, sgst={item.sgst_rate}%)")
            
    c3_msg = "CGST and SGST rates are identical and valid." if c3_passed else f"CGST/SGST rate parity or valid rate violation in: {', '.join(c3_violations)}."
    c3 = ValidationCheck(
        check_name="cgst_sgst_rate_parity",
        passed=c3_passed,
        message=c3_msg,
        severity="error",
        confidence_penalty=0.0 if c3_passed else 0.20
    )
    checks.append(c3)

    # CHECK 4 — amount_accuracy
    c4_passed = True
    c4_violations = []
    for idx, item in enumerate(invoice.line_items):
        expected_cgst = round(item.taxable_value * item.cgst_rate / 100.0, 2)
        expected_sgst = round(item.taxable_value * item.sgst_rate / 100.0, 2)
        expected_igst = round(item.taxable_value * item.igst_rate / 100.0, 2)
        
        cgst_diff = abs(item.cgst_amount - expected_cgst)
        sgst_diff = abs(item.sgst_amount - expected_sgst)
        igst_diff = abs(item.igst_amount - expected_igst)
        
        if cgst_diff > 1.0 or sgst_diff > 1.0 or igst_diff > 1.0:
            c4_passed = False
            details = []
            if cgst_diff > 1.0:
                details.append(f"CGST diff {cgst_diff:.2f}")
            if sgst_diff > 1.0:
                details.append(f"SGST diff {sgst_diff:.2f}")
            if igst_diff > 1.0:
                details.append(f"IGST diff {igst_diff:.2f}")
            c4_violations.append(f"Line item {idx+1} ({', '.join(details)})")
            
    c4_msg = "Tax amounts are mathematically accurate." if c4_passed else f"Tax amount accuracy violations: {'; '.join(c4_violations)}."
    c4 = ValidationCheck(
        check_name="amount_accuracy",
        passed=c4_passed,
        message=c4_msg,
        severity="error",
        confidence_penalty=0.0 if c4_passed else 0.25
    )
    checks.append(c4)

    # CHECK 5 — line_items_sum
    subtotal = invoice.subtotal if invoice.subtotal is not None else 0.0
    expected_subtotal = sum(item.taxable_value for item in invoice.line_items)
    c5_passed = (invoice.subtotal is not None) and (abs(subtotal - expected_subtotal) <= 1.0)
    c5_msg = "Subtotal matches sum of line items." if c5_passed else f"Subtotal mismatch: got {invoice.subtotal}, expected {expected_subtotal} based on line items."
    c5 = ValidationCheck(
        check_name="line_items_sum",
        passed=c5_passed,
        message=c5_msg,
        severity="error",
        confidence_penalty=0.0 if c5_passed else 0.30
    )
    checks.append(c5)

    # CHECK 6 — grand_total_balance
    subtotal = invoice.subtotal if invoice.subtotal is not None else 0.0
    grand_total = invoice.grand_total if invoice.grand_total is not None else 0.0
    # grand_total is now always computed from components in the enrichment agent,
    # so expected = subtotal + cgst + sgst + igst (round_off is always 0).
    expected_grand_total = subtotal + invoice.total_cgst + invoice.total_sgst + invoice.total_igst
    c6_passed = (invoice.grand_total is not None) and (abs(grand_total - expected_grand_total) <= 2.0)
    c6_msg = "Grand total matches expected sum." if c6_passed else f"Grand total mismatch: got {invoice.grand_total}, expected {expected_grand_total}."
    c6 = ValidationCheck(
        check_name="grand_total_balance",
        passed=c6_passed,
        message=c6_msg,
        severity="error",
        confidence_penalty=0.0 if c6_passed else 0.40
    )
    checks.append(c6)

    # CHECK 7 — hsn_code
    # Valid lengths: 4 (goods chapter heading), 6 (goods sub-heading / service SAC), 8 (goods tariff item)
    missing_or_invalid_count = 0
    for item in invoice.line_items:
        hsn = item.hsn_code
        if hsn:  # If HSN is present, validate its length
            hsn_clean = re.sub(r"\D", "", hsn)
            if len(hsn_clean) not in {4, 6, 8}:
                missing_or_invalid_count += 1
        # Missing HSN is NOT penalised — many service invoices omit it legally

    c7_passed = missing_or_invalid_count == 0
    c7_msg = "All HSN/SAC codes are valid." if c7_passed else f"Found {missing_or_invalid_count} line item(s) with invalid HSN/SAC code length (must be 4, 6 or 8 digits)."
    c7 = ValidationCheck(
        check_name="hsn_code",
        passed=c7_passed,
        message=c7_msg,
        severity="warning",
        confidence_penalty=0.05 * missing_or_invalid_count
    )
    checks.append(c7)

    # CHECK 8 — invoice_date
    # Old invoices (> 180 days) are still valid GST documents — no penalty.
    # Only penalise: missing date (can't process) or future date (definitely wrong).
    c8_passed = True
    c8_severity: Literal["error", "warning"] = "warning"
    c8_penalty = 0.0
    c8_msg = "Invoice date is valid."

    if not invoice.invoice_date:
        c8_passed = False
        c8_severity = "warning"
        c8_penalty = 0.10
        c8_msg = "Invoice date is missing."
    else:
        try:
            inv_date = datetime.strptime(invoice.invoice_date.strip(), "%Y-%m-%d").date()
            today = date.today()
            if inv_date > today:
                c8_passed = False
                c8_severity = "error"
                c8_penalty = 0.20
                c8_msg = f"Invoice date {invoice.invoice_date} is in the future."
            elif (today - inv_date).days > 180:
                # Informational only — no penalty, invoice is still valid
                c8_passed = True
                c8_severity = "warning"
                c8_penalty = 0.0
                c8_msg = f"Invoice date {invoice.invoice_date} is older than 180 days (informational only)."
        except ValueError:
            c8_passed = False
            c8_severity = "warning"
            c8_penalty = 0.10
            c8_msg = f"Invalid invoice date format: {invoice.invoice_date}."

    c8 = ValidationCheck(
        check_name="invoice_date",
        passed=c8_passed,
        message=c8_msg,
        severity=c8_severity,
        confidence_penalty=c8_penalty
    )
    checks.append(c8)

    # CHECK 9 — debit_credit_balance
    c9_passed = True
    c9_msg = "Debit and credit ledger configuration is balanced."
    if not ledger.credit_ledger or not ledger.debit_ledger:
        c9_passed = False
        c9_msg = "Ledger mapping is missing credit_ledger or debit_ledger."
        
    c9 = ValidationCheck(
        check_name="debit_credit_balance",
        passed=c9_passed,
        message=c9_msg,
        severity="error",
        confidence_penalty=0.0 if c9_passed else 0.40
    )
    checks.append(c9)

    blocking_errors = [c.check_name for c in checks if not c.passed and c.severity == "error"]
    total_penalty = sum(c.confidence_penalty for c in checks if not c.passed)
    all_passed = all(c.passed for c in checks)

    return ValidationReport(
        all_passed=all_passed,
        checks=checks,
        total_penalty=total_penalty,
        blocking_errors=blocking_errors
    )
