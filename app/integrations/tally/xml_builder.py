from __future__ import annotations

import xml.etree.ElementTree as ET
from app.schemas.invoice import ExtractedInvoice
from app.schemas.ledger import LedgerMapping


def build_voucher_xml(invoice: ExtractedInvoice, ledger: LedgerMapping) -> str:
    """
    Builds a TallyPrime compatible XML request for posting a purchase voucher.
    """
    envelope = ET.Element("ENVELOPE")
    
    header = ET.SubElement(envelope, "HEADER")
    ET.SubElement(header, "TALLYREQUEST").text = "Import Data"
    
    body = ET.SubElement(envelope, "BODY")
    import_data = ET.SubElement(body, "IMPORTDATA")
    
    req_desc = ET.SubElement(import_data, "REQUESTDESC")
    ET.SubElement(req_desc, "REPORTNAME").text = "Vouchers"
    
    static_variables = ET.SubElement(req_desc, "STATICVARIABLES")
    ET.SubElement(static_variables, "SVEXPORTFORMAT").text = "$$SysName:XML"
    
    req_data = ET.SubElement(import_data, "REQUESTDATA")
    tally_msg = ET.SubElement(req_data, "TALLYMESSAGE", {"UDF": "Signature"})
    
    voucher = ET.SubElement(tally_msg, "VOUCHER", {"VCHTYPE": "Purchase", "ACTION": "Create"})
    
    # Header fields
    ET.SubElement(voucher, "DATE").text = (invoice.invoice_date or "").replace("-", "")
    ET.SubElement(voucher, "VOUCHERNUMBER").text = invoice.invoice_number or ""
    ET.SubElement(voucher, "PARTYLEDGERNAME").text = ledger.credit_ledger
    ET.SubElement(voucher, "PERSISTEDVIEW").text = "Accounting Voucher View"
    
    # Credit ledger entry (Sundry Creditor / Vendor)
    cred_entry = ET.SubElement(voucher, "ALLLEDGERENTRIES.LIST")
    ET.SubElement(cred_entry, "LEDGERNAME").text = ledger.credit_ledger
    ET.SubElement(cred_entry, "ISDEEMEDPOSITIVE").text = "No"  # Credit is negative in Tally double-entry logic
    ET.SubElement(cred_entry, "AMOUNT").text = f"{invoice.grand_total or 0.0}"
    
    # Debit ledger entry (Expense)
    deb_entry = ET.SubElement(voucher, "ALLLEDGERENTRIES.LIST")
    ET.SubElement(deb_entry, "LEDGERNAME").text = ledger.debit_ledger
    ET.SubElement(deb_entry, "ISDEEMEDPOSITIVE").text = "Yes"
    ET.SubElement(deb_entry, "AMOUNT").text = f"-{invoice.subtotal or 0.0}"  # Debit is positive/negative inverse
    
    # Tax ledger entries if present
    if ledger.cgst_ledger and invoice.total_cgst:
        cgst_entry = ET.SubElement(voucher, "ALLLEDGERENTRIES.LIST")
        ET.SubElement(cgst_entry, "LEDGERNAME").text = ledger.cgst_ledger
        ET.SubElement(cgst_entry, "ISDEEMEDPOSITIVE").text = "Yes"
        ET.SubElement(cgst_entry, "AMOUNT").text = f"-{invoice.total_cgst}"
        
    if ledger.sgst_ledger and invoice.total_sgst:
        sgst_entry = ET.SubElement(voucher, "ALLLEDGERENTRIES.LIST")
        ET.SubElement(sgst_entry, "LEDGERNAME").text = ledger.sgst_ledger
        ET.SubElement(sgst_entry, "ISDEEMEDPOSITIVE").text = "Yes"
        ET.SubElement(sgst_entry, "AMOUNT").text = f"-{invoice.total_sgst}"
        
    if ledger.igst_ledger and invoice.total_igst:
        igst_entry = ET.SubElement(voucher, "ALLLEDGERENTRIES.LIST")
        ET.SubElement(igst_entry, "LEDGERNAME").text = ledger.igst_ledger
        ET.SubElement(igst_entry, "ISDEEMEDPOSITIVE").text = "Yes"
        ET.SubElement(igst_entry, "AMOUNT").text = f"-{invoice.total_igst}"
        
    return ET.tostring(envelope, encoding="utf-8").decode("utf-8")
