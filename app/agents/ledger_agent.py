from __future__ import annotations

"""
Ledger Agent — LangGraph node that resolves an ExtractedInvoice to a
LedgerMapping by querying the ledger_rules table in priority order.

Input state:  { "invoice": ExtractedInvoice }
Output state: { "ledger_mapping": LedgerMapping }
"""

import logging
import re
from typing import Any, Optional

from sqlalchemy import and_, select

from app.config import settings
from app.database import get_db_session
from app.models.ledger_rule import LedgerRule
from app.models.company import Company
from app.schemas.invoice import ExtractedInvoice
from app.schemas.ledger import LedgerMapping

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fallback defaults used when no rule matches
# ---------------------------------------------------------------------------

_FALLBACK_RULE = {
    "tally_ledger_name": "Miscellaneous Expenses",
    "tally_parent_ledger": "Indirect Expenses",
    "cgst_ledger": "CGST Payable",
    "sgst_ledger": "SGST Payable",
    "igst_ledger": "IGST Payable",
    "credit_ledger": "Sundry Creditors",
}


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------


class LedgerMappingAgent:
    """
    Wraps ledger matching logic. Instantiate once and call resolve() per invoice.
    """

    # ------------------------------------------------------------------
    # Step 1 — Tax type determination
    # ------------------------------------------------------------------

    def _determine_tax_type(self, invoice: ExtractedInvoice, company_state: str = None) -> str:
        """
        Determine tax type by reading what's actually on the invoice first.
        The invoice's own CGST/SGST/IGST amounts are the ground truth — they
        override any state-code heuristic, which can be wrong when the company
        state code setting doesn't match the real registration state.
        """
        has_cgst = (invoice.total_cgst or 0.0) > 0.0
        has_igst = (invoice.total_igst or 0.0) > 0.0
        has_line_cgst = any((item.cgst_amount or 0.0) > 0.0 for item in invoice.line_items)
        has_line_igst = any((item.igst_amount or 0.0) > 0.0 for item in invoice.line_items)

        # If the invoice explicitly shows CGST charges → intra-state
        if has_cgst or has_line_cgst:
            return "cgst_sgst"

        # If the invoice explicitly shows IGST charges → inter-state
        if has_igst or has_line_igst:
            return "igst"

        # No tax amounts on invoice — fall back to state code comparison
        comp_state = company_state or settings.COMPANY_STATE_CODE
        vendor_state = invoice.vendor_state_code or ""
        if vendor_state and comp_state and vendor_state != comp_state:
            return "igst"

        # Default → intra-state CGST/SGST
        return "cgst_sgst"

    # ------------------------------------------------------------------
    # Step 2 — Rule matching
    # ------------------------------------------------------------------

    async def _match_rule(
        self, invoice: ExtractedInvoice
    ) -> tuple[Optional[LedgerRule], str]:
        """
        Returns (matched_rule, match_type).
        match_type is one of: "exact" | "regex" | "category_default" | "fallback"
        """
        vendor_name = (invoice.vendor_name or "").strip()
        category = (invoice.category or "").lower()

        async with get_db_session() as session:
            # 1. Load all active rules across ALL categories to look for a vendor match first
            stmt = (
                select(LedgerRule)
                .where(LedgerRule.is_active.is_(True))
                .order_by(LedgerRule.priority.desc())
            )
            result = await session.execute(stmt)
            all_rules = list(result.scalars().all())

        if not all_rules:
            logger.debug("No active ledger rules found in the database.")
            return None, "fallback"

        # a. Exact match — check globally across all categories first
        for rule in all_rules:
            if rule.provider_name_display.lower() == vendor_name.lower():
                logger.info("Global Exact match → rule id=%d %r", rule.id, rule.provider_name_display)
                return rule, "exact"

        # b. Regex match — check globally across all categories
        for rule in all_rules:
            try:
                if rule.provider_pattern and re.search(
                    rule.provider_pattern, vendor_name, re.IGNORECASE
                ):
                    logger.info("Global Regex match → rule id=%d pattern=%r", rule.id, rule.provider_pattern)
                    return rule, "regex"
            except re.error as e:
                logger.warning("Invalid regex in rule id=%d: %s", rule.id, e)

        # c. If no direct vendor match is found, filter rules by the predicted category
        category_rules = [r for r in all_rules if r.category == category]
        if not category_rules:
            logger.debug("No active rules found for category %r", category)
            return None, "fallback"

        # d. Category default — first active rule for this category
        logger.info("Category default match → rule id=%d", category_rules[0].id)
        return category_rules[0], "category_default"

    # ------------------------------------------------------------------
    # Step 3 — Build LedgerMapping
    # ------------------------------------------------------------------

    def _build_mapping(
        self,
        rule: Optional[LedgerRule],
        match_type: str,
        tax_type: str,
    ) -> LedgerMapping:
        confidence_penalty = {
            "exact": 0.0,
            "regex": 0.0,
            "category_default": 0.10,
            "fallback": 0.20,
        }.get(match_type, 0.20)

        if rule:
            cgst_l: Optional[str] = rule.cgst_ledger
            sgst_l: Optional[str] = rule.sgst_ledger
            igst_l: Optional[str] = rule.igst_ledger
            credit = rule.credit_ledger
            debit  = rule.tally_ledger_name
        else:
            cgst_l = _FALLBACK_RULE["cgst_ledger"]
            sgst_l = _FALLBACK_RULE["sgst_ledger"]
            igst_l = _FALLBACK_RULE["igst_ledger"]
            credit = _FALLBACK_RULE["credit_ledger"]
            debit  = _FALLBACK_RULE["tally_ledger_name"]

        # Null out ledgers that don't apply for the detected tax type
        if tax_type == "igst":
            cgst_l = None
            sgst_l = None
        elif tax_type == "cgst_sgst":
            igst_l = None
        else:  # exempt
            cgst_l = sgst_l = igst_l = None

        return LedgerMapping(
            debit_ledger=debit,
            credit_ledger=credit,
            cgst_ledger=cgst_l,
            sgst_ledger=sgst_l,
            igst_ledger=igst_l,
            tax_type=tax_type,
            match_type=match_type,
            confidence_penalty=confidence_penalty,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def resolve(self, invoice: ExtractedInvoice, company_id: int = 1) -> LedgerMapping:
        company_state = settings.COMPANY_STATE_CODE
        if company_id:
            async with get_db_session() as session:
                res = await session.execute(select(Company).where(Company.id == company_id))
                company = res.scalar_one_or_none()
                if company:
                    company_state = company.state_code

        tax_type = self._determine_tax_type(invoice, company_state)
        logger.info(
            "Tax type for vendor %r (state %s) → %s",
            invoice.vendor_name,
            invoice.vendor_state_code,
            tax_type,
        )
        rule, match_type = await self._match_rule(invoice)
        if rule is None:
            logger.warning(
                "No rule matched for vendor %r category %r — using fallback",
                invoice.vendor_name,
                invoice.category,
            )
        return self._build_mapping(rule, match_type, tax_type)


# ---------------------------------------------------------------------------
# LangGraph node entry point
# ---------------------------------------------------------------------------

_agent = LedgerMappingAgent()


async def ledger_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node.
    Input state keys:  invoice (ExtractedInvoice)
    Output state keys: ledger_mapping (LedgerMapping)
    """
    invoice: ExtractedInvoice = state["invoice"]
    company_id: int = state.get("company_id", 1)
    ledger_mapping = await _agent.resolve(invoice, company_id)
    return {"ledger_mapping": ledger_mapping}
