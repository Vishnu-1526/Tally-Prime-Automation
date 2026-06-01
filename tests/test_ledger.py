from __future__ import annotations

import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
import pytest

from app.agents.ledger_agent import LedgerMappingAgent, ledger_node, _FALLBACK_RULE
from app.schemas.invoice import ExtractedInvoice, LineItem
from app.schemas.ledger import LedgerMapping
from app.models.ledger_rule import LedgerRule
from app.config import settings


# ---------------------------------------------------------------------------
# Test Tax Type Determination
# ---------------------------------------------------------------------------

class TestDetermineTaxType:
    def test_intra_state_explicit_total_cgst(self):
        agent = LedgerMappingAgent()
        invoice = ExtractedInvoice(
            segment_id="test",
            line_items=[],
            total_cgst=10.0,
            total_sgst=10.0,
            total_igst=0.0,
            grand_total=120.0,
            raw_text=""
        )
        assert agent._determine_tax_type(invoice) == "cgst_sgst"

    def test_intra_state_explicit_line_cgst(self):
        agent = LedgerMappingAgent()
        invoice = ExtractedInvoice(
            segment_id="test",
            line_items=[
                LineItem(description="Item A", taxable_value=100.0, cgst_amount=9.0, sgst_amount=9.0)
            ],
            total_cgst=0.0,
            total_sgst=0.0,
            total_igst=0.0,
            grand_total=118.0,
            raw_text=""
        )
        assert agent._determine_tax_type(invoice) == "cgst_sgst"

    def test_inter_state_explicit_total_igst(self):
        agent = LedgerMappingAgent()
        invoice = ExtractedInvoice(
            segment_id="test",
            line_items=[],
            total_cgst=0.0,
            total_sgst=0.0,
            total_igst=18.0,
            grand_total=118.0,
            raw_text=""
        )
        assert agent._determine_tax_type(invoice) == "igst"

    def test_inter_state_explicit_line_igst(self):
        agent = LedgerMappingAgent()
        invoice = ExtractedInvoice(
            segment_id="test",
            line_items=[
                LineItem(description="Item A", taxable_value=100.0, igst_amount=18.0)
            ],
            total_cgst=0.0,
            total_sgst=0.0,
            total_igst=0.0,
            grand_total=118.0,
            raw_text=""
        )
        assert agent._determine_tax_type(invoice) == "igst"

    def test_state_code_comparison_inter_state(self):
        agent = LedgerMappingAgent()
        # Mock settings COMPANY_STATE_CODE to 27
        with patch.object(settings, "COMPANY_STATE_CODE", "27"):
            invoice = ExtractedInvoice(
                segment_id="test",
                line_items=[],
                vendor_state_code="29",  # Different state!
                total_cgst=0.0,
                total_sgst=0.0,
                total_igst=0.0,
                grand_total=100.0,
                raw_text=""
            )
            assert agent._determine_tax_type(invoice) == "igst"

    def test_state_code_comparison_intra_state(self):
        agent = LedgerMappingAgent()
        # Mock settings COMPANY_STATE_CODE to 27
        with patch.object(settings, "COMPANY_STATE_CODE", "27"):
            invoice = ExtractedInvoice(
                segment_id="test",
                line_items=[],
                vendor_state_code="27",  # Same state!
                total_cgst=0.0,
                total_sgst=0.0,
                total_igst=0.0,
                grand_total=100.0,
                raw_text=""
            )
            assert agent._determine_tax_type(invoice) == "cgst_sgst"


# ---------------------------------------------------------------------------
# Test Rule Matching (Async Database Mocks)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRuleMatching:
    async def test_exact_match(self):
        agent = LedgerMappingAgent()
        invoice = ExtractedInvoice(
            segment_id="test",
            line_items=[],
            vendor_name="JioFiber",
            category="internet",
            raw_text=""
        )

        # Mock database rules
        mock_rule = LedgerRule(
            id=1,
            category="internet",
            provider_pattern=r"Jio\s*Fiber",
            provider_name_display="JioFiber",
            tally_ledger_name="Internet Charges",
            tally_parent_ledger="Indirect Expenses",
            cgst_ledger="CGST @9%",
            sgst_ledger="SGST @9%",
            igst_ledger="IGST @18%",
            credit_ledger="Sundry Creditors",
            priority=50,
            is_active=True
        )

        with patch("app.agents.ledger_agent.get_db_session") as mock_db_session:
            mock_session = AsyncMock()
            mock_db_session.return_value.__aenter__.return_value = mock_session
            mock_result = MagicMock()
            mock_result.scalars.return_value.all.return_value = [mock_rule]
            mock_session.execute.return_value = mock_result

            rule, match_type = await agent._match_rule(invoice)
            assert match_type == "exact"
            assert rule is not None
            assert rule.id == 1

    async def test_regex_match(self):
        agent = LedgerMappingAgent()
        invoice = ExtractedInvoice(
            segment_id="test",
            line_items=[],
            vendor_name="Reliance Jio Infocomm Ltd",
            category="internet",
            raw_text=""
        )

        mock_rule = LedgerRule(
            id=2,
            category="internet",
            provider_pattern=r"Reliance Jio|Jio Infocomm",
            provider_name_display="Jio",
            tally_ledger_name="Internet Charges",
            tally_parent_ledger="Indirect Expenses",
            cgst_ledger="CGST @9%",
            sgst_ledger="SGST @9%",
            igst_ledger="IGST @18%",
            credit_ledger="Sundry Creditors",
            priority=50,
            is_active=True
        )

        with patch("app.agents.ledger_agent.get_db_session") as mock_db_session:
            mock_session = AsyncMock()
            mock_db_session.return_value.__aenter__.return_value = mock_session
            mock_result = MagicMock()
            mock_result.scalars.return_value.all.return_value = [mock_rule]
            mock_session.execute.return_value = mock_result

            rule, match_type = await agent._match_rule(invoice)
            assert match_type == "regex"
            assert rule is not None
            assert rule.id == 2

    async def test_category_default(self):
        agent = LedgerMappingAgent()
        invoice = ExtractedInvoice(
            segment_id="test",
            line_items=[],
            vendor_name="Some Strange ISP",
            category="internet",
            raw_text=""
        )

        mock_rule = LedgerRule(
            id=3,
            category="internet",
            provider_pattern=r"BSNL|Hathway",
            provider_name_display="Generic ISP",
            tally_ledger_name="Internet Charges",
            tally_parent_ledger="Indirect Expenses",
            cgst_ledger="CGST @9%",
            sgst_ledger="SGST @9%",
            igst_ledger="IGST @18%",
            credit_ledger="Sundry Creditors",
            priority=40,
            is_active=True
        )

        with patch("app.agents.ledger_agent.get_db_session") as mock_db_session:
            mock_session = AsyncMock()
            mock_db_session.return_value.__aenter__.return_value = mock_session
            mock_result = MagicMock()
            mock_result.scalars.return_value.all.return_value = [mock_rule]
            mock_session.execute.return_value = mock_result

            rule, match_type = await agent._match_rule(invoice)
            assert match_type == "category_default"
            assert rule is not None
            assert rule.id == 3

    async def test_fallback_when_no_rules_exist(self):
        agent = LedgerMappingAgent()
        invoice = ExtractedInvoice(
            segment_id="test",
            line_items=[],
            vendor_name="Unknown Vendor",
            category="unknown",
            raw_text=""
        )

        with patch("app.agents.ledger_agent.get_db_session") as mock_db_session:
            mock_session = AsyncMock()
            mock_db_session.return_value.__aenter__.return_value = mock_session
            mock_result = MagicMock()
            mock_result.scalars.return_value.all.return_value = []
            mock_session.execute.return_value = mock_result

            rule, match_type = await agent._match_rule(invoice)
            assert match_type == "fallback"
            assert rule is None


# ---------------------------------------------------------------------------
# Test Mapping Construction
# ---------------------------------------------------------------------------

class TestMappingConstruction:
    def test_build_mapping_cgst_sgst(self):
        agent = LedgerMappingAgent()
        rule = LedgerRule(
            id=1,
            category="internet",
            provider_pattern=".*",
            provider_name_display="Test Provider",
            tally_ledger_name="Internet Charges",
            tally_parent_ledger="Indirect Expenses",
            cgst_ledger="CGST @9%",
            sgst_ledger="SGST @9%",
            igst_ledger="IGST @18%",
            credit_ledger="Sundry Creditors"
        )

        mapping = agent._build_mapping(rule, "exact", "cgst_sgst")
        assert mapping.debit_ledger == "Internet Charges"
        assert mapping.credit_ledger == "Sundry Creditors"
        assert mapping.cgst_ledger == "CGST @9%"
        assert mapping.sgst_ledger == "SGST @9%"
        assert mapping.igst_ledger is None
        assert mapping.tax_type == "cgst_sgst"
        assert mapping.match_type == "exact"
        assert mapping.confidence_penalty == 0.0

    def test_build_mapping_igst(self):
        agent = LedgerMappingAgent()
        rule = LedgerRule(
            id=1,
            category="internet",
            provider_pattern=".*",
            provider_name_display="Test Provider",
            tally_ledger_name="Internet Charges",
            tally_parent_ledger="Indirect Expenses",
            cgst_ledger="CGST @9%",
            sgst_ledger="SGST @9%",
            igst_ledger="IGST @18%",
            credit_ledger="Sundry Creditors"
        )

        mapping = agent._build_mapping(rule, "regex", "igst")
        assert mapping.debit_ledger == "Internet Charges"
        assert mapping.credit_ledger == "Sundry Creditors"
        assert mapping.cgst_ledger is None
        assert mapping.sgst_ledger is None
        assert mapping.igst_ledger == "IGST @18%"
        assert mapping.tax_type == "igst"
        assert mapping.match_type == "regex"
        assert mapping.confidence_penalty == 0.0

    def test_build_mapping_exempt(self):
        agent = LedgerMappingAgent()
        rule = LedgerRule(
            id=1,
            category="internet",
            provider_pattern=".*",
            provider_name_display="Test Provider",
            tally_ledger_name="Internet Charges",
            tally_parent_ledger="Indirect Expenses",
            cgst_ledger="CGST @9%",
            sgst_ledger="SGST @9%",
            igst_ledger="IGST @18%",
            credit_ledger="Sundry Creditors"
        )

        mapping = agent._build_mapping(rule, "category_default", "exempt")
        assert mapping.debit_ledger == "Internet Charges"
        assert mapping.credit_ledger == "Sundry Creditors"
        assert mapping.cgst_ledger is None
        assert mapping.sgst_ledger is None
        assert mapping.igst_ledger is None
        assert mapping.tax_type == "exempt"
        assert mapping.match_type == "category_default"
        assert mapping.confidence_penalty == 0.10

    def test_build_mapping_fallback(self):
        agent = LedgerMappingAgent()
        mapping = agent._build_mapping(None, "fallback", "cgst_sgst")
        assert mapping.debit_ledger == _FALLBACK_RULE["tally_ledger_name"]
        assert mapping.credit_ledger == _FALLBACK_RULE["credit_ledger"]
        assert mapping.cgst_ledger == _FALLBACK_RULE["cgst_ledger"]
        assert mapping.sgst_ledger == _FALLBACK_RULE["sgst_ledger"]
        assert mapping.igst_ledger is None
        assert mapping.tax_type == "cgst_sgst"
        assert mapping.match_type == "fallback"
        assert mapping.confidence_penalty == 0.20


# ---------------------------------------------------------------------------
# Test End-To-End Resolve & LangGraph Node Entry Point
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestResolveAndNode:
    async def test_resolve_e2e(self):
        agent = LedgerMappingAgent()
        invoice = ExtractedInvoice(
            segment_id="test",
            line_items=[],
            vendor_name="Asianet Satellite Communications Limited",
            category="internet",
            total_cgst=944.73,
            total_sgst=944.73,
            grand_total=12386.46,
            raw_text=""
        )

        mock_rule = LedgerRule(
            id=10,
            category="internet",
            provider_pattern=r"Asianet",
            provider_name_display="Asianet Satellite Communications",
            tally_ledger_name="Internet Charges",
            tally_parent_ledger="Indirect Expenses",
            cgst_ledger="CGST @9%",
            sgst_ledger="SGST @9%",
            igst_ledger="IGST @18%",
            credit_ledger="Sundry Creditors",
            priority=50,
            is_active=True
        )

        with patch("app.agents.ledger_agent.get_db_session") as mock_db_session:
            mock_session = AsyncMock()
            mock_db_session.return_value.__aenter__.return_value = mock_session
            mock_result = MagicMock()
            mock_result.scalars.return_value.all.return_value = [mock_rule]
            mock_session.execute.return_value = mock_result

            mapping = await agent.resolve(invoice)
            assert isinstance(mapping, LedgerMapping)
            assert mapping.debit_ledger == "Internet Charges"
            assert mapping.cgst_ledger == "CGST @9%"
            assert mapping.sgst_ledger == "SGST @9%"
            assert mapping.igst_ledger is None
            assert mapping.tax_type == "cgst_sgst"
            assert mapping.match_type == "regex"

    async def test_ledger_node_langgraph_entry(self):
        invoice = ExtractedInvoice(
            segment_id="test",
            line_items=[],
            vendor_name="Airtel Broadband",
            category="internet",
            total_cgst=10.0,
            total_sgst=10.0,
            raw_text=""
        )

        mock_rule = LedgerRule(
            id=5,
            category="internet",
            provider_pattern=r"Airtel",
            provider_name_display="Airtel Broadband",
            tally_ledger_name="Internet Charges",
            tally_parent_ledger="Indirect Expenses",
            cgst_ledger="CGST @9%",
            sgst_ledger="SGST @9%",
            igst_ledger="IGST @18%",
            credit_ledger="Sundry Creditors"
        )

        with patch("app.agents.ledger_agent.get_db_session") as mock_db_session, \
             patch("app.agents.ledger_agent._agent.resolve") as mock_resolve:
            
            mock_mapping = LedgerMapping(
                debit_ledger="Internet Charges",
                credit_ledger="Sundry Creditors",
                cgst_ledger="CGST @9%",
                sgst_ledger="SGST @9%",
                tax_type="cgst_sgst",
                match_type="exact",
                confidence_penalty=0.0
            )
            mock_resolve.return_value = mock_mapping

            result = await ledger_node({"invoice": invoice})
            assert "ledger_mapping" in result
            assert result["ledger_mapping"] == mock_mapping
