"""create_tables_and_seed_ledger_rules

Revision ID: 0001
Revises:
Create Date: 2026-05-28

Creates the documents and ledger_rules tables, then seeds all default
vendor-to-ledger mapping rules for the Tally Prime pipeline.
"""
from typing import Sequence, Union
from datetime import datetime

from alembic import op
import sqlalchemy as sa

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # Table: documents
    # ------------------------------------------------------------------
    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("document_uuid", sa.String(36), nullable=False),
        sa.Column("original_filename", sa.String(512), nullable=False),
        sa.Column("file_path", sa.String(1024), nullable=False),
        sa.Column("file_size_bytes", sa.Integer(), nullable=True),
        sa.Column("sha256_hash", sa.String(64), nullable=False),
        sa.Column("overall_status", sa.String(32), nullable=True),
        sa.Column("pipeline_result_json", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_uuid", name="uq_documents_uuid"),
        sa.UniqueConstraint("sha256_hash", name="uq_documents_sha256"),
    )
    op.create_index("ix_documents_sha256", "documents", ["sha256_hash"], unique=True)
    op.create_index("ix_documents_uuid", "documents", ["document_uuid"], unique=True)

    # ------------------------------------------------------------------
    # Table: ledger_rules
    # ------------------------------------------------------------------
    op.create_table(
        "ledger_rules",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("category", sa.String(64), nullable=False),
        sa.Column("provider_pattern", sa.String(255), nullable=False),
        sa.Column("provider_name_display", sa.String(255), nullable=False),
        sa.Column("tally_ledger_name", sa.String(255), nullable=False),
        sa.Column("tally_parent_ledger", sa.String(255), nullable=False),
        sa.Column("cgst_ledger", sa.String(128), nullable=False),
        sa.Column("sgst_ledger", sa.String(128), nullable=False),
        sa.Column("igst_ledger", sa.String(128), nullable=False),
        sa.Column("credit_ledger", sa.String(255), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False, default=10),
        sa.Column("is_active", sa.Boolean(), nullable=False, default=True),
        sa.Column("created_by", sa.String(16), nullable=False, default="seed"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_ledger_rules_category", "ledger_rules", ["category"])
    op.create_index("ix_ledger_rules_priority", "ledger_rules", ["priority"])
    op.create_index(
        "ix_ledger_rules_category_priority",
        "ledger_rules",
        ["category", "priority"],
    )

    # ------------------------------------------------------------------
    # Seed default ledger rules
    # ------------------------------------------------------------------
    ledger_rules = sa.table(
        "ledger_rules",
        sa.column("category", sa.String),
        sa.column("provider_pattern", sa.String),
        sa.column("provider_name_display", sa.String),
        sa.column("tally_ledger_name", sa.String),
        sa.column("tally_parent_ledger", sa.String),
        sa.column("cgst_ledger", sa.String),
        sa.column("sgst_ledger", sa.String),
        sa.column("igst_ledger", sa.String),
        sa.column("credit_ledger", sa.String),
        sa.column("priority", sa.Integer),
        sa.column("is_active", sa.Boolean),
        sa.column("created_by", sa.String),
    )

    seed_data = [
        # ── Electricity ────────────────────────────────────────────────
        dict(
            category="electricity",
            provider_pattern=r"TSSPDCL|TSPDCL",
            provider_name_display="TSSPDCL",
            tally_ledger_name="Electricity Charges",
            tally_parent_ledger="Indirect Expenses",
            cgst_ledger="CGST @9%",
            sgst_ledger="SGST @9%",
            igst_ledger="IGST @18%",
            credit_ledger="Sundry Creditors",
            priority=50,
            is_active=True,
            created_by="seed",
        ),
        dict(
            category="electricity",
            provider_pattern=r"BESCOM|MSEDCL|CESC|WBSEDCL|TANGEDCO|APEPDCL|UHBVN|UPPCL",
            provider_name_display="Generic Electricity Board",
            tally_ledger_name="Electricity Charges",
            tally_parent_ledger="Indirect Expenses",
            cgst_ledger="CGST @9%",
            sgst_ledger="SGST @9%",
            igst_ledger="IGST @18%",
            credit_ledger="Sundry Creditors",
            priority=40,
            is_active=True,
            created_by="seed",
        ),
        # ── Internet ───────────────────────────────────────────────────
        dict(
            category="internet",
            provider_pattern=r"Airtel",
            provider_name_display="Airtel Broadband",
            tally_ledger_name="Internet Charges",
            tally_parent_ledger="Indirect Expenses",
            cgst_ledger="CGST @9%",
            sgst_ledger="SGST @9%",
            igst_ledger="IGST @18%",
            credit_ledger="Sundry Creditors",
            priority=50,
            is_active=True,
            created_by="seed",
        ),
        dict(
            category="internet",
            provider_pattern=r"Jio\s*Fiber|JioFiber",
            provider_name_display="JioFiber",
            tally_ledger_name="Internet Charges",
            tally_parent_ledger="Indirect Expenses",
            cgst_ledger="CGST @9%",
            sgst_ledger="SGST @9%",
            igst_ledger="IGST @18%",
            credit_ledger="Sundry Creditors",
            priority=50,
            is_active=True,
            created_by="seed",
        ),
        dict(
            category="internet",
            provider_pattern=r"ACT\s*Fibernet|ACT Fibernet",
            provider_name_display="ACT Fibernet",
            tally_ledger_name="Internet Charges",
            tally_parent_ledger="Indirect Expenses",
            cgst_ledger="CGST @9%",
            sgst_ledger="SGST @9%",
            igst_ledger="IGST @18%",
            credit_ledger="Sundry Creditors",
            priority=50,
            is_active=True,
            created_by="seed",
        ),
        dict(
            category="internet",
            provider_pattern=r"Asianet\s*Satellite|Asianet\s*Fiber|Asianet",
            provider_name_display="Asianet Satellite Communications",
            tally_ledger_name="Internet Charges",
            tally_parent_ledger="Indirect Expenses",
            cgst_ledger="CGST @9%",
            sgst_ledger="SGST @9%",
            igst_ledger="IGST @18%",
            credit_ledger="Sundry Creditors",
            priority=50,
            is_active=True,
            created_by="seed",
        ),
        dict(
            category="internet",
            provider_pattern=r"BSNL|Hathway|YOU Broadband|Excitel",
            provider_name_display="Generic ISP",
            tally_ledger_name="Internet Charges",
            tally_parent_ledger="Indirect Expenses",
            cgst_ledger="CGST @9%",
            sgst_ledger="SGST @9%",
            igst_ledger="IGST @18%",
            credit_ledger="Sundry Creditors",
            priority=40,
            is_active=True,
            created_by="seed",
        ),
        # ── Travel ─────────────────────────────────────────────────────
        dict(
            category="travel",
            provider_pattern=r"Ola|Uber",
            provider_name_display="Ola/Uber Cab",
            tally_ledger_name="Conveyance Expenses",
            tally_parent_ledger="Indirect Expenses",
            cgst_ledger="CGST @2.5%",
            sgst_ledger="SGST @2.5%",
            igst_ledger="IGST @5%",
            credit_ledger="Sundry Creditors",
            priority=50,
            is_active=True,
            created_by="seed",
        ),
        dict(
            category="travel",
            provider_pattern=r"IRCTC|Indian Railways",
            provider_name_display="IRCTC",
            tally_ledger_name="Travel Expenses",
            tally_parent_ledger="Indirect Expenses",
            cgst_ledger="CGST @2.5%",
            sgst_ledger="SGST @2.5%",
            igst_ledger="IGST @5%",
            credit_ledger="Sundry Creditors",
            priority=50,
            is_active=True,
            created_by="seed",
        ),
        dict(
            category="travel",
            provider_pattern=r"IndiGo|SpiceJet|Air India|Vistara|Go First",
            provider_name_display="Domestic Airline",
            tally_ledger_name="Travel Expenses",
            tally_parent_ledger="Indirect Expenses",
            cgst_ledger="CGST @6%",
            sgst_ledger="SGST @6%",
            igst_ledger="IGST @12%",
            credit_ledger="Sundry Creditors",
            priority=45,
            is_active=True,
            created_by="seed",
        ),
        # ── Rent ───────────────────────────────────────────────────────
        dict(
            category="rent",
            provider_pattern=r"rent|lease|landlord|premises",
            provider_name_display="Generic Rent",
            tally_ledger_name="Rent Expenses",
            tally_parent_ledger="Indirect Expenses",
            cgst_ledger="CGST @9%",
            sgst_ledger="SGST @9%",
            igst_ledger="IGST @18%",
            credit_ledger="Sundry Creditors",
            priority=10,
            is_active=True,
            created_by="seed",
        ),
        # ── Pantry ─────────────────────────────────────────────────────
        dict(
            category="pantry",
            provider_pattern=r"pantry|canteen|beverages|water|aquaguard",
            provider_name_display="Generic Pantry",
            tally_ledger_name="Staff Welfare Expenses",
            tally_parent_ledger="Indirect Expenses",
            cgst_ledger="CGST @2.5%",
            sgst_ledger="SGST @2.5%",
            igst_ledger="IGST @5%",
            credit_ledger="Sundry Creditors",
            priority=10,
            is_active=True,
            created_by="seed",
        ),
        # ── Infrastructure ─────────────────────────────────────────────
        dict(
            category="infrastructure",
            provider_pattern=r".*",
            provider_name_display="Generic Infrastructure",
            tally_ledger_name="Repairs & Maintenance",
            tally_parent_ledger="Indirect Expenses",
            cgst_ledger="CGST @9%",
            sgst_ledger="SGST @9%",
            igst_ledger="IGST @18%",
            credit_ledger="Sundry Creditors",
            priority=10,
            is_active=True,
            created_by="seed",
        ),
        # ── Professional Services ──────────────────────────────────────
        dict(
            category="professional_services",
            provider_pattern=r".*",
            provider_name_display="Generic Professional Services",
            tally_ledger_name="Professional Fees",
            tally_parent_ledger="Indirect Expenses",
            cgst_ledger="CGST @9%",
            sgst_ledger="SGST @9%",
            igst_ledger="IGST @18%",
            credit_ledger="Sundry Creditors",
            priority=10,
            is_active=True,
            created_by="seed",
        ),
        # ── Fallback / Miscellaneous ───────────────────────────────────
        dict(
            category="unknown",
            provider_pattern=r".*",
            provider_name_display="Fallback",
            tally_ledger_name="Miscellaneous Expenses",
            tally_parent_ledger="Indirect Expenses",
            cgst_ledger="CGST Payable",
            sgst_ledger="SGST Payable",
            igst_ledger="IGST Payable",
            credit_ledger="Sundry Creditors",
            priority=1,
            is_active=True,
            created_by="seed",
        ),
    ]

    op.bulk_insert(ledger_rules, seed_data)


def downgrade() -> None:
    op.drop_table("ledger_rules")
    op.drop_table("documents")
