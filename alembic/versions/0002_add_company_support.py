"""add_company_support

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-31

Creates the companies table, adds company_id to the documents table,
seeds the default company from settings, and maps existing documents.
"""
from typing import Sequence, Union
from datetime import datetime

from alembic import op
import sqlalchemy as sa
from app.config import settings

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create table: companies
    companies_table = op.create_table(
        "companies",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("gstin", sa.String(15), nullable=False),
        sa.Column("state_code", sa.String(2), nullable=False),
        sa.Column("tally_host", sa.String(255), nullable=False, server_default="localhost"),
        sa.Column("tally_port", sa.Integer(), nullable=False, server_default="9000"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("gstin", name="uq_companies_gstin"),
    )

    # 2. Add column: company_id to documents
    op.add_column("documents", sa.Column("company_id", sa.Integer(), nullable=True))

    # 3. Seed default company using settings
    company_name = "KODRYX AI PRIVATE LIMITED"
    company_gstin = settings.COMPANY_GSTIN or "36AALCK7998P2Z6"
    company_state = settings.COMPANY_STATE_CODE or "36"
    tally_host = settings.TALLY_HOST or "localhost"
    tally_port = settings.TALLY_PORT or 9000

    op.bulk_insert(
        companies_table,
        [
            {
                "name": company_name,
                "gstin": company_gstin,
                "state_code": company_state,
                "tally_host": tally_host,
                "tally_port": tally_port,
                "is_active": True,
            }
        ]
    )

    # 4. Update all existing documents to map to default company (ID = 1)
    op.execute("UPDATE documents SET company_id = 1")

    # 5. Create ForeignKey Constraint and Index for company_id in documents
    op.create_foreign_key(
        "fk_documents_company_id",
        "documents",
        "companies",
        ["company_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_documents_company_id", "documents", ["company_id"])


def downgrade() -> None:
    op.drop_index("ix_documents_company_id", table_name="documents")
    op.drop_constraint("fk_documents_company_id", "documents", type_="foreignkey")
    op.drop_column("documents", "company_id")
    op.drop_table("companies")
