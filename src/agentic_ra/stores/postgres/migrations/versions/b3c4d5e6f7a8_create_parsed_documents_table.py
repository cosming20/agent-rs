"""create parsed_documents table

Revision ID: b3c4d5e6f7a8
Revises: a2b3c4d5e6f7
Create Date: 2026-03-26 12:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "b3c4d5e6f7a8"
down_revision: Union[str, None] = "a2b3c4d5e6f7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "parsed_documents",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("source_path", sa.Text(), nullable=False),
        sa.Column(
            "docling_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_parsed_documents_source_path"),
        "parsed_documents",
        ["source_path"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_parsed_documents_source_path"), table_name="parsed_documents"
    )
    op.drop_table("parsed_documents")
