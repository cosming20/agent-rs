"""create memory_entries table

Revision ID: a2b3c4d5e6f7
Revises: 1da85e5b5d58
Create Date: 2026-03-26 10:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "a2b3c4d5e6f7"
down_revision: Union[str, None] = "1da85e5b5d58"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "memory_entries",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("key", sa.String(length=255), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_memory_entries_user_id"),
        "memory_entries",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_memory_entries_key"),
        "memory_entries",
        ["key"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_memory_entries_key"), table_name="memory_entries")
    op.drop_index(op.f("ix_memory_entries_user_id"), table_name="memory_entries")
    op.drop_table("memory_entries")
