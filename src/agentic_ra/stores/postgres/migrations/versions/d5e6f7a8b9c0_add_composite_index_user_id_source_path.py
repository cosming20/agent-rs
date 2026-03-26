"""add composite index user_id_source_path on parsed_documents

Revision ID: d5e6f7a8b9c0
Revises: 71bd3ba7d1c8
Create Date: 2026-03-26 18:00:00.000000
"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "d5e6f7a8b9c0"
down_revision: Union[str, None] = "71bd3ba7d1c8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_parsed_documents_user_id_source_path",
        "parsed_documents",
        ["user_id", "source_path"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_parsed_documents_user_id_source_path",
        table_name="parsed_documents",
    )
