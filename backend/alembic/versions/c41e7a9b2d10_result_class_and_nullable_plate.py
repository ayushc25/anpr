"""result class (A/B/C), nullable events.plate_number, review crop paths

A vehicle the system saw but could not name - a class B result with no usable
plate string, or a class C (no plate visible) - has no plate to store. NULL is
the honest value: an empty string or a placeholder would match the wrong thing
in a plate search, a registry lookup or an alert rule.

Existing rows keep their plate and get result_class NULL (they predate the
classification and are not retro-labelled).

Revision ID: c41e7a9b2d10
Revises: 9a67a0a34836
Create Date: 2026-10-08 12:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

revision = 'c41e7a9b2d10'
down_revision = '9a67a0a34836'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column('events', 'plate_number', existing_type=sa.String(length=32), nullable=True)
    op.add_column('events', sa.Column('result_class', sa.String(length=1), nullable=True))
    op.add_column('events', sa.Column('result_reason', sa.String(length=48), nullable=True))
    op.add_column('events', sa.Column('result_detail', sa.JSON(), nullable=True))
    op.add_column('events', sa.Column('review_crop_paths', sa.JSON(), nullable=True))
    op.create_index(op.f('ix_events_result_class'), 'events', ['result_class'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_events_result_class'), table_name='events')
    op.drop_column('events', 'review_crop_paths')
    op.drop_column('events', 'result_detail')
    op.drop_column('events', 'result_reason')
    op.drop_column('events', 'result_class')
    # Rows with no plate cannot satisfy NOT NULL. Give them the placeholder the
    # column's old contract required rather than failing the downgrade, and
    # say so: this is the one lossy step in the migration.
    op.execute("UPDATE events SET plate_number = 'UNKNOWN' WHERE plate_number IS NULL")
    op.alter_column('events', 'plate_number', existing_type=sa.String(length=32), nullable=False)
