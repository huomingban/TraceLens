"""Create the TraceLens persistence schema.

Revision ID: 20260905_01
Revises:
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.mysql import LONGTEXT

revision: str = "20260905_01"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("username", sa.String(255), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("role", sa.String(32), server_default="USER", nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("1"), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("username", name="uq_users_username"),
    )
    op.create_table(
        "video_owners",
        sa.Column("video_id", sa.String(255), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("video_id"),
    )
    op.create_table(
        "videos",
        sa.Column("video_id", sa.String(255), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("stored_path", sa.String(1024), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), server_default="COMPLETED", nullable=False),
        sa.Column("ocr_status", sa.String(32), server_default="UNKNOWN", nullable=False),
        sa.Column("transcript_text", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("video_id"),
    )
    op.create_table(
        "evidence",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("video_id", sa.String(255), nullable=False),
        sa.Column("start_seconds", sa.Float(), nullable=False),
        sa.Column("end_seconds", sa.Float(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("source", sa.String(32), server_default="ASR", nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_evidence_video_timeline", "evidence", ["video_id", "start_seconds"])
    op.create_table(
        "agent_sessions",
        sa.Column("session_id", sa.String(100), nullable=False),
        sa.Column("video_id", sa.String(255), nullable=True),
        sa.Column("title", sa.String(255), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("session_id"),
    )
    op.create_table(
        "agent_messages",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(100), nullable=False),
        sa.Column("role", sa.String(32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["agent_sessions.session_id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "agent_reports",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(100), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("answerable", sa.Boolean(), nullable=False),
        sa.Column("support_level", sa.String(32), nullable=False),
        sa.Column("report_json", sa.Text().with_variant(LONGTEXT(), "mysql"), nullable=False),
        sa.Column("trace_json", sa.Text().with_variant(LONGTEXT(), "mysql"), nullable=True),
        sa.Column("report_type", sa.String(32), server_default="INITIAL", nullable=False),
        sa.Column("parent_report_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["agent_sessions.session_id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "media_tasks",
        sa.Column("task_id", sa.String(100), nullable=False),
        sa.Column("video_id", sa.String(255), nullable=False),
        sa.Column("task_type", sa.String(32), nullable=False),
        sa.Column("state", sa.String(32), server_default="QUEUED", nullable=False),
        sa.Column("progress_current", sa.Integer(), server_default="0", nullable=False),
        sa.Column("progress_total", sa.Integer(), server_default="0", nullable=False),
        sa.Column("progress_message", sa.Text(), nullable=True),
        sa.Column("result_json", sa.Text().with_variant(LONGTEXT(), "mysql"), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("question", sa.Text(), nullable=True),
        sa.Column("session_id", sa.String(100), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("task_id"),
    )


def downgrade() -> None:
    op.drop_table("media_tasks")
    op.drop_table("agent_reports")
    op.drop_table("agent_messages")
    op.drop_table("agent_sessions")
    op.drop_index("ix_evidence_video_timeline", table_name="evidence")
    op.drop_table("evidence")
    op.drop_table("videos")
    op.drop_table("video_owners")
    op.drop_table("users")
