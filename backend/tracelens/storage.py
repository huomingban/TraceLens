"""Database persistence for evidence, media resources, and Agent sessions.

The application uses a small compatibility wrapper while the older API modules
are migrated away from positional SQL. SQLAlchemy owns the connection,
transactions, and dialect selection; existing callers can keep using
``row["column"]`` and ``?`` parameters during the transition.
"""
from __future__ import annotations
import hashlib
import json
import re
import uuid
from contextlib import contextmanager
from collections.abc import Mapping, Iterator
from pathlib import Path
from typing import Any
from fastapi import HTTPException
from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, MetaData, String,
    Table, Text, create_engine, func,
)
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql import text
from .config import DATABASE_URL, DB_PATH, UPLOADS_DIR, env_flag
from .models import Evidence

metadata = MetaData()
users = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", String(255), nullable=False, unique=True),
    Column("password_hash", String(255), nullable=False),
    Column("role", String(32), nullable=False, server_default="USER"),
    Column("is_active", Boolean, nullable=False, server_default="1"),
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
)
video_owners = Table(
    "video_owners", metadata,
    Column("video_id", String(255), primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id"), nullable=False),
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
)
evidence = Table(
    "evidence", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("video_id", String(255), nullable=False),
    Column("start_seconds", Float, nullable=False),
    Column("end_seconds", Float, nullable=False),
    Column("text", Text, nullable=False),
    Column("source", String(32), nullable=False, server_default="ASR"),
)
videos = Table(
    "videos", metadata,
    Column("video_id", String(255), primary_key=True),
    Column("filename", String(255), nullable=False),
    Column("stored_path", String(1024), nullable=False),
    Column("content_hash", String(64), nullable=False),
    Column("status", String(32), nullable=False, server_default="COMPLETED"),
    Column("ocr_status", String(32), nullable=False, server_default="UNKNOWN"),
    Column("transcript_text", Text, nullable=True),
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
    Column("updated_at", DateTime, nullable=False, server_default=func.now()),
)
agent_sessions = Table(
    "agent_sessions", metadata,
    Column("session_id", String(100), primary_key=True),
    Column("video_id", String(255), nullable=True),
    Column("title", String(255), nullable=True),
    Column("summary", Text, nullable=True),
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
    Column("updated_at", DateTime, nullable=False, server_default=func.now()),
)
agent_messages = Table(
    "agent_messages", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", String(100), ForeignKey("agent_sessions.session_id"), nullable=False),
    Column("role", String(32), nullable=False),
    Column("content", Text, nullable=False),
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
)
agent_reports = Table(
    "agent_reports", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", String(100), ForeignKey("agent_sessions.session_id"), nullable=False),
    Column("question", Text, nullable=False),
    Column("answer", Text, nullable=False),
    Column("answerable", Boolean, nullable=False),
    Column("support_level", String(32), nullable=False),
    Column("report_json", Text().with_variant(LONGTEXT(), "mysql"), nullable=False),
    Column("trace_json", Text().with_variant(LONGTEXT(), "mysql"), nullable=True),
    Column("report_type", String(32), nullable=False, server_default="INITIAL"),
    Column("parent_report_id", Integer, nullable=True),
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
)
media_tasks = Table(
    "media_tasks", metadata,
    Column("task_id", String(100), primary_key=True),
    Column("video_id", String(255), nullable=False),
    Column("task_type", String(32), nullable=False),
    Column("state", String(32), nullable=False, server_default="QUEUED"),
    Column("progress_current", Integer, nullable=False, server_default="0"),
    Column("progress_total", Integer, nullable=False, server_default="0"),
    Column("progress_message", Text, nullable=True),
    Column("result_json", Text().with_variant(LONGTEXT(), "mysql"), nullable=True),
    Column("error", Text, nullable=True),
    Column("question", Text, nullable=True),
    Column("session_id", String(100), nullable=True),
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
    Column("started_at", DateTime, nullable=True),
    Column("finished_at", DateTime, nullable=True),
    Column("updated_at", DateTime, nullable=False, server_default=func.now()),
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=1800,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)


class CompatRow(Mapping[str, Any]):
    """SQLAlchemy Row with the old sqlite3.Row indexing behavior."""

    def __init__(self, row: Any):
        self._row = row
        self._mapping = row._mapping

    def __getitem__(self, key: str | int) -> Any:
        return self._row[key] if isinstance(key, int) else self._mapping[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)

    def keys(self):
        return self._mapping.keys()


class CompatResult:
    def __init__(self, result: Any):
        self._result = result
        self.lastrowid = getattr(result, "lastrowid", None)
        if self.lastrowid is None:
            try:
                self.lastrowid = result.inserted_primary_key[0]
            except Exception:
                pass

    def fetchone(self) -> CompatRow | None:
        row = self._result.fetchone()
        return CompatRow(row) if row is not None else None

    def fetchall(self) -> list[CompatRow]:
        return [CompatRow(row) for row in self._result.fetchall()]


class ConnectionAdapter:
    def __init__(self, connection: Any):
        self._connection = connection

    @property
    def sa_connection(self) -> Any:
        return self._connection

    @property
    def dialect_name(self) -> str:
        return self._connection.dialect.name

    @staticmethod
    def _bind(sql: str, params: Any) -> tuple[Any, dict[str, Any]]:
        if isinstance(params, dict):
            return text(sql), params
        values = tuple(params or ())
        index = 0
        output: list[str] = []
        for character in sql:
            if character == "?":
                name = f"p{index}"
                output.append(f":{name}")
                index += 1
            else:
                output.append(character)
        if index != len(values):
            raise ValueError(f"SQL parameter count mismatch: expected {index}, got {len(values)}")
        return text("".join(output)), {f"p{i}": value for i, value in enumerate(values)}

    def execute(self, sql: str, params: Any = None) -> CompatResult:
        statement, bound = self._bind(sql, params)
        return CompatResult(self._connection.execute(statement, bound))

    def executemany(self, sql: str, params_list: list[tuple[Any, ...]]) -> CompatResult:
        statement, _ = self._bind(sql, params_list[0] if params_list else ())
        values = []
        for params in params_list:
            _, bound = self._bind(sql, params)
            values.append(bound)
        return CompatResult(self._connection.execute(statement, values))


@contextmanager
def get_connection() -> Iterator[ConnectionAdapter]:
    with engine.begin() as connection:
        yield ConnectionAdapter(connection)


def upsert_video(connection: ConnectionAdapter, values: dict[str, Any], update_columns: list[str]) -> None:
    """Insert or update a video using the active database dialect."""
    if connection.dialect_name == "sqlite":
        from sqlalchemy.dialects.sqlite import insert
        statement = insert(videos).values(**values)
        statement = statement.on_conflict_do_update(
            index_elements=[videos.c.video_id],
            set_={column: getattr(statement.excluded, column) for column in update_columns},
        )
    else:
        from sqlalchemy.dialects.mysql import insert
        statement = insert(videos).values(**values)
        statement = statement.on_duplicate_key_update(
            **{column: getattr(statement.inserted, column) for column in update_columns}
        )
    connection.sa_connection.execute(statement)


def init_db() -> None:
    """Create a local schema when explicitly allowed.

    Production uses ``alembic upgrade head`` and sets AUTO_CREATE_SCHEMA=false.
    Keeping this local fallback makes the existing SQLite test workflow simple.
    """
    if DATABASE_URL.startswith("sqlite") and env_flag("AUTO_CREATE_SCHEMA", True):
        metadata.create_all(engine)


def create_user(username: str, password_hash: str) -> dict[str, Any]:
    with get_connection() as connection:
        try:
            cursor = connection.execute(
                "INSERT INTO users(username, password_hash) VALUES (?, ?)",
                (username, password_hash),
            )
        except IntegrityError as error:
            raise ValueError("username already exists") from error
        return {"id": cursor.lastrowid, "username": username, "role": "USER", "is_active": True}


def get_user_by_username(username: str) -> dict[str, Any] | None:
    with get_connection() as connection:
        row = connection.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["is_active"] = bool(result.get("is_active"))
    return result


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    with get_connection() as connection:
        row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["is_active"] = bool(result.get("is_active"))
    return result


def claim_video(video_id: str, user_id: int) -> None:
    with get_connection() as connection:
        row = connection.execute("SELECT user_id FROM video_owners WHERE video_id = ?", (video_id,)).fetchone()
        if row is None:
            connection.execute("INSERT INTO video_owners(video_id, user_id) VALUES (?, ?)", (video_id, user_id))
        elif int(row["user_id"]) != int(user_id):
            raise HTTPException(status_code=404, detail="video not found")


def user_owns_video(video_id: str, user_id: int) -> bool:
    with get_connection() as connection:
        row = connection.execute("SELECT user_id FROM video_owners WHERE video_id = ?", (video_id,)).fetchone()
    return row is not None and int(row["user_id"]) == int(user_id)


def create_media_task(
    video_id: str,
    task_type: str,
    *,
    question: str | None = None,
    session_id: str | None = None,
) -> str:
    task_id = uuid.uuid4().hex
    with get_connection() as connection:
        connection.execute(
            "INSERT INTO media_tasks(task_id, video_id, task_type, question, session_id) VALUES (?, ?, ?, ?, ?)",
            (task_id, video_id, task_type.upper(), question, session_id),
        )
    return task_id


def update_media_task(task_id: str, **values: Any) -> None:
    allowed = {
        "state", "progress_current", "progress_total", "progress_message",
        "result_json", "error", "started_at", "finished_at",
    }
    values = {key: value for key, value in values.items() if key in allowed}
    if not values:
        return
    assignments = ", ".join(f"{key} = ?" for key in values)
    with get_connection() as connection:
        connection.execute(
            f"UPDATE media_tasks SET {assignments}, updated_at = CURRENT_TIMESTAMP WHERE task_id = ?",
            (*values.values(), task_id),
        )


def get_media_task(task_id: str) -> dict[str, Any] | None:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM media_tasks WHERE task_id = ?", (task_id,)
        ).fetchone()
    return dict(row) if row else None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def register_legacy_videos() -> None:
    """Reconcile upload directories with the persistent video resource table."""
    if not UPLOADS_DIR.is_dir():
        return
    with get_connection() as connection:
        for video_dir in UPLOADS_DIR.iterdir():
            if not video_dir.is_dir() or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", video_dir.name):
                continue
            existing = connection.execute(
                "SELECT video_id, filename, stored_path, content_hash FROM videos WHERE video_id = ?",
                (video_dir.name,),
            ).fetchone()
            candidates = [
                path for path in video_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpeg", ".mpg"}
            ]
            if not candidates:
                continue
            path = max(candidates, key=lambda item: item.stat().st_mtime)
            digest = file_sha256(path)
            if existing is None:
                evidence_count = connection.execute(
                    "SELECT COUNT(*) FROM evidence WHERE video_id = ?", (video_dir.name,)
                ).fetchone()[0]
                connection.execute(
                    """
                    INSERT INTO videos(video_id, filename, stored_path, content_hash, status, transcript_text)
                    VALUES (?, ?, ?, ?, 'COMPLETED', ?)
                    """,
                    (
                        video_dir.name,
                        path.name,
                        str(path),
                        digest,
                        "\n".join(
                            row[0] for row in connection.execute(
                                "SELECT text FROM evidence WHERE video_id = ? ORDER BY start_seconds, id",
                                (video_dir.name,),
                            ).fetchall()
                        ) if evidence_count else None,
                    ),
                )
            elif (
                existing["filename"] != path.name
                or existing["stored_path"] != str(path)
                or existing["content_hash"] != digest
            ):
                connection.execute(
                    "UPDATE videos SET filename = ?, stored_path = ?, content_hash = ?, updated_at = CURRENT_TIMESTAMP WHERE video_id = ?",
                    (path.name, str(path), digest, video_dir.name),
                )
def get_or_create_session(session_id: str | None, video_id: str | None) -> str:
    if session_id and not re.fullmatch(r"[A-Za-z0-9-]{8,100}", session_id):
        raise HTTPException(status_code=422, detail="invalid session_id")
    with get_connection() as connection:
        if session_id:
            row = connection.execute(
                "SELECT video_id FROM agent_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is not None:
                if row["video_id"] != video_id:
                    raise HTTPException(status_code=409, detail="session belongs to another video")
                return session_id
        new_session_id = uuid.uuid4().hex
        connection.execute(
            "INSERT INTO agent_sessions(session_id, video_id, title) VALUES (?, ?, ?)",
            (new_session_id, video_id, None),
        )
        return new_session_id


def get_session_history(session_id: str, limit: int = 12) -> list[dict[str, str]]:
    limit = max(2, min(int(limit), 20))
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT role, content FROM agent_messages WHERE session_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]


def get_latest_agent_result(session_id: str) -> dict[str, Any] | None:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT report_json FROM agent_reports WHERE session_id = ? AND report_type = 'INITIAL' "
            "ORDER BY id DESC LIMIT 1",
            (session_id,),
        ).fetchone()
    if not row or not row["report_json"]:
        return None
    try:
        value = json.loads(row["report_json"])
    except (TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def save_agent_turn(session_id: str, question: str, result: dict[str, Any]) -> None:
    # Every submitted question gets a durable record, including safe refusals
    # and provider failures. The UI can then explain what happened and users
    # can remove the record explicitly.
    answer = str(result.get("answer") or result.get("error") or "未生成回答").strip()
    report_type = "FOLLOW_UP" if result.get("kind") == "follow_up" else "INITIAL"
    with get_connection() as connection:
        parent = None
        if report_type == "FOLLOW_UP":
            parent = connection.execute(
                "SELECT id FROM agent_reports WHERE session_id = ? AND report_type = 'INITIAL' "
                "ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        connection.executemany(
            "INSERT INTO agent_messages(session_id, role, content) VALUES (?, ?, ?)",
            [(session_id, "user", question), (session_id, "assistant", answer)],
        )
        connection.execute(
            """
            INSERT INTO agent_reports(
                session_id, question, answer, answerable, support_level, report_json, trace_json,
                report_type, parent_report_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                question,
                answer,
                int(bool(result.get("grounded"))),
                str(result.get("support_level") or ("DIRECT" if result.get("grounded") else "INSUFFICIENT")),
                json.dumps(result, ensure_ascii=False),
                json.dumps(result.get("tool_trace") or result.get("trace") or [], ensure_ascii=False),
                report_type,
                parent["id"] if parent else None,
            ),
        )
        connection.execute(
            "UPDATE agent_sessions SET title = COALESCE(title, ?), updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
            (question[:80], session_id),
        )
        recent = connection.execute(
            "SELECT role, content FROM agent_messages WHERE session_id = ? ORDER BY id DESC LIMIT 12",
            (session_id,),
        ).fetchall()
        recent.reverse()
        summary = "\n".join(
            f"{item['role']}: {str(item['content'])[:1200]}"
            for item in recent
        )[-6000:]
        connection.execute(
            "UPDATE agent_sessions SET summary = ?, updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
            (summary, session_id),
        )


def get_session_summary(session_id: str) -> str:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT summary FROM agent_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    return str(row["summary"] or "") if row else ""
