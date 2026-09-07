"""Keep automated tests isolated from the developer's local media store."""

import os
import tempfile
from pathlib import Path


_test_root = Path(tempfile.gettempdir()) / f"video-evidence-agent-tests-{os.getpid()}"
os.environ.setdefault("VIDEO_EVIDENCE_DB_PATH", str(_test_root / "agent.sqlite3"))
os.environ.setdefault("VIDEO_EVIDENCE_UPLOADS_DIR", str(_test_root / "uploads"))
os.environ.setdefault("VIDEO_EVIDENCE_QDRANT_DIR", str(_test_root / "qdrant"))
os.environ.setdefault("AUTH_REQUIRED", "false")
os.environ.setdefault("EMBEDDING_ENABLED", "false")
