"""Application paths and environment-backed runtime configuration."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / "backend" / ".env"

if load_dotenv is not None:
    load_dotenv(ENV_PATH)

# VIDEO_EVIDENCE_DB_PATH is deliberately an explicit local-test override.
# Otherwise DATABASE_URL is the source of truth for persistent storage.
DB_PATH = Path(os.getenv("VIDEO_EVIDENCE_DB_PATH", str(PROJECT_ROOT / "data" / "agent.sqlite3"))).resolve()
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if os.getenv("VIDEO_EVIDENCE_DB_PATH"):
    DATABASE_URL = f"sqlite:///{DB_PATH.as_posix()}"
elif not DATABASE_URL:
    DATABASE_URL = f"sqlite:///{DB_PATH.as_posix()}"
UPLOADS_DIR = Path(os.getenv("VIDEO_EVIDENCE_UPLOADS_DIR", str(PROJECT_ROOT / "data" / "uploads"))).resolve()
LOCAL_QDRANT_DIR = Path(os.getenv("VIDEO_EVIDENCE_QDRANT_DIR", str(PROJECT_ROOT / "data" / "qdrant"))).resolve()

for directory in (DB_PATH.parent, UPLOADS_DIR, LOCAL_QDRANT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("tracelens")


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def llm_settings() -> dict[str, Any]:
    """Read DeepSeek settings, with Kimi names kept as a migration fallback."""
    deepseek_configured = "DEEPSEEK_API_KEY" in os.environ or "DEEPSEEK_BASE_URL" in os.environ
    api_key = (os.getenv("DEEPSEEK_API_KEY", "") if deepseek_configured else os.getenv("KIMI_API_KEY", "")).strip()
    base_url = (os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1") if deepseek_configured else os.getenv(
        "KIMI_BASE_URL", "https://api.deepseek.com/v1"
    )).strip()
    model = (os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash") if deepseek_configured else os.getenv("KIMI_MODEL", "deepseek-v4-flash")).strip()
    # Older local .env files used a model name that this DeepSeek endpoint rejects.
    if model in {"deepseek-v4-chat", "deepseek-chat"}:
        model = "deepseek-v4-flash"
    enabled = env_flag("DEEPSEEK_ENABLED", True) if deepseek_configured else env_flag("KIMI_ENABLED", True)
    timeout = float(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", os.getenv("KIMI_TIMEOUT_SECONDS", "60")))
    temperature = float(os.getenv("DEEPSEEK_TEMPERATURE", os.getenv("KIMI_TEMPERATURE", "0.2")))
    thinking_enabled = env_flag(
        "DEEPSEEK_THINKING_ENABLED",
        env_flag("KIMI_THINKING_ENABLED", False),
    )
    trust_env = env_flag("DEEPSEEK_TRUST_ENV", env_flag("KIMI_TRUST_ENV", False))
    proxy = os.getenv("DEEPSEEK_PROXY", "").strip() or os.getenv("KIMI_PROXY", "").strip() or None
    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "enabled": enabled,
        "timeout": timeout,
        "temperature": temperature,
        "thinking_enabled": thinking_enabled,
        "trust_env": trust_env,
        "proxy": proxy,
    }


def kimi_settings() -> dict[str, Any]:
    """Backward-compatible alias for older imports and local tests."""
    return llm_settings()


@contextmanager
def direct_connection_if_configured():
    """Temporarily bypass broken system proxies during model downloads."""
    if env_flag("WHISPER_TRUST_ENV", False):
        yield
        return
    proxy_names = (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy",
    )
    saved = {name: os.environ.pop(name, None) for name in proxy_names}
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value
