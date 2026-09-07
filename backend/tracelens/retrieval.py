"""Lexical, embedding, and local Qdrant evidence retrieval."""
from __future__ import annotations
import math
import os
import re
import socket
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
try:
    from qdrant_client import QdrantClient
except Exception:
    QdrantClient = None
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
from .config import LOCAL_QDRANT_DIR, PROJECT_ROOT, env_flag, logger
from .models import Evidence
from .storage import get_connection

_QDRANT_CLIENT: Any | None = None
_EMBEDDING_MODEL: Any | None = None
_EMBEDDING_LOAD_FAILED = False


class SearchRetriever(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        sources: list[str] | None = None,
    ) -> dict[str, Any]: ...


class EvidenceRetriever:
    """Deterministic lexical retriever used as the first candidate source.

    The reference project deliberately keeps lexical retrieval behind a small
    interface.  This gives the structured Agent a stable contract whether
    Qdrant is available or not, and makes Chinese transcript/OCR search work
    without relying on an embedding model being downloaded at runtime.
    """

    def __init__(self, segments: list[dict[str, Any]]) -> None:
        self.segments = [dict(item) for item in segments]

    @staticmethod
    def terms(value: str) -> set[str]:
        normalized = str(value).lower()
        # Keep ASCII word boundaries. Removing all whitespace turns a query
        # such as "project retrieval" into one artificial token and makes
        # English transcript retrieval silently fail.
        terms = set(re.findall(r"[a-z0-9_]{2,}", normalized))
        chinese = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
        terms.update(chinese[index:index + 2] for index in range(max(0, len(chinese) - 1)))
        return terms

    @classmethod
    def score(cls, query: str, content: str) -> dict[str, float]:
        compact_query = re.sub(r"\s+", "", str(query).lower())
        compact_content = re.sub(r"\s+", "", str(content).lower())
        if not compact_query or not compact_content:
            return {"score": 0.0, "termCoverage": 0.0, "exactBonus": 0.0}
        terms = cls.terms(query)
        matched = sum(1 for term in terms if term in compact_content)
        coverage = matched / max(1, len(terms))
        exact_bonus = 1.0 if compact_query in compact_content else 0.0
        return {
            "score": round(exact_bonus + coverage, 4),
            "termCoverage": round(coverage, 4),
            "exactBonus": exact_bonus,
        }

    @classmethod
    def relevance(cls, query: str, content: str) -> float:
        return cls.score(query, content)["score"]

    def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        sources: list[str] | None = None,
    ) -> dict[str, Any]:
        normalized = " ".join(str(query).split()).strip()
        if not normalized:
            raise ValueError("检索词不能为空")
        limit = max(1, min(int(top_k), 40))
        allowed = {str(item).upper() for item in sources or []}
        ranked = []
        for item in self.segments:
            if allowed and str(item.get("source", "")).upper() not in allowed:
                continue
            details = self.score(normalized, str(item.get("content", "")))
            ranked.append({
                **item,
                "score": details["score"],
                "scoreDetails": {
                    "termCoverage": details["termCoverage"],
                    "exactBonus": details["exactBonus"],
                },
            })
        ranked.sort(key=lambda item: (
            -float(item["score"]),
            int(item.get("startMs", 0)),
            str(item.get("segmentId", "")),
        ))
        positive = [item for item in ranked if float(item["score"]) > 0]
        return {
            "ok": True,
            "query": normalized,
            "retrievalMode": "HYBRID_LEXICAL_BASELINE",
            "matches": (positive or ranked)[:limit],
            "matchedCount": len(positive),
            "fallbackToTimelineStart": not bool(positive),
        }

def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def parse_time_hints(value: str) -> list[int]:
    """Extract explicit mm:ss, hh:mm:ss, minute, and second anchors in ms."""
    text = str(value)
    hints: list[int] = []
    occupied: list[tuple[int, int]] = []

    def add(seconds: float, span: tuple[int, int]) -> None:
        if any(span[0] < end and span[1] > start for start, end in occupied):
            return
        milliseconds = max(0, int(round(seconds * 1000)))
        if milliseconds not in hints:
            hints.append(milliseconds)
        occupied.append(span)

    for match in re.finditer(r"(?<!\d)(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?!\d)", text):
        add(int(match.group(1) or 0) * 3600 + int(match.group(2)) * 60 + int(match.group(3)), match.span())
    for match in re.finditer(r"第?\s*(\d+(?:\.\d+)?)\s*(?:分钟|分)(?:\s*(\d+(?:\.\d+)?)\s*秒)?", text):
        add(float(match.group(1)) * 60 + float(match.group(2) or 0), match.span())
    for match in re.finditer(r"第?\s*(\d+(?:\.\d+)?)\s*(?:秒|s)\b", text, flags=re.I):
        add(float(match.group(1)), match.span())
    # Also recognize natural-language anchors such as "3 minutes around"
    # and their Chinese equivalents. Modifiers are intentionally ignored.
    for match in re.finditer(
        r"(?:(?:\u7b2c|\u7ea6|\u5927\u7ea6|\u5728|around|about|at)\s*)?"
        r"(\d+(?:\.\d+)?)\s*(?:\u5206\u949f|\u5206|minutes?|mins?)"
        r"(?:\s*(?:\u5de6\u53f3|\u9644\u8fd1|around|about))?",
        text,
        flags=re.I,
    ):
        add(float(match.group(1)) * 60, match.span())
    for match in re.finditer(
        r"(?:(?:\u7b2c|\u7ea6|\u5927\u7ea6|\u5728|around|about|at)\s*)?"
        r"(\d+(?:\.\d+)?)\s*(?:\u79d2|seconds?|secs?)"
        r"(?:\s*(?:\u5de6\u53f3|\u9644\u8fd1|around|about))?",
        text,
        flags=re.I,
    ):
        add(float(match.group(1)), match.span())
    return hints[:5]


def tokenize(text: str) -> set[str]:
    normalized = text.lower()
    tokens = {token for token in re.findall(r"[a-z0-9_]+", normalized) if len(token) > 1}
    chinese = re.findall(r"[\u4e00-\u9fff]", normalized)
    tokens.update("".join(chinese[index:index + 2]) for index in range(len(chinese) - 1))
    return tokens


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def get_embedding_model() -> Any | None:
    global _EMBEDDING_MODEL, _EMBEDDING_LOAD_FAILED
    if not env_flag("EMBEDDING_ENABLED", False) or _EMBEDDING_LOAD_FAILED:
        return None
    if _EMBEDDING_MODEL is None:
        try:
            backend = os.getenv("EMBEDDING_BACKEND", "sentence-transformers").strip().lower()
            if backend in {"onnx", "bge-onnx", "local-onnx"}:
                from .embedding_retrieval import (
                    DEFAULT_ONNX_MODEL_FILE,
                    DEFAULT_ONNX_REPO,
                    DEFAULT_ONNX_REVISION,
                    LocalOnnxEmbeddingBackend,
                )
                cache_dir = Path(os.getenv(
                    "EMBEDDING_MODEL_ROOT",
                    str(PROJECT_ROOT / "data" / "models" / "embeddings"),
                )).resolve()
                _EMBEDDING_MODEL = LocalOnnxEmbeddingBackend(
                    cache_dir,
                    repo_id=os.getenv("EMBEDDING_MODEL_REPO", DEFAULT_ONNX_REPO),
                    revision=os.getenv("EMBEDDING_MODEL_REVISION", DEFAULT_ONNX_REVISION),
                    model_file=os.getenv("EMBEDDING_MODEL_FILE", DEFAULT_ONNX_MODEL_FILE),
                    allow_download=env_flag("EMBEDDING_ALLOW_DOWNLOAD", True),
                    cpu_threads=int(os.getenv("EMBEDDING_CPU_THREADS", "2")),
                    batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
                )
                return _EMBEDDING_MODEL
            if SentenceTransformer is None:
                return None
            model_name = os.getenv(
                "EMBEDDING_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            )
            cache_dir = Path(os.getenv(
                "EMBEDDING_MODEL_ROOT",
                str(PROJECT_ROOT / "data" / "models" / "embeddings"),
            )).resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)
            allow_download = env_flag("EMBEDDING_ALLOW_DOWNLOAD", True)
            try:
                import torch
                torch.set_num_threads(max(1, int(os.getenv("EMBEDDING_CPU_THREADS", "2"))))
            except Exception:
                pass
            _EMBEDDING_MODEL = SentenceTransformer(
                model_name,
                cache_folder=str(cache_dir),
                model_kwargs={"local_files_only": not allow_download},
            )
        except Exception:
            _EMBEDDING_LOAD_FAILED = True
            logger.exception("Unable to load the embedding model")
            return None
    return _EMBEDDING_MODEL


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    if model is None:
        return []
    try:
        batch_size = max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", "32")))
        backend = os.getenv("EMBEDDING_BACKEND", "sentence-transformers").strip().lower()
        if backend in {"onnx", "bge-onnx", "local-onnx"}:
            return model.encode(texts).tolist()
        return model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
    except Exception:
        logger.exception("Unable to encode embedding texts")
        return []


def is_qdrant_running() -> bool:
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def get_qdrant_client() -> Any | None:
    global _QDRANT_CLIENT
    if QdrantClient is None:
        return None
    if _QDRANT_CLIENT is not None:
        return _QDRANT_CLIENT
    try:
        if env_flag("QDRANT_IN_MEMORY", False):
            _QDRANT_CLIENT = QdrantClient(":memory:")
        elif is_qdrant_running():
            host = os.getenv("QDRANT_HOST", "localhost")
            port = int(os.getenv("QDRANT_PORT", "6333"))
            _QDRANT_CLIENT = QdrantClient(host=host, port=port, timeout=5)
        else:
            _QDRANT_CLIENT = QdrantClient(path=str(LOCAL_QDRANT_DIR))
        return _QDRANT_CLIENT
    except Exception:
        logger.exception("Unable to initialize Qdrant")
        return None


def ensure_qdrant_collection(
    collection_name: str = "video_evidence",
    vector_size: int | None = None,
) -> None:
    client = get_qdrant_client()
    if client is None:
        return
    if vector_size is None:
        model = get_embedding_model()
        if model is None:
            return
        vectors = embed_texts(["dimension probe"])
        if not vectors:
            return
        vector_size = len(vectors[0])
    try:
        collection = client.get_collection(collection_name)
        configured = getattr(collection.config.params.vectors, "size", None)
        if configured is None or int(configured) == int(vector_size):
            return
        # A model change (MiniLM 384 -> BGE 512) invalidates old vectors, but
        # the source evidence remains authoritative in MySQL and is rebuilt
        # on the next sync.
        client.delete_collection(collection_name)
    except Exception:
        pass
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={"size": vector_size, "distance": "Cosine"},
        )
    except Exception:
        pass


def sync_evidence_to_qdrant(video_id: str | None = None) -> None:
    client = get_qdrant_client()
    if client is None:
        return
    try:
        with get_connection() as connection:
            rows = connection.execute(
                "SELECT id, video_id, start_seconds, end_seconds, text, source FROM evidence "
                "WHERE (? IS NULL OR video_id = ?) ORDER BY start_seconds",
                (video_id, video_id),
            ).fetchall()
        if not rows:
            return
        texts = [row["text"] for row in rows]
        vectors = embed_texts(texts)
        if not vectors:
            return
        ensure_qdrant_collection(vector_size=len(vectors[0]))
        if video_id is not None:
            client.delete(
                collection_name="video_evidence",
                points_selector={
                    "filter": {"must": [{"key": "video_id", "match": {"value": video_id}}]}
                },
                wait=True,
            )
        points = [
            {
                "id": row["id"],
                "vector": vectors[index],
                "payload": {
                    "id": row["id"],
                    "video_id": row["video_id"],
                    "start_seconds": row["start_seconds"],
                    "end_seconds": row["end_seconds"],
                    "text": row["text"],
                    "source": row["source"],
                },
            }
            for index, row in enumerate(rows)
        ]
        client.upsert(collection_name="video_evidence", points=points, wait=True)
    except Exception:
        return


def search_qdrant(question: str, video_id: str | None, limit: int = 5) -> list[Evidence]:
    client = get_qdrant_client()
    if client is None:
        return []
    try:
        vectors = embed_texts([question])
        if not vectors:
            return []
        ensure_qdrant_collection(vector_size=len(vectors[0]))
        query_filter = None
        if video_id is not None:
            query_filter = {"must": [{"key": "video_id", "match": {"value": video_id}}]}
        hits = client.search(
            collection_name="video_evidence",
            query_vector=vectors[0],
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        results: list[Evidence] = []
        minimum_score = float(os.getenv("MIN_SEMANTIC_SCORE", "0.35"))
        for hit in hits:
            if getattr(hit, "score", 0.0) < minimum_score:
                continue
            payload = hit.payload or {}
            results.append(
                Evidence(
                    id=int(payload.get("id", hit.id)),
                    video_id=str(payload.get("video_id", video_id or "unknown")),
                    start_seconds=float(payload.get("start_seconds", 0.0)),
                    end_seconds=float(payload.get("end_seconds", 0.0)),
                    text=str(payload.get("text", "")),
                    source=str(payload.get("source", "ASR")),
                )
            )
        return results
    except Exception:
        return []


def search_evidence(question: str, video_id: str | None, limit: int = 5) -> list[Evidence]:
    question_tokens = tokenize(question)
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT id, video_id, start_seconds, end_seconds, text, source FROM evidence "
            "WHERE (? IS NULL OR video_id = ?) ORDER BY start_seconds",
            (video_id, video_id),
        ).fetchall()

    scored: list[tuple[int, Evidence]] = []
    for row in rows:
        evidence = Evidence(**dict(row))
        overlap = len(question_tokens & tokenize(evidence.text))
        if overlap:
            scored.append((overlap, evidence))
    scored.sort(key=lambda item: (-item[0], item[1].start_seconds))

    semantic_results = search_qdrant(question, video_id, limit)
    combined: list[Evidence] = []
    seen_ids: set[int] = set()
    for item in semantic_results + [evidence for _, evidence in scored[:limit]]:
        if item.id not in seen_ids:
            combined.append(item)
            seen_ids.add(item.id)
    return combined[:limit]


def search_timeline(
    question: str,
    video_id: str | None,
    limit: int = 8,
    sources: list[str] | None = None,
) -> dict[str, Any]:
    """Return stable, source-filtered evidence matches for API and Agent tools."""
    allowed = {str(source).upper() for source in (sources or [])}
    candidates = search_evidence(question, video_id, max(1, min(limit * 2, 20)))
    matches = [item for item in candidates if not allowed or item.source.upper() in allowed]
    if not allowed:
        ocr = [item for item in matches if item.source.upper() == "OCR"]
        asr = [item for item in matches if item.source.upper() != "OCR"]
        ocr_quota = min(len(ocr), max(1, (limit * 3 + 4) // 5))
        matches = (ocr[:ocr_quota] + asr)[:limit]
    else:
        matches = matches[:limit]
    return {
        "ok": True,
        "query": " ".join(str(question).split()),
        "retrievalMode": "HYBRID_LOCAL",
        "sources": sorted(allowed),
        "matchedCount": len(matches),
        "matches": [public_evidence(item) for item in matches],
    }


def evidence_window(
    video_id: str | None,
    timestamp_ms: int,
    before_ms: int = 15000,
    after_ms: int = 15000,
) -> dict[str, Any]:
    """Load continuous ASR/OCR evidence surrounding a timeline anchor."""
    if not video_id:
        return {"ok": True, "timestampMs": max(0, timestamp_ms), "segments": []}
    start_ms = max(0, int(timestamp_ms) - max(0, int(before_ms)))
    end_ms = max(start_ms, int(timestamp_ms) + max(0, int(after_ms)))
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT id, video_id, start_seconds, end_seconds, text, source FROM evidence "
            "WHERE video_id = ? AND start_seconds * 1000 <= ? AND end_seconds * 1000 >= ? "
            "ORDER BY start_seconds, id",
            (video_id, end_ms, start_ms),
        ).fetchall()
    return {
        "ok": True,
        "videoId": video_id,
        "timestampMs": int(timestamp_ms),
        "startMs": start_ms,
        "endMs": end_ms,
        "segments": [
            {
                "evidenceId": item["id"],
                "source": item["source"],
                "startMs": round(item["start_seconds"] * 1000),
                "endMs": round(item["end_seconds"] * 1000),
                "content": item["text"],
            }
            for item in rows
        ],
    }


def plan_evidence_requirements(query: str) -> dict[str, Any]:
    """Create a deterministic evidence-slot plan before model verification.

    The reference project uses this boundary to separate question
    decomposition from retrieval. The local implementation intentionally
    stays provider-independent and supplies conservative query slots.
    """
    normalized = " ".join(str(query).split()).strip()
    if not normalized:
        return {"strategy": "SINGLE_QUERY", "requirements": []}

    parts = [
        item.strip(" ，,、；;。！？!?")
        for item in re.split(r"\s*(?:以及|并且|同时|分别|和|与|及|以及|以及|以及|以及|以及|以及|and|also)\s*", normalized, flags=re.I)
        if item.strip(" ，,、；;。！？!?")
    ]
    if not parts:
        parts = [normalized]
    if len(parts) > 4:
        parts = parts[:4]

    strategy = "SINGLE_QUERY"
    if len(parts) > 1:
        strategy = "MULTI_REQUIREMENT"
    if re.search(r"比较|区别|不同|对比|versus|vs\.?", normalized, flags=re.I):
        strategy = "COMPARISON_DECOMPOSITION"
    return {
        "strategy": strategy,
        "requirements": [{"query": part} for part in parts],
    }

def public_evidence(item: Evidence) -> dict[str, Any]:
    """Return the small, stable evidence shape exposed to the model."""
    return {
        "evidence_id": item.id,
        "start_ms": round(item.start_seconds * 1000),
        "end_ms": round(item.end_seconds * 1000),
        "timestamp": f"{format_timestamp(item.start_seconds)} - {format_timestamp(item.end_seconds)}",
        "text": item.text,
        "source": item.source,
    }


def load_video_evidence(video_id: str | None) -> list[Evidence]:
    if not video_id:
        return []
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT id, video_id, start_seconds, end_seconds, text, source FROM evidence "
            "WHERE video_id = ? ORDER BY start_seconds, id",
            (video_id,),
        ).fetchall()
    return [Evidence(**dict(row)) for row in rows]


# The following wrappers are intentionally kept in this module so the
# runtime Agent and the HTTP evidence API use the same retrieval semantics.
# They mirror the reference project's contextual and coverage-aware layers.


def _segment_key(item: dict[str, Any]) -> str:
    return str(item.get("segmentId") or ":".join((
        str(item.get("source", "")),
        str(item.get("startMs", 0)),
        str(item.get("endMs", 0)),
    )))


def _interval_distance(timestamp_ms: int, item: dict[str, Any]) -> int:
    start = int(item.get("startMs", 0))
    end = max(start, int(item.get("endMs", start)))
    if start <= timestamp_ms <= end:
        return 0
    return min(abs(timestamp_ms - start), abs(timestamp_ms - end))


def _parse_number(value: str) -> float:
    return float({"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
                  "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}.get(value, value))


def parse_time_hints(value: str) -> list[int]:
    """Parse explicit and approximate time expressions into milliseconds."""
    text = str(value)
    hints: list[int] = []
    occupied: list[tuple[int, int]] = []

    def add(seconds: float, span: tuple[int, int]) -> None:
        if any(span[0] < end and span[1] > start for start, end in occupied):
            return
        timestamp = max(0, int(round(seconds * 1000)))
        if timestamp not in hints:
            hints.append(timestamp)
        occupied.append(span)

    for match in re.finditer(r"(?<!\d)(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?!\d)", text):
        add(int(match.group(1) or 0) * 3600 + int(match.group(2)) * 60 + int(match.group(3)), match.span())
    number = r"(?:\d+(?:\.\d+)?|[一二两三四五六七八九十])"
    for match in re.finditer(
        rf"(?:第|约|大约|在|around|about|at)?\s*({number})\s*(?:分钟|分|minutes?|mins?)"
        rf"(?:\s*({number})\s*秒)?\s*(?:左右|附近|around|about)?",
        text, flags=re.I,
    ):
        add(_parse_number(match.group(1)) * 60 + (_parse_number(match.group(2)) if match.group(2) else 0), match.span())
    for match in re.finditer(
        rf"(?:第|约|大约|在|around|about|at)?\s*({number})\s*(?:秒|seconds?|secs?)\s*(?:左右|附近|around|about)?",
        text, flags=re.I,
    ):
        add(_parse_number(match.group(1)), match.span())
    return hints[:8]


def _question_clauses(value: str) -> list[str]:
    parts = re.split(r"[？?；;]+", str(value))
    return [re.sub(r"^[，,。.!！\s]+|[，,。.!！\s]+$", "", part).strip()
            for part in parts if len(part.strip()) >= 4]


def plan_evidence_requirements(query: str) -> dict[str, Any]:
    """Deterministically split multi-part, comparison, and list questions."""
    normalized = " ".join(str(query).split()).strip()
    if not normalized:
        return {"query": "", "strategy": "SINGLE_REQUIREMENT", "requirements": []}
    clauses = _question_clauses(normalized)
    if len(clauses) == 1:
        # Chinese users often join independent asks with conjunctions rather
        # than a question mark. Keep each clause as its own coverage slot.
        joined = re.split(r"\s*(?:以及|并且|同时|另外|还想知道|and|also)\s*", normalized, flags=re.I)
        if len(joined) > 1 and all(len(item.strip()) >= 4 for item in joined):
            clauses = [item.strip(" ，,、；;。！？!?") for item in joined]
    if len(clauses) > 1:
        requirements = [{"requirementId": f"requirement-{i}", "query": clause,
                         "kind": "CLAUSE", "markers": []}
                        for i, clause in enumerate(clauses, 1)]
        strategy = "CLAUSE_DECOMPOSITION"
    else:
        comparison = re.search(
            r"(?P<left>[A-Za-z][\w.+-]*)\s*(?:和|与|、|vs\.?|versus)\s*"
            r"(?P<right>[A-Za-z][\w.+-]*)\s*分别(?P<tail>.+)", normalized, re.I,
        )
        if comparison:
            tail = comparison.group("tail").strip("？?。 ")
            requirements = [
                {"requirementId": "requirement-1", "query": f"{comparison.group('left')} {tail}", "kind": "COMPARISON_SIDE", "markers": []},
                {"requirementId": "requirement-2", "query": f"{comparison.group('right')} {tail}", "kind": "COMPARISON_SIDE", "markers": []},
            ]
            strategy = "COMPARISON_DECOMPOSITION"
        else:
            enumeration = re.search(
                r"(?P<subject>.+?)(?:有|包含|包括)哪\s*"
                r"(?P<count>[二两三四五六七八九2-9])\s*(?:个|种|项|条|类)"
                r"(?P<label>[^？?。]{1,30})",
                normalized,
            ) or re.search(
                r"(?P<subject>.+?)哪\s*"
                r"(?P<count>[二两三四五六七八九2-9])\s*(?:个|种|项|条|类)"
                r"(?P<label>[^？?。]{1,30})",
                normalized,
            )
            if enumeration:
                number_map = {"二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
                              "六": 6, "七": 7, "八": 8, "九": 9}
                count = int(enumeration.group("count")) if enumeration.group("count").isdigit() else number_map[enumeration.group("count")]
                subject = enumeration.group("subject").strip(" ，,、：:")
                label = enumeration.group("label").strip(" ，,、：:？?")
                requirements = [
                    {
                        "requirementId": f"requirement-{index}",
                        "query": f"{subject} {label} 第{index}项",
                        "kind": "ENUMERATED_ITEM",
                        "markers": [f"第{index}项", f"第{index}个", f"第{index}种"],
                    }
                    for index in range(1, min(count, 6) + 1)
                ]
                strategy = "ENUMERATION_DECOMPOSITION"
            else:
                requirements = [{"requirementId": "requirement-1", "query": normalized,
                                 "kind": "PRIMARY", "markers": []}]
                strategy = "SINGLE_REQUIREMENT"
    return {"query": normalized, "strategy": strategy,
            "requirementCount": len(requirements), "requirements": requirements}


def required_query_anchors(query: str) -> list[dict[str, str]]:
    anchors = []
    seen: set[str] = set()
    for value in re.findall(r"[A-Za-z][A-Za-z0-9_.+-]{1,}", str(query)):
        normalized = re.sub(r"[^a-z0-9]+", "", value.lower())
        if normalized and normalized not in {"video", "agent", "asr", "ocr"} and normalized not in seen:
            seen.add(normalized)
            anchors.append({"text": value, "normalized": normalized, "kind": "ASCII_ENTITY"})
    return anchors


class ContextualEvidenceRetriever:
    """Adds time-anchor and neighboring-context candidates to base search."""

    _VISUAL_HINT = re.compile(r"画面|屏幕|图中|图片|字幕|显示|截图|文字")
    _CONTEXT_HINT = re.compile(r"这段|这个|这些|这几个|当时|之后|之前|随后|接下来|分别|哪些|顺序")

    def __init__(self, segments: list[dict[str, Any]], base: SearchRetriever,
                 *, candidate_depth: int = 24, time_window_ms: int = 30000,
                 neighbor_window_ms: int = 20000, max_neighbors: int = 4) -> None:
        self.segments = [dict(item) for item in segments]
        self.base = base
        self.candidate_depth = max(8, min(int(candidate_depth), 80))
        self.time_window_ms = max(1000, int(time_window_ms))
        self.neighbor_window_ms = max(1000, int(neighbor_window_ms))
        self.max_neighbors = max(0, min(int(max_neighbors), 8))

    def _neighbors(self, anchor: dict[str, Any], allowed: set[str]) -> list[dict[str, Any]]:
        anchor_start = int(anchor.get("startMs", 0))
        anchor_end = max(anchor_start, int(anchor.get("endMs", anchor_start)))
        anchor_source = str(anchor.get("source", "")).upper()
        candidates = []
        for item in self.segments:
            if _segment_key(item) == _segment_key(anchor) or (allowed and str(item.get("source", "")).upper() not in allowed):
                continue
            start = int(item.get("startMs", 0)); end = max(start, int(item.get("endMs", start)))
            gap = max(0, start - anchor_end, anchor_start - end)
            if gap <= self.neighbor_window_ms:
                candidates.append((int(str(item.get("source", "")).upper() == anchor_source), gap, start, _segment_key(item), dict(item)))
        candidates.sort(key=lambda item: item[:4])
        return [item[4] for item in candidates[:self.max_neighbors]]

    def search(self, query: str, *, top_k: int = 8, sources: list[str] | None = None) -> dict[str, Any]:
        normalized = " ".join(str(query).split()).strip()
        limit = max(1, min(int(top_k), 30))
        allowed = {str(item).upper() for item in sources or []}
        base = self.base.search(normalized, top_k=max(limit, self.candidate_depth), sources=sources)
        base_matches = [dict(item) for item in base.get("matches", [])]
        hints = parse_time_hints(normalized)
        ranked: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add(items: list[dict[str, Any]], reason: str) -> None:
            for item in items:
                key = _segment_key(item)
                if key in seen:
                    continue
                enriched = dict(item)
                enriched["selectionReason"] = reason
                ranked.append(enriched); seen.add(key)

        if hints:
            time_matches = []
            prefer_ocr = bool(self._VISUAL_HINT.search(normalized))
            for item in self.segments:
                source = str(item.get("source", "")).upper()
                if allowed and source not in allowed:
                    continue
                distance = min((_interval_distance(hint, item) for hint in hints), default=self.time_window_ms + 1)
                if distance <= self.time_window_ms:
                    candidate = dict(item)
                    candidate["scoreDetails"] = {**dict(candidate.get("scoreDetails") or {}), "timeDistanceMs": distance}
                    time_matches.append((0 if prefer_ocr and source == "OCR" else 1, distance, int(item.get("startMs", 0)), _segment_key(item), candidate))
            time_matches.sort(key=lambda item: item[:4])
            add([item[4] for item in time_matches], "TIME_ANCHOR")
            add(base_matches, "BASE_RANK")
        elif base_matches and self._CONTEXT_HINT.search(normalized):
            add(base_matches[:1], "BASE_ANCHOR")
            add(self._neighbors(base_matches[0], allowed), "ADJACENT_CONTEXT")
            add(base_matches[1:], "BASE_RANK")
        else:
            add(base_matches, "BASE_RANK")
        return {**base, "retrievalMode": "CONTEXTUAL_" + str(base.get("retrievalMode", "UNKNOWN")),
                "matches": ranked[:limit], "matchedCount": len(ranked),
                "timeHintsMs": hints, "fallbackToTimelineStart": False}


class CoverageAwareEvidenceRetriever:
    """Retrieve independently for every requirement and preserve coverage metadata."""

    def __init__(self, segments: list[dict[str, Any]], base: SearchRetriever,
                 *, candidate_depth: int = 24, context_window_ms: int = 20000,
                 context_per_requirement: int = 2, enable_anchor_gate: bool = False) -> None:
        self.segments = [dict(item) for item in segments]
        self.base = base
        self.candidate_depth = max(8, min(int(candidate_depth), 80))
        self.context_window_ms = max(1000, int(context_window_ms))
        self.context_per_requirement = max(0, min(int(context_per_requirement), 4))
        self.enable_anchor_gate = bool(enable_anchor_gate)

    def _neighbors(self, anchor: dict[str, Any], query: str, allowed: set[str]) -> list[dict[str, Any]]:
        contextual = ContextualEvidenceRetriever(self.segments, self.base,
                                                  neighbor_window_ms=self.context_window_ms,
                                                  max_neighbors=max(4, self.context_per_requirement * 3))
        return contextual._neighbors(anchor, allowed)

    @staticmethod
    def _status(requirement: dict[str, Any], matches: list[dict[str, Any]]) -> dict[str, Any]:
        query = str(requirement.get("query", ""))
        coverage = max((EvidenceRetriever.score(query, item.get("content", "")).get("termCoverage", 0.0)
                        for item in matches[:8]), default=0.0)
        positive = [item for item in matches if float(item.get("score", 0.0)) > 0]
        # The lexical retriever may return a timeline fallback to give the
        # model context. A fallback row is not evidence that answers a slot.
        satisfied = bool(positive)
        return {**requirement, "candidateCount": len(matches), "positiveCandidateCount": len(positive),
                "maxTermCoverage": round(coverage, 4), "satisfied": satisfied,
                "status": "SATISFIED" if satisfied else "MISSING_COVERAGE"}

    def search(self, query: str, *, top_k: int = 8, sources: list[str] | None = None) -> dict[str, Any]:
        normalized = " ".join(str(query).split()).strip()
        limit = max(1, min(int(top_k), 30))
        plan = plan_evidence_requirements(normalized)
        requirements = plan.get("requirements", []) or [{"requirementId": "requirement-1", "query": normalized, "kind": "PRIMARY", "markers": []}]
        allowed = {str(item).upper() for item in sources or []}
        all_ranked: list[dict[str, Any]] = []
        by_key: dict[str, dict[str, Any]] = {}
        statuses = []

        def add(item: dict[str, Any], requirement_id: str, rank: int, reason: str) -> None:
            key = _segment_key(item)
            target = by_key.setdefault(key, dict(item))
            if target is not item and key not in {_segment_key(x) for x in all_ranked}:
                all_ranked.append(target)
            elif target is item:
                all_ranked.append(target)
            ids = target.setdefault("coverageRequirementIds", [])
            if requirement_id not in ids: ids.append(requirement_id)
            ranks = target.setdefault("coverageRequirementRanks", {})
            ranks[requirement_id] = min(rank, int(ranks.get(requirement_id, rank)))
            reasons = target.setdefault("coverageSelectionReasons", [])
            if reason not in reasons: reasons.append(reason)

        primary = self.base.search(normalized, top_k=self.candidate_depth, sources=sources)
        for requirement in requirements:
            requirement_id = str(requirement.get("requirementId") or "requirement-1")
            result = self.base.search(str(requirement.get("query") or normalized), top_k=self.candidate_depth, sources=sources)
            matches = [dict(item) for item in result.get("matches", [])]
            # Independent source searches prevent dense ASR rows from filling
            # all slots before OCR has a chance to contribute.
            if not allowed or allowed == {"ASR", "OCR"}:
                source_matches = []
                for source in ("OCR", "ASR"):
                    source_matches.extend(self.base.search(str(requirement.get("query") or normalized), top_k=self.candidate_depth, sources=[source]).get("matches", []))
                unique = {_segment_key(item): dict(item) for item in [*source_matches, *matches]}
                matches = list(unique.values())
            statuses.append(self._status(requirement, matches))
            for rank, item in enumerate(matches[:8], 1):
                add(item, requirement_id, rank, "REQUIREMENT_PRIMARY" if rank == 1 else "REQUIREMENT_RANK")
            if matches:
                for item in self._neighbors(matches[0], str(requirement.get("query") or normalized), allowed)[:self.context_per_requirement]:
                    add(item, requirement_id, 2, "REQUIREMENT_CONTEXT")

        for rank, item in enumerate(primary.get("matches", []), 1):
            add(dict(item), "primary-query", rank, "PRIMARY_RANK")
        satisfied = sum(bool(item["satisfied"]) for item in statuses)
        # Sort by source-aware relevance first, but retain OCR in the visible
        # result when it exists. The ledger itself still contains all slots.
        hints = parse_time_hints(normalized)
        ranked = sorted(all_ranked, key=lambda item: (
            min((_interval_distance(hint, item) for hint in hints), default=10**12),
            0 if str(item.get("source", "")).upper() == "OCR" else 1,
            -float(item.get("score", 0.0)), int(item.get("startMs", 0)), _segment_key(item)))
        return {**primary, "retrievalMode": "COVERAGE_AWARE_" + str(primary.get("retrievalMode", "UNKNOWN")),
                "matches": ranked[:limit], "matchedCount": len(ranked), "fallbackToTimelineStart": False,
                "coveragePlan": plan, "evidenceSufficiency": {
                    "decision": "SUFFICIENT_CANDIDATES" if satisfied == len(statuses) else "PARTIAL_EVIDENCE",
                    "fullyCovered": bool(statuses) and satisfied == len(statuses),
                    "requirementCount": len(statuses), "satisfiedRequirementCount": satisfied,
                    "requirements": statuses,
                }}


@dataclass(frozen=True)
class RetrieverProfile:
    profile_id: str
    retrieval_mode: str
    factory: Callable[[list[dict[str, Any]]], SearchRetriever]
    description: str
    settings: dict[str, Any] = field(default_factory=dict)

    def create(self, segments: list[dict[str, Any]]) -> SearchRetriever:
        return self.factory([dict(item) for item in segments])

    def metadata(self) -> dict[str, Any]:
        return {"profileId": self.profile_id, "retrievalMode": self.retrieval_mode,
                "description": self.description, **self.settings}
