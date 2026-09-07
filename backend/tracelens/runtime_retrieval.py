"""Build the reference-style retrieval stack for one persisted video."""

from __future__ import annotations

import os
from typing import Any

from .models import Evidence
from .retrieval import (
    CoverageAwareEvidenceRetriever,
    EvidenceRetriever,
    ContextualEvidenceRetriever,
    get_embedding_model,
    get_qdrant_client,
    load_video_evidence,
    search_qdrant,
)


def _segment(item: Evidence) -> dict[str, Any]:
    return {
        "evidenceId": item.id,
        "segmentId": f"evidence-{item.id}",
        "startMs": round(item.start_seconds * 1000),
        "endMs": round(item.end_seconds * 1000),
        "content": item.text,
        "source": str(item.source or "ASR").upper(),
    }


class QdrantHybridEvidenceRetriever:
    """Fuse persisted Qdrant hits and lexical hits with reciprocal ranks.

    Qdrant stores vectors across API and Worker processes, while lexical
    search preserves reliable exact matching for transcript/OCR terminology.
    This small adapter deliberately degrades to lexical search when either
    the local model or Qdrant is unavailable.
    """

    def __init__(self, video_id: str | None, segments: list[dict[str, Any]]) -> None:
        self.video_id = video_id
        self.segments = [dict(item) for item in segments]
        self.lexical = EvidenceRetriever(self.segments)
        self.by_id = {
            int(item["evidenceId"]): item
            for item in self.segments
            if str(item.get("evidenceId", "")).isdigit()
        }

    def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        sources: list[str] | None = None,
    ) -> dict[str, Any]:
        limit = max(1, min(int(top_k), 40))
        lexical = self.lexical.search(query, top_k=max(limit, 24), sources=sources)
        if get_embedding_model() is None or get_qdrant_client() is None:
            return lexical
        semantic = search_qdrant(query, self.video_id, limit=max(limit, 24))
        allowed = {str(item).upper() for item in sources or []}
        semantic_matches = [
            dict(self.by_id[item.id])
            for item in semantic
            if item.id in self.by_id
            and (not allowed or str(self.by_id[item.id].get("source", "")).upper() in allowed)
        ]
        if not semantic_matches:
            return lexical

        candidates: dict[str, dict[str, Any]] = {}

        def add(items: list[dict[str, Any]], component: str) -> None:
            for rank, item in enumerate(items, start=1):
                key = str(item.get("segmentId"))
                candidate = candidates.setdefault(key, {
                    **item,
                    "score": 0.0,
                    "scoreDetails": {
                        "lexicalRank": None,
                        "denseRank": None,
                        "lexicalScore": None,
                        "denseScore": None,
                    },
                })
                candidate["score"] += 0.5 / (60 + rank)
                candidate["scoreDetails"][f"{component}Rank"] = rank
                candidate["scoreDetails"][f"{component}Score"] = item.get("score")

        lexical_matches = [] if lexical.get("fallbackToTimelineStart") else lexical.get("matches", [])
        add([dict(item) for item in lexical_matches], "lexical")
        add(semantic_matches, "dense")
        ranked = sorted(
            candidates.values(),
            key=lambda item: (
                -float(item["score"]),
                int(item.get("startMs", 0)),
                str(item.get("segmentId", "")),
            ),
        )
        for item in ranked:
            item["score"] = round(float(item["score"]), 8)
        return {
            "ok": True,
            "query": " ".join(str(query).split()),
            "retrievalMode": "QDRANT_HYBRID_LEXICAL_DENSE_RRF",
            "matches": ranked[:limit],
            "matchedCount": len(ranked),
            "fallbackToTimelineStart": False,
        }


def build_runtime_retriever(video_id: str | None) -> CoverageAwareEvidenceRetriever:
    """Return Coverage -> Context -> Qdrant hybrid, with lexical fallback."""
    segments = [_segment(item) for item in load_video_evidence(video_id)]
    lexical = EvidenceRetriever(segments)
    profile = os.getenv("EVIDENCE_RETRIEVER_PROFILE", "coverage-aware-qdrant-hybrid-v2").strip().lower()
    base: Any = lexical
    if profile in {
        "qdrant", "qdrant-hybrid", "contextual-qdrant-hybrid-v1",
        "coverage-qdrant-hybrid-v2", "coverage-aware-qdrant-hybrid-v2",
    }:
        base = QdrantHybridEvidenceRetriever(video_id, segments)
    contextual = ContextualEvidenceRetriever(segments, base)
    return CoverageAwareEvidenceRetriever(
        segments,
        contextual,
        candidate_depth=24,
        context_window_ms=20000,
        context_per_requirement=2,
        # Missing-anchor refusal belongs to the Verifier: ASR/OCR can contain
        # legitimate transcription variants of the user's wording.
        enable_anchor_gate=False,
    )


__all__ = ["QdrantHybridEvidenceRetriever", "build_runtime_retriever"]
