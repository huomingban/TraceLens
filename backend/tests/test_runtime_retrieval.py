from __future__ import annotations

import tracelens.runtime_retrieval as runtime
from tracelens.models import Evidence


def _segments() -> list[dict[str, object]]:
    return [
        {
            "evidenceId": 1,
            "segmentId": "evidence-1",
            "startMs": 0,
            "endMs": 4000,
            "content": "The project uses exact lexical retrieval.",
            "source": "ASR",
        },
        {
            "evidenceId": 2,
            "segmentId": "evidence-2",
            "startMs": 5000,
            "endMs": 9000,
            "content": "Semantic vectors are persisted in Qdrant.",
            "source": "OCR",
        },
    ]


def test_qdrant_hybrid_fuses_semantic_and_lexical_ranks(monkeypatch) -> None:
    retriever = runtime.QdrantHybridEvidenceRetriever("demo", _segments())
    monkeypatch.setattr(runtime, "get_embedding_model", lambda: object())
    monkeypatch.setattr(runtime, "get_qdrant_client", lambda: object())
    monkeypatch.setattr(runtime, "search_qdrant", lambda *args, **kwargs: [
        Evidence(2, "demo", 5, 9, "Semantic vectors are persisted in Qdrant.", "OCR"),
    ])

    result = retriever.search("project retrieval", top_k=4)

    assert result["retrievalMode"] == "QDRANT_HYBRID_LEXICAL_DENSE_RRF"
    assert {item["evidenceId"] for item in result["matches"]} == {1, 2}
    dense_match = next(item for item in result["matches"] if item["evidenceId"] == 2)
    assert dense_match["scoreDetails"]["denseRank"] == 1


def test_qdrant_hybrid_falls_back_to_lexical_when_embedding_is_unavailable(monkeypatch) -> None:
    retriever = runtime.QdrantHybridEvidenceRetriever("demo", _segments())
    monkeypatch.setattr(runtime, "get_embedding_model", lambda: None)

    result = retriever.search("exact lexical", top_k=4)

    assert result["retrievalMode"] == "HYBRID_LEXICAL_BASELINE"
    assert result["matches"][0]["evidenceId"] == 1
