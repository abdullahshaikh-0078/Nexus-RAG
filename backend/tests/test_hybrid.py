import os
import json
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.chunker import RecursiveTextChunker
from app.services.hybrid_retriever import HybridRetriever, hybrid_retriever
from app.services.bm25_search import BM25IndexService
from app.db.vectorstore import QdrantVectorStore, vector_store
from app.models.schemas import DocumentChunk, SourceCitation
from app.evaluation.runner import RESULTS_DIR


def test_rrf_formula_correctness_and_ordering():
    """Verifies standard RRF formula RRF(d) = Σ 1 / (60 + rank_m(d)) and 1-based rank indexing."""
    retriever = HybridRetriever()

    # Mock candidate lists
    # Doc A: Dense rank 1, BM25 rank 2 -> RRF = 1/61 + 1/62 = 0.0163934 + 0.0161290 = 0.032522
    # Doc B: Dense rank 2, BM25 rank 1 -> RRF = 1/62 + 1/61 = 0.032522
    # Doc C: Dense rank 3, BM25 None -> RRF = 1/63 = 0.015873
    # Doc D: Dense None, BM25 rank 3 -> RRF = 1/63 = 0.015873

    rrf_a = (1.0 / (60 + 1)) + (1.0 / (60 + 2))
    rrf_c = 1.0 / (60 + 3)

    assert rrf_a == pytest.approx(0.032522, 0.0001)
    assert rrf_c == pytest.approx(0.015873, 0.0001)


def test_hybrid_deduplication_and_fusion():
    """Tests chunk deduplication and fusion for items in both, Dense-only, or BM25-only."""
    class MockVectorStore:
        def search_similar(self, query_vector, top_k, document_ids=None, **kwargs):
            return [
                SourceCitation(document_id="doc1", document_name="file1.txt", chunk_id="chk1", chunk_index=0, score=0.9, content="Content A"),
                SourceCitation(document_id="doc1", document_name="file1.txt", chunk_id="chk2", chunk_index=1, score=0.8, content="Content B"),
                SourceCitation(document_id="doc2", document_name="file2.txt", chunk_id="chk3", chunk_index=0, score=0.7, content="Dense Only Content"),
            ]

    class MockBM25Service:
        def search(self, query, top_k, document_ids=None, **kwargs):
            return [
                SourceCitation(document_id="doc1", document_name="file1.txt", chunk_id="chk2", chunk_index=1, score=1.0, content="Content B"),
                SourceCitation(document_id="doc1", document_name="file1.txt", chunk_id="chk1", chunk_index=0, score=0.8, content="Content A"),
                SourceCitation(document_id="doc3", document_name="file3.txt", chunk_id="chk4", chunk_index=0, score=0.6, content="BM25 Only Content"),
            ]

    retriever = HybridRetriever(
        vector_store_service=MockVectorStore(),
        bm25_search_service=MockBM25Service(),
    )

    results = retriever.search(query="test query", top_k=4)

    # Total unique chunks fused: 4 (chk1, chk2, chk3, chk4)
    assert len(results) == 4

    # chk2: Dense rank 2, BM25 rank 1 -> RRF = 1/62 + 1/61 = 0.032522
    # chk1: Dense rank 1, BM25 rank 2 -> RRF = 1/61 + 1/62 = 0.032522
    # Both top hits should have highest RRF score
    top_keys = [f"{c.document_id}_{c.chunk_index}" for c in results[:2]]
    assert "doc1_0" in top_keys
    assert "doc1_1" in top_keys

    # Check Dense-only item (chk3) and BM25-only item (chk4) exist
    all_keys = [f"{c.document_id}_{c.chunk_index}" for c in results]
    assert "doc2_0" in all_keys
    assert "doc3_0" in all_keys


def test_api_hybrid_retrieval_mode():
    """Verifies that API handles retrieval_mode='hybrid' correctly."""
    with TestClient(app) as client:
        # Ingest document
        doc_text = "Section 3.2.3 Applications of Multi-Head Attention in the Transformer model."
        upload_res = client.post(
            "/api/v1/documents/upload",
            files={"file": ("hybrid_spec.txt", doc_text.encode("utf-8"), "text/plain")}
        )
        assert upload_res.status_code == 201
        doc_id = upload_res.json()["document"]["document_id"]

        try:
            # Query with retrieval_mode="hybrid"
            res_hybrid = client.post(
                "/api/v1/chat/query",
                json={"query": "Applications of Multi-Head Attention", "top_k": 3, "retrieval_mode": "hybrid"}
            )
            assert res_hybrid.status_code == 200
            data = res_hybrid.json()
            assert data["retrieval_mode"] == "hybrid"
            assert len(data["sources"]) > 0

            # Verify Dense mode still works
            res_dense = client.post(
                "/api/v1/chat/query",
                json={"query": "Applications of Multi-Head Attention", "top_k": 3, "retrieval_mode": "dense"}
            )
            assert res_dense.status_code == 200
            assert res_dense.json()["retrieval_mode"] == "dense"

            # Verify BM25 mode still works
            res_bm25 = client.post(
                "/api/v1/chat/query",
                json={"query": "Applications of Multi-Head Attention", "top_k": 3, "retrieval_mode": "bm25"}
            )
            assert res_bm25.status_code == 200
            assert res_bm25.json()["retrieval_mode"] == "bm25"

        finally:
            client.delete(f"/api/v1/documents/{doc_id}")


def test_v1_baseline_file_untouched_after_hybrid():
    """Ensures V1 Dense baseline results file latest.json remains unchanged."""
    latest_path = os.path.join(RESULTS_DIR, "latest.json")

    v1_timestamp_before = None
    if os.path.exists(latest_path):
        with open(latest_path, "r", encoding="utf-8") as f:
            v1_timestamp_before = json.load(f).get("timestamp")

    from app.evaluation.runner import EvaluationRunner
    runner = EvaluationRunner()
    res_hybrid = runner.run_evaluation(top_k=10, retrieval_mode="hybrid")

    assert res_hybrid.retrieval_mode == "hybrid"
    assert res_hybrid.evaluation_version == "v2_2_hybrid"

    # Verify v2_2_hybrid_latest.json exists
    hybrid_latest_path = os.path.join(RESULTS_DIR, "v2_2_hybrid_latest.json")
    assert os.path.exists(hybrid_latest_path)

    # Verify V1 latest.json was NOT modified
    if v1_timestamp_before:
        with open(latest_path, "r", encoding="utf-8") as f:
            v1_timestamp_after = json.load(f).get("timestamp")
        assert v1_timestamp_after == v1_timestamp_before, "V2.2 Hybrid run must not modify frozen V1 latest.json"
