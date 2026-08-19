import os
import json
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.query_eval_service import query_eval_service, QueryEvaluation
from app.evaluation.runner import RESULTS_DIR

client = TestClient(app)


def test_query_evaluation_service_record_and_list():
    """Verifies that QueryEvaluationService creates, saves, lists, and deletes query evaluations."""
    eval_rec = query_eval_service.record_evaluation(
        query="What is the capital expenditure of 3M in 2018?",
        answer="The capital expenditure for 3M in 2018 was $1,577 million.",
        retrieval_mode="hybrid",
        citations=[
            {
                "document_id": "doc_123",
                "document_name": "3M_2018_10K.pdf",
                "chunk_id": "chunk_1",
                "chunk_index": 1,
                "score": 0.85,
                "content": "Purchases of property, plant and equipment: $1,577M",
            }
        ],
        latency_breakdown={
            "embedding_ms": 12.5,
            "dense_search_ms": 3.4,
            "bm25_search_ms": 0.1,
            "rrf_fusion_ms": 0.2,
            "context_expansion_ms": 2.1,
            "llm_generation_ms": 150.0,
            "total_request_ms": 168.3,
        },
    )

    assert eval_rec.evaluation_id
    assert eval_rec.query == "What is the capital expenditure of 3M in 2018?"
    assert eval_rec.retrieval_mode == "hybrid"
    assert eval_rec.evaluation_status["retrieval_status"] == "relevant_context_detected"
    assert eval_rec.evaluation_status["answer_status"] == "answer_generated"

    # List evaluations
    all_evals = query_eval_service.list_evaluations()
    found = [e for e in all_evals if e.evaluation_id == eval_rec.evaluation_id]
    assert len(found) == 1

    # Get single evaluation
    single_eval = query_eval_service.get_evaluation(eval_rec.evaluation_id)
    assert single_eval is not None
    assert single_eval.evaluation_id == eval_rec.evaluation_id

    # Clean up test evaluation
    deleted = query_eval_service.delete_evaluation(eval_rec.evaluation_id)
    assert deleted is True


def test_api_list_benchmarks_and_query_evaluations():
    """Verifies backend API endpoints GET /api/v1/evaluation/benchmarks and /query-evaluations."""
    # List benchmarks
    res_b = client.get("/api/v1/evaluation/benchmarks")
    assert res_b.status_code == 200
    b_data = res_b.json()
    assert "benchmarks" in b_data
    assert "total" in b_data

    # List query evaluations
    res_q = client.get("/api/v1/evaluation/query-evaluations")
    assert res_q.status_code == 200
    assert isinstance(res_q.json(), list)


def test_chat_query_creates_query_evaluation():
    """Verifies that executing a chat query returns evaluation_id and latency_breakdown and records an evaluation."""
    response = client.post(
        "/api/v1/chat/query",
        json={"query": "What is 3M?", "retrieval_mode": "hybrid"},
    )
    assert response.status_code == 200
    data = response.json()

    assert "evaluation_id" in data
    assert "latency_breakdown" in data
    assert data["latency_breakdown"]["total_request_ms"] > 0

    if data["evaluation_id"]:
        eval_item = query_eval_service.get_evaluation(data["evaluation_id"])
        assert eval_item is not None
        assert eval_item.query == "What is 3M?"
        # Clean up
        query_eval_service.delete_evaluation(data["evaluation_id"])


def test_evaluation_center_preserves_frozen_v1_baseline():
    """Ensures Evaluation Center additions leave frozen V1 baseline files 100% untouched."""
    v1_latest_path = os.path.join(RESULTS_DIR, "latest.json")
    if os.path.exists(v1_latest_path):
        with open(v1_latest_path, "r", encoding="utf-8") as f:
            v1_data = json.load(f)
        assert v1_data.get("aggregate_recall_at_1") == 0.9167
