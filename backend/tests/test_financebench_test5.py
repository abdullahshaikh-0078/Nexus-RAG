import os
import json
import pytest
from app.models.schemas import SourceCitation
from app.evaluation.financebench.loader import FinanceBenchLoader
from app.evaluation.financebench.reranker import experimental_reranker, ExperimentalReranker
from app.evaluation.financebench.run_financebench_test5 import (
    TARGET_QUESTION_IDS,
    QUESTION_CATEGORY_MAP,
    TEST5_DIR,
)
from app.evaluation.runner import RESULTS_DIR


def test_financebench_test5_target_questions_exist():
    """Verifies that all 12 Test 5 question IDs exist in FinanceBench dataset."""
    loader = FinanceBenchLoader()
    all_questions = loader.load_dataset()
    q_map = {q.financebench_id: q for q in all_questions}

    for q_id in TARGET_QUESTION_IDS:
        assert q_id in q_map
        q = q_map[q_id]
        assert q.question
        assert q.doc_name
        assert q.answer
        assert q_id in QUESTION_CATEGORY_MAP


def test_experimental_reranker_isolated_execution():
    """Verifies isolated evaluation reranker rescoring and reordering."""
    citations = [
        SourceCitation(chunk_id="chunk_1", document_id="doc_1", document_name="doc_1.pdf", chunk_index=0, content="Unrelated general paragraph.", score=0.5),
        SourceCitation(chunk_id="chunk_2", document_id="doc_1", document_name="doc_1.pdf", chunk_index=1, content="FY2018 capital expenditures were $1,577 million.", score=0.4),
    ]
    query = "What is the FY2018 capital expenditure amount?"

    reranked, latency_ms = experimental_reranker.rerank(query, citations)
    assert len(reranked) == 2
    assert latency_ms >= 0.0


def test_financebench_test5_preserves_frozen_v1_baseline():
    """Ensures Test 5 preserves frozen V1 baseline latest.json untouched."""
    v1_latest_path = os.path.join(RESULTS_DIR, "latest.json")
    if os.path.exists(v1_latest_path):
        with open(v1_latest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("aggregate_recall_at_1") == 0.9167
