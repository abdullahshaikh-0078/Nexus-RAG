import os
import json
import pytest
from app.evaluation.financebench.loader import FinanceBenchLoader
from app.evaluation.financebench.run_financebench_test4 import (
    TARGET_QUESTION_IDS,
    QUESTION_DIAGNOSTIC_CATEGORY,
    check_evidence_presence,
    classify_primary_failure,
    TEST4_DIR,
)
from app.evaluation.runner import RESULTS_DIR


def test_financebench_test4_target_questions_exist():
    """Verifies that all 10 Test 4 diagnostic question IDs exist in FinanceBench dataset."""
    loader = FinanceBenchLoader()
    all_questions = loader.load_dataset()
    q_map = {q.financebench_id: q for q in all_questions}

    for q_id in TARGET_QUESTION_IDS:
        assert q_id in q_map
        q = q_map[q_id]
        assert q.question
        assert q.doc_name
        assert q.answer
        assert q_id in QUESTION_DIAGNOSTIC_CATEGORY


def test_evidence_presence_check():
    """Verifies evidence presence detection helper."""
    texts = [
        "Unrelated chunk about software engineering.",
        "Purchases of property, plant and equipment was $1,577 million in FY2018.",
    ]
    expected = ["$1,577 million"]

    assert check_evidence_presence(texts, expected, top_k=2) is True
    assert check_evidence_presence(texts, expected, top_k=1) is False


def test_classify_primary_failure():
    """Verifies diagnostic failure classification logic."""
    # Ranking failure case: evidence at top-10 but not top-1
    f_rank = classify_primary_failure(
        ev_10=True, ev_1=False, context_status="CONTEXT_PRESENT", answer="Answer", gt_answer="Answer", q_id="financebench_id_03029"
    )
    assert f_rank == "ranking_failure"

    # Table fragmentation case: no evidence at top-10
    f_tab = classify_primary_failure(
        ev_10=False, ev_1=False, context_status="CONTEXT_MISSING", answer="Answer", gt_answer="Answer", q_id="financebench_id_03029"
    )
    assert f_tab == "table_fragmentation"


def test_financebench_test4_preserves_frozen_v1_baseline():
    """Ensures Test 4 preserves frozen V1 baseline latest.json untouched."""
    v1_latest_path = os.path.join(RESULTS_DIR, "latest.json")
    if os.path.exists(v1_latest_path):
        with open(v1_latest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("aggregate_recall_at_1") == 0.9167
