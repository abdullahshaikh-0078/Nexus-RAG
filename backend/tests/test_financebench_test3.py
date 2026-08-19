import os
import json
import pytest
from app.evaluation.financebench.loader import FinanceBenchLoader
from app.evaluation.financebench.run_financebench_test3 import (
    TARGET_QUESTION_IDS,
    QUESTION_CATEGORY_MAP,
    classify_failure_taxonomy,
)
from app.evaluation.runner import RESULTS_DIR


def test_financebench_test3_target_questions_exist():
    """Verifies that all 21 Test 3 question IDs exist in FinanceBench dataset."""
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


def test_failure_taxonomy_classification():
    """Verifies granular failure taxonomy classification logic."""
    assert classify_failure_taxonomy("financebench_id_001", 1.0) == "success"
    assert classify_failure_taxonomy("financebench_id_07966", 0.0) == "multi_year_calculation"
    assert classify_failure_taxonomy("financebench_id_00941", 0.0) == "footnote_reference"
    assert classify_failure_taxonomy("financebench_id_00499", 0.0) == "terminology_mismatch"
    assert classify_failure_taxonomy("financebench_id_03029", 0.0) == "table_fragmentation"


def test_financebench_test3_preserves_frozen_v1_baseline():
    """Ensures Test 3 preserves frozen V1 baseline latest.json untouched."""
    v1_latest_path = os.path.join(RESULTS_DIR, "latest.json")
    if os.path.exists(v1_latest_path):
        with open(v1_latest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("aggregate_recall_at_1") == 0.9167
