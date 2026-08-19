import os
import json
import pytest
from app.evaluation.run_diagnostic_eval import (
    DIAGNOSTIC_DATASET_PATH,
    DIAGNOSTIC_RESULTS_DIR,
)
from app.evaluation.runner import RESULTS_DIR


def test_diagnostic_dataset_loading_and_categories():
    """Verifies that v2_diagnostic.json dataset exists, loads properly, and contains 36 cases across 12 categories."""
    assert os.path.exists(DIAGNOSTIC_DATASET_PATH)
    with open(DIAGNOSTIC_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    test_cases = data.get("test_cases", [])
    assert len(test_cases) == 36

    categories = set(c["category"] for c in test_cases)
    assert len(categories) == 12


def test_diagnostic_run_preserves_frozen_v1_baseline():
    """Ensures diagnostic evaluation runs store results in diagnostic_tests/ and do NOT modify frozen V1 latest.json."""
    v1_latest_path = os.path.join(RESULTS_DIR, "latest.json")
    v1_timestamp_before = None

    if os.path.exists(v1_latest_path):
        with open(v1_latest_path, "r", encoding="utf-8") as f:
            v1_timestamp_before = json.load(f).get("timestamp")

    from app.evaluation.run_diagnostic_eval import run_diagnostic_evaluation
    results = run_diagnostic_evaluation()

    # Verify diagnostic result files exist
    hybrid_latest = os.path.join(DIAGNOSTIC_RESULTS_DIR, "v2_diagnostic_hybrid_latest.json")
    assert os.path.exists(hybrid_latest)

    # Verify V1 baseline latest.json was NOT modified
    if v1_timestamp_before:
        with open(v1_latest_path, "r", encoding="utf-8") as f:
            v1_timestamp_after = json.load(f).get("timestamp")
        assert v1_timestamp_after == v1_timestamp_before, "Diagnostic eval must not overwrite frozen V1 latest.json"
