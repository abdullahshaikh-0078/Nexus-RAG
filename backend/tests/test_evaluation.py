import os
import json
import pytest
from app.evaluation.metrics import RetrievalEvaluator, QuestionEvalResult, EvaluationRunResult
from app.evaluation.runner import EvaluationRunner, RESULTS_DIR


def test_recall_at_k_calculation():
    retrieved = [
        "The Transformer is a neural network architecture based on self-attention mechanisms.",
        "Convolutional networks use spatial filtering for image classification.",
        "Recurrent neural networks process sequential data with hidden states.",
    ]
    expected = [
        "The Transformer is a neural network architecture based on self-attention mechanisms.",
    ]

    assert RetrievalEvaluator.calculate_recall_at_k(retrieved, expected, k=1) == 1.0
    assert RetrievalEvaluator.calculate_recall_at_k(retrieved, expected, k=3) == 1.0

    unmatched = ["Quantum computing uses qubits."]
    assert RetrievalEvaluator.calculate_recall_at_k(unmatched, expected, k=1) == 0.0


def test_mrr_at_k_calculation():
    retrieved = [
        "Unrelated chunk about databases.",
        "The Transformer is a neural network architecture based on self-attention mechanisms.",
        "Another unrelated chunk.",
    ]
    expected = [
        "The Transformer is a neural network architecture based on self-attention mechanisms.",
    ]

    mrr = RetrievalEvaluator.calculate_mrr_at_k(retrieved, expected, k=10)
    assert mrr == pytest.approx(0.5)


def test_ndcg_at_k_calculation():
    retrieved = [
        "Chunk 1: Attention mechanisms in transformers",
        "Chunk 2: Target snippet: attention mechanism for global dependencies",
    ]
    expected = ["attention mechanism"]

    ndcg = RetrievalEvaluator.calculate_ndcg_at_k(retrieved, expected, k=10)
    assert ndcg > 0.0


def test_evaluation_dataset_loading():
    runner = EvaluationRunner()
    dataset = runner.load_dataset()
    assert dataset["version"] == "v1_baseline"
    test_cases = dataset["test_cases"]
    assert len(test_cases) == 12
    for case in test_cases:
        assert "question_id" in case
        assert "question" in case
        assert "expected_snippets" in case


def test_evaluation_runner_execution_and_atomic_write():
    latest_path = os.path.join(RESULTS_DIR, "latest.json")
    original_latest = None
    if os.path.exists(latest_path):
        with open(latest_path, "r", encoding="utf-8") as f:
            original_latest = f.read()

    try:
        runner = EvaluationRunner()
        result = runner.run_evaluation(top_k=5)
        assert isinstance(result, EvaluationRunResult)
        assert result.evaluation_version == "v1_baseline"
        assert os.path.exists(latest_path)
    finally:
        # Restore frozen V1 latest.json
        if original_latest:
            with open(latest_path, "w", encoding="utf-8") as f:
                f.write(original_latest)


def test_bm25_evaluation_preserves_v1_baseline():
    runner = EvaluationRunner()
    v1_latest_path = os.path.join(RESULTS_DIR, "latest.json")
    v1_timestamp_before = None
    if os.path.exists(v1_latest_path):
        with open(v1_latest_path, "r", encoding="utf-8") as f:
            v1_timestamp_before = json.load(f).get("timestamp")

    res_bm25 = runner.run_evaluation(top_k=10, retrieval_mode="bm25")
    assert res_bm25.dataset_version == "v1_baseline"
    assert res_bm25.evaluation_version == "v2_1_bm25"
    assert res_bm25.retrieval_mode == "bm25"
    assert res_bm25.total_questions == 12

    bm25_latest_path = os.path.join(RESULTS_DIR, "v2_1_bm25_latest.json")
    assert os.path.exists(bm25_latest_path)

    if v1_timestamp_before:
        with open(v1_latest_path, "r", encoding="utf-8") as f:
            v1_timestamp_after = json.load(f).get("timestamp")
        assert v1_timestamp_after == v1_timestamp_before, "BM25 evaluation must not overwrite frozen V1 latest.json"


def test_evaluation_runner_atomic_error_resilience(tmp_path):
    invalid_dataset_file = str(tmp_path / "invalid.json")
    with open(invalid_dataset_file, "w", encoding="utf-8") as f:
        json.dump({"test_cases": []}, f)

    runner = EvaluationRunner(dataset_path=invalid_dataset_file)
    with pytest.raises(ValueError, match="Evaluation dataset contains no test cases"):
        runner.run_evaluation()
