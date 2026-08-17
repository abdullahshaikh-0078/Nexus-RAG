import os
import json
import pytest
from app.evaluation.metrics import RetrievalEvaluator, QuestionEvalResult, EvaluationRunResult
from app.evaluation.runner import EvaluationRunner, RESULTS_DIR


def test_recall_at_k_calculation():
    retrieved = [
        "Chunk 1: Title and authors",
        "Chunk 2: The Transformer architecture relies solely on attention mechanisms",
        "Chunk 3: Conclusions and future work",
    ]
    expected = ["attention mechanisms"]

    assert RetrievalEvaluator.calculate_recall_at_k(retrieved, expected, k=1) == 0.0
    assert RetrievalEvaluator.calculate_recall_at_k(retrieved, expected, k=2) == 1.0
    assert RetrievalEvaluator.calculate_recall_at_k(retrieved, expected, k=3) == 1.0


def test_mrr_at_k_calculation():
    retrieved = [
        "Chunk 1: Unrelated intro",
        "Chunk 2: Unrelated background",
        "Chunk 3: Target phrase: eschewing recurrence and convolutions",
    ]
    expected = ["eschewing recurrence"]

    mrr = RetrievalEvaluator.calculate_mrr_at_k(retrieved, expected, k=10)
    assert mrr == pytest.approx(1.0 / 3.0, 0.001)

    # When expected is at rank 1
    retrieved_top = ["Chunk 1: Target phrase: eschewing recurrence"]
    assert RetrievalEvaluator.calculate_mrr_at_k(retrieved_top, expected, k=10) == 1.0


def test_ndcg_at_k_calculation():
    retrieved = [
        "Chunk 1: Unrelated snippet",
        "Chunk 2: Target snippet: attention mechanism for global dependencies",
    ]
    expected = ["attention mechanism"]

    ndcg = RetrievalEvaluator.calculate_ndcg_at_k(retrieved, expected, k=10)
    assert ndcg > 0.0
    assert ndcg <= 1.0


def test_evaluation_dataset_loading():
    runner = EvaluationRunner()
    dataset = runner.load_dataset()
    assert dataset["version"] == "v1_baseline"
    test_cases = dataset["test_cases"]
    assert len(test_cases) >= 10
    for case in test_cases:
        assert "question_id" in case
        assert "question" in case
        assert "expected_snippets" in case


def test_evaluation_runner_execution_and_atomic_write():
    runner = EvaluationRunner()
    result = runner.run_evaluation(top_k=5)
    assert isinstance(result, EvaluationRunResult)
    assert result.evaluation_version == "v1_baseline"

    # Verify latest.json exists and contains matching timestamp
    latest_path = os.path.join(RESULTS_DIR, "latest.json")
    assert os.path.exists(latest_path)

    with open(latest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    latest_obj = EvaluationRunResult(**data)
    assert latest_obj.timestamp == result.timestamp


def test_evaluation_runner_atomic_error_resilience(tmp_path):
    # Test invalid dataset handling
    invalid_dataset_file = str(tmp_path / "invalid.json")
    with open(invalid_dataset_file, "w", encoding="utf-8") as f:
        json.dump({"test_cases": []}, f)

    runner = EvaluationRunner(dataset_path=invalid_dataset_file)
    with pytest.raises(ValueError, match="Evaluation dataset contains no test cases"):
        runner.run_evaluation()
