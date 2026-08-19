import os
import json
import pytest
from app.evaluation.financebench.loader import FinanceBenchLoader
from app.evaluation.financebench.validator import validate_financebench_environment
from app.evaluation.financebench.runner import FinanceBenchRunner, FINANCEBENCH_RESULTS_DIR
from app.evaluation.runner import RESULTS_DIR


def test_financebench_environment_validation():
    """Verifies that FinanceBench validator finds 150 questions, 84 required PDFs, and 0 missing PDFs."""
    report = validate_financebench_environment()
    assert report["valid"] is True
    assert report["total_questions"] == 150
    assert report["unique_documents_count"] == 84
    assert report["available_matching_pdfs_count"] == 84
    assert report["missing_pdfs_count"] == 0


def test_financebench_loader_parsing():
    """Verifies that FinanceBenchLoader parses annotation JSONL lines accurately."""
    loader = FinanceBenchLoader()
    questions = loader.load_dataset()
    assert len(questions) == 150

    q_3m = loader.get_questions_by_doc_name("3M_2018_10K")
    assert len(q_3m) >= 2
    for q in q_3m:
        assert q.doc_name == "3M_2018_10K"
        assert q.company == "3M"
        assert q.question
        assert q.answer


def test_financebench_runner_single_doc_ingest_and_eval():
    """Tests selective single-document FinanceBench ingestion and evaluation run."""
    runner = FinanceBenchRunner()
    doc_name = "3M_2018_10K"

    # Test PDF path resolution
    pdf_path = runner.resolve_pdf_path(doc_name)
    assert os.path.exists(pdf_path)

    # Test single-document evaluation run
    res = runner.run_document_evaluation(doc_name=doc_name, top_k=5, retrieval_mode="hybrid")
    assert res.total_questions >= 2
    assert res.retrieval_mode == "hybrid"
    assert res.evaluation_version == f"financebench_{doc_name}"

    # Verify result saved under results/financebench/
    latest_file = os.path.join(FINANCEBENCH_RESULTS_DIR, f"financebench_{doc_name}_hybrid_latest.json")
    assert os.path.exists(latest_file)


def test_financebench_preserves_frozen_v1_baseline():
    """Ensures FinanceBench runner stores outputs in results/financebench/ and leaves frozen V1 latest.json untouched."""
    v1_latest_path = os.path.join(RESULTS_DIR, "latest.json")
    v1_timestamp_before = None

    if os.path.exists(v1_latest_path):
        with open(v1_latest_path, "r", encoding="utf-8") as f:
            v1_timestamp_before = json.load(f).get("timestamp")

    runner = FinanceBenchRunner()
    runner.run_document_evaluation(doc_name="3M_2018_10K", top_k=3, retrieval_mode="dense")

    if v1_timestamp_before:
        with open(v1_latest_path, "r", encoding="utf-8") as f:
            v1_timestamp_after = json.load(f).get("timestamp")
        assert v1_timestamp_after == v1_timestamp_before, "FinanceBench run must not modify frozen V1 latest.json"
