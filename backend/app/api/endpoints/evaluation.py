import os
import glob
import json
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, status, Query
from app.evaluation.runner import EvaluationRunner, RESULTS_DIR
from app.evaluation.metrics import EvaluationRunResult
from app.services.query_eval_service import query_eval_service, QueryEvaluation

router = APIRouter()


@router.get("/results", response_model=EvaluationRunResult)
async def get_latest_evaluation_results(mode: str = Query("dense", description="Retrieval mode: 'dense', 'bm25', or 'hybrid'")):
    """Returns the latest saved evaluation result for the requested retrieval mode."""
    mode_lower = mode.lower()
    if mode_lower == "hybrid":
        file_name = "v2_2_hybrid_latest.json"
    elif mode_lower == "bm25":
        file_name = "v2_1_bm25_latest.json"
    else:
        file_name = "latest.json"

    latest_path = os.path.join(RESULTS_DIR, file_name)

    if not os.path.exists(latest_path):
        runner = EvaluationRunner()
        return runner.run_evaluation(top_k=10, retrieval_mode=mode_lower)

    try:
        with open(latest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return EvaluationRunResult(**data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read evaluation result for mode '{mode}': {str(e)}",
        )


@router.get("/results/all")
async def get_all_evaluation_comparison():
    """Returns side-by-side comparison data for V1 Dense Baseline, V2.1 BM25, and V2.2 Hybrid."""
    runner = EvaluationRunner()
    v1_path = os.path.join(RESULTS_DIR, "latest.json")
    v2_1_path = os.path.join(RESULTS_DIR, "v2_1_bm25_latest.json")
    v2_2_path = os.path.join(RESULTS_DIR, "v2_2_hybrid_latest.json")

    v1_data = None
    if os.path.exists(v1_path):
        with open(v1_path, "r", encoding="utf-8") as f:
            v1_data = json.load(f)
    else:
        v1_res = runner.run_evaluation(top_k=10, retrieval_mode="dense")
        v1_data = v1_res.model_dump()

    v2_1_data = None
    if os.path.exists(v2_1_path):
        with open(v2_1_path, "r", encoding="utf-8") as f:
            v2_1_data = json.load(f)
    else:
        v2_1_res = runner.run_evaluation(top_k=10, retrieval_mode="bm25")
        v2_1_data = v2_1_res.model_dump()

    v2_2_data = None
    if os.path.exists(v2_2_path):
        with open(v2_2_path, "r", encoding="utf-8") as f:
            v2_2_data = json.load(f)
    else:
        v2_2_res = runner.run_evaluation(top_k=10, retrieval_mode="hybrid")
        v2_2_data = v2_2_res.model_dump()

    return {
        "v1_dense": v1_data,
        "v2_1_bm25": v2_1_data,
        "v2_2_hybrid": v2_2_data,
    }


@router.post("/run", response_model=EvaluationRunResult)
async def run_evaluation_suite(mode: str = Query("dense", description="Retrieval mode: 'dense', 'bm25', or 'hybrid'")):
    """Triggers an on-demand evaluation run across the test dataset for the specified retrieval mode."""
    try:
        runner = EvaluationRunner()
        result = runner.run_evaluation(top_k=10, retrieval_mode=mode.lower())
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute evaluation suite for mode '{mode}': {str(e)}",
        )


# --- Benchmark Evaluations Catalog API ---

@router.get("/benchmarks")
async def list_benchmark_evaluations():
    """Catalog of all benchmark evaluation runs (V1, V2.1, V2.2, Secondary, Diagnostic, FinanceBench)."""
    items: List[Dict[str, Any]] = []

    # 1. Base results directory
    all_json_files = glob.glob(os.path.join(RESULTS_DIR, "**", "*.json"), recursive=True)
    for fpath in all_json_files:
        fname = os.path.basename(fpath)
        rel_path = os.path.relpath(fpath, RESULTS_DIR)

        # Skip query_evaluations directory items
        if "query_evaluations" in rel_path:
            continue

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = json.load(f)

            b_id = fname.replace(".json", "")

            # Detect benchmark type
            b_type = "official_v1"
            if "financebench" in rel_path or "financebench" in fname:
                b_type = "financebench"
            elif "secondary_tests" in rel_path:
                b_type = "secondary_test"
            elif "diagnostic_tests" in rel_path:
                b_type = "diagnostic_test"
            elif "v2_2" in fname:
                b_type = "v2_2_hybrid"
            elif "v2_1" in fname:
                b_type = "v2_1_bm25"

            items.append({
                "benchmark_id": b_id,
                "file_name": fname,
                "relative_path": rel_path,
                "benchmark_type": b_type,
                "dataset_version": content.get("dataset_version") or content.get("summary", {}).get("test_name", "v1_baseline"),
                "evaluation_version": content.get("evaluation_version") or content.get("summary", {}).get("document_name", "v1_baseline"),
                "retrieval_mode": content.get("retrieval_mode") or "hybrid",
                "timestamp": content.get("timestamp") or content.get("summary", {}).get("timestamp"),
                "total_questions": content.get("total_questions") or content.get("summary", {}).get("total_questions", 0),
                "recall_at_1": content.get("aggregate_recall_at_1", 0.0),
                "recall_at_3": content.get("aggregate_recall_at_3", 0.0),
                "recall_at_5": content.get("aggregate_recall_at_5", 0.0),
                "recall_at_10": content.get("aggregate_recall_at_10", 0.0),
                "mrr_at_10": content.get("aggregate_mrr_at_10", 0.0),
                "ndcg_at_10": content.get("aggregate_ndcg_at_10", 0.0),
                "average_retrieval_latency_ms": content.get("average_retrieval_latency_ms", 0.0),
            })
        except Exception:
            continue

    items.sort(key=lambda x: x.get("timestamp") or "", reverse=True)
    return {"total": len(items), "benchmarks": items}


@router.get("/benchmarks/{benchmark_id:path}")
async def get_benchmark_detail(benchmark_id: str):
    """Retrieves raw detail for a specific benchmark evaluation file."""
    search_pattern = os.path.join(RESULTS_DIR, "**", f"{benchmark_id}.json")
    matches = glob.glob(search_pattern, recursive=True)

    if not matches:
        # Check direct filename match
        direct_path = os.path.join(RESULTS_DIR, f"{benchmark_id}.json")
        if os.path.exists(direct_path):
            matches = [direct_path]

    if not matches:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark evaluation '{benchmark_id}' not found.",
        )

    target_file = matches[0]
    try:
        with open(target_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load benchmark evaluation detail: {str(e)}",
        )


# --- Per-Query Evaluation Endpoints ---

@router.get("/query-evaluations", response_model=List[QueryEvaluation])
async def list_query_evaluations(limit: int = Query(50, ge=1, le=200)):
    """Lists per-query evaluation events recorded from chat interactions."""
    return query_eval_service.list_evaluations(limit=limit)


@router.get("/query-evaluations/{eval_id}", response_model=QueryEvaluation)
async def get_query_evaluation(eval_id: str):
    """Gets step-by-step latency and chunk rank detail for a query evaluation."""
    eval_item = query_eval_service.get_evaluation(eval_id)
    if not eval_item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Query evaluation '{eval_id}' not found.",
        )
    return eval_item


@router.delete("/query-evaluations/{eval_id}")
async def delete_query_evaluation(eval_id: str):
    """Deletes a query evaluation record."""
    success = query_eval_service.delete_evaluation(eval_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Query evaluation '{eval_id}' not found or could not be deleted.",
        )
    return {"success": True, "message": f"Query evaluation '{eval_id}' deleted."}
