import os
import json
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status, Query
from app.evaluation.runner import EvaluationRunner, RESULTS_DIR
from app.evaluation.metrics import EvaluationRunResult

router = APIRouter()


@router.get("/results", response_model=EvaluationRunResult)
async def get_latest_evaluation_results(mode: str = Query("dense", description="Retrieval mode: 'dense' or 'bm25'")):
    """
    Returns the latest saved evaluation result for the requested retrieval mode.
    """
    mode_lower = mode.lower()
    file_name = "v2_1_bm25_latest.json" if mode_lower == "bm25" else "latest.json"
    latest_path = os.path.join(RESULTS_DIR, file_name)

    if not os.path.exists(latest_path):
        # Trigger evaluation run if no prior result exists
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
    """
    Returns side-by-side comparison data for V1 Dense Baseline and V2.1 BM25.
    """
    runner = EvaluationRunner()
    v1_path = os.path.join(RESULTS_DIR, "latest.json")
    v2_1_path = os.path.join(RESULTS_DIR, "v2_1_bm25_latest.json")

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

    return {
        "v1_dense": v1_data,
        "v2_1_bm25": v2_1_data,
    }


@router.post("/run", response_model=EvaluationRunResult)
async def run_evaluation_suite(mode: str = Query("dense", description="Retrieval mode: 'dense' or 'bm25'")):
    """
    Triggers an on-demand evaluation run across the test dataset for the specified retrieval mode.
    """
    try:
        runner = EvaluationRunner()
        result = runner.run_evaluation(top_k=10, retrieval_mode=mode.lower())
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute evaluation suite for mode '{mode}': {str(e)}",
        )
