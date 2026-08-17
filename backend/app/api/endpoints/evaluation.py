import os
import json
from fastapi import APIRouter, HTTPException, status
from app.evaluation.runner import EvaluationRunner, RESULTS_DIR
from app.evaluation.metrics import EvaluationRunResult

router = APIRouter()


@router.get("/results", response_model=EvaluationRunResult)
async def get_latest_evaluation_results():
    """
    Returns the latest saved V1 Dense Retrieval evaluation baseline result.
    """
    latest_path = os.path.join(RESULTS_DIR, "latest.json")
    if not os.path.exists(latest_path):
        # Trigger evaluation run if no prior baseline exists
        runner = EvaluationRunner()
        return runner.run_evaluation(top_k=10)

    try:
        with open(latest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return EvaluationRunResult(**data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read evaluation baseline result: {str(e)}",
        )


@router.post("/run", response_model=EvaluationRunResult)
async def run_evaluation_suite():
    """
    Triggers an on-demand baseline evaluation run across the test dataset.
    """
    try:
        runner = EvaluationRunner()
        result = runner.run_evaluation(top_k=10)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute evaluation suite: {str(e)}",
        )
