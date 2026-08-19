import os
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

QUERY_EVALS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "evaluation", "results", "query_evaluations"
)


class QueryEvaluation(BaseModel):
    evaluation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    query: str
    document_ids: Optional[List[str]] = None
    retrieval_mode: str = "hybrid"
    dense_results: List[Dict[str, Any]] = Field(default_factory=list)
    bm25_results: List[Dict[str, Any]] = Field(default_factory=list)
    hybrid_results: List[Dict[str, Any]] = Field(default_factory=list)
    final_context: List[Dict[str, Any]] = Field(default_factory=list)
    answer: str
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    latency_breakdown: Dict[str, float] = Field(default_factory=dict)
    evaluation_status: Dict[str, str] = Field(
        default_factory=lambda: {
            "retrieval_status": "relevant_context_detected",
            "answer_status": "answer_generated",
        }
    )
    failure_category: Optional[str] = None
    groundedness_score: Optional[float] = None
    answer_correctness_score: Optional[float] = None
    ground_truth: Optional[Dict[str, Any]] = None
    retrieval_metrics: Optional[Dict[str, Any]] = None
    answer_metrics: Optional[Dict[str, Any]] = None


class QueryEvaluationService:
    """Manages recording, listing, reading, and deleting query execution logs."""

    def __init__(self, storage_dir: str = QUERY_EVALS_DIR):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def record_evaluation(
        self,
        query: str,
        answer: str,
        retrieval_mode: str,
        citations: List[Dict[str, Any]],
        latency_breakdown: Dict[str, float],
        document_ids: Optional[List[str]] = None,
        dense_results: Optional[List[Dict[str, Any]]] = None,
        bm25_results: Optional[List[Dict[str, Any]]] = None,
        hybrid_results: Optional[List[Dict[str, Any]]] = None,
        ground_truth: Optional[Dict[str, Any]] = None,
        failure_category: Optional[str] = None,
        groundedness_score: Optional[float] = None,
        answer_correctness_score: Optional[float] = None,
    ) -> QueryEvaluation:
        """Creates and persists a QueryEvaluation record."""
        retrieval_status = "no_relevant_context"
        if citations:
            retrieval_status = "relevant_context_detected"

        answer_status = "no_answer"
        if answer and answer.strip():
            if "insufficient information" in answer.lower() or "not contain" in answer.lower():
                answer_status = "insufficient_evidence"
            else:
                answer_status = "answer_generated"

        eval_record = QueryEvaluation(
            query=query,
            answer=answer,
            document_ids=document_ids,
            retrieval_mode=retrieval_mode,
            dense_results=dense_results or [],
            bm25_results=bm25_results or [],
            hybrid_results=hybrid_results or [],
            final_context=citations,
            citations=citations,
            latency_breakdown=latency_breakdown,
            evaluation_status={
                "retrieval_status": retrieval_status,
                "answer_status": answer_status,
            },
            failure_category=failure_category,
            groundedness_score=groundedness_score,
            answer_correctness_score=answer_correctness_score,
            ground_truth=ground_truth,
        )

        file_path = os.path.join(self.storage_dir, f"query_eval_{eval_record.evaluation_id}.json")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(eval_record.model_dump(), f, indent=2)
            logger.info(f"Recorded query evaluation event '{eval_record.evaluation_id}'")
        except Exception as e:
            logger.error(f"Failed to persist query evaluation record: {str(e)}")

        return eval_record

    def list_evaluations(self, limit: int = 50) -> List[QueryEvaluation]:
        """Lists stored query evaluations sorted by timestamp descending."""
        evals: List[QueryEvaluation] = []
        if not os.path.exists(self.storage_dir):
            return evals

        for fname in os.listdir(self.storage_dir):
            if fname.startswith("query_eval_") and fname.endswith(".json"):
                fpath = os.path.join(self.storage_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    evals.append(QueryEvaluation(**data))
                except Exception as e:
                    logger.warning(f"Error reading query evaluation file {fname}: {str(e)}")

        evals.sort(key=lambda x: x.timestamp, reverse=True)
        return evals[:limit]

    def get_evaluation(self, eval_id: str) -> Optional[QueryEvaluation]:
        """Retrieves single QueryEvaluation by ID."""
        file_path = os.path.join(self.storage_dir, f"query_eval_{eval_id}.json")
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return QueryEvaluation(**data)
        except Exception as e:
            logger.error(f"Error loading query evaluation {eval_id}: {str(e)}")
            return None

    def delete_evaluation(self, eval_id: str) -> bool:
        """Deletes single QueryEvaluation file."""
        file_path = os.path.join(self.storage_dir, f"query_eval_{eval_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False


query_eval_service = QueryEvaluationService()
