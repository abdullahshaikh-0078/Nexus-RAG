import math
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class QuestionEvalResult(BaseModel):
    question_id: str
    question: str
    category: str
    first_relevant_rank: Optional[int] = None  # None if not found in top K
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float
    ndcg_at_10: float
    retrieval_latency_ms: float
    retrieved_chunk_ids: List[str] = Field(default_factory=list)
    retrieved_snippets: List[str] = Field(default_factory=list)


class EvaluationRunResult(BaseModel):
    evaluation_version: str = "v1_baseline"
    timestamp: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int
    total_questions: int
    aggregate_recall_at_1: float
    aggregate_recall_at_3: float
    aggregate_recall_at_5: float
    aggregate_recall_at_10: float
    aggregate_mrr_at_10: float
    aggregate_ndcg_at_10: float
    average_retrieval_latency_ms: float
    question_results: List[QuestionEvalResult]


class RetrievalEvaluator:
    """Calculates scientific retrieval metrics (Recall@K, MRR@10, NDCG@10)."""

    @staticmethod
    def is_chunk_relevant(retrieved_text: str, expected_snippets: List[str]) -> bool:
        """Checks if a retrieved chunk text contains any of the expected ground truth snippets."""
        if not retrieved_text or not expected_snippets:
            return False
        text_lower = retrieved_text.lower()
        for snippet in expected_snippets:
            if snippet.lower() in text_lower:
                return True
        return False

    @staticmethod
    def calculate_recall_at_k(
        retrieved_texts: List[str], expected_snippets: List[str], k: int
    ) -> float:
        """Recall@K: 1.0 if at least one relevant chunk appears in top K, else 0.0."""
        top_k_texts = retrieved_texts[:k]
        for text in top_k_texts:
            if RetrievalEvaluator.is_chunk_relevant(text, expected_snippets):
                return 1.0
        return 0.0

    @staticmethod
    def calculate_first_relevant_rank(
        retrieved_texts: List[str], expected_snippets: List[str], max_k: int = 10
    ) -> Optional[int]:
        """Finds 1-based rank of the first relevant chunk in top max_k results."""
        for idx, text in enumerate(retrieved_texts[:max_k], 1):
            if RetrievalEvaluator.is_chunk_relevant(text, expected_snippets):
                return idx
        return None

    @staticmethod
    def calculate_mrr_at_k(
        retrieved_texts: List[str], expected_snippets: List[str], k: int = 10
    ) -> float:
        """MRR@K: Reciprocal rank (1 / rank) of first relevant chunk, or 0.0 if not in top K."""
        rank = RetrievalEvaluator.calculate_first_relevant_rank(retrieved_texts, expected_snippets, max_k=k)
        if rank is not None:
            return 1.0 / rank
        return 0.0

    @staticmethod
    def calculate_ndcg_at_k(
        retrieved_texts: List[str], expected_snippets: List[str], k: int = 10
    ) -> float:
        """NDCG@K: Normalized Discounted Cumulative Gain for binary relevance."""
        top_k_texts = retrieved_texts[:k]
        dcg = 0.0
        relevant_found = 0

        for idx, text in enumerate(top_k_texts, 1):
            if RetrievalEvaluator.is_chunk_relevant(text, expected_snippets):
                rel = 1.0
                dcg += rel / math.log2(idx + 1)
                relevant_found += 1

        if dcg == 0.0:
            return 0.0

        # Calculate Ideal DCG (IDCG) assuming relevant items are ranked at top positions
        idcg = 0.0
        num_ideal = min(len(expected_snippets), k)
        for idx in range(1, num_ideal + 1):
            idcg += 1.0 / math.log2(idx + 1)

        if idcg == 0.0:
            return 0.0

        return dcg / idcg
