import time
import logging
from typing import List, Tuple
from app.models.schemas import SourceCitation

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder
    _CROSS_ENCODER_AVAILABLE = True
except ImportError:
    _CROSS_ENCODER_AVAILABLE = False
    logger.warning("sentence_transformers CrossEncoder not available.")


class ExperimentalReranker:
    """
    Isolated evaluation reranking stage using cross-encoder/ms-marco-MiniLM-L-6-v2.
    Does NOT modify production hybrid retriever or vector store.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None and _CROSS_ENCODER_AVAILABLE:
            logger.info(f"Loading isolated evaluation CrossEncoder model '{self.model_name}'...")
            self._model = CrossEncoder(self.model_name)

    def rerank(self, query: str, citations: List[SourceCitation]) -> Tuple[List[SourceCitation], float]:
        """
        Reranks a list of candidate SourceCitation objects for a given query string.
        Returns (reranked_citations, rerank_latency_ms).
        """
        if not citations:
            return citations, 0.0

        t0 = time.time()
        self._load_model()

        if self._model is None:
            # Fallback if sentence_transformers isn't loaded
            return citations, round((time.time() - t0) * 1000, 2)

        pairs = [[query, c.content] for c in citations]
        scores = self._model.predict(pairs)

        # Pair citations with scores and sort descending
        scored_citations = list(zip(citations, scores))
        scored_citations.sort(key=lambda x: x[1], reverse=True)

        reranked_list = []
        for rank_idx, (cit, sc) in enumerate(scored_citations, 1):
            cit_copy = cit.model_copy()
            cit_copy.score = float(sc)
            cit_copy.rrf_score = float(sc)
            reranked_list.append(cit_copy)

        t_rerank_ms = round((time.time() - t0) * 1000, 2)
        return reranked_list, t_rerank_ms


experimental_reranker = ExperimentalReranker()
