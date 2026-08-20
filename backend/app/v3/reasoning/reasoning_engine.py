import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from app.v3.reasoning.schemas import (
    QueryClassification,
    FinancialFact,
    CalculationResult,
)
from app.v3.reasoning.query_classifier import query_classifier
from app.v3.reasoning.fact_extractor import fact_extractor
from app.v3.reasoning.calculator import calculation_engine
from app.v3.query_expansion.rewriter import v3_query_rewriter
from app.v3.retrieval.v3_retriever import v3_retriever
from app.models.schemas import SourceCitation

logger = logging.getLogger(__name__)


class V3ReasoningEngine:
    """
    Main V3 Financial Reasoning Engine.
    Orchestrates Query Classification -> Fact Retrieval -> Fact Extraction -> Calculation -> Grounded Synthesis.
    """

    def process_query(
        self,
        query: str,
        top_k: int = 4,
        document_ids: Optional[List[str]] = None,
        chunking_strategy: str = "table_aware",
        chat_id: Optional[str] = None,
    ) -> Tuple[List[SourceCitation], Dict[str, float], Optional[Dict[str, Any]], Optional[CalculationResult]]:
        t0 = time.time()

        # 1. Classify Query Intent
        classification: QueryClassification = query_classifier.classify(query)

        # 2. Rewrite query for retrieval (V3.3 query expansion)
        t0_exp = time.time()
        bm25_rewritten, exp_trace = v3_query_rewriter.rewrite_for_bm25(query)
        t_exp = round((time.time() - t0_exp) * 1000, 2)

        # 3. Retrieve Context Chunks & Tables via V3 Isolated Hybrid Retrieval
        t0_h = time.time()
        citations = v3_retriever.search(
            query=bm25_rewritten,
            top_k=top_k * 2 if classification.is_calculation_required else top_k,
            document_ids=document_ids,
            chunking_strategy=chunking_strategy,
            chat_id=chat_id,
        )
        t_h = round((time.time() - t0_h) * 1000, 2)

        calc_result: Optional[CalculationResult] = None
        t_calc = 0.0

        # 4. If Calculation Required, Extract Facts and Run Calculation Engine
        if classification.is_calculation_required and classification.target_metric_id:
            t0_c = time.time()
            facts: List[FinancialFact] = fact_extractor.extract_facts_from_citations(
                citations=citations,
                target_years=classification.target_years,
            )

            calc_result = calculation_engine.calculate(
                metric_id=classification.target_metric_id,
                facts=facts,
                target_years=classification.target_years,
            )
            t_calc = round((time.time() - t0_c) * 1000, 2)

        breakdown = {
            "query_expansion_ms": t_exp,
            "rrf_fusion_ms": t_h,
            "calculation_engine_ms": t_calc,
            "total_request_ms": round((time.time() - t0) * 1000, 2),
        }

        exp_meta = exp_trace.model_dump()
        exp_meta["classification"] = classification.model_dump()

        return citations[:top_k], breakdown, exp_meta, calc_result


v3_reasoning_engine = V3ReasoningEngine()
