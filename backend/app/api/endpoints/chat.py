import time
import logging
from fastapi import APIRouter, HTTPException, status
from app.services.embedder import embedding_service
from app.services.bm25_search import bm25_service
from app.services.hybrid_retriever import hybrid_retriever
from app.db.vectorstore import vector_store
from app.db.mongodb import mongo_db
from app.services.llm_service import llm_service
from app.services.query_eval_service import query_eval_service
from app.models.schemas import ChatQueryRequest, ChatQueryResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/query", response_model=ChatQueryResponse)
async def query_rag(request: ChatQueryRequest):
    """
    RAG Query Flow:
    1. Check retrieval_mode ('dense', 'bm25', or 'hybrid').
    2. Measure step-by-step latency for embedding, search, fusion, expansion, and LLM synthesis.
    3. Retrieve top-k context chunks & expand adjacent context.
    4. Generate LLM answer.
    5. Record per-query evaluation event and return response with evaluation_id & latency_breakdown.
    """
    start_time = time.time()
    query_str = request.query.strip()
    if not query_str:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty.",
        )

    mode = (request.retrieval_mode or "hybrid").lower()
    t_embed = 0.0
    t_dense = 0.0
    t_bm25 = 0.0
    t_rrf = 0.0
    t_expand = 0.0

    try:
        t0 = time.time()
        if mode == "bm25":
            raw_citations = bm25_service.search(
                query=query_str,
                top_k=request.top_k,
                document_ids=request.document_ids,
            )
            t_bm25 = round((time.time() - t0) * 1000, 2)
        elif mode == "dense":
            t0_emb = time.time()
            query_vector = embedding_service.embed_text(query_str)
            t_embed = round((time.time() - t0_emb) * 1000, 2)

            t0_d = time.time()
            raw_citations = vector_store.search_similar(
                query_vector=query_vector,
                top_k=request.top_k,
                document_ids=request.document_ids,
            )
            t_dense = round((time.time() - t0_d) * 1000, 2)
        else:
            mode = "hybrid"
            t0_h = time.time()
            raw_citations = hybrid_retriever.search(
                query=query_str,
                top_k=request.top_k,
                document_ids=request.document_ids,
            )
            t_rrf = round((time.time() - t0_h) * 1000, 2)

        # Context Expansion
        t0_exp = time.time()
        citations = vector_store.expand_adjacent_context(raw_citations, window=1)
        t_expand = round((time.time() - t0_exp) * 1000, 2)

        # LLM Synthesis
        t0_gen = time.time()
        answer, provider_used, model_used = llm_service.generate_answer(
            query=query_str, citations=citations
        )
        t_gen = round((time.time() - t0_gen) * 1000, 2)

        elapsed = round(time.time() - start_time, 3)
        total_ms = round(elapsed * 1000, 2)

        latency_breakdown = {
            "embedding_ms": t_embed,
            "dense_search_ms": t_dense,
            "bm25_search_ms": t_bm25,
            "rrf_fusion_ms": t_rrf,
            "context_expansion_ms": t_expand,
            "llm_generation_ms": t_gen,
            "total_request_ms": total_ms,
        }

        # Audit log in MongoDB (graceful if MongoDB is uninitialized)
        citation_dicts = [c.model_dump() for c in citations]
        try:
            await mongo_db.log_chat_interaction(
                query=query_str,
                answer=answer,
                sources=citation_dicts,
                provider=provider_used,
            )
        except Exception as mongo_err:
            logger.warning(f"MongoDB chat interaction logging skipped: {str(mongo_err)}")

        # Record Per-Query Evaluation Event
        eval_record = query_eval_service.record_evaluation(
            query=query_str,
            answer=answer,
            retrieval_mode=mode,
            citations=citation_dicts,
            latency_breakdown=latency_breakdown,
            document_ids=request.document_ids,
        )

        return ChatQueryResponse(
            query=query_str,
            answer=answer,
            sources=citations,
            retrieval_mode=mode,
            llm_provider=provider_used,
            model_name=model_used,
            processing_time_seconds=elapsed,
            evaluation_id=eval_record.evaluation_id,
            latency_breakdown=latency_breakdown,
        )

    except Exception as e:
        logger.exception(f"Error executing RAG query '{query_str}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process RAG query: {str(e)}",
        )
