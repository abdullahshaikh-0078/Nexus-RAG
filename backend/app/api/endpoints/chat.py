import time
import logging
from fastapi import APIRouter, HTTPException, status
from app.services.embedder import embedding_service
from app.services.bm25_search import bm25_service
from app.db.vectorstore import vector_store
from app.db.mongodb import mongo_db
from app.services.llm_service import llm_service
from app.models.schemas import ChatQueryRequest, ChatQueryResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/query", response_model=ChatQueryResponse)
async def query_rag(request: ChatQueryRequest):
    """
    RAG Query Flow:
    1. Check retrieval_mode ('dense' or 'bm25').
    2. Retrieve top-k context chunks from Qdrant (dense) or BM25 index (bm25).
    3. Expand adjacent chunk window for contiguous context.
    4. Send retrieved context + query to LLM synthesizer.
    5. Log transaction and return answer with source citations & retrieval mode.
    """
    start_time = time.time()
    query_str = request.query.strip()
    if not query_str:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty.",
        )

    mode = (request.retrieval_mode or "dense").lower()

    try:
        if mode == "bm25":
            # 2. Retrieve top-k context chunks from BM25 index
            raw_citations = bm25_service.search(
                query=query_str,
                top_k=request.top_k,
                document_ids=request.document_ids,
            )
        else:
            # 1. Embed query into dense vector
            query_vector = embedding_service.embed_text(query_str)
            # 2. Retrieve top-k context chunks from Qdrant
            raw_citations = vector_store.search_similar(
                query_vector=query_vector,
                top_k=request.top_k,
                document_ids=request.document_ids,
            )

        # 3. Expand adjacent chunk window for contiguous context
        citations = vector_store.expand_adjacent_context(raw_citations, window=1)

        # 4. LLM synthesis
        answer, provider_used, model_used = llm_service.generate_answer(
            query=query_str, citations=citations
        )

        elapsed = round(time.time() - start_time, 3)

        # 5. Audit log in MongoDB
        citation_dicts = [c.model_dump() for c in citations]
        await mongo_db.log_chat_interaction(
            query=query_str,
            answer=answer,
            sources=citation_dicts,
            provider=provider_used,
        )

        return ChatQueryResponse(
            query=query_str,
            answer=answer,
            sources=citations,
            retrieval_mode=mode,
            llm_provider=provider_used,
            model_name=model_used,
            processing_time_seconds=elapsed,
        )

    except Exception as e:
        logger.exception(f"Error executing RAG query '{query_str}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process RAG query: {str(e)}",
        )
