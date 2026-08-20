import logging
from typing import List, Dict, Any, Optional
from app.models.schemas import SourceCitation
from app.db.vectorstore import vector_store, QdrantVectorStore
from app.services.bm25_search import bm25_service, BM25IndexService
from app.services.embedder import embedding_service

logger = logging.getLogger(__name__)


class V3Retriever:
    """
    Isolated V3 Retrieval Engine.
    Executes Hybrid Reciprocal Rank Fusion (RRF) between V3 Dense Vector Search
    and V3 BM25 Lexical Search, filtered strictly by version="v3" and strategy.
    """

    def __init__(
        self,
        vector_store_service: QdrantVectorStore = vector_store,
        bm25_search_service: BM25IndexService = bm25_service,
    ):
        self.vector_store = vector_store_service
        self.bm25_service = bm25_search_service

    def search(
        self,
        query: str,
        top_k: int = 4,
        document_ids: Optional[List[str]] = None,
        chunking_strategy: str = "table_aware",
        rrf_k: int = 60,
        fetch_k: int = 20,
        chat_id: Optional[str] = None,
    ) -> List[SourceCitation]:
        """
        Executes isolated V3 Hybrid Retrieval for query and returns top K SourceCitation objects.
        """
        if not query or not query.strip():
            return []

        query_str = query.strip()
        logger.info(f"[V3][RETRIEVAL] Query='{query_str}' version=v3 strategy={chunking_strategy} chat_id={chat_id} top_k={top_k}")

        # 1. Retrieve V3 Dense Candidate List
        query_vector = embedding_service.embed_text(query_str)
        dense_citations = self.vector_store.search_similar(
            query_vector=query_vector,
            top_k=fetch_k,
            document_ids=document_ids,
            version="v3",
            chunking_strategy=chunking_strategy,
            chat_id=chat_id,
        )

        # 2. Retrieve V3 BM25 Candidate List
        bm25_citations = self.bm25_service.search(
            query=query_str,
            top_k=fetch_k,
            document_ids=document_ids,
            version="v3",
            chunking_strategy=chunking_strategy,
            chat_id=chat_id,
        )

        # Strict Contamination Guard: No legacy V1/V2 chunks allowed in V3 retrieval pipeline
        for c in dense_citations:
            if getattr(c, "version", "v3") != "v3":
                raise ValueError(f"Legacy chunk contamination detected in V3 Dense search: chunk_id={c.chunk_id}")
        for c in bm25_citations:
            if getattr(c, "version", "v3") != "v3":
                raise ValueError(f"Legacy chunk contamination detected in V3 BM25 search: chunk_id={c.chunk_id}")

        # 3. Reciprocal Rank Fusion (RRF)
        unified_chunks: Dict[str, Dict[str, Any]] = {}

        # Dense Ranks
        for rank, citation in enumerate(dense_citations, start=1):
            cid = citation.chunk_id
            if cid not in unified_chunks:
                unified_chunks[cid] = {
                    "citation": citation,
                    "dense_rank": rank,
                    "bm25_rank": None,
                    "rrf_score": 0.0,
                }
            else:
                unified_chunks[cid]["dense_rank"] = rank

        # BM25 Ranks
        for rank, citation in enumerate(bm25_citations, start=1):
            cid = citation.chunk_id
            if cid not in unified_chunks:
                unified_chunks[cid] = {
                    "citation": citation,
                    "dense_rank": None,
                    "bm25_rank": rank,
                    "rrf_score": 0.0,
                }
            else:
                unified_chunks[cid]["bm25_rank"] = rank

        # Calculate RRF Scores: RRF_score = sum(1 / (k + rank))
        fused_results = []
        for cid, entry in unified_chunks.items():
            rrf_score = 0.0
            if entry["dense_rank"] is not None:
                rrf_score += 1.0 / (rrf_k + entry["dense_rank"])
            if entry["bm25_rank"] is not None:
                rrf_score += 1.0 / (rrf_k + entry["bm25_rank"])

            entry["rrf_score"] = rrf_score

            base_cit: SourceCitation = entry["citation"]
            fused_cit = SourceCitation(
                document_id=base_cit.document_id,
                document_name=base_cit.document_name,
                chunk_id=base_cit.chunk_id,
                chunk_index=base_cit.chunk_index,
                score=round(rrf_score, 6),
                content=base_cit.content,
                dense_rank=entry["dense_rank"],
                bm25_rank=entry["bm25_rank"],
                rrf_score=round(rrf_score, 6),
                page_number=base_cit.page_number,
                section=base_cit.section,
                content_type=base_cit.content_type,
                table_id=base_cit.table_id,
                table_title=base_cit.table_title,
                row_range=base_cit.row_range,
                column_range=base_cit.column_range,
                bbox=base_cit.bbox,
                strategy=chunking_strategy,
                version="v3",
            )
            fused_results.append(fused_cit)

        # Sort by RRF score descending
        fused_results.sort(key=lambda x: x.rrf_score or 0.0, reverse=True)
        final_top_k = fused_results[:top_k]

        logger.info(f"[V3][RETRIEVAL] Found {len(final_top_k)} V3 RRF hybrid results (top score={final_top_k[0].rrf_score if final_top_k else 0.0})")
        return final_top_k


v3_retriever = V3Retriever()
