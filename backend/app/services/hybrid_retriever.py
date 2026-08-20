import logging
from typing import List, Dict, Any, Optional
from app.models.schemas import SourceCitation
from app.db.vectorstore import vector_store, QdrantVectorStore
from app.services.bm25_search import bm25_service, BM25IndexService
from app.services.embedder import embedding_service

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Orchestrates Dense Vector Retrieval and BM25 Lexical Retrieval 
    using Reciprocal Rank Fusion (RRF) for unified, score-scale-invariant ranking.
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
        rrf_k: int = 60,
        fetch_k: int = 20,
        chat_id: Optional[str] = None,
    ) -> List[SourceCitation]:
        """
        Executes Dense + BM25 retrieval, performs Reciprocal Rank Fusion (RRF),
        deduplicates chunks, and returns top K unified SourceCitations.
        """
        if not query or not query.strip():
            return []

        query_str = query.strip()

        # 1. Retrieve Dense candidate list (fetch_k)
        query_vector = embedding_service.embed_text(query_str)
        dense_citations = self.vector_store.search_similar(
            query_vector=query_vector,
            top_k=fetch_k,
            document_ids=document_ids,
            chat_id=chat_id,
        )

        # 2. Retrieve BM25 candidate list (fetch_k)
        bm25_citations = self.bm25_service.search(
            query=query_str,
            top_k=fetch_k,
            document_ids=document_ids,
            chat_id=chat_id,
        )

        # 3. Reciprocal Rank Fusion (RRF) & Chunk Deduplication
        unified_chunks: Dict[str, Dict[str, Any]] = {}

        # Process Dense candidates (1-based rank)
        for rank, citation in enumerate(dense_citations, 1):
            key = f"{citation.document_id}_{citation.chunk_index}"
            if key not in unified_chunks:
                unified_chunks[key] = {
                    "citation": citation,
                    "dense_rank": rank,
                    "bm25_rank": None,
                    "dense_score": citation.score,
                    "bm25_score": 0.0,
                    "rrf_score": 0.0,
                }
            else:
                unified_chunks[key]["dense_rank"] = rank
                unified_chunks[key]["dense_score"] = citation.score

        # Process BM25 candidates (1-based rank)
        for rank, citation in enumerate(bm25_citations, 1):
            key = f"{citation.document_id}_{citation.chunk_index}"
            if key not in unified_chunks:
                unified_chunks[key] = {
                    "citation": citation,
                    "dense_rank": None,
                    "bm25_rank": rank,
                    "dense_score": 0.0,
                    "bm25_score": citation.score,
                    "rrf_score": 0.0,
                }
            else:
                unified_chunks[key]["bm25_rank"] = rank
                unified_chunks[key]["bm25_score"] = citation.score

        # Calculate RRF score for each chunk: RRF(d) = Σ 1 / (k + rank_m(d))
        for item in unified_chunks.values():
            rrf_score = 0.0
            if item["dense_rank"] is not None:
                rrf_score += 1.0 / (rrf_k + item["dense_rank"])
            if item["bm25_rank"] is not None:
                rrf_score += 1.0 / (rrf_k + item["bm25_rank"])
            item["rrf_score"] = rrf_score

        # 4. Sort candidates by RRF score descending
        sorted_items = sorted(
            unified_chunks.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )[:top_k]

        # 5. Build final SourceCitation list
        results: List[SourceCitation] = []
        for item in sorted_items:
            orig_cite: SourceCitation = item["citation"]
            # Create citation object with unified RRF score
            res_citation = SourceCitation(
                document_id=orig_cite.document_id,
                document_name=orig_cite.document_name,
                chunk_id=orig_cite.chunk_id,
                chunk_index=orig_cite.chunk_index,
                score=round(item["rrf_score"], 6),
                content=orig_cite.content,
                dense_rank=item["dense_rank"],
                bm25_rank=item["bm25_rank"],
                rrf_score=round(item["rrf_score"], 6),
            )
            results.append(res_citation)

        logger.info(
            f"Hybrid RRF retrieval returned {len(results)} fused chunks for query '{query_str[:30]}...' "
            f"(Dense candidates: {len(dense_citations)}, BM25 candidates: {len(bm25_citations)})"
        )

        return results


hybrid_retriever = HybridRetriever()
