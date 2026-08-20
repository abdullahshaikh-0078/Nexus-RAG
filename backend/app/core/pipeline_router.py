import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

from app.services.embedder import embedding_service
from app.services.bm25_search import bm25_service
from app.services.hybrid_retriever import hybrid_retriever
from app.db.vectorstore import vector_store
from app.v3.parsing.structural_parser import v3_structural_parser
from app.v3.chunking.engine import v3_chunking_engine
from app.v3.schemas.chunk_schema import ChunkingConfig, V3Chunk
from app.v3.query_expansion.expander import v3_query_expander
from app.v3.query_expansion.rewriter import v3_query_rewriter
from app.models.schemas import SourceCitation
from app.services.bm25_search import BM25IndexService, DocumentChunk

from app.v3.reasoning.reasoning_engine import v3_reasoning_engine
from app.v3.reasoning.schemas import CalculationResult

logger = logging.getLogger(__name__)

Tuple_Route_Result = Tuple[List[SourceCitation], Dict[str, float], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]


class VersionInfo(BaseModel):
    version_id: str
    display_name: str
    description: str
    status: str
    pipeline_type: str


class PipelineRouter:
    """
    Central Version Registry & Pipeline Router.
    Routes RAG requests cleanly to V1 (Dense), V2.1 (BM25), V2.2 (Hybrid), or V3 (Structural RAG)
    via thin non-invasive adapters.
    """

    def __init__(self):
        self._versions: Dict[str, VersionInfo] = {
            "v1": VersionInfo(
                version_id="v1",
                display_name="V1 — Dense Retrieval",
                description="Dense Vector Similarity Retrieval using Qdrant (Frozen Baseline)",
                status="frozen_baseline",
                pipeline_type="dense",
            ),
            "v2.1": VersionInfo(
                version_id="v2.1",
                display_name="V2.1 — BM25",
                description="Lexical BM25 Search Engine",
                status="production",
                pipeline_type="lexical",
            ),
            "v2.2": VersionInfo(
                version_id="v2.2",
                display_name="V2.2 — Hybrid Retrieval",
                description="Unified Dense + BM25 Retrieval fused via Reciprocal Rank Fusion (RRF)",
                status="production",
                pipeline_type="hybrid_rrf",
            ),
            "v3": VersionInfo(
                version_id="v3",
                display_name="V3 — Structural RAG",
                description="Layout-aware Structural PDF Parsing, Multi-Strategy Chunking, Query Expansion, and Financial Reasoning",
                status="experimental",
                pipeline_type="structural_rag",
            ),
        }

    def list_versions(self) -> List[VersionInfo]:
        return list(self._versions.values())

    def get_version(self, version_id: str) -> Optional[VersionInfo]:
        v_key = version_id.lower().strip()
        if v_key == "dense":
            v_key = "v1"
        elif v_key == "bm25":
            v_key = "v2.1"
        elif v_key == "hybrid":
            v_key = "v2.2"
        return self._versions.get(v_key)

    def route_query(
        self,
        query: str,
        top_k: int = 4,
        document_ids: Optional[List[str]] = None,
        version: str = "v2.2",
        chunking_strategy: str = "table_aware",
        chat_id: Optional[str] = None,
    ) -> Tuple_Route_Result:
        v_info = self.get_version(version)
        if not v_info:
            raise ValueError(
                f"Unknown system version '{version}'. Available versions: {[v.version_id for v in self.list_versions()]}"
            )

        v_id = v_info.version_id
        t0 = time.time()

        if v_id == "v1":
            t0_emb = time.time()
            query_vec = embedding_service.embed_text(query)
            t_emb = round((time.time() - t0_emb) * 1000, 2)

            t0_srch = time.time()
            citations = vector_store.search_similar(
                query_vector=query_vec,
                top_k=top_k,
                document_ids=document_ids,
                chat_id=chat_id,
            )
            t_srch = round((time.time() - t0_srch) * 1000, 2)

            breakdown = {
                "embedding_ms": t_emb,
                "dense_search_ms": t_srch,
                "total_request_ms": round((time.time() - t0) * 1000, 2),
            }
            return citations, breakdown, None, None

        elif v_id == "v2.1":
            t0_bm25 = time.time()
            citations = bm25_service.search(
                query=query,
                top_k=top_k,
                document_ids=document_ids,
                chat_id=chat_id,
            )
            t_bm25 = round((time.time() - t0_bm25) * 1000, 2)

            breakdown = {
                "bm25_search_ms": t_bm25,
                "total_request_ms": round((time.time() - t0) * 1000, 2),
            }
            return citations, breakdown, None, None

        elif v_id == "v2.2":
            t0_h = time.time()
            raw_cits = hybrid_retriever.search(
                query=query,
                top_k=top_k,
                document_ids=document_ids,
                chat_id=chat_id,
            )
            t_h = round((time.time() - t0_h) * 1000, 2)
            breakdown = {
                "rrf_fusion_ms": t_h,
                "total_request_ms": round((time.time() - t0) * 1000, 2),
            }
            return raw_cits, breakdown, None, None

        elif v_id == "v3":
            from app.v3.ingestion.ingestion_service import v3_ingestion_service
            from app.v3.ingestion.v3_chunking_policy import v3_chunking_policy

            resolved_strategy = v3_chunking_policy.select_strategy(
                requested_strategy=chunking_strategy
            )

            if document_ids:
                for doc_id in document_ids:
                    v3_ingestion_service.ensure_v3_indexed([doc_id], strategy=resolved_strategy, chat_id=chat_id)

            raw_cits, breakdown, exp_meta, calc_result = v3_reasoning_engine.process_query(
                query=query,
                top_k=top_k,
                document_ids=document_ids,
                chunking_strategy=resolved_strategy,
                chat_id=chat_id,
            )
            calc_dict = calc_result.model_dump() if calc_result else None
            return raw_cits, breakdown, exp_meta, calc_dict

        else:
            raise ValueError(f"Unhandled pipeline version '{v_id}'")


pipeline_router = PipelineRouter()
