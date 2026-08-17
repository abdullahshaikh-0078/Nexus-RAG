import uuid
import os
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from app.core.config import settings
from app.models.schemas import DocumentChunk, SourceCitation

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """Manages Qdrant vector database connection, collections, search, and context expansion."""

    def __init__(self):
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.dimension = settings.EMBEDDING_DIMENSION
        self.client: Optional[QdrantClient] = None

    def initialize(self):
        """Initializes client and creates collection if absent."""
        if self.client is not None:
            return

        if settings.QDRANT_URL:
            logger.info(f"Connecting to Qdrant cluster at {settings.QDRANT_URL}")
            self.client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
            )
        else:
            storage_path = os.path.abspath(settings.QDRANT_STORAGE_PATH)
            os.makedirs(storage_path, exist_ok=True)
            logger.info(f"Initializing local disk Qdrant store at {storage_path}")
            self.client = QdrantClient(path=storage_path)

        self._ensure_collection()

    def _ensure_collection(self):
        """Ensures vector collection exists with cosine distance metric."""
        if not self.client:
            return

        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            logger.info(f"Creating Qdrant collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.dimension,
                    distance=qmodels.Distance.COSINE,
                ),
            )

        self._ensure_payload_indices()

    def _ensure_payload_indices(self):
        """Ensures required payload field indices (document_id KEYWORD, chunk_index INTEGER) exist."""
        if not self.client:
            return

        try:
            logger.info(f"Ensuring KEYWORD payload index for 'document_id' in collection '{self.collection_name}'")
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="document_id",
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
        except Exception as e:
            logger.debug(f"Note on document_id payload index: {str(e)}")

        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="chunk_index",
                field_schema=qmodels.PayloadSchemaType.INTEGER,
            )
        except Exception as e:
            logger.debug(f"Note on chunk_index payload index: {str(e)}")

    def upsert_chunks(
        self, chunks: List[DocumentChunk], embeddings: List[List[float]], filename: str
    ) -> bool:
        """Stores chunks and vectors into Qdrant."""
        self.initialize()
        if not chunks or not self.client:
            return False

        points = []
        for chunk, vector in zip(chunks, embeddings):
            payload = {
                "document_id": chunk.document_id,
                "document_name": filename,
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.text,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
            }
            point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            points.append(
                qmodels.PointStruct(
                    id=point_uuid,
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        return True

    def search_similar(
        self, query_vector: List[float], top_k: int = 4, document_ids: Optional[List[str]] = None
    ) -> List[SourceCitation]:
        """Searches nearest neighbor chunks for a query vector."""
        self.initialize()
        if not self.client:
            return []

        query_filter = None
        if document_ids:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="document_id",
                        match=qmodels.MatchAny(any=document_ids),
                    )
                ]
            )

        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=query_filter,
            )
            search_results = response.points
        elif hasattr(self.client, "search"):
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=query_filter,
            )
        else:
            search_results = []

        citations = []
        for hit in search_results:
            payload = getattr(hit, "payload", {}) or {}
            citation = SourceCitation(
                document_id=payload.get("document_id", ""),
                document_name=payload.get("document_name", "Unknown"),
                chunk_id=payload.get("chunk_id", str(getattr(hit, "id", ""))),
                chunk_index=payload.get("chunk_index", 0),
                score=round(float(getattr(hit, "score", 0.0)), 4),
                content=payload.get("content", ""),
            )
            citations.append(citation)

        return citations

    def get_chunks_by_indices(
        self, document_id: str, chunk_indices: List[int]
    ) -> List[SourceCitation]:
        """Fetches specific chunk indices for a document from Qdrant."""
        self.initialize()
        if not self.client or not chunk_indices:
            return []

        filter_condition = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="document_id",
                    match=qmodels.MatchValue(value=document_id),
                ),
                qmodels.FieldCondition(
                    key="chunk_index",
                    match=qmodels.MatchAny(any=chunk_indices),
                ),
            ]
        )

        try:
            scroll_res, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_condition,
                limit=len(chunk_indices) * 2,
                with_payload=True,
            )
            citations = []
            for point in scroll_res:
                payload = point.payload or {}
                citations.append(
                    SourceCitation(
                        document_id=payload.get("document_id", ""),
                        document_name=payload.get("document_name", "Unknown"),
                        chunk_id=payload.get("chunk_id", str(point.id)),
                        chunk_index=payload.get("chunk_index", 0),
                        score=0.0,
                        content=payload.get("content", ""),
                    )
                )
            return citations
        except Exception as e:
            logger.warning(f"Failed to fetch neighbor chunks: {str(e)}")
            return []

    def expand_adjacent_context(
        self, citations: List[SourceCitation], window: int = 1
    ) -> List[SourceCitation]:
        """
        Expands retrieved top-K citations by including adjacent neighbor chunks 
        from the same document, merging contiguous chunks into complete context blocks.
        """
        if not citations:
            return []

        # Collect needed neighbor indices per document
        docs_indices: Dict[str, set] = {}
        citation_scores: Dict[str, float] = {}

        for cite in citations:
            doc_id = cite.document_id
            if doc_id not in docs_indices:
                docs_indices[doc_id] = set()
            
            # Store primary score
            key = f"{doc_id}_{cite.chunk_index}"
            citation_scores[key] = cite.score

            # Add primary index + neighbors (e.g. index+1, index-1)
            for idx in range(max(0, cite.chunk_index - window), cite.chunk_index + window + 1):
                docs_indices[doc_id].add(idx)

        # Fetch neighbor chunks
        all_citations: List[SourceCitation] = []
        for doc_id, indices in docs_indices.items():
            fetched = self.get_chunks_by_indices(doc_id, list(indices))
            all_citations.extend(fetched)

        # Map and attach scores
        for c in all_citations:
            key = f"{c.document_id}_{c.chunk_index}"
            if key in citation_scores:
                c.score = citation_scores[key]
            else:
                # Assign neighbor a score slightly below parent
                parent_scores = [
                    cite.score for cite in citations if cite.document_id == c.document_id
                ]
                c.score = max(parent_scores, default=0.5)

        # Group by document and merge contiguous chunk indices
        merged_citations = self._merge_contiguous_chunks(all_citations)
        return merged_citations

    def _merge_contiguous_chunks(
        self, citations: List[SourceCitation]
    ) -> List[SourceCitation]:
        """Merges contiguous chunks (e.g. index 30 & 31) into unified citation blocks."""
        if not citations:
            return []

        # Group by document_id
        doc_groups: Dict[str, List[SourceCitation]] = {}
        for c in citations:
            doc_groups.setdefault(c.document_id, []).append(c)

        merged_results: List[SourceCitation] = []

        for doc_id, group in doc_groups.items():
            sorted_group = sorted(group, key=lambda x: x.chunk_index)
            
            curr_chunk = sorted_group[0]
            curr_text = curr_chunk.content
            curr_indices = [curr_chunk.chunk_index]
            max_score = curr_chunk.score

            for next_chunk in sorted_group[1:]:
                if next_chunk.chunk_index in curr_indices:
                    continue
                # If contiguous (chunk_index == max(curr_indices) + 1)
                if next_chunk.chunk_index == max(curr_indices) + 1:
                    # Cleanly combine texts
                    next_text = next_chunk.content
                    overlap_pos = self._find_overlap(curr_text, next_text)
                    if overlap_pos > 0:
                        curr_text = curr_text + next_text[overlap_pos:]
                    else:
                        curr_text = curr_text + "\n\n" + next_text

                    curr_indices.append(next_chunk.chunk_index)
                    max_score = max(max_score, next_chunk.score)
                else:
                    # Append completed group
                    merged_results.append(
                        SourceCitation(
                            document_id=doc_id,
                            document_name=curr_chunk.document_name,
                            chunk_id=f"{doc_id}_merged_{curr_indices[0]}_{curr_indices[-1]}",
                            chunk_index=curr_indices[0],
                            score=max_score,
                            content=curr_text,
                        )
                    )
                    curr_chunk = next_chunk
                    curr_text = next_chunk.content
                    curr_indices = [next_chunk.chunk_index]
                    max_score = next_chunk.score

            merged_results.append(
                SourceCitation(
                    document_id=doc_id,
                    document_name=curr_chunk.document_name,
                    chunk_id=f"{doc_id}_merged_{curr_indices[0]}_{curr_indices[-1]}",
                    chunk_index=curr_indices[0],
                    score=max_score,
                    content=curr_text,
                )
            )

        # Sort merged citations by score descending
        return sorted(merged_results, key=lambda x: x.score, reverse=True)

    @staticmethod
    def _find_overlap(text1: str, text2: str) -> int:
        """Finds overlapping character suffix of text1 and prefix of text2."""
        max_search = min(len(text1), len(text2), 200)
        for i in range(max_search, 10, -1):
            suffix = text1[-i:]
            if text2.startswith(suffix):
                return len(suffix)
        return 0

    def delete_document_chunks(self, document_id: str) -> bool:
        """Deletes all chunks associated with a document_id."""
        self.initialize()
        if not self.client:
            return False

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="document_id",
                            match=qmodels.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )
        return True

    def check_health(self) -> Dict[str, Any]:
        """Returns Qdrant connection status."""
        try:
            self.initialize()
            if not self.client:
                return {"status": "unhealthy", "details": "Client not initialized"}
            collections = self.client.get_collections().collections
            return {
                "status": "healthy",
                "details": f"Connected. Collections count: {len(collections)}",
            }
        except Exception as e:
            return {"status": "degraded", "details": f"Qdrant error: {str(e)}"}


vector_store = QdrantVectorStore()
