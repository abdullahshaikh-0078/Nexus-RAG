import re
import math
import logging
from typing import List, Dict, Any, Optional
from app.models.schemas import DocumentChunk, SourceCitation

logger = logging.getLogger(__name__)


def default_tokenize(text: str) -> List[str]:
    """Simple, efficient alphanumeric tokenizer for BM25 indexing."""
    if not text:
        return []
    return re.findall(r"\w+", text.lower())


class RobustBM25:
    """
    Robust BM25Okapi implementation with non-negative IDF guarantees 
    (Lucene/Okapi formulation) ensuring accuracy on small & large corpora.
    """

    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus_tokens)
        self.doc_len = [len(doc) for doc in corpus_tokens]
        self.avg_doc_len = (sum(self.doc_len) / self.corpus_size) if self.corpus_size > 0 else 0.0
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self._calc_idf(corpus_tokens)

    def _calc_idf(self, corpus_tokens: List[List[str]]):
        df: Dict[str, int] = {}
        for doc in corpus_tokens:
            freqs: Dict[str, int] = {}
            for token in doc:
                freqs[token] = freqs.get(token, 0) + 1
            self.doc_freqs.append(freqs)
            for token in freqs:
                df[token] = df.get(token, 0) + 1

        for token, num_docs in df.items():
            # Non-negative Robertson-Spärck Jones / Lucene BM25 IDF formula
            idf_val = math.log(1.0 + (self.corpus_size - num_docs + 0.5) / (num_docs + 0.5))
            self.idf[token] = max(0.0, idf_val)

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.corpus_size
        if self.avg_doc_len == 0:
            return scores

        for token in query_tokens:
            if token not in self.idf:
                continue
            idf = self.idf[token]
            for i, freqs in enumerate(self.doc_freqs):
                tf = freqs.get(token, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_len[i] / self.avg_doc_len))
                scores[i] += idf * (tf * (self.k1 + 1.0)) / denom

        return scores


class BM25IndexService:
    """Independent lexical BM25 retrieval service."""

    def __init__(self):
        self.chunk_payloads: List[Dict[str, Any]] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[RobustBM25] = None

    def index_chunks(self, chunks: List[DocumentChunk], filename: str):
        """Indexes document chunks into the BM25 lexical corpus."""
        if not chunks:
            return

        for chunk in chunks:
            payload = {
                "document_id": chunk.document_id,
                "document_name": filename,
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.text,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
            }
            tokens = default_tokenize(chunk.text)
            self.chunk_payloads.append(payload)
            self.corpus_tokens.append(tokens)

        self._rebuild_index()
        logger.info(f"Indexed {len(chunks)} chunks into BM25 store for '{filename}'. Total corpus size: {len(self.chunk_payloads)}")

    def hydrate_from_vector_store(self, vector_store) -> int:
        """Hydrates in-memory BM25 index from persistent Qdrant vector store chunks on application startup."""
        try:
            vector_store.initialize()
            if not vector_store.client:
                return 0

            scroll_res, _ = vector_store.client.scroll(
                collection_name=vector_store.collection_name,
                limit=10000,
                with_payload=True,
            )
            if not scroll_res:
                return 0

            self.clear()
            for point in scroll_res:
                payload = point.payload or {}
                doc_id = payload.get("document_id", "")
                filename = payload.get("document_name", "unknown")
                chunk_id = payload.get("chunk_id", str(point.id))
                chunk_index = payload.get("chunk_index", 0)
                content = payload.get("content", "")

                if content:
                    p = {
                        "document_id": doc_id,
                        "document_name": filename,
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_index,
                        "content": content,
                        "start_char": payload.get("start_char", 0),
                        "end_char": payload.get("end_char", len(content)),
                    }
                    tokens = default_tokenize(content)
                    self.chunk_payloads.append(p)
                    self.corpus_tokens.append(tokens)

            self._rebuild_index()
            logger.info(f"Hydrated BM25 index with {len(self.chunk_payloads)} chunks from Qdrant storage.")
            return len(self.chunk_payloads)
        except Exception as e:
            logger.warning(f"Failed to hydrate BM25 index from vector store: {str(e)}")
            return 0

    def _rebuild_index(self):
        """Rebuilds RobustBM25 index over corpus tokens."""
        if self.corpus_tokens and len(self.corpus_tokens) > 0:
            self.bm25 = RobustBM25(self.corpus_tokens)
        else:
            self.bm25 = None

    def search(
        self, query: str, top_k: int = 4, document_ids: Optional[List[str]] = None
    ) -> List[SourceCitation]:
        """Searches BM25 lexical index for top K matching chunks."""
        if not self.bm25 or not self.chunk_payloads:
            # Lazy hydration check if backend restarted
            from app.db.vectorstore import vector_store
            self.hydrate_from_vector_store(vector_store)

        if not self.bm25 or not self.chunk_payloads:
            return []

        tokenized_query = default_tokenize(query)
        if not tokenized_query:
            return []

        raw_scores = self.bm25.get_scores(tokenized_query)

        # Filter candidate indices by document_ids if provided
        candidate_indices = []
        for idx, payload in enumerate(self.chunk_payloads):
            if document_ids and payload.get("document_id") not in document_ids:
                continue
            score = float(raw_scores[idx])
            # Only include hits with positive BM25 relevance score
            if score > 0.0:
                candidate_indices.append((idx, score))

        # Sort candidates by score descending
        sorted_candidates = sorted(candidate_indices, key=lambda x: x[1], reverse=True)[:top_k]

        max_score = max([s for _, s in sorted_candidates], default=1.0)
        if max_score <= 0.0:
            max_score = 1.0

        citations = []
        for idx, score in sorted_candidates:
            payload = self.chunk_payloads[idx]
            # Normalize BM25 score relative to top hit for visual scaling
            norm_score = round(score / max_score if max_score > 0 else 0.0, 4)
            citation = SourceCitation(
                document_id=payload.get("document_id", ""),
                document_name=payload.get("document_name", "Unknown"),
                chunk_id=payload.get("chunk_id", ""),
                chunk_index=payload.get("chunk_index", 0),
                score=norm_score,
                content=payload.get("content", ""),
            )
            citations.append(citation)

        return citations

    def delete_document(self, document_id: str) -> bool:
        """Removes all chunks associated with document_id from BM25 index."""
        initial_count = len(self.chunk_payloads)
        new_payloads = []
        new_corpus = []

        for payload, tokens in zip(self.chunk_payloads, self.corpus_tokens):
            if payload.get("document_id") != document_id:
                new_payloads.append(payload)
                new_corpus.append(tokens)

        self.chunk_payloads = new_payloads
        self.corpus_tokens = new_corpus
        self._rebuild_index()

        removed = initial_count - len(self.chunk_payloads)
        logger.info(f"Removed {removed} chunks for document '{document_id}' from BM25 index.")
        return True

    def clear(self):
        """Clears all indexed BM25 payloads."""
        self.chunk_payloads.clear()
        self.corpus_tokens.clear()
        self.bm25 = None


bm25_service = BM25IndexService()
