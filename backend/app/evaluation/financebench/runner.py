import os
import json
import time
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from app.core.config import settings
from app.services.document_parser import UnifiedDocumentParser
from app.services.chunker import RecursiveTextChunker
from app.services.embedder import embedding_service
from app.services.bm25_search import bm25_service
from app.services.hybrid_retriever import hybrid_retriever
from app.db.vectorstore import vector_store
from app.evaluation.metrics import RetrievalEvaluator, QuestionEvalResult, EvaluationRunResult
from app.evaluation.financebench.loader import FinanceBenchLoader, FinanceBenchQuestion

logger = logging.getLogger(__name__)

FINANCEBENCH_RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "results", "financebench"
)


class FinanceBenchRunner:
    """Executes isolated RAG retrieval evaluation on real-world FinanceBench SEC 10-K PDFs."""

    def __init__(
        self,
        loader: Optional[FinanceBenchLoader] = None,
        pdf_dir: Optional[str] = None,
    ):
        self.loader = loader or FinanceBenchLoader()
        self.pdf_dir = pdf_dir or settings.FINANCEBENCH_PDF_DIR
        os.makedirs(FINANCEBENCH_RESULTS_DIR, exist_ok=True)

    def resolve_pdf_path(self, doc_name: str) -> str:
        """Resolves absolute path to PDF file given doc_name."""
        pdf_filename = f"{doc_name}.pdf" if not doc_name.lower().endswith(".pdf") else doc_name
        pdf_path = os.path.join(self.pdf_dir, pdf_filename)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"FinanceBench PDF not found for doc_name '{doc_name}' at {pdf_path}")
        return pdf_path

    def ingest_document(self, doc_name: str) -> str:
        """Parses, chunks, embeds, and indexes a single FinanceBench PDF on demand."""
        pdf_path = self.resolve_pdf_path(doc_name)
        doc_id = f"doc_financebench_{doc_name}"

        logger.info(f"Ingesting FinanceBench document '{doc_name}' from {pdf_path}...")

        with open(pdf_path, "rb") as f:
            file_bytes = f.read()

        # Parse PDF text using UnifiedDocumentParser
        text, _ = UnifiedDocumentParser.extract_text(file_bytes, filename=os.path.basename(pdf_path))

        # Chunk document
        chunker = RecursiveTextChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        chunks = chunker.chunk_document(text, document_id=doc_id)
        chunk_texts = [c.text for c in chunks]

        # Generate embeddings
        embeddings = embedding_service.embed_batch(chunk_texts)

        # Upsert into Qdrant vector store and BM25 index
        filename = f"{doc_name}.pdf"
        vector_store.upsert_chunks(chunks=chunks, embeddings=embeddings, filename=filename)
        bm25_service.index_chunks(chunks=chunks, filename=filename)

        logger.info(f"Successfully ingested '{doc_name}': {len(chunks)} chunks, {len(text)} characters.")
        return doc_id

    def run_document_evaluation(
        self, doc_name: str, top_k: int = 10, retrieval_mode: str = "hybrid"
    ) -> EvaluationRunResult:
        """
        Runs RAG retrieval evaluation for all FinanceBench questions associated
        with a single specified SEC 10-K PDF document.
        """
        mode = retrieval_mode.lower()
        questions = self.loader.get_questions_by_doc_name(doc_name)

        if not questions:
            raise ValueError(f"No FinanceBench questions found in dataset for doc_name '{doc_name}'")

        # Ensure target document is ingested into vector store & BM25 index
        doc_id = self.ingest_document(doc_name)

        question_results: List[QuestionEvalResult] = []
        latencies: List[float] = []

        logger.info(
            f"Executing FinanceBench evaluation for '{doc_name}' "
            f"({len(questions)} questions, mode: '{mode}')..."
        )

        for q in questions:
            q_id = q.financebench_id
            question_str = q.question
            category = q.question_type or "financebench"

            # Gather expected ground-truth evidence text snippets
            expected_snippets = []
            for ev in q.evidence:
                if ev.evidence_text:
                    expected_snippets.append(ev.evidence_text)
                if ev.evidence_text_full_page:
                    expected_snippets.append(ev.evidence_text_full_page[:300])

            if not expected_snippets:
                # Fallback to golden answer text if evidence snippet is empty
                expected_snippets = [q.answer]

            # Measure isolated retrieval latency
            start_time = time.time()
            if mode == "hybrid":
                raw_citations = hybrid_retriever.search(
                    query=question_str,
                    top_k=top_k,
                    document_ids=[doc_id],
                )
            elif mode == "bm25":
                raw_citations = bm25_service.search(
                    query=question_str,
                    top_k=top_k,
                    document_ids=[doc_id],
                )
            else:
                query_vector = embedding_service.embed_text(question_str)
                raw_citations = vector_store.search_similar(
                    query_vector=query_vector,
                    top_k=top_k,
                    document_ids=[doc_id],
                )

            citations = vector_store.expand_adjacent_context(raw_citations, window=1)
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            latencies.append(elapsed_ms)

            retrieved_texts = [c.content for c in citations]
            retrieved_ids = [c.chunk_id for c in citations]

            # Compute evaluation metrics
            r1 = RetrievalEvaluator.calculate_recall_at_k(retrieved_texts, expected_snippets, k=1)
            r3 = RetrievalEvaluator.calculate_recall_at_k(retrieved_texts, expected_snippets, k=3)
            r5 = RetrievalEvaluator.calculate_recall_at_k(retrieved_texts, expected_snippets, k=5)
            r10 = RetrievalEvaluator.calculate_recall_at_k(retrieved_texts, expected_snippets, k=10)
            first_rank = RetrievalEvaluator.calculate_first_relevant_rank(retrieved_texts, expected_snippets, max_k=10)
            mrr10 = RetrievalEvaluator.calculate_mrr_at_k(retrieved_texts, expected_snippets, k=10)
            ndcg10 = RetrievalEvaluator.calculate_ndcg_at_k(retrieved_texts, expected_snippets, k=10)

            q_res = QuestionEvalResult(
                question_id=q_id,
                question=question_str,
                category=category,
                first_relevant_rank=first_rank,
                recall_at_1=r1,
                recall_at_3=r3,
                recall_at_5=r5,
                recall_at_10=r10,
                mrr_at_10=mrr10,
                ndcg_at_10=ndcg10,
                retrieval_latency_ms=elapsed_ms,
                retrieved_chunk_ids=retrieved_ids,
                retrieved_snippets=[t[:120] for t in retrieved_texts],
            )
            question_results.append(q_res)

        total_q = len(question_results)
        agg_r1 = round(sum(q.recall_at_1 for q in question_results) / total_q, 4) if total_q else 0.0
        agg_r3 = round(sum(q.recall_at_3 for q in question_results) / total_q, 4) if total_q else 0.0
        agg_r5 = round(sum(q.recall_at_5 for q in question_results) / total_q, 4) if total_q else 0.0
        agg_r10 = round(sum(q.recall_at_10 for q in question_results) / total_q, 4) if total_q else 0.0
        agg_mrr = round(sum(q.mrr_at_10 for q in question_results) / total_q, 4) if total_q else 0.0
        agg_ndcg = round(sum(q.ndcg_at_10 for q in question_results) / total_q, 4) if total_q else 0.0
        avg_latency = round(sum(latencies) / total_q, 2) if total_q else 0.0

        now_iso = datetime.now(timezone.utc).isoformat()
        run_result = EvaluationRunResult(
            dataset_version="financebench_open_source",
            evaluation_version=f"financebench_{doc_name}",
            retrieval_mode=mode,
            timestamp=now_iso,
            embedding_model="N/A (BM25 Lexical Search)" if mode == "bm25" else settings.EMBEDDING_MODEL_NAME,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            retrieval_top_k=top_k,
            total_questions=total_q,
            aggregate_recall_at_1=agg_r1,
            aggregate_recall_at_3=agg_r3,
            aggregate_recall_at_5=agg_r5,
            aggregate_recall_at_10=agg_r10,
            aggregate_mrr_at_10=agg_mrr,
            aggregate_ndcg_at_10=agg_ndcg,
            average_retrieval_latency_ms=avg_latency,
            question_results=question_results,
        )

        # Isolated Persistence in results/financebench/
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        versioned_file = os.path.join(
            FINANCEBENCH_RESULTS_DIR, f"financebench_{doc_name}_{mode}_{timestamp_str}.json"
        )
        latest_file = os.path.join(
            FINANCEBENCH_RESULTS_DIR, f"financebench_{doc_name}_{mode}_latest.json"
        )

        result_dict = run_result.model_dump()
        with open(versioned_file, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2)
        with open(latest_file, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2)

        logger.info(
            f"FinanceBench evaluation for '{doc_name}' complete! Saved {versioned_file} and {latest_file}."
        )
        return run_result
