import os
import json
import time
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any
from app.core.config import settings
from app.services.chunker import RecursiveTextChunker
from app.services.embedder import embedding_service
from app.services.bm25_search import bm25_service
from app.db.vectorstore import vector_store
from app.evaluation.metrics import (
    RetrievalEvaluator,
    QuestionEvalResult,
    EvaluationRunResult,
)

logger = logging.getLogger(__name__)

DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "datasets", "v1_baseline.json"
)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


class EvaluationRunner:
    """Executes evaluation runs against the V1 Dense Retrieval or V2.1 BM25 Lexical engine."""

    def __init__(self, dataset_path: str = DATASET_PATH):
        self.dataset_path = dataset_path
        os.makedirs(RESULTS_DIR, exist_ok=True)

    def load_dataset(self) -> Dict[str, Any]:
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Evaluation dataset not found at {self.dataset_path}")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_evaluation(self, top_k: int = 10, retrieval_mode: str = "dense") -> EvaluationRunResult:
        mode = retrieval_mode.lower()
        version_name = "v2_1_bm25" if mode == "bm25" else "v1_baseline"
        logger.info(f"Starting NEXUS RAG Evaluation Run (Mode: '{mode}', Version: '{version_name}')...")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        dataset = self.load_dataset()
        test_cases = dataset.get("test_cases", [])

        if not test_cases:
            raise ValueError("Evaluation dataset contains no test cases.")

        # Ensure document is ingested into vector store and BM25 index
        doc_id = self._ensure_test_document_ingested()

        question_results: List[QuestionEvalResult] = []
        latencies: List[float] = []

        for case in test_cases:
            q_id = case["question_id"]
            question = case["question"]
            category = case["category"]
            expected_snippets = case["expected_snippets"]

            # Measure isolated retrieval latency
            start_time = time.time()
            if mode == "bm25":
                raw_citations = bm25_service.search(
                    query=question,
                    top_k=top_k,
                    document_ids=[doc_id] if doc_id else None,
                )
            else:
                query_vector = embedding_service.embed_text(question)
                raw_citations = vector_store.search_similar(
                    query_vector=query_vector,
                    top_k=top_k,
                    document_ids=[doc_id] if doc_id else None,
                )
            citations = vector_store.expand_adjacent_context(raw_citations, window=1)
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            latencies.append(elapsed_ms)

            retrieved_texts = [c.content for c in citations]
            retrieved_ids = [c.chunk_id for c in citations]

            # Calculate metrics
            r1 = RetrievalEvaluator.calculate_recall_at_k(retrieved_texts, expected_snippets, k=1)
            r3 = RetrievalEvaluator.calculate_recall_at_k(retrieved_texts, expected_snippets, k=3)
            r5 = RetrievalEvaluator.calculate_recall_at_k(retrieved_texts, expected_snippets, k=5)
            r10 = RetrievalEvaluator.calculate_recall_at_k(retrieved_texts, expected_snippets, k=10)
            first_rank = RetrievalEvaluator.calculate_first_relevant_rank(retrieved_texts, expected_snippets, max_k=10)
            mrr10 = RetrievalEvaluator.calculate_mrr_at_k(retrieved_texts, expected_snippets, k=10)
            ndcg10 = RetrievalEvaluator.calculate_ndcg_at_k(retrieved_texts, expected_snippets, k=10)

            q_res = QuestionEvalResult(
                question_id=q_id,
                question=question,
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
            dataset_version=dataset.get("version", "v1_baseline"),
            evaluation_version=version_name,
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

        # Atomic Result Persistence Strategy
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        versioned_file = os.path.join(RESULTS_DIR, f"{version_name}_{timestamp_str}.json")
        latest_file_name = "v2_1_bm25_latest.json" if mode == "bm25" else "latest.json"
        latest_file = os.path.join(RESULTS_DIR, latest_file_name)
        temp_latest_file = os.path.join(RESULTS_DIR, f"{latest_file_name}.tmp")

        result_dict = run_result.model_dump()

        # 1. Write timestamped versioned result file
        with open(versioned_file, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2)

        # 2. Write latest.json atomically with schema validation
        try:
            with open(temp_latest_file, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, indent=2)

            with open(temp_latest_file, "r", encoding="utf-8") as f:
                validation_data = json.load(f)
            EvaluationRunResult(**validation_data)

            os.replace(temp_latest_file, latest_file)
            logger.info(f"Evaluation run complete! Saved {versioned_file} and updated {latest_file} atomically.")
        except Exception as e:
            logger.error(f"Atomic update of latest.json failed: {str(e)}. Preserving previous latest.json.")
            if os.path.exists(temp_latest_file):
                os.remove(temp_latest_file)
            raise e

        return run_result

    def _ensure_test_document_ingested(self) -> str:
        """Ingests Attention Is All You Need paper text for evaluation runner."""
        paper_text = (
            "Attention Is All You Need\n"
            "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin\n\n"
            "Abstract\n"
            "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks "
            "that include an encoder and a decoder. The best performing models also connect the encoder and decoder "
            "through an attention mechanism. We propose a new simple network architecture, the Transformer, "
            "based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. "
            "Experiments on two machine translation tasks show these models to be superior in quality while being "
            "more parallelizable and requiring significantly less time to train.\n\n"
            "1 Introduction\n"
            "Recurrent neural networks, particularly long short-term memory and gated recurrent neural networks, "
            "have been firmly established as state of the art approaches in sequence modeling and transduction problems. "
            "In this work we offer the Transformer, a model architecture eschewing recurrence and instead relying entirely "
            "on an attention mechanism to draw global dependencies between input and output.\n\n"
            "3.2.3 Applications of Attention in our Model\n"
            "The Transformer uses multi-head attention in three different ways:\n"
            "1. In encoder-decoder attention layers, the queries come from the previous decoder layer, and the memory keys "
            "and values come from the output of the encoder. This allows every position in the decoder to attend over all "
            "positions in the input sequence.\n"
            "2. The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries "
            "come from the same place, in this case, the output of the previous layer in the encoder.\n"
            "3. Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions "
            "in the decoder up to and including that position.\n\n"
            "3.2.2 Multi-Head Attention\n"
            "Multi-head attention allows the model to jointly attend to information from different representation subspaces "
            "at different positions. With a single attention head, averaging inhibits this.\n\n"
            "3.5 Positional Encoding\n"
            "Since our model contains no recurrence and no convolution, in order for the model to make use of the order "
            "of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence.\n\n"
            "5.4 Maximum Path Lengths\n"
            "A self-attention layer connects all positions with a constant number of sequentially executed operations, "
            "whereas a recurrent layer requires O(n) sequential operations. In terms of computational complexity, self-attention "
            "layers are faster than recurrent layers when the sequence length n is smaller than the representation dimensionality d."
        )

        doc_id = "doc_eval_attention_paper"
        chunker = RecursiveTextChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        chunks = chunker.chunk_document(paper_text, document_id=doc_id)
        chunk_texts = [c.text for c in chunks]
        embeddings = embedding_service.embed_batch(chunk_texts)

        vector_store.upsert_chunks(
            chunks=chunks,
            embeddings=embeddings,
            filename="attention_paper.txt",
        )
        bm25_service.index_chunks(chunks=chunks, filename="attention_paper.txt")
        return doc_id


if __name__ == "__main__":
    runner = EvaluationRunner()
    res = runner.run_evaluation(top_k=10)
    print("=" * 60)
    print("NEXUS RAG V1 DENSE RETRIEVAL BASELINE RESULTS")
    print("=" * 60)
    print(f"Timestamp: {res.timestamp}")
    print(f"Embedding Model: {res.embedding_model}")
    print(f"Chunk Size: {res.chunk_size} | Overlap: {res.chunk_overlap}")
    print(f"Total Test Cases: {res.total_questions}")
    print(f"Recall@1:  {res.aggregate_recall_at_1 * 100:.1f}%")
    print(f"Recall@3:  {res.aggregate_recall_at_3 * 100:.1f}%")
    print(f"Recall@5:  {res.aggregate_recall_at_5 * 100:.1f}%")
    print(f"Recall@10: {res.aggregate_recall_at_10 * 100:.1f}%")
    print(f"MRR@10:    {res.aggregate_mrr_at_10:.4f}")
    print(f"NDCG@10:   {res.aggregate_ndcg_at_10:.4f}")
    print(f"Avg Latency: {res.average_retrieval_latency_ms:.2f} ms")
    print("=" * 60)
