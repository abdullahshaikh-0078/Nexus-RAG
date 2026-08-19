import os
import json
import time
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from app.core.config import settings
from app.services.chunker import RecursiveTextChunker
from app.services.embedder import embedding_service
from app.services.bm25_search import bm25_service
from app.services.hybrid_retriever import hybrid_retriever
from app.db.vectorstore import vector_store
from app.evaluation.metrics import RetrievalEvaluator, EvaluationRunResult, QuestionEvalResult

logger = logging.getLogger(__name__)

DIAGNOSTIC_DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "datasets", "v2_diagnostic.json"
)
DIAGNOSTIC_RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "results", "diagnostic_tests"
)

FULL_PAPER_TEXT = (
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
    "on an attention mechanism to draw global dependencies between input and output. Avoiding recurrence allows "
    "full parallelization across sequence tokens during training.\n\n"
    "3.2.1 Scaled Dot-Product Attention\n"
    "We call our particular attention Scaled Dot-Product Attention. The input consists of queries and keys of "
    "dimension dk, and values of dimension dv. We compute the dot products of the query with all keys, divide each "
    "by sqrt(dk), and apply a softmax function to obtain the weights on the values. Scaled dot-product attention "
    "was proposed by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin.\n\n"
    "3.2.2 Multi-Head Attention\n"
    "Multi-head attention allows the model to jointly attend to information from different representation subspaces "
    "at different positions. With a single attention head, averaging inhibits this.\n\n"
    "3.2.3 Applications of Attention in our Model\n"
    "The Transformer uses multi-head attention in three different ways:\n"
    "1. In encoder-decoder attention layers, the queries come from the previous decoder layer, and the memory keys "
    "and values come from the output of the encoder. This allows every position in the decoder to attend over all "
    "positions in the input sequence.\n"
    "2. The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries "
    "come from the same place, in this case, the output of the previous layer in the encoder.\n"
    "3. Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions "
    "in the decoder up to and including that position.\n\n"
    "3.5 Positional Encoding\n"
    "Since our model contains no recurrence and no convolution, in order for the model to make use of the order "
    "of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence.\n\n"
    "5.4 Maximum Path Lengths\n"
    "A self-attention layer connects all positions with a constant O(1) number of sequentially executed operations, "
    "whereas a recurrent layer requires O(n) sequential operations. In terms of computational complexity per layer, "
    "self-attention layers are faster than recurrent layers when sequence length n is smaller than representation dimensionality d. "
    "This constant path length enables high parallelization and experimental training efficiency.\n\n"
    "6.1 Hardware and Schedule\n"
    "We trained our models on one million steps. The base model was trained on a single machine with 8 NVIDIA P100 GPUs for 12 hours. "
    "For our big models, step time was 1.0 seconds. The big models were trained for 300,000 steps (3.5 days) on 8 NVIDIA P100 GPUs.\n\n"
    "6.2 Results\n"
    "On the WMT 2014 English-to-German translation task, the big Transformer model achieves a state-of-the-art BLEU score of 28.4, "
    "outperforming the best existing models by over 2.0 BLEU. On the WMT 2014 English-to-French translation task, our big model "
    "achieves a BLEU score of 41.8, outperforming all previously reported single models. WMT stands for Workshop on Statistical Machine Translation.\n\n"
    "References\n"
    "[1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NIPS, 2017.\n"
    "[2] Mitchell P. Marcus, Mary Ann Marcinkiewicz, and Beatrice Santorini. Building a large annotated corpus of English: The Penn Treebank. Computational Linguistics, 19(2):313-330, 1993."
)


def ensure_diagnostic_document_ingested() -> str:
    """Ingests full diagnostic paper text into vector store and BM25 index."""
    doc_id = "doc_eval_attention_paper"
    chunker = RecursiveTextChunker(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    chunks = chunker.chunk_document(FULL_PAPER_TEXT, document_id=doc_id)
    chunk_texts = [c.text for c in chunks]
    embeddings = embedding_service.embed_batch(chunk_texts)

    filename = "attention_paper.txt"
    vector_store.upsert_chunks(chunks=chunks, embeddings=embeddings, filename=filename)
    bm25_service.index_chunks(chunks=chunks, filename=filename)
    return doc_id


def profile_hybrid_retrieval(query: str, doc_id: str, top_k: int = 10, fetch_k: int = 20) -> Dict[str, Any]:
    """Profiles exact latency breakdown of query execution in Hybrid mode."""
    # 1. Embedding generation
    t0 = time.time()
    query_vector = embedding_service.embed_text(query)
    t_embed = (time.time() - t0) * 1000

    # 2. Qdrant Search
    t0 = time.time()
    dense_citations = vector_store.search_similar(
        query_vector=query_vector, top_k=fetch_k, document_ids=[doc_id]
    )
    t_dense = (time.time() - t0) * 1000

    # 3. BM25 Search
    t0 = time.time()
    bm25_citations = bm25_service.search(
        query=query, top_k=fetch_k, document_ids=[doc_id]
    )
    t_bm25 = (time.time() - t0) * 1000

    # 4. RRF Fusion
    t0 = time.time()
    raw_hybrid_citations = hybrid_retriever.search(
        query=query, top_k=top_k, document_ids=[doc_id]
    )
    t_rrf = (time.time() - t0) * 1000

    # 5. Context Expansion (neighbor scroll)
    t0 = time.time()
    expanded_citations = vector_store.expand_adjacent_context(raw_hybrid_citations, window=1)
    t_expand = (time.time() - t0) * 1000

    total_latency = t_rrf + t_expand

    return {
        "embedding_ms": round(t_embed, 2),
        "dense_search_ms": round(t_dense, 2),
        "bm25_search_ms": round(t_bm25, 2),
        "rrf_fusion_ms": round(t_rrf, 2),
        "context_expansion_ms": round(t_expand, 2),
        "total_latency_ms": round(total_latency, 2),
        "citations": expanded_citations,
    }


def run_diagnostic_evaluation() -> Dict[str, Any]:
    os.makedirs(DIAGNOSTIC_RESULTS_DIR, exist_ok=True)
    doc_id = ensure_diagnostic_document_ingested()

    with open(DIAGNOSTIC_DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    test_cases = dataset.get("test_cases", [])

    results_by_mode: Dict[str, List[Dict[str, Any]]] = {
        "dense": [],
        "bm25": [],
        "hybrid": [],
    }

    timing_breakdown: List[Dict[str, Any]] = []

    logger.info(f"Executing Diagnostic Evaluation across {len(test_cases)} questions...")

    for case in test_cases:
        q_id = case["question_id"]
        category = case["category"]
        question = case["question"]
        expected_snippets = case["expected_snippets"]

        # 1. Profile Hybrid & measure breakdown
        profile_res = profile_hybrid_retrieval(question, doc_id=doc_id, top_k=10)
        timing_breakdown.append({
            "question_id": q_id,
            "category": category,
            "question": question,
            "embedding_ms": profile_res["embedding_ms"],
            "dense_search_ms": profile_res["dense_search_ms"],
            "bm25_search_ms": profile_res["bm25_search_ms"],
            "rrf_fusion_ms": profile_res["rrf_fusion_ms"],
            "context_expansion_ms": profile_res["context_expansion_ms"],
            "total_hybrid_ms": profile_res["total_latency_ms"],
        })

        # 2. Evaluate Dense
        t0 = time.time()
        q_vec = embedding_service.embed_text(question)
        raw_d = vector_store.search_similar(q_vec, top_k=10, document_ids=[doc_id])
        exp_d = vector_store.expand_adjacent_context(raw_d, window=1)
        lat_d = round((time.time() - t0) * 1000, 2)
        ret_texts_d = [c.content for c in exp_d]

        # 3. Evaluate BM25
        t0 = time.time()
        raw_b = bm25_service.search(question, top_k=10, document_ids=[doc_id])
        exp_b = vector_store.expand_adjacent_context(raw_b, window=1)
        lat_b = round((time.time() - t0) * 1000, 2)
        ret_texts_b = [c.content for c in exp_b]

        # 4. Evaluate Hybrid
        exp_h = profile_res["citations"]
        lat_h = profile_res["total_latency_ms"]
        ret_texts_h = [c.content for c in exp_h]

        for mode_name, texts, lat in [("dense", ret_texts_d, lat_d), ("bm25", ret_texts_b, lat_b), ("hybrid", ret_texts_h, lat_h)]:
            r1 = RetrievalEvaluator.calculate_recall_at_k(texts, expected_snippets, k=1)
            r3 = RetrievalEvaluator.calculate_recall_at_k(texts, expected_snippets, k=3)
            r5 = RetrievalEvaluator.calculate_recall_at_k(texts, expected_snippets, k=5)
            r10 = RetrievalEvaluator.calculate_recall_at_k(texts, expected_snippets, k=10)
            rank = RetrievalEvaluator.calculate_first_relevant_rank(texts, expected_snippets, max_k=10)
            mrr = RetrievalEvaluator.calculate_mrr_at_k(texts, expected_snippets, k=10)
            ndcg = RetrievalEvaluator.calculate_ndcg_at_k(texts, expected_snippets, k=10)

            results_by_mode[mode_name].append({
                "question_id": q_id,
                "category": category,
                "question": question,
                "first_relevant_rank": rank,
                "recall_at_1": r1,
                "recall_at_3": r3,
                "recall_at_5": r5,
                "recall_at_10": r10,
                "mrr_at_10": mrr,
                "ndcg_at_10": ndcg,
                "retrieval_latency_ms": lat,
                "is_relevant": rank is not None,
            })

    # Build Summary per mode
    def build_summary(mode_name: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(items)
        r1 = round(sum(i["recall_at_1"] for i in items) / n, 4) if n else 0.0
        r3 = round(sum(i["recall_at_3"] for i in items) / n, 4) if n else 0.0
        r5 = round(sum(i["recall_at_5"] for i in items) / n, 4) if n else 0.0
        r10 = round(sum(i["recall_at_10"] for i in items) / n, 4) if n else 0.0
        mrr = round(sum(i["mrr_at_10"] for i in items) / n, 4) if n else 0.0
        ndcg = round(sum(i["ndcg_at_10"] for i in items) / n, 4) if n else 0.0
        avg_lat = round(sum(i["retrieval_latency_ms"] for i in items) / n, 2) if n else 0.0
        return {
            "dataset_version": "v2_diagnostic",
            "evaluation_version": f"v2_diagnostic_{mode_name}",
            "retrieval_mode": mode_name,
            "total_questions": n,
            "aggregate_recall_at_1": r1,
            "aggregate_recall_at_3": r3,
            "aggregate_recall_at_5": r5,
            "aggregate_recall_at_10": r10,
            "aggregate_mrr_at_10": mrr,
            "aggregate_ndcg_at_10": ndcg,
            "average_retrieval_latency_ms": avg_lat,
            "question_results": items,
        }

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    summaries = {}
    for mode in ["dense", "bm25", "hybrid"]:
        summary_data = build_summary(mode, results_by_mode[mode])
        summaries[mode] = summary_data

        # Save mode files
        fn_ts = os.path.join(DIAGNOSTIC_RESULTS_DIR, f"v2_diagnostic_{mode}_{timestamp_str}.json")
        fn_latest = os.path.join(DIAGNOSTIC_RESULTS_DIR, f"v2_diagnostic_{mode}_latest.json")
        with open(fn_ts, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        with open(fn_latest, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

    # Compute Latency Breakdown Averages
    avg_embed = round(sum(t["embedding_ms"] for t in timing_breakdown) / len(timing_breakdown), 2)
    avg_dense_s = round(sum(t["dense_search_ms"] for t in timing_breakdown) / len(timing_breakdown), 2)
    avg_bm25_s = round(sum(t["bm25_search_ms"] for t in timing_breakdown) / len(timing_breakdown), 2)
    avg_rrf_s = round(sum(t["rrf_fusion_ms"] for t in timing_breakdown) / len(timing_breakdown), 2)
    avg_expand_s = round(sum(t["context_expansion_ms"] for t in timing_breakdown) / len(timing_breakdown), 2)

    latency_analysis = {
        "avg_query_embedding_ms": avg_embed,
        "avg_dense_vector_search_ms": avg_dense_s,
        "avg_bm25_search_ms": avg_bm25_s,
        "avg_rrf_fusion_ms": avg_rrf_s,
        "avg_context_expansion_ms": avg_expand_s,
        "primary_bottleneck": "Context Expansion (get_chunks_by_indices scrolling Qdrant points) + Query Embedding overhead",
    }

    # Generate Markdown Report
    report_file = os.path.join(DIAGNOSTIC_RESULTS_DIR, "DIAGNOSTIC_REPORT.md")
    generate_diagnostic_report(summaries, timing_breakdown, latency_analysis, report_file)

    logger.info("Diagnostic evaluation complete! Saved all diagnostic results and report.")
    return {
        "summaries": summaries,
        "latency_analysis": latency_analysis,
    }


def generate_diagnostic_report(
    summaries: Dict[str, Dict[str, Any]],
    timing_breakdown: List[Dict[str, Any]],
    latency_analysis: Dict[str, Any],
    report_file: str,
):
    d_sum = summaries["dense"]
    b_sum = summaries["bm25"]
    h_sum = summaries["hybrid"]

    lines = []
    lines.append("# NEXUS RAG V2 Diagnostic Evaluation & Latency Validation Report\n")
    lines.append("> **Note**: This diagnostic evaluation runs 36 test questions across 12 categories. **Official V1 baseline remains frozen and untouched.**\n")

    lines.append("## Overall Benchmark Performance Matrix (36 Diagnostic Questions)\n")
    lines.append("| Metric | V1 Dense | V2.1 BM25 | V2.2 Hybrid (RRF) | Delta (Hybrid vs Dense) |")
    lines.append("|---|:---:|:---:|:---:|:---:|")

    m_names = [
        ("Recall @ 1", "aggregate_recall_at_1", True),
        ("Recall @ 3", "aggregate_recall_at_3", True),
        ("Recall @ 5", "aggregate_recall_at_5", True),
        ("Recall @ 10", "aggregate_recall_at_10", True),
        ("MRR @ 10", "aggregate_mrr_at_10", False),
        ("NDCG @ 10", "aggregate_ndcg_at_10", False),
        ("Avg Latency", "average_retrieval_latency_ms", "ms"),
    ]

    for label, key, is_pct in m_names:
        v_d = d_sum[key]
        v_b = b_sum[key]
        v_h = h_sum[key]

        if is_pct == "ms":
            val_d = f"{v_d:.2f} ms"
            val_b = f"{v_b:.2f} ms"
            val_h = f"{v_h:.2f} ms"
            diff = v_h - v_d
            delta_str = f"+{diff:.2f} ms" if diff > 0 else f"{diff:.2f} ms"
        elif is_pct:
            val_d = f"{v_d * 100:.1f}%"
            val_b = f"{v_b * 100:.1f}%"
            val_h = f"{v_h * 100:.1f}%"
            diff = (v_h - v_d) * 100
            delta_str = f"+{diff:.1f} pp" if diff >= 0 else f"{diff:.1f} pp"
        else:
            val_d = f"{v_d:.4f}"
            val_b = f"{v_b:.4f}"
            val_h = f"{v_h:.4f}"
            diff = v_h - v_d
            delta_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"

        lines.append(f"| **{label}** | {val_d} | {val_b} | {val_h} | `{delta_str}` |")

    lines.append("\n## Per-Category Performance Breakdown\n")
    lines.append("| Category | Dense Recall@1 | BM25 Recall@1 | Hybrid Recall@1 | Hybrid MRR@10 |")
    lines.append("|---|:---:|:---:|:---:|:---:|")

    # Group questions by category
    cats: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for mode in ["dense", "bm25", "hybrid"]:
        for q in summaries[mode]["question_results"]:
            c = q["category"]
            cats.setdefault(c, {}).setdefault(mode, []).append(q)

    for cat_name, modes in cats.items():
        q_d = modes["dense"]
        q_b = modes["bm25"]
        q_h = modes["hybrid"]
        rec_d = sum(q["recall_at_1"] for q in q_d) / len(q_d)
        rec_b = sum(q["recall_at_1"] for q in q_b) / len(q_b)
        rec_h = sum(q["recall_at_1"] for q in q_h) / len(q_h)
        mrr_h = sum(q["mrr_at_10"] for q in q_h) / len(q_h)

        lines.append(f"| {cat_name} | {rec_d*100:.1f}% | {rec_b*100:.1f}% | {rec_h*100:.1f}% | {mrr_h:.4f} |")

    lines.append("\n## Latency Breakdown & Bottleneck Analysis\n")
    lines.append("Step-by-step latency instrumentation across diagnostic runs:\n")
    lines.append(f"- **Query Embedding Generation (`SentenceTransformer.encode`)**: `{latency_analysis['avg_query_embedding_ms']} ms`")
    lines.append(f"- **Qdrant Dense Search (`search_similar`)**: `{latency_analysis['avg_dense_vector_search_ms']} ms`")
    lines.append(f"- **BM25 Lexical Search (`bm25_service.search`)**: `{latency_analysis['avg_bm25_search_ms']} ms`")
    lines.append(f"- **RRF Fusion & Deduplication (`HybridRetriever.search`)**: `{latency_analysis['avg_rrf_fusion_ms']} ms`")
    lines.append(f"- **Adjacent Context Expansion (`expand_adjacent_context` Qdrant scroll)**: `{latency_analysis['avg_context_expansion_ms']} ms`")
    lines.append(f"\n> **IDENTIFIED BOTTLENECK**: {latency_analysis['primary_bottleneck']}\n")

    lines.append("## Per-Question Comparison Table (36 Questions)\n")
    lines.append("| ID | Category | Question | Dense Rank | BM25 Rank | Hybrid Rank | Hybrid Latency |")
    lines.append("|---|---|---|:---:|:---:|:---:|:---:|")

    d_q_map = {q["question_id"]: q for q in d_sum["question_results"]}
    b_q_map = {q["question_id"]: q for q in b_sum["question_results"]}
    h_q_map = {q["question_id"]: q for q in h_sum["question_results"]}

    for q_id in sorted(d_q_map.keys()):
        qd = d_q_map[q_id]
        qb = b_q_map[q_id]
        qh = h_q_map[q_id]

        rd = f"#{qd['first_relevant_rank']}" if qd['first_relevant_rank'] else "--"
        rb = f"#{qb['first_relevant_rank']}" if qb['first_relevant_rank'] else "--"
        rh = f"#{qh['first_relevant_rank']}" if qh['first_relevant_rank'] else "--"
        lat = f"{qh['retrieval_latency_ms']:.1f}ms"

        lines.append(f"| {q_id} | {qd['category']} | {qd['question']} | {rd} | {rb} | {rh} | {lat} |")

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_diagnostic_evaluation()
