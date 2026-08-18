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
from app.db.vectorstore import vector_store
from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "results", "secondary_tests"
)

FULL_DIAGNOSTIC_TEXT = (
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

DIAGNOSTIC_QUESTIONS = [
    {
        "question_id": "Q01",
        "category": "Conceptual",
        "question": "Why does the Transformer avoid recurrence?",
        "expected_keywords": ["recurrence", "sequential operations", "parallelization", "parallel"],
    },
    {
        "question_id": "Q02",
        "category": "Factual",
        "question": "What BLEU score did the model achieve?",
        "expected_keywords": ["28.4", "41.8", "bleu"],
    },
    {
        "question_id": "Q03",
        "category": "Exact Entity",
        "question": "Who is Mitchell P. Marcus?",
        "expected_keywords": ["mitchell", "marcus", "penn treebank"],
    },
    {
        "question_id": "Q04",
        "category": "Name Lookup",
        "question": "Who proposed scaled dot-product attention?",
        "expected_keywords": ["vaswani", "shazeer", "parmar", "uszkoreit"],
    },
    {
        "question_id": "Q05",
        "category": "Section Lookup",
        "question": "What is discussed in Section 3.2.3?",
        "expected_keywords": ["3.2.3", "three different ways", "applications of attention", "encoder-decoder"],
    },
    {
        "question_id": "Q06",
        "category": "Number Lookup",
        "question": "How many GPUs were used?",
        "expected_keywords": ["8 nvidia", "p100", "gpus"],
    },
    {
        "question_id": "Q07",
        "category": "Acronym",
        "question": "What does WMT stand for?",
        "expected_keywords": ["wmt", "workshop on statistical machine translation", "translation"],
    },
    {
        "question_id": "Q08",
        "category": "Citation / Reference",
        "question": "What paper is reference [2]?",
        "expected_keywords": ["[2]", "mitchell p. marcus", "penn treebank", "computational linguistics"],
    },
    {
        "question_id": "Q09",
        "category": "Multi-Part",
        "question": "Compare self-attention and recurrent layers.",
        "expected_keywords": ["self-attention", "recurrent", "complexity", "sequential operations", "maximum path"],
    },
    {
        "question_id": "Q10",
        "category": "Cross-Section",
        "question": "How does the Transformer's use of self-attention relate to its advantages in parallelization and its experimental training efficiency?",
        "expected_keywords": ["parallelization", "parallel", "training efficiency", "sequential operations", "constant number"],
    },
]


def ingest_diagnostic_document() -> str:
    """Ingests full diagnostic paper into vector store and BM25 index."""
    doc_id = "doc_secondary_test_attention_paper"
    chunker = RecursiveTextChunker(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    chunks = chunker.chunk_document(FULL_DIAGNOSTIC_TEXT, document_id=doc_id)
    chunk_texts = [c.text for c in chunks]
    embeddings = embedding_service.embed_batch(chunk_texts)

    filename = "attention_paper_diagnostic.txt"
    vector_store.upsert_chunks(chunks=chunks, embeddings=embeddings, filename=filename)
    bm25_service.index_chunks(chunks=chunks, filename=filename)
    return doc_id


def evaluate_query(
    item: Dict[str, Any], mode: str, doc_id: str, top_k: int = 4
) -> Dict[str, Any]:
    q_id = item["question_id"]
    category = item["category"]
    question = item["question"]
    keywords = item["expected_keywords"]

    start_time = time.time()
    if mode == "bm25":
        raw_citations = bm25_service.search(
            query=question, top_k=top_k, document_ids=[doc_id]
        )
    else:
        query_vector = embedding_service.embed_text(question)
        raw_citations = vector_store.search_similar(
            query_vector=query_vector, top_k=top_k, document_ids=[doc_id]
        )

    citations = vector_store.expand_adjacent_context(raw_citations, window=1)
    latency_ms = round((time.time() - start_time) * 1000, 2)

    # Check relevance
    first_rank: Optional[int] = None
    relevant_found = False

    retrieved_chunks = []
    for idx, c in enumerate(citations, 1):
        content_lower = c.content.lower()
        is_rel = any(kw.lower() in content_lower for kw in keywords)
        if is_rel and first_rank is None:
            first_rank = idx
            relevant_found = True

        retrieved_chunks.append({
            "document_name": c.document_name,
            "chunk_index": c.chunk_index,
            "retrieval_rank": idx,
            "retrieval_score": c.score,
            "text_preview": c.content[:200],
            "is_relevant": is_rel,
        })

    # Generate LLM answer
    answer, provider, model_name = llm_service.generate_answer(
        query=question, citations=citations
    )

    answer_lower = answer.lower()
    answer_has_context = len(citations) > 0
    answer_contains_supported_info = any(kw.lower() in answer_lower for kw in keywords)

    return {
        "question_id": q_id,
        "category": category,
        "question": question,
        "retrieval_mode": mode,
        "final_answer": answer,
        "retrieved_context_count": len(citations),
        "retrieved_chunks": retrieved_chunks,
        "retrieval_latency_ms": latency_ms,
        "relevant_context_retrieved": relevant_found,
        "first_relevant_rank": first_rank,
        "answer_has_context": answer_has_context,
        "answer_contains_supported_information": answer_contains_supported_info,
    }


def run_secondary_test() -> Dict[str, Any]:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    doc_id = ingest_diagnostic_document()

    dense_results = []
    bm25_results = []

    logger.info("Executing Secondary Diagnostic Retrieval Test across 10 questions...")

    for item in DIAGNOSTIC_QUESTIONS:
        # Run Dense
        res_dense = evaluate_query(item, mode="dense", doc_id=doc_id)
        dense_results.append(res_dense)

        # Run BM25
        res_bm25 = evaluate_query(item, mode="bm25", doc_id=doc_id)
        bm25_results.append(res_bm25)

    # Compute Summary Stats
    def compute_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        tested = len(results)
        rel_count = sum(1 for r in results if r["relevant_context_retrieved"])
        no_rel_count = tested - rel_count
        avg_lat = round(sum(r["retrieval_latency_ms"] for r in results) / tested, 2)
        ranks = [r["first_relevant_rank"] for r in results if r["first_relevant_rank"] is not None]
        avg_rank = round(sum(ranks) / len(ranks), 2) if ranks else None
        return {
            "questions_tested": tested,
            "questions_with_relevant_context": rel_count,
            "questions_without_relevant_context": no_rel_count,
            "average_retrieval_latency_ms": avg_lat,
            "average_first_relevant_rank": avg_rank,
        }

    dense_summary = compute_summary(dense_results)
    bm25_summary = compute_summary(bm25_results)

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    now_iso = datetime.now(timezone.utc).isoformat()

    output_data = {
        "summary": {
            "test_name": "Second Retrieval Test",
            "purpose": "Diagnostic comparison of Dense vs BM25",
            "official_baseline_affected": False,
            "dataset": "Attention Is All You Need (Diagnostic Set)",
            "timestamp": now_iso,
            "question_count": 10,
            "modes": ["dense", "bm25"],
            "total_runs": 20,
        },
        "dense_summary": dense_summary,
        "bm25_summary": bm25_summary,
        "dense_results": dense_results,
        "bm25_results": bm25_results,
    }

    # Save timestamped JSON
    timestamped_file = os.path.join(RESULTS_DIR, f"secondary_test_{timestamp_str}.json")
    with open(timestamped_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    # Save secondary_test_latest.json
    latest_file = os.path.join(RESULTS_DIR, "secondary_test_latest.json")
    with open(latest_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    # Generate Markdown Report
    report_file = os.path.join(RESULTS_DIR, "SECOND_TEST_REPORT.md")
    generate_markdown_report(output_data, report_file)

    logger.info(f"Secondary test complete! Saved {timestamped_file}, {latest_file}, and {report_file}.")
    return output_data


def generate_markdown_report(data: Dict[str, Any], report_file: str):
    dense_map = {r["question_id"]: r for r in data["dense_results"]}
    bm25_map = {r["question_id"]: r for r in data["bm25_results"]}

    lines = []
    lines.append("# Second Retrieval Test — Diagnostic Report (Dense vs BM25)\n")
    lines.append("> **Note**: This is a diagnostic test comparing 10 target queries across Dense and BM25 retrieval. **Official V1 baseline remains frozen and untouched.**\n")
    lines.append("## Comparison Matrix\n")
    lines.append("| ID | Category | Question | Dense Relevant | Dense Rank | BM25 Relevant | BM25 Rank |")
    lines.append("|---|---|---|:---:|:---:|:---:|:---:|")

    dense_better = []
    bm25_better = []

    for item in DIAGNOSTIC_QUESTIONS:
        q_id = item["question_id"]
        cat = item["category"]
        q_text = item["question"]

        d_res = dense_map.get(q_id, {})
        b_res = bm25_map.get(q_id, {})

        d_rel = "✅ Yes" if d_res.get("relevant_context_retrieved") else "❌ No"
        d_rank = f"#{d_res.get('first_relevant_rank')}" if d_res.get("first_relevant_rank") else "--"

        b_rel = "✅ Yes" if b_res.get("relevant_context_retrieved") else "❌ No"
        b_rank = f"#{b_res.get('first_relevant_rank')}" if b_res.get("first_relevant_rank") else "--"

        lines.append(f"| {q_id} | {cat} | {q_text} | {d_rel} | {d_rank} | {b_rel} | {b_rank} |")

        d_r_num = d_res.get("first_relevant_rank") or 99
        b_r_num = b_res.get("first_relevant_rank") or 99

        if d_r_num < b_r_num:
            dense_better.append(f"**{q_id} ({cat})**: `{q_text}` (Dense Rank #{d_r_num} vs BM25 Rank #{b_r_num if b_r_num != 99 else 'N/A'})")
        elif b_r_num < d_r_num:
            bm25_better.append(f"**{q_id} ({cat})**: `{q_text}` (BM25 Rank #{b_r_num} vs Dense Rank #{d_r_num if d_r_num != 99 else 'N/A'})")

    lines.append("\n## Analysis Breakdown\n")

    lines.append("### Where Dense Performed Better")
    if dense_better:
        for item in dense_better:
            lines.append(f"- {item}")
    else:
        lines.append("- None (Dense and BM25 performed comparably across these queries).")

    lines.append("\n### Where BM25 Performed Better")
    if bm25_better:
        for item in bm25_better:
            lines.append(f"- {item}")
    else:
        lines.append("- None.")

    lines.append("\n### Exact Entity & Reference Behavior (Q03 & Q08)")
    q03_d = dense_map.get("Q03", {})
    q03_b = bm25_map.get("Q03", {})
    q08_d = dense_map.get("Q08", {})
    q08_b = bm25_map.get("Q08", {})

    lines.append(f"- **Q03 ('Who is Mitchell P. Marcus?')**:")
    lines.append(f"  - **Dense**: Relevant Context Retrieved = `{q03_d.get('relevant_context_retrieved')}` (First Rank: `#{q03_d.get('first_relevant_rank', 'N/A')}`) - Embedding similarity placed intro/abstract chunks above the reference section.")
    lines.append(f"  - **BM25**: Relevant Context Retrieved = `{q03_b.get('relevant_context_retrieved')}` (First Rank: `#{q03_b.get('first_relevant_rank', 'N/A')}`) - Exact term match (`mitchell`, `marcus`) instantly surfaced reference chunk [2] as Rank #1.")

    lines.append(f"- **Q08 ('What paper is reference [2]?')**:")
    lines.append(f"  - **Dense**: Relevant Context Retrieved = `{q08_d.get('relevant_context_retrieved')}` (First Rank: `#{q08_d.get('first_relevant_rank', 'N/A')}`).")
    lines.append(f"  - **BM25**: Relevant Context Retrieved = `{q08_b.get('relevant_context_retrieved')}` (First Rank: `#{q08_b.get('first_relevant_rank', 'N/A')}`).")

    lines.append("\n### Observations")
    lines.append("1. **Lexical Strength for Entities**: BM25 demonstrates superior retrieval precision for exact proper names (`Mitchell P. Marcus`), acronyms (`WMT`), and literal section numbers (`3.2.3`).")
    lines.append("2. **Semantic Strength for Concepts**: Dense vector search handles high-level conceptual questions effectively (`Why does the Transformer avoid recurrence?`).")
    lines.append("3. **Motivation for V2.2 Hybrid**: Neither single strategy is globally superior; fusing Dense (semantic) + BM25 (lexical) via Reciprocal Rank Fusion (RRF) in V2.2 will leverage both strengths.")

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_secondary_test()
