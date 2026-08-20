import os
import json
import time
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from app.core.config import settings
from app.services.embedder import embedding_service
from app.services.bm25_search import bm25_service
from app.services.hybrid_retriever import hybrid_retriever
from app.db.vectorstore import vector_store
from app.services.llm_service import llm_service
from app.evaluation.metrics import RetrievalEvaluator
from app.evaluation.financebench.loader import FinanceBenchLoader, FinanceBenchQuestion
from app.evaluation.financebench.runner import FinanceBenchRunner, FINANCEBENCH_RESULTS_DIR

logger = logging.getLogger(__name__)

TARGET_QUESTION_IDS = [
    "financebench_id_00499",
    "financebench_id_01226",
    "financebench_id_01865",
    "financebench_id_00807",
    "financebench_id_00941",
    "financebench_id_01858",
    "financebench_id_02987",
    "financebench_id_07966",
]


def classify_failure_mode(q_id: str, first_rank: Optional[int], recall_1: float) -> str:
    """Classifies financial domain retrieval failure mode."""
    if recall_1 == 1.0:
        return "success"
    if q_id in ["financebench_id_02987", "financebench_id_07966"]:
        return "multi_year_calculation"
    if q_id in ["financebench_id_00807", "financebench_id_00499"]:
        return "terminology_mismatch"
    if q_id in ["financebench_id_00941"]:
        return "footnote_reference"
    return "table_fragmentation"


def run_financebench_test2() -> Dict[str, Any]:
    loader = FinanceBenchLoader()
    runner = FinanceBenchRunner(loader=loader)

    all_questions = loader.load_dataset()
    target_questions = [q for q in all_questions if q.financebench_id in TARGET_QUESTION_IDS]

    logger.info(f"Loaded {len(target_questions)} target FinanceBench Test 2 questions.")

    # Group by doc_name and ingest required PDFs
    unique_docs = list(set(q.doc_name for q in target_questions))
    doc_id_map: Dict[str, str] = {}

    for d_name in unique_docs:
        doc_id_map[d_name] = runner.ingest_document(d_name)

    modes = ["dense", "bm25", "hybrid"]
    all_runs: List[Dict[str, Any]] = []

    for q in target_questions:
        q_id = q.financebench_id
        q_text = q.question
        doc_name = q.doc_name
        doc_id = doc_id_map[doc_name]
        gt_answer = q.answer
        gt_justification = q.justification

        expected_snippets = []
        for ev in q.evidence:
            if ev.evidence_text:
                expected_snippets.append(ev.evidence_text)
            if ev.evidence_text_full_page:
                expected_snippets.append(ev.evidence_text_full_page[:400])

        if not expected_snippets:
            expected_snippets = [gt_answer]

        for mode in modes:
            t0_ret = time.time()
            if mode == "hybrid":
                raw_citations = hybrid_retriever.search(
                    query=q_text, top_k=10, document_ids=[doc_id]
                )
            elif mode == "bm25":
                raw_citations = bm25_service.search(
                    query=q_text, top_k=10, document_ids=[doc_id]
                )
            else:
                q_vec = embedding_service.embed_text(q_text)
                raw_citations = vector_store.search_similar(
                    query_vector=q_vec, top_k=10, document_ids=[doc_id]
                )

            citations = vector_store.expand_adjacent_context(raw_citations, window=1)
            t_ret_ms = round((time.time() - t0_ret) * 1000, 2)

            retrieved_texts = [c.content for c in citations]

            r1 = RetrievalEvaluator.calculate_recall_at_k(retrieved_texts, expected_snippets, k=1)
            r5 = RetrievalEvaluator.calculate_recall_at_k(retrieved_texts, expected_snippets, k=5)
            r10 = RetrievalEvaluator.calculate_recall_at_k(retrieved_texts, expected_snippets, k=10)
            first_rank = RetrievalEvaluator.calculate_first_relevant_rank(retrieved_texts, expected_snippets, max_k=10)
            mrr10 = RetrievalEvaluator.calculate_mrr_at_k(retrieved_texts, expected_snippets, k=10)
            ndcg10 = RetrievalEvaluator.calculate_ndcg_at_k(retrieved_texts, expected_snippets, k=10)

            failure_cat = classify_failure_mode(q_id, first_rank, r1)

            t0_gen = time.time()
            gen_answer, provider, model_used = llm_service.generate_answer(
                query=q_text, citations=citations
            )
            t_gen_ms = round((time.time() - t0_gen) * 1000, 2)
            total_latency_ms = round(t_ret_ms + t_gen_ms, 2)

            retrieved_chunk_info = []
            for idx, c in enumerate(citations, 1):
                retrieved_chunk_info.append({
                    "rank": idx,
                    "chunk_id": c.chunk_id,
                    "chunk_index": c.chunk_index,
                    "score": c.score,
                    "dense_rank": getattr(c, "dense_rank", None),
                    "bm25_rank": getattr(c, "bm25_rank", None),
                    "rrf_score": getattr(c, "rrf_score", None),
                    "content_preview": c.content[:250],
                })

            run_record = {
                "question_id": q_id,
                "document_name": doc_name,
                "question": q_text,
                "retrieval_mode": mode,
                "first_relevant_rank": first_rank,
                "recall_at_1": r1,
                "recall_at_5": r5,
                "recall_at_10": r10,
                "mrr_at_10": mrr10,
                "ndcg_at_10": ndcg10,
                "failure_category": failure_cat,
                "ground_truth_answer": gt_answer,
                "ground_truth_justification": gt_justification,
                "generated_answer": gen_answer,
                "llm_provider": provider,
                "llm_model": model_used,
                "retrieved_chunk_count": len(citations),
                "retrieved_chunks": retrieved_chunk_info,
                "retrieval_latency_ms": t_ret_ms,
                "generation_latency_ms": t_gen_ms,
                "total_latency_ms": total_latency_ms,
            }
            all_runs.append(run_record)

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    now_iso = datetime.now(timezone.utc).isoformat()

    output_data = {
        "summary": {
            "test_name": "FinanceBench Test 2 — Multi-Document 8-Question Diagnostic",
            "timestamp": now_iso,
            "total_questions": len(target_questions),
            "documents_evaluated": unique_docs,
            "question_ids": TARGET_QUESTION_IDS,
            "modes_evaluated": modes,
            "official_v1_baseline_untouched": True,
        },
        "runs": all_runs,
    }

    # Save JSON files
    ts_json = os.path.join(
        FINANCEBENCH_RESULTS_DIR, f"financebench_test2_{timestamp_str}.json"
    )
    latest_json = os.path.join(
        FINANCEBENCH_RESULTS_DIR, "financebench_test2_latest.json"
    )

    with open(ts_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    # Generate Report
    report_file = os.path.join(FINANCEBENCH_RESULTS_DIR, "FINANCEBENCH_TEST2_REPORT.md")
    generate_test2_markdown_report(output_data, report_file)

    logger.info(f"Test 2 Complete! Saved {ts_json}, {latest_json}, and {report_file}.")
    return output_data


def generate_test2_markdown_report(data: Dict[str, Any], report_file: str):
    summary = data["summary"]
    runs = data["runs"]

    lines = []
    lines.append("# FinanceBench Test 2 — Multi-Document 8-Question Diagnostic Report\n")
    lines.append("> **Note**: Diagnostic evaluation across 8 FinanceBench questions (3 SEC 10-K/10-Q reports). **Frozen V1 baseline remains untouched.**\n")

    lines.append("## Target Details\n")
    lines.append(f"- **Evaluated Documents**: `{', '.join(summary['documents_evaluated'])}`")
    lines.append(f"- **Total Questions**: `{summary['total_questions']}`")
    lines.append(f"- **Question IDs**: `{', '.join(summary['question_ids'])}`")
    lines.append(f"- **Execution Timestamp**: `{summary['timestamp']}`\n")

    lines.append("## Mode-by-Mode Benchmark Performance Comparison\n")
    lines.append("| Mode | Recall @ 1 | Recall @ 5 | MRR @ 10 | NDCG @ 10 | Avg Retrieval Latency | Avg Generation Latency | Avg Total Latency |")
    lines.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for mode in ["dense", "bm25", "hybrid"]:
        mode_runs = [r for r in runs if r["retrieval_mode"] == mode]
        n = len(mode_runs)
        r1 = sum(r["recall_at_1"] for r in mode_runs) / n if n else 0.0
        r5 = sum(r["recall_at_5"] for r in mode_runs) / n if n else 0.0
        mrr = sum(r["mrr_at_10"] for r in mode_runs) / n if n else 0.0
        ndcg = sum(r["ndcg_at_10"] for r in mode_runs) / n if n else 0.0
        lat_ret = sum(r["retrieval_latency_ms"] for r in mode_runs) / n if n else 0.0
        lat_gen = sum(r["generation_latency_ms"] for r in mode_runs) / n if n else 0.0
        lat_tot = sum(r["total_latency_ms"] for r in mode_runs) / n if n else 0.0

        mode_label = "V1 — Dense" if mode == "dense" else ("V2.1 — BM25" if mode == "bm25" else "V2.2 — Hybrid RRF")
        lines.append(
            f"| **{mode_label}** | {r1*100:.1f}% | {r5*100:.1f}% | {mrr:.4f} | {ndcg:.4f} | {lat_ret:.2f} ms | {lat_gen:.2f} ms | {lat_tot:.2f} ms |"
        )

    lines.append("\n## Failure Mode Taxonomy & Domain Breakdown\n")
    lines.append("1. **`terminology_mismatch`**: Query financial terms (e.g., 'quick ratio', 'capital-intensive') do not literally match line item phrasing in 10-K tables.")
    lines.append("2. **`multi_year_calculation`**: Financial metrics requiring 3-year historical average calculation across separate table columns.")
    lines.append("3. **`table_fragmentation`**: Fixed 1000-character chunking splits multi-column balance sheets across chunk boundaries.")
    lines.append("4. **`footnote_reference`**: Small font footnote text embedded below financial tables.")

    lines.append("\n## Per-Question Detailed Results\n")

    for q_id in summary["question_ids"]:
        q_runs = [r for r in runs if r["question_id"] == q_id]
        if not q_runs:
            continue
        fq = q_runs[0]
        lines.append(f"### Question `{q_id}` (`{fq['document_name']}`)")
        lines.append(f"**Question**: `{fq['question']}`\n")
        lines.append(f"**Ground-Truth Answer**: `{fq['ground_truth_answer']}`\n")
        lines.append(f"**Failure Category**: `{fq['failure_category']}`\n")

        for r in q_runs:
            m_label = r["retrieval_mode"].upper()
            rank_str = f"#{r['first_relevant_rank']}" if r["first_relevant_rank"] else "❌ Not in Top 10"
            lines.append(f"#### Mode: `{m_label}` (Relevant Rank: `{rank_str}` | Latency: `{r['total_latency_ms']}ms`)")
            lines.append(f"**Generated Answer**:\n>{r['generated_answer']}\n")

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_financebench_test2()
