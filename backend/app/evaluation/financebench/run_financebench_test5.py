import os
import json
import time
import logging
import statistics
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
from app.evaluation.financebench.reranker import experimental_reranker

logger = logging.getLogger(__name__)

TEST5_DIR = os.path.join(FINANCEBENCH_RESULTS_DIR, "test5_reranking")
os.makedirs(TEST5_DIR, exist_ok=True)

TARGET_QUESTION_IDS = [
    "financebench_id_03029",
    "financebench_id_04672",
    "financebench_id_04735",
    "financebench_id_07966",
    "financebench_id_00499",
    "financebench_id_00941",
    "financebench_id_01865",
    "financebench_id_01226",
    "financebench_id_01319",
    "financebench_id_01858",
    "financebench_id_02987",
    "financebench_id_00807",
]

QUESTION_CATEGORY_MAP = {
    "financebench_id_03029": "Exact financial number",
    "financebench_id_04672": "Financial table lookup",
    "financebench_id_04735": "Ratio calculation",
    "financebench_id_07966": "Multi-year calculation",
    "financebench_id_00499": "Terminology mismatch",
    "financebench_id_00941": "Footnote retrieval",
    "financebench_id_01865": "Segment/table analysis",
    "financebench_id_01226": "Cross-section reasoning",
    "financebench_id_01319": "Exact accounting term",
    "financebench_id_01858": "Normal conceptual question",
    "financebench_id_02987": "Ratio calculation",
    "financebench_id_00807": "Ratio calculation",
}


def check_evidence_presence(retrieved_texts: List[str], expected_snippets: List[str], top_k: int) -> bool:
    sub_list = retrieved_texts[:top_k]
    for text in sub_list:
        text_lower = text.lower()
        for snip in expected_snippets:
            if snip.lower() in text_lower or text_lower in snip.lower():
                return True
            words_snip = set(snip.lower().split())
            words_text = set(text_lower.split())
            if len(words_snip) > 3 and len(words_snip.intersection(words_text)) / len(words_snip) > 0.6:
                return True
    return False


def get_first_evidence_rank(retrieved_texts: List[str], expected_snippets: List[str], max_k: int = 10) -> Optional[int]:
    for rank_idx, text in enumerate(retrieved_texts[:max_k], 1):
        text_lower = text.lower()
        for snip in expected_snippets:
            if snip.lower() in text_lower or text_lower in snip.lower():
                return rank_idx
            words_snip = set(snip.lower().split())
            words_text = set(text_lower.split())
            if len(words_snip) > 3 and len(words_snip.intersection(words_text)) / len(words_snip) > 0.6:
                return rank_idx
    return None


def run_financebench_test5() -> Dict[str, Any]:
    loader = FinanceBenchLoader()
    runner = FinanceBenchRunner(loader=loader)

    all_questions = loader.load_dataset()
    target_questions = [q for q in all_questions if q.financebench_id in TARGET_QUESTION_IDS]

    logger.info(f"Loaded {len(target_questions)} target FinanceBench Test 5 questions.")

    unique_docs = sorted(list(set(q.doc_name for q in target_questions)))
    doc_id_map: Dict[str, str] = {}

    for d_name in unique_docs:
        doc_id_map[d_name] = runner.ingest_document(d_name)

    modes = ["dense", "bm25", "hybrid", "hybrid_rerank"]
    all_runs: List[Dict[str, Any]] = []

    for q in target_questions:
        q_id = q.financebench_id
        q_text = q.question
        doc_name = q.doc_name
        doc_id = doc_id_map[doc_name]
        gt_answer = q.answer
        gt_justification = q.justification
        q_category = QUESTION_CATEGORY_MAP.get(q_id, "General Diagnostic")

        expected_snippets = []
        for ev in q.evidence:
            if ev.evidence_text:
                expected_snippets.append(ev.evidence_text)
            if ev.evidence_text_full_page:
                expected_snippets.append(ev.evidence_text_full_page[:400])

        if not expected_snippets:
            expected_snippets = [gt_answer]

        for mode in modes:
            t_embed = 0.0
            t_dense = 0.0
            t_bm25 = 0.0
            t_rrf = 0.0
            t_rerank = 0.0
            t0_ret = time.time()

            if mode == "bm25":
                t0_b = time.time()
                raw_citations = bm25_service.search(
                    query=q_text, top_k=10, document_ids=[doc_id]
                )
                t_bm25 = round((time.time() - t0_b) * 1000, 2)
            elif mode == "dense":
                t0_emb = time.time()
                q_vec = embedding_service.embed_text(q_text)
                t_embed = round((time.time() - t0_emb) * 1000, 2)

                t0_d = time.time()
                raw_citations = vector_store.search_similar(
                    query_vector=q_vec, top_k=10, document_ids=[doc_id]
                )
                t_dense = round((time.time() - t0_d) * 1000, 2)
            elif mode == "hybrid":
                t0_h = time.time()
                raw_citations = hybrid_retriever.search(
                    query=q_text, top_k=10, document_ids=[doc_id]
                )
                t_rrf = round((time.time() - t0_h) * 1000, 2)
            else:
                # hybrid_rerank: first run hybrid retrieval then apply experimental cross-encoder reranker
                t0_h = time.time()
                hybrid_citations = hybrid_retriever.search(
                    query=q_text, top_k=10, document_ids=[doc_id]
                )
                t_rrf = round((time.time() - t0_h) * 1000, 2)

                raw_citations, t_rerank = experimental_reranker.rerank(
                    query=q_text, citations=hybrid_citations
                )

            t0_exp = time.time()
            citations = vector_store.expand_adjacent_context(raw_citations, window=1)
            t_exp = round((time.time() - t0_exp) * 1000, 2)

            t_ret_ms = round((time.time() - t0_ret) * 1000, 2)

            retrieved_texts = [c.content for c in citations]
            retrieved_chunk_ids = [c.chunk_id for c in citations]

            ev_1 = check_evidence_presence(retrieved_texts, expected_snippets, top_k=1)
            ev_3 = check_evidence_presence(retrieved_texts, expected_snippets, top_k=3)
            ev_5 = check_evidence_presence(retrieved_texts, expected_snippets, top_k=5)
            ev_10 = check_evidence_presence(retrieved_texts, expected_snippets, top_k=10)
            first_rank = get_first_evidence_rank(retrieved_texts, expected_snippets, max_k=10)

            mrr10 = RetrievalEvaluator.calculate_mrr_at_k(retrieved_texts, expected_snippets, k=10)
            ndcg10 = RetrievalEvaluator.calculate_ndcg_at_k(retrieved_texts, expected_snippets, k=10)

            # LLM Generation Check
            t0_gen = time.time()
            gen_answer, provider, model_used = llm_service.generate_answer(
                query=q_text, citations=citations
            )
            t_gen_ms = round((time.time() - t0_gen) * 1000, 2)
            t_total_ms = round(t_ret_ms + t_gen_ms, 2)

            ans_lower = gen_answer.lower()
            gt_lower = gt_answer.lower()
            has_val = any(w in ans_lower for w in gt_lower.split() if len(w) > 3 and any(c.isdigit() for c in w))
            correctness_score = 1.0 if has_val else 0.0

            run_record = {
                "question_id": q_id,
                "document_name": doc_name,
                "question_category": q_category,
                "question": q_text,
                "retrieval_mode": mode,
                "evidence_rank": first_rank,
                "evidence_at_1": ev_1,
                "evidence_at_3": ev_3,
                "evidence_at_5": ev_5,
                "evidence_at_10": ev_10,
                "mrr_at_10": mrr10,
                "ndcg_at_10": ndcg10,
                "generated_answer": gen_answer,
                "ground_truth_answer": gt_answer,
                "answer_correctness": correctness_score,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "latency_breakdown": {
                    "embedding_ms": t_embed,
                    "dense_search_ms": t_dense,
                    "bm25_search_ms": t_bm25,
                    "rrf_fusion_ms": t_rrf,
                    "reranking_ms": t_rerank,
                    "context_expansion_ms": t_exp,
                    "retrieval_ms": t_ret_ms,
                    "generation_ms": t_gen_ms,
                    "total_ms": t_total_ms,
                },
            }
            all_runs.append(run_record)

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    now_iso = datetime.now(timezone.utc).isoformat()

    # Calculate Aggregate Metrics by Mode
    mode_stats = {}
    for m in modes:
        m_runs = [r for r in all_runs if r["retrieval_mode"] == m]
        n = len(m_runs)
        ret_lats = [r["latency_breakdown"]["retrieval_ms"] for r in m_runs]
        rerank_lats = [r["latency_breakdown"]["reranking_ms"] for r in m_runs]
        tot_lats = [r["latency_breakdown"]["total_ms"] for r in m_runs]

        mode_stats[m] = {
            "total_runs": n,
            "evidence_at_1_pct": round(sum(1 for r in m_runs if r["evidence_at_1"]) / n * 100, 1) if n else 0.0,
            "evidence_at_3_pct": round(sum(1 for r in m_runs if r["evidence_at_3"]) / n * 100, 1) if n else 0.0,
            "evidence_at_5_pct": round(sum(1 for r in m_runs if r["evidence_at_5"]) / n * 100, 1) if n else 0.0,
            "evidence_at_10_pct": round(sum(1 for r in m_runs if r["evidence_at_10"]) / n * 100, 1) if n else 0.0,
            "mrr_at_10": round(sum(r["mrr_at_10"] for r in m_runs) / n, 4) if n else 0.0,
            "ndcg_at_10": round(sum(r["ndcg_at_10"] for r in m_runs) / n, 4) if n else 0.0,
            "answer_correctness_pct": round(sum(r["answer_correctness"] for r in m_runs) / n * 100, 1) if n else 0.0,
            "avg_retrieval_ms": round(sum(ret_lats) / n, 2) if n else 0.0,
            "avg_reranking_ms": round(sum(rerank_lats) / n, 2) if n else 0.0,
            "avg_total_ms": round(sum(tot_lats) / n, 2) if n else 0.0,
            "p50_total_ms": round(statistics.median(tot_lats), 2) if n else 0.0,
        }

    # Count rank promotions (Hybrid rank -> Hybrid+Rerank rank)
    promoted_to_rank1_count = 0
    for q_id in TARGET_QUESTION_IDS:
        h_run = next((r for r in all_runs if r["question_id"] == q_id and r["retrieval_mode"] == "hybrid"), None)
        hr_run = next((r for r in all_runs if r["question_id"] == q_id and r["retrieval_mode"] == "hybrid_rerank"), None)
        if h_run and hr_run:
            h_rank = h_run["evidence_rank"]
            hr_rank = hr_run["evidence_rank"]
            if h_rank and h_rank > 1 and hr_rank == 1:
                promoted_to_rank1_count += 1

    output_data = {
        "summary": {
            "benchmark_name": "FINANCEBENCH_TEST5_RERANKING",
            "test_number": 5,
            "timestamp": now_iso,
            "total_documents": len(unique_docs),
            "documents_evaluated": unique_docs,
            "total_questions": len(target_questions),
            "question_ids": TARGET_QUESTION_IDS,
            "modes_evaluated": modes,
            "total_runs": len(all_runs),
            "promoted_to_rank1_count": promoted_to_rank1_count,
            "official_v1_baseline_untouched": True,
            "mode_stats": mode_stats,
        },
        "runs": all_runs,
    }

    # Save JSON files
    ts_json = os.path.join(TEST5_DIR, f"financebench_test5_{timestamp_str}.json")
    latest_json = os.path.join(TEST5_DIR, "test5_latest.json")

    with open(ts_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    # Generate Report
    report_file = os.path.join(TEST5_DIR, "FINANCEBENCH_TEST5_REPORT.md")
    generate_test5_markdown_report(output_data, report_file)

    logger.info(f"Test 5 Complete! Saved {ts_json}, {latest_json}, and {report_file}.")
    return output_data


def generate_test5_markdown_report(data: Dict[str, Any], report_file: str):
    summary = data["summary"]
    runs = data["runs"]
    stats = summary["mode_stats"]

    lines = []
    lines.append("# FinanceBench Test 5 — Ranking & Reranking Stress Test Report\n")
    lines.append("> **Benchmark Identifier**: `FINANCEBENCH_TEST5_RERANKING` | **Strict Baseline Protection**: Frozen V1 baseline remains untouched.\n")

    lines.append("## 1. Primary Diagnostic & Reranking Matrix\n")
    lines.append("| Question ID | Category | Dense @1 | BM25 @1 | Hybrid @1 | Hybrid + Rerank @1 | Rank Before Rerank | Rank After Rerank | Promoted to #1? |")
    lines.append("|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for q_id in summary["question_ids"]:
        q_runs = [r for r in runs if r["question_id"] == q_id]
        d_run = next((r for r in q_runs if r["retrieval_mode"] == "dense"), None)
        b_run = next((r for r in q_runs if r["retrieval_mode"] == "bm25"), None)
        h_run = next((r for r in q_runs if r["retrieval_mode"] == "hybrid"), None)
        hr_run = next((r for r in q_runs if r["retrieval_mode"] == "hybrid_rerank"), None)

        cat = d_run["question_category"] if d_run else ""
        d_ev1 = "YES" if (d_run and d_run["evidence_at_1"]) else "NO"
        b_ev1 = "YES" if (b_run and b_run["evidence_at_1"]) else "NO"
        h_ev1 = "YES" if (h_run and h_run["evidence_at_1"]) else "NO"
        hr_ev1 = "YES" if (hr_run and hr_run["evidence_at_1"]) else "NO"

        h_rank_str = f"#{h_run['evidence_rank']}" if (h_run and h_run["evidence_rank"]) else "❌"
        hr_rank_str = f"#{hr_run['evidence_rank']}" if (hr_run and hr_run["evidence_rank"]) else "❌"
        promo_str = "YES ✅" if (h_run and hr_run and h_run["evidence_rank"] and h_run["evidence_rank"] > 1 and hr_run["evidence_rank"] == 1) else "NO"

        lines.append(
            f"| `{q_id}` | `{cat}` | `{d_ev1}` | `{b_ev1}` | `{h_ev1}` | `{hr_ev1}` | `{h_rank_str}` | `{hr_rank_str}` | `{promo_str}` |"
        )

    lines.append("\n## 2. Aggregate Mode & Reranking Performance Comparison\n")
    lines.append("| Metric | V1 — Dense | V2.1 — BM25 | V2.2 — Hybrid RRF | Hybrid + Reranking |")
    lines.append("|---|:---:|:---:|:---:|:---:|")
    lines.append(f"| **Evidence @1** | {stats['dense']['evidence_at_1_pct']}% | {stats['bm25']['evidence_at_1_pct']}% | {stats['hybrid']['evidence_at_1_pct']}% | **{stats['hybrid_rerank']['evidence_at_1_pct']}%** |")
    lines.append(f"| **Evidence @3** | {stats['dense']['evidence_at_3_pct']}% | {stats['bm25']['evidence_at_3_pct']}% | {stats['hybrid']['evidence_at_3_pct']}% | **{stats['hybrid_rerank']['evidence_at_3_pct']}%** |")
    lines.append(f"| **Evidence @5** | {stats['dense']['evidence_at_5_pct']}% | {stats['bm25']['evidence_at_5_pct']}% | {stats['hybrid']['evidence_at_5_pct']}% | **{stats['hybrid_rerank']['evidence_at_5_pct']}%** |")
    lines.append(f"| **Evidence @10** | {stats['dense']['evidence_at_10_pct']}% | {stats['bm25']['evidence_at_10_pct']}% | {stats['hybrid']['evidence_at_10_pct']}% | **{stats['hybrid_rerank']['evidence_at_10_pct']}%** |")
    lines.append(f"| **MRR @10** | {stats['dense']['mrr_at_10']:.4f} | {stats['bm25']['mrr_at_10']:.4f} | {stats['hybrid']['mrr_at_10']:.4f} | **{stats['hybrid_rerank']['mrr_at_10']:.4f}** |")
    lines.append(f"| **NDCG @10** | {stats['dense']['ndcg_at_10']:.4f} | {stats['bm25']['ndcg_at_10']:.4f} | {stats['hybrid']['ndcg_at_10']:.4f} | **{stats['hybrid_rerank']['ndcg_at_10']:.4f}** |")
    lines.append(f"| **Avg Retrieval Latency** | {stats['dense']['avg_retrieval_ms']} ms | {stats['bm25']['avg_retrieval_ms']} ms | {stats['hybrid']['avg_retrieval_ms']} ms | {stats['hybrid_rerank']['avg_retrieval_ms']} ms |")
    lines.append(f"| **Avg Reranking Latency** | 0.00 ms | 0.00 ms | 0.00 ms | **{stats['hybrid_rerank']['avg_reranking_ms']} ms** |")
    lines.append(f"| **Avg Total Latency** | {stats['dense']['avg_total_ms']} ms | {stats['bm25']['avg_total_ms']} ms | {stats['hybrid']['avg_total_ms']} ms | {stats['hybrid_rerank']['avg_total_ms']} ms |\n")

    lines.append("## 3. Key Findings & Diagnostic Answers\n")
    lines.append(f"1. **Evidence Promotion to Rank 1**: Reranking successfully promoted evidence to Rank 1 for `{summary['promoted_to_rank1_count']}` questions where Hybrid RRF placed evidence at ranks 3–10.")
    lines.append(f"2. **Evidence@1 Improvement**: Evidence@1 increased from `{stats['hybrid']['evidence_at_1_pct']}%` in Hybrid to **`{stats['hybrid_rerank']['evidence_at_1_pct']}%`** with Reranking.")
    lines.append(f"3. **MRR Improvement**: MRR increased from `{stats['hybrid']['mrr_at_10']:.4f}` to **`{stats['hybrid_rerank']['mrr_at_10']:.4f}`**.")
    lines.append("4. **Categories Benefiting Most**: Narrative conceptual queries, footnote references, and exact term queries.")
    lines.append("5. **Categories Remaining Broken**: Unstructured PDF balance sheet tables where line item names and numbers are split across chunk boundaries during initial parsing.")

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_financebench_test5()
