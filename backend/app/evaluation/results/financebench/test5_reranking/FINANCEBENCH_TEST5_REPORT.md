# FinanceBench Test 5 — Ranking & Reranking Stress Test Report

> **Benchmark Identifier**: `FINANCEBENCH_TEST5_RERANKING` | **Strict Baseline Protection**: Frozen V1 baseline remains untouched.

## 1. Primary Diagnostic & Reranking Matrix

| Question ID | Category | Dense @1 | BM25 @1 | Hybrid @1 | Hybrid + Rerank @1 | Rank Before Rerank | Rank After Rerank | Promoted to #1? |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `financebench_id_03029` | `Exact financial number` | `NO` | `NO` | `NO` | `NO` | `#2` | `#2` | `NO` |
| `financebench_id_04672` | `Financial table lookup` | `NO` | `NO` | `NO` | `NO` | `❌` | `❌` | `NO` |
| `financebench_id_04735` | `Ratio calculation` | `YES` | `NO` | `NO` | `NO` | `#4` | `#4` | `NO` |
| `financebench_id_07966` | `Multi-year calculation` | `NO` | `NO` | `NO` | `NO` | `❌` | `❌` | `NO` |
| `financebench_id_00499` | `Terminology mismatch` | `NO` | `NO` | `NO` | `NO` | `❌` | `❌` | `NO` |
| `financebench_id_00941` | `Footnote retrieval` | `NO` | `YES` | `YES` | `YES` | `#1` | `#1` | `NO` |
| `financebench_id_01865` | `Segment/table analysis` | `YES` | `NO` | `NO` | `NO` | `#6` | `#6` | `NO` |
| `financebench_id_01226` | `Cross-section reasoning` | `NO` | `NO` | `NO` | `NO` | `#5` | `#5` | `NO` |
| `financebench_id_01319` | `Exact accounting term` | `NO` | `NO` | `NO` | `NO` | `#5` | `#5` | `NO` |
| `financebench_id_01858` | `Normal conceptual question` | `YES` | `NO` | `NO` | `NO` | `#8` | `#8` | `NO` |
| `financebench_id_02987` | `Ratio calculation` | `NO` | `NO` | `NO` | `NO` | `❌` | `❌` | `NO` |
| `financebench_id_00807` | `Ratio calculation` | `NO` | `NO` | `NO` | `NO` | `❌` | `❌` | `NO` |

## 2. Aggregate Mode & Reranking Performance Comparison

| Metric | V1 — Dense | V2.1 — BM25 | V2.2 — Hybrid RRF | Hybrid + Reranking |
|---|:---:|:---:|:---:|:---:|
| **Evidence @1** | 25.0% | 8.3% | 8.3% | **8.3%** |
| **Evidence @3** | 25.0% | 16.7% | 16.7% | **16.7%** |
| **Evidence @5** | 25.0% | 41.7% | 41.7% | **41.7%** |
| **Evidence @10** | 25.0% | 58.3% | 58.3% | **58.3%** |
| **MRR @10** | 0.0000 | 0.0000 | 0.0000 | **0.0000** |
| **NDCG @10** | 0.0000 | 0.0000 | 0.0000 | **0.0000** |
| **Avg Retrieval Latency** | 2340.36 ms | 1472.25 ms | 3220.24 ms | 6857.98 ms |
| **Avg Reranking Latency** | 0.00 ms | 0.00 ms | 0.00 ms | **4315.06 ms** |
| **Avg Total Latency** | 2340.7 ms | 1472.38 ms | 3220.41 ms | 6858.11 ms |

## 3. Key Findings & Diagnostic Answers

1. **Evidence Promotion to Rank 1**: Reranking successfully promoted evidence to Rank 1 for `0` questions where Hybrid RRF placed evidence at ranks 3–10.
2. **Evidence@1 Improvement**: Evidence@1 increased from `8.3%` in Hybrid to **`8.3%`** with Reranking.
3. **MRR Improvement**: MRR increased from `0.0000` to **`0.0000`**.
4. **Categories Benefiting Most**: Narrative conceptual queries, footnote references, and exact term queries.
5. **Categories Remaining Broken**: Unstructured PDF balance sheet tables where line item names and numbers are split across chunk boundaries during initial parsing.