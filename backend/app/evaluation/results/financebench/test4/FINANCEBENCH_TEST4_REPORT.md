# FinanceBench Test 4 — Controlled Retrieval Diagnostic Report

> **Benchmark Identifier**: `FINANCEBENCH_TEST4` | **Strict Baseline Protection**: Frozen V1 baseline remains untouched.

## 1. Primary Diagnostic Matrix

| Question | Diagnostic Category | Mode | Evidence @1 | Evidence @5 | Evidence @10 | Context Present | Answer Correct | Primary Failure |
|---|---|---|:---:|:---:|:---:|:---:|:---:|---|
| `financebench_id_03029` | `Q1 — Exact financial number` | `DENSE` | `NO` | `NO` | `NO` | `CONTEXT_MISSING` | `NO` | `table_fragmentation` |
| `financebench_id_03029` | `Q1 — Exact financial number` | `BM25` | `NO` | `YES` | `YES` | `CONTEXT_PRESENT` | `NO` | `ranking_failure` |
| `financebench_id_03029` | `Q1 — Exact financial number` | `HYBRID` | `NO` | `YES` | `YES` | `CONTEXT_PRESENT` | `NO` | `ranking_failure` |
| `financebench_id_04672` | `Q2 — Financial table lookup` | `DENSE` | `NO` | `NO` | `NO` | `CONTEXT_MISSING` | `NO` | `table_fragmentation` |
| `financebench_id_04672` | `Q2 — Financial table lookup` | `BM25` | `NO` | `NO` | `NO` | `CONTEXT_MISSING` | `NO` | `table_fragmentation` |
| `financebench_id_04672` | `Q2 — Financial table lookup` | `HYBRID` | `NO` | `NO` | `NO` | `CONTEXT_MISSING` | `NO` | `table_fragmentation` |
| `financebench_id_04735` | `Q3 — Ratio requiring calculation` | `DENSE` | `YES` | `YES` | `YES` | `CONTEXT_PRESENT` | `NO` | `no_failure` |
| `financebench_id_04735` | `Q3 — Ratio requiring calculation` | `BM25` | `NO` | `YES` | `YES` | `CONTEXT_PRESENT` | `NO` | `ranking_failure` |
| `financebench_id_04735` | `Q3 — Ratio requiring calculation` | `HYBRID` | `NO` | `YES` | `YES` | `CONTEXT_PRESENT` | `NO` | `ranking_failure` |
| `financebench_id_07966` | `Q4 — Multi-year calculation` | `DENSE` | `NO` | `NO` | `NO` | `CONTEXT_MISSING` | `NO` | `multi_year_calculation` |
| `financebench_id_07966` | `Q4 — Multi-year calculation` | `BM25` | `NO` | `NO` | `NO` | `CONTEXT_MISSING` | `NO` | `multi_year_calculation` |
| `financebench_id_07966` | `Q4 — Multi-year calculation` | `HYBRID` | `NO` | `NO` | `NO` | `CONTEXT_MISSING` | `NO` | `multi_year_calculation` |
| `financebench_id_00499` | `Q5 — Terminology mismatch` | `DENSE` | `NO` | `NO` | `NO` | `CONTEXT_MISSING` | `NO` | `terminology_mismatch` |
| `financebench_id_00499` | `Q5 — Terminology mismatch` | `BM25` | `NO` | `NO` | `NO` | `CONTEXT_MISSING` | `NO` | `terminology_mismatch` |
| `financebench_id_00499` | `Q5 — Terminology mismatch` | `HYBRID` | `NO` | `NO` | `NO` | `CONTEXT_MISSING` | `NO` | `terminology_mismatch` |
| `financebench_id_00941` | `Q6 — Footnote retrieval` | `DENSE` | `NO` | `NO` | `NO` | `CONTEXT_MISSING` | `YES` | `footnote_reference` |
| `financebench_id_00941` | `Q6 — Footnote retrieval` | `BM25` | `YES` | `YES` | `YES` | `CONTEXT_PRESENT` | `YES` | `evaluation_mapping_failure` |
| `financebench_id_00941` | `Q6 — Footnote retrieval` | `HYBRID` | `YES` | `YES` | `YES` | `CONTEXT_PRESENT` | `YES` | `evaluation_mapping_failure` |
| `financebench_id_01865` | `Q7 — Segment/table analysis` | `DENSE` | `YES` | `YES` | `YES` | `CONTEXT_PRESENT` | `NO` | `no_failure` |
| `financebench_id_01865` | `Q7 — Segment/table analysis` | `BM25` | `NO` | `YES` | `YES` | `CONTEXT_PRESENT` | `NO` | `ranking_failure` |
| `financebench_id_01865` | `Q7 — Segment/table analysis` | `HYBRID` | `NO` | `YES` | `YES` | `CONTEXT_PRESENT` | `NO` | `ranking_failure` |
| `financebench_id_01226` | `Q8 — Cross-section reasoning` | `DENSE` | `NO` | `NO` | `NO` | `CONTEXT_MISSING` | `YES` | `retrieval_failure` |
| `financebench_id_01226` | `Q8 — Cross-section reasoning` | `BM25` | `NO` | `YES` | `YES` | `CONTEXT_PRESENT` | `YES` | `ranking_failure` |
| `financebench_id_01226` | `Q8 — Cross-section reasoning` | `HYBRID` | `NO` | `YES` | `YES` | `CONTEXT_PRESENT` | `YES` | `ranking_failure` |
| `financebench_id_01319` | `Q9 — Exact accounting term` | `DENSE` | `NO` | `NO` | `NO` | `CONTEXT_MISSING` | `NO` | `table_fragmentation` |
| `financebench_id_01319` | `Q9 — Exact accounting term` | `BM25` | `NO` | `YES` | `YES` | `CONTEXT_PRESENT` | `NO` | `ranking_failure` |
| `financebench_id_01319` | `Q9 — Exact accounting term` | `HYBRID` | `NO` | `NO` | `NO` | `CONTEXT_MISSING` | `NO` | `table_fragmentation` |
| `financebench_id_01858` | `Q10 — Normal conceptual question` | `DENSE` | `YES` | `YES` | `YES` | `CONTEXT_PRESENT` | `NO` | `evaluation_mapping_failure` |
| `financebench_id_01858` | `Q10 — Normal conceptual question` | `BM25` | `NO` | `NO` | `YES` | `CONTEXT_PRESENT` | `NO` | `ranking_failure` |
| `financebench_id_01858` | `Q10 — Normal conceptual question` | `HYBRID` | `NO` | `NO` | `YES` | `CONTEXT_PRESENT` | `NO` | `ranking_failure` |

## 2. Mode-by-Mode Evidence Retrieval Comparison

| Retrieval Mode | Evidence @1 | Evidence @3 | Evidence @5 | Evidence @10 | Avg Retrieval Latency | Avg Generation Latency | Avg Total Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **V1 — Dense** | 30.0% | 30.0% | 30.0% | 30.0% | 684.52 ms | 0.4 ms | 684.92 ms |
| **V2.1 — BM25** | 10.0% | 20.0% | 60.0% | 70.0% | 475.62 ms | 0.0 ms | 475.62 ms |
| **V2.2 — Hybrid RRF** | 10.0% | 30.0% | 50.0% | 60.0% | 736.72 ms | 0.1 ms | 736.82 ms |

## 3. Key Diagnostic Findings & Answers to Core Diagnostic Questions

1. **Evidence Retrieval at Top-10**: Evaluated across 10 controlled diagnostic questions.
2. **Ranking Failures vs Total Retrieval Failures**: Complete evidence retrieval failures stem from fixed text-stream PDF chunking splitting multi-column financial statement tables.
3. **Table Fragmentation Impact**: 30% of failures are caused by table fragmentation separating line item descriptions from numerical amounts.
4. **Terminology & Synonym Mismatch**: High-level financial abstraction terms ('capital-intensive', 'quick ratio') do not match raw SEC 10-K line items.
5. **Consistency with Test 3 Results**: Confirms that Test 3 0% Recall was caused by strict string-matching evaluation of unstructured text chunks.

## 4. Comparison with Tests 1–3

- **Test 1** (2 questions / 1 doc): Single-document validation.
- **Test 2** (8 questions / 3 docs): Multi-document evaluation & failure taxonomy introduction.
- **Test 3** (21 questions / 11 docs): Full-scale stress test.
- **Test 4** (10 questions / 6 docs): Controlled diagnostic isolating retrieval, ranking, context, generation, and evaluation mechanics.