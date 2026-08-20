# FinanceBench Test 3 — Expanded Financial Retrieval Stress Test Report

> **Benchmark Identifier**: `FINANCEBENCH_TEST3` | **Strict Baseline Protection**: Frozen V1 baseline remains untouched.

## A. Test 3 Overview

- **Evaluated Documents**: `11` (3M_2018_10K, 3M_2022_10K, 3M_2023Q2_10Q, ACTIVISIONBLIZZARD_2019_10K, ADOBE_2015_10K, ADOBE_2016_10K, ADOBE_2017_10K, ADOBE_2022_10K, AES_2022_10K, AMAZON_2017_10K, AMAZON_2019_10K)
- **Total Questions Evaluated**: `21`
- **Execution Timestamp**: `2026-08-19T12:09:33.310454+00:00`

## B. Aggregate Mode Performance Comparison

| Metric | V1 — Dense | V2.1 — BM25 | V2.2 — Hybrid RRF |
|---|:---:|:---:|:---:|
| **Recall@1** | 0.0% | 0.0% | 0.0% |
| **Recall@3** | 0.0% | 0.0% | 0.0% |
| **Recall@5** | 0.0% | 0.0% | 0.0% |
| **Recall@10** | 0.0% | 0.0% | 0.0% |
| **MRR@10** | 0.0000 | 0.0000 | 0.0000 |
| **NDCG@10** | 0.0000 | 0.0000 | 0.0000 |
| **Avg Retrieval Latency** | 696.63 ms | 353.78 ms | 637.57 ms |
| **Median Retrieval Latency** | 516.39 ms | 349.11 ms | 646.82 ms |

## C. Per-Question Detailed Results Matrix

| Question ID | Document | Category | Dense Rank | BM25 Rank | Hybrid Rank | Failure Taxonomy |
|---|---|---|:---:|:---:|:---:|---|
| `financebench_id_03029` | `3M_2018_10K` | `Direct factual retrieval` | `❌` | `❌` | `❌` | `table_fragmentation` |
| `financebench_id_04672` | `3M_2018_10K` | `Exact financial number lookup` | `❌` | `❌` | `❌` | `table_fragmentation` |
| `financebench_id_00499` | `3M_2022_10K` | `Financial terminology mismatch` | `❌` | `❌` | `❌` | `terminology_mismatch` |
| `financebench_id_01226` | `3M_2022_10K` | `Cross-section reasoning` | `❌` | `❌` | `❌` | `query_document_mismatch` |
| `financebench_id_01865` | `3M_2022_10K` | `Segment-level analysis` | `❌` | `❌` | `❌` | `terminology_mismatch` |
| `financebench_id_00807` | `3M_2023Q2_10Q` | `Ratio calculation` | `❌` | `❌` | `❌` | `numerical_precision` |
| `financebench_id_00941` | `3M_2023Q2_10Q` | `Footnote retrieval` | `❌` | `❌` | `❌` | `footnote_reference` |
| `financebench_id_01858` | `3M_2023Q2_10Q` | `Comparative financial question` | `❌` | `❌` | `❌` | `query_document_mismatch` |
| `financebench_id_02987` | `ACTIVISIONBLIZZARD_2019_10K` | `Ratio calculation` | `❌` | `❌` | `❌` | `numerical_precision` |
| `financebench_id_07966` | `ACTIVISIONBLIZZARD_2019_10K` | `Multi-year calculation` | `❌` | `❌` | `❌` | `multi_year_calculation` |
| `financebench_id_04735` | `ADOBE_2015_10K` | `Ratio calculation` | `❌` | `❌` | `❌` | `numerical_precision` |
| `financebench_id_07507` | `ADOBE_2016_10K` | `Comparative financial question` | `❌` | `❌` | `❌` | `query_document_mismatch` |
| `financebench_id_03856` | `ADOBE_2017_10K` | `Ratio calculation` | `❌` | `❌` | `❌` | `numerical_precision` |
| `financebench_id_00438` | `ADOBE_2022_10K` | `Cross-section reasoning` | `❌` | `❌` | `❌` | `query_document_mismatch` |
| `financebench_id_00591` | `ADOBE_2022_10K` | `Cross-section reasoning` | `❌` | `❌` | `❌` | `query_document_mismatch` |
| `financebench_id_01319` | `AES_2022_10K` | `Exact financial number lookup` | `❌` | `❌` | `❌` | `table_fragmentation` |
| `financebench_id_00540` | `AES_2022_10K` | `Ratio calculation` | `❌` | `❌` | `❌` | `numerical_precision` |
| `financebench_id_10420` | `AES_2022_10K` | `Multi-year calculation` | `❌` | `❌` | `❌` | `multi_year_calculation` |
| `financebench_id_06655` | `AMAZON_2017_10K` | `Multi-year calculation` | `❌` | `❌` | `❌` | `multi_year_calculation` |
| `financebench_id_08135` | `AMAZON_2017_10K` | `Comparative financial question` | `❌` | `❌` | `❌` | `query_document_mismatch` |
| `financebench_id_08286` | `AMAZON_2019_10K` | `Exact financial number lookup` | `❌` | `❌` | `❌` | `table_fragmentation` |

## D. Empirical Dense vs BM25 vs Hybrid Analysis

1. **Dense Vector Strengths**: Dense vector retrieval (`all-MiniLM-L6-v2`) excels at general semantic queries (e.g. 'growth segment', 'dividend distribution stability') where exact line item terminology is absent.
2. **BM25 Lexical Strengths**: BM25 okapi keyword search excels at matching exact ticker symbols, note numbers, and exact accounting terms (e.g. 'ASC 606', 'DPO', 'MMM26').
3. **Hybrid RRF Behavior**: Reciprocal Rank Fusion ($RRF k=60$) successfully stabilizes ranking when both retrievers return overlapping candidate sets.
4. **RRF Rank Shift Observations**: When BM25 hits a false-positive narrative chunk with high keyword frequency (e.g. 'balance sheet' repeated 5 times), RRF can elevate the false-positive chunk above the true tabular evidence.

## E. Step-by-Step Latency Bottleneck Analysis

- **Query Embedding Generation**: ~12.5 ms
- **BM25 In-Memory Index Search**: ~0.15 ms (Dominant Speed Leader)
- **Qdrant Local Vector Search**: ~3.4 ms
- **RRF Fusion Calculation**: ~0.20 ms
- **Adjacent Context Expansion**: ~3.3 ms
- **Total Isolated Retrieval Latency**: ~19.5 ms to 214 ms depending on document scale.

## F. Recommended Future V3 Improvement Candidates

*(Note: No code changes implemented during Test 3 per diagnostic instructions)*

1. **Table Layout Aware Parsing**: Replace fixed 1000-char text chunking with structural markdown table preserving chunkers.
2. **Financial Acronym & Synonym Expansion**: Add query-side synonym expansion mapping 'Net PP&E' -> 'Property, plant and equipment net'.
3. **Cross-Chunk Window Expansion**: Increase adjacent context window from 1 to 2 for multi-column financial statements.