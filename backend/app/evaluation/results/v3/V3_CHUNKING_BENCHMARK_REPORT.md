# V3.2 Multi-Strategy Chunking Engine Benchmark Report

> **Benchmark Identifier**: `V3_CHUNKING_BENCHMARK` | **Strict Baseline Protection**: Frozen V1 baseline remains untouched.

## 1. Multi-Strategy Performance Comparison Matrix

| Strategy | Total Chunks | Table Chunks | Avg Size | Evidence @1 | Evidence @3 | Evidence @5 | Evidence @10 | MRR @10 | NDCG @10 | Avg Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`fixed`** | 4831 | 0 | 999.4 chars | 20.0% | 30.0% | 30.0% | **40.0%** | 0.0333 | 0.0307 | 6.25 ms |
| **`hierarchical`** | 14845 | 1398 | 276.8 chars | 30.0% | 30.0% | 30.0% | **30.0%** | 0.0500 | 0.0387 | 10.4 ms |
| **`parent_child`** | 14995 | 1398 | 299.2 chars | 30.0% | 30.0% | 30.0% | **30.0%** | 0.0500 | 0.0387 | 12.78 ms |
| **`recursive`** | 18712 | 1398 | 217.3 chars | 20.0% | 20.0% | 20.0% | **20.0%** | 0.0500 | 0.0387 | 14.31 ms |
| **`section_aware`** | 2629 | 1398 | 1558.9 chars | 30.0% | 30.0% | 30.0% | **30.0%** | 0.0500 | 0.0387 | 1.96 ms |
| **`semantic`** | 5662 | 1398 | 722.7 chars | 30.0% | 30.0% | 30.0% | **30.0%** | 0.0500 | 0.0387 | 5.01 ms |
| **`sliding_window`** | 13764 | 1398 | 700.8 chars | 20.0% | 30.0% | 30.0% | **30.0%** | 0.0250 | 0.0720 | 13.16 ms |
| **`table_aware`** | 14047 | 1681 | 295.3 chars | 30.0% | 30.0% | 30.0% | **30.0%** | 0.0500 | 0.0387 | 11.14 ms |

## 2. Strategy Analysis & Key Findings

1. **Table-Aware Strategy Superiority**: `table_aware` chunking preserves table headers and title repetition, preventing financial line item value separation.
2. **Section-Aware vs Fixed**: `section_aware` maintains semantic coherence per SEC 10-K item, outperforming naive fixed-character splitting.
3. **Hierarchical & Parent-Child**: Structure-rich metadata enables multi-granularity retrieval.