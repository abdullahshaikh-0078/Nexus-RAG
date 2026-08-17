# NEXUS RAG Evaluation Results Store

This directory contains versioned evaluation result JSON files for NEXUS RAG.

## Contents

- `latest.json`: Pointer to the most recent successful baseline evaluation run result.
- `v1_baseline_<timestamp>.json`: Timestamped evaluation snapshot for the V1 Dense Retrieval baseline.

## Result Schema Overview

Each result file contains:
- `evaluation_version`: Version identifier (e.g. `v1_baseline`)
- `timestamp`: ISO 8601 UTC timestamp
- `embedding_model`: Model identifier (`all-MiniLM-L6-v2`)
- `chunk_size` & `chunk_overlap`: RAG chunking parameters (1000 / 150)
- `retrieval_top_k`: Search top-K limit
- `total_questions`: Total test cases in evaluation dataset
- `aggregate_recall_at_1`, `@3`, `@5`, `@10`: Recall metrics across dataset
- `aggregate_mrr_at_10`: Mean Reciprocal Rank
- `aggregate_ndcg_at_10`: Normalized Discounted Cumulative Gain
- `average_retrieval_latency_ms`: Isolated retrieval search & expansion time (ms)
- `question_results`: Detailed per-question metrics breakdown
