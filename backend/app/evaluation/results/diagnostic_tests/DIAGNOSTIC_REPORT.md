# NEXUS RAG V2 Diagnostic Evaluation & Latency Validation Report

> **Note**: This diagnostic evaluation runs 36 test questions across 12 categories. **Official V1 baseline remains frozen and untouched.**

## Overall Benchmark Performance Matrix (36 Diagnostic Questions)

| Metric | V1 Dense | V2.1 BM25 | V2.2 Hybrid (RRF) | Delta (Hybrid vs Dense) |
|---|:---:|:---:|:---:|:---:|
| **Recall @ 1** | 77.8% | 52.8% | 52.8% | `-25.0 pp` |
| **Recall @ 3** | 77.8% | 63.9% | 55.6% | `-22.2 pp` |
| **Recall @ 5** | 77.8% | 63.9% | 55.6% | `-22.2 pp` |
| **Recall @ 10** | 77.8% | 63.9% | 55.6% | `-22.2 pp` |
| **MRR @ 10** | 0.7778 | 0.5833 | 0.5417 | `-0.2361` |
| **NDCG @ 10** | 0.4627 | 0.3524 | 0.3202 | `-0.1425` |
| **Avg Latency** | 2558.41 ms | 1360.39 ms | 2665.54 ms | `+107.13 ms` |

## Per-Category Performance Breakdown

| Category | Dense Recall@1 | BM25 Recall@1 | Hybrid Recall@1 | Hybrid MRR@10 |
|---|:---:|:---:|:---:|:---:|
| Conceptual | 100.0% | 100.0% | 100.0% | 1.0000 |
| Factual | 100.0% | 0.0% | 0.0% | 0.0000 |
| Exact Entity / Name Lookup | 100.0% | 33.3% | 33.3% | 0.3333 |
| Section-Number Lookup | 66.7% | 66.7% | 66.7% | 0.6667 |
| Number / Value Lookup | 100.0% | 0.0% | 0.0% | 0.0000 |
| Acronym Lookup | 66.7% | 33.3% | 33.3% | 0.5000 |
| Citation / Reference Lookup | 100.0% | 0.0% | 0.0% | 0.0000 |
| Technical Terminology Lookup | 66.7% | 66.7% | 66.7% | 0.6667 |
| Multi-Part Questions | 66.7% | 66.7% | 66.7% | 0.6667 |
| Cross-Section Questions | 100.0% | 100.0% | 100.0% | 1.0000 |
| Exact Wording Matters | 0.0% | 66.7% | 66.7% | 0.6667 |
| Semantic Understanding | 66.7% | 100.0% | 100.0% | 1.0000 |

## Latency Breakdown & Bottleneck Analysis

Step-by-step latency instrumentation across diagnostic runs:

- **Query Embedding Generation (`SentenceTransformer.encode`)**: `40.48 ms`
- **Qdrant Dense Search (`search_similar`)**: `1443.12 ms`
- **BM25 Lexical Search (`bm25_service.search`)**: `62.38 ms`
- **RRF Fusion & Deduplication (`HybridRetriever.search`)**: `1484.44 ms`
- **Adjacent Context Expansion (`expand_adjacent_context` Qdrant scroll)**: `1181.1 ms`

> **IDENTIFIED BOTTLENECK**: Context Expansion (get_chunks_by_indices scrolling Qdrant points) + Query Embedding overhead

## Per-Question Comparison Table (36 Questions)

| ID | Category | Question | Dense Rank | BM25 Rank | Hybrid Rank | Hybrid Latency |
|---|---|---|:---:|:---:|:---:|:---:|
| D01 | Conceptual | Why does the Transformer avoid recurrence in sequence transduction? | #1 | #1 | #1 | 2486.0ms |
| D02 | Conceptual | How do attention mechanisms allow drawing global dependencies? | #1 | #1 | #1 | 2476.0ms |
| D03 | Conceptual | What is the primary architectural difference between recurrent neural networks and the Transformer? | #1 | #1 | #1 | 2351.0ms |
| D04 | Factual | What BLEU score did the big Transformer model achieve on WMT 2014 English-to-German? | #1 | -- | -- | 2551.1ms |
| D05 | Factual | What BLEU score was achieved on the WMT 2014 English-to-French translation task? | #1 | -- | -- | 3665.8ms |
| D06 | Factual | How long was the base model trained on 8 NVIDIA P100 GPUs? | #1 | -- | -- | 3611.9ms |
| D07 | Exact Entity / Name Lookup | Who is Mitchell P. Marcus? | #1 | #2 | -- | 2776.2ms |
| D08 | Exact Entity / Name Lookup | Who are the lead authors of the Attention Is All You Need paper? | #1 | #1 | #1 | 2367.1ms |
| D09 | Exact Entity / Name Lookup | Who co-authored the Penn Treebank paper with Mitchell P. Marcus? | #1 | -- | -- | 2289.0ms |
| D10 | Section-Number Lookup | What is discussed in Section 3.2.3? | #1 | #1 | #1 | 2238.3ms |
| D11 | Section-Number Lookup | Which section describes Scaled Dot-Product Attention? | -- | -- | -- | 2123.7ms |
| D12 | Section-Number Lookup | What topic is covered in Section 5.4? | #1 | #1 | #1 | 2621.1ms |
| D13 | Number / Value Lookup | How many GPUs were used for training the big Transformer model? | #1 | -- | -- | 2142.9ms |
| D14 | Number / Value Lookup | What was the step time for the big models during training? | #1 | -- | -- | 2116.3ms |
| D15 | Number / Value Lookup | By how much did the big model outperform existing models on English-to-German? | #1 | -- | -- | 2150.2ms |
| D16 | Acronym Lookup | What does WMT stand for? | #1 | #2 | #2 | 2260.5ms |
| D17 | Acronym Lookup | What recurrent neural network variations are mentioned in Introduction? | -- | #1 | #1 | 2174.7ms |
| D18 | Acronym Lookup | What conference proceedings published the Attention is All You Need paper in 2017? | #1 | #2 | -- | 4158.1ms |
| D19 | Citation / Reference Lookup | What paper is cited as reference [2]? | #1 | #2 | -- | 3863.6ms |
| D20 | Citation / Reference Lookup | Which citation corresponds to the Attention Is All You Need paper itself? | #1 | -- | -- | 3477.6ms |
| D21 | Citation / Reference Lookup | In what journal was the Penn Treebank paper published? | #1 | -- | -- | 2127.2ms |
| D22 | Technical Terminology Lookup | How is Scaled Dot-Product Attention calculated? | -- | -- | -- | 2609.4ms |
| D23 | Technical Terminology Lookup | What is the purpose of Positional Encoding in the Transformer? | #1 | #1 | #1 | 2448.0ms |
| D24 | Technical Terminology Lookup | Why does multi-head attention outperform a single attention head? | #1 | #1 | #1 | 2174.6ms |
| D25 | Multi-Part Questions | What are the three distinct ways multi-head attention is used in the Transformer model? | #1 | #1 | #1 | 2199.1ms |
| D26 | Multi-Part Questions | What are the key inputs and dimensions involved in computing attention weights? | -- | -- | -- | 2223.0ms |
| D27 | Multi-Part Questions | How do self-attention layers differ between the encoder and the decoder? | #1 | #1 | #1 | 2433.9ms |
| D28 | Cross-Section Questions | How does avoiding recurrence enable parallelization during model training? | #1 | #1 | #1 | 2459.6ms |
| D29 | Cross-Section Questions | How does maximum path length in self-attention compare to recurrent layers? | #1 | #1 | #1 | 2310.1ms |
| D30 | Cross-Section Questions | What relationship exists between sequence length n and representation dimensionality d for self-attention speed? | #1 | #1 | #1 | 2263.4ms |
| D31 | Exact Wording Matters | What exact phrase describes the network architecture proposed in the paper? | -- | #1 | #1 | 2308.3ms |
| D32 | Exact Wording Matters | What phrase is used to describe drawing global dependencies in the introduction? | -- | #1 | #1 | 2568.2ms |
| D33 | Exact Wording Matters | How is the scaling factor in dot-product attention explicitly written? | -- | -- | -- | 2446.2ms |
| D34 | Semantic Understanding | Why are Transformer models superior in quality while requiring less training time? | -- | #1 | #1 | 4791.6ms |
| D35 | Semantic Understanding | Why must positional encodings be added if sequence order information is needed? | #1 | #1 | #1 | 4252.6ms |
| D36 | Semantic Understanding | What advantage does constant path length provide for experimental training efficiency? | #1 | #1 | #1 | 2443.2ms |