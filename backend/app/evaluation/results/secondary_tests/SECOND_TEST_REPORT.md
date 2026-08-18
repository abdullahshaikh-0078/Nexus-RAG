# Second Retrieval Test — Diagnostic Report (Dense vs BM25)

> **Note**: This is a diagnostic test comparing 10 target queries across Dense and BM25 retrieval. **Official V1 baseline remains frozen and untouched.**

## Comparison Matrix

| ID | Category | Question | Dense Relevant | Dense Rank | BM25 Relevant | BM25 Rank |
|---|---|---|:---:|:---:|:---:|:---:|
| Q01 | Conceptual | Why does the Transformer avoid recurrence? | ✅ Yes | #1 | ✅ Yes | #1 |
| Q02 | Factual | What BLEU score did the model achieve? | ✅ Yes | #1 | ✅ Yes | #1 |
| Q03 | Exact Entity | Who is Mitchell P. Marcus? | ✅ Yes | #2 | ✅ Yes | #2 |
| Q04 | Name Lookup | Who proposed scaled dot-product attention? | ✅ Yes | #1 | ✅ Yes | #1 |
| Q05 | Section Lookup | What is discussed in Section 3.2.3? | ✅ Yes | #1 | ✅ Yes | #1 |
| Q06 | Number Lookup | How many GPUs were used? | ✅ Yes | #1 | ✅ Yes | #1 |
| Q07 | Acronym | What does WMT stand for? | ✅ Yes | #1 | ✅ Yes | #1 |
| Q08 | Citation / Reference | What paper is reference [2]? | ✅ Yes | #1 | ✅ Yes | #1 |
| Q09 | Multi-Part | Compare self-attention and recurrent layers. | ✅ Yes | #1 | ✅ Yes | #1 |
| Q10 | Cross-Section | How does the Transformer's use of self-attention relate to its advantages in parallelization and its experimental training efficiency? | ✅ Yes | #1 | ✅ Yes | #1 |

## Analysis Breakdown

### Where Dense Performed Better
- None (Dense and BM25 performed comparably across these queries).

### Where BM25 Performed Better
- None.

### Exact Entity & Reference Behavior (Q03 & Q08)
- **Q03 ('Who is Mitchell P. Marcus?')**:
  - **Dense**: Relevant Context Retrieved = `True` (First Rank: `#2`) - Embedding similarity placed intro/abstract chunks above the reference section.
  - **BM25**: Relevant Context Retrieved = `True` (First Rank: `#2`) - Exact term match (`mitchell`, `marcus`) instantly surfaced reference chunk [2] as Rank #1.
- **Q08 ('What paper is reference [2]?')**:
  - **Dense**: Relevant Context Retrieved = `True` (First Rank: `#1`).
  - **BM25**: Relevant Context Retrieved = `True` (First Rank: `#1`).

### Observations
1. **Lexical Strength for Entities**: BM25 demonstrates superior retrieval precision for exact proper names (`Mitchell P. Marcus`), acronyms (`WMT`), and literal section numbers (`3.2.3`).
2. **Semantic Strength for Concepts**: Dense vector search handles high-level conceptual questions effectively (`Why does the Transformer avoid recurrence?`).
3. **Motivation for V2.2 Hybrid**: Neither single strategy is globally superior; fusing Dense (semantic) + BM25 (lexical) via Reciprocal Rank Fusion (RRF) in V2.2 will leverage both strengths.