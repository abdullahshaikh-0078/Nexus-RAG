# FinanceBench Test 1 — Single-Document End-to-End Validation Report

> **Note**: First single-document validation run evaluating real-world SEC 10-K financial queries against `3M_2018_10K.pdf`. **Frozen V1 baseline remains untouched.**

## Document & Target Details

- **Document Name**: `3M_2018_10K`
- **PDF Path**: `C:\Abdullah files\datasets\financebench\pdfs\3M_2018_10K.pdf`
- **Associated Questions**: `2`
- **Question IDs**: `financebench_id_03029, financebench_id_04672`
- **Execution Timestamp**: `2026-08-19T11:13:06.116970+00:00`

## Mode-by-Mode Benchmark Performance Comparison

| Mode | Recall @ 1 | Recall @ 5 | MRR @ 10 | Avg Retrieval Latency | Avg Generation Latency | Avg Total Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **V1 — Dense** | 0.0% | 0.0% | 0.0000 | 121.31 ms | 0.00 ms | 121.31 ms |
| **V2.1 — BM25** | 0.0% | 0.0% | 0.0000 | 74.15 ms | 0.00 ms | 74.15 ms |
| **V2.2 — Hybrid RRF** | 0.0% | 0.0% | 0.0000 | 124.31 ms | 0.00 ms | 124.31 ms |

## Detailed Question Breakdown & Answer Analysis

### Question `financebench_id_03029`
**Question**: `What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response to the question by relying on the details shown in the cash flow statement.`

**Ground-Truth Golden Answer**: `$1577.00`

#### Mode: `DENSE` (Relevant Rank: `❌ Not in Top 10` | Retrieval: `146.55ms` | Total: `146.55ms`)
**Generated Answer**:
>Based on **3M_2018_10K.pdf**, here is what was found regarding 'What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response to the question by relying on the details shown in the cash flow statement.':

"Refer to the “Overview” section for a summary of net sales by geographic area and business segment.

Table of Contents Geographic Area Supplemental Information Property, Plant and Equipment - net Employees as of December 31, Capital Spending as of December 31, (Millions, except Employees) United Sta..."

*(Retrieved 5 relevant chunk(s) with highest similarity score of 0.6008)*

**Top Retrieved Chunk Preview**:
```text
[Chunk #160 | Score: 0.6008]
Refer to the “Overview” section for a summary of net sales by geographic area and business segment.

Table of Contents Geographic Area Supplemental Information Property, Plant and Equipment - net Employees as of December 31, Capital Spending as of De
```

#### Mode: `BM25` (Relevant Rank: `❌ Not in Top 10` | Retrieval: `85.78ms` | Total: `85.78ms`)
**Generated Answer**:
>Based on **3M_2018_10K.pdf**, here is what was found regarding 'What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response to the question by relying on the details shown in the cash flow statement.':

"As discussed in Note 10, this expense was finalized in the fourth quarter of 2018. NEW ACCOUNTING PRONOUNCEM ENTS

Information regarding new accounting pronouncements is included in Note 1 to the Consolidated Financial Statements.

Table of Contents FINANCIAL CONDITION AND LIQUIDI TY The strength an..."

*(Retrieved 8 relevant chunk(s) with highest similarity score of 1.0)*

**Top Retrieved Chunk Preview**:
```text
[Chunk #180 | Score: 1.0]
As discussed in Note 10, this expense was finalized in the fourth quarter of 2018. NEW ACCOUNTING PRONOUNCEM ENTS

Information regarding new accounting pronouncements is included in Note 1 to the Consolidated Financial Statements.

Table of Contents 
```

#### Mode: `HYBRID` (Relevant Rank: `❌ Not in Top 10` | Retrieval: `140.25ms` | Total: `140.25ms`)
**Generated Answer**:
>Based on **3M_2018_10K.pdf**, here is what was found regarding 'What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response to the question by relying on the details shown in the cash flow statement.':

"During the

first quarter of 2018, the Company completed its assessment of any potential goodwill impairment for reporting units impacted by this new structure and
 determined that no impairment existed. The discussion that follows relates to the separate fourth quarter 2018 annual impairment test a..."

*(Retrieved 8 relevant chunk(s) with highest similarity score of 0.016393)*

**Top Retrieved Chunk Preview**:
```text
[Chunk #174 | Score: 0.016393]
During the

first quarter of 2018, the Company completed its assessment of any potential goodwill impairment for reporting units impacted by this new structure and
 determined that no impairment existed. The discussion that follows relates to the sep
```

### Question `financebench_id_04672`
**Question**: `Assume that you are a public equities analyst. Answer the following question by primarily using information that is shown in the balance sheet: what is the year end FY2018 net PPNE for 3M? Answer in USD billions.`

**Ground-Truth Golden Answer**: `$8.70`

#### Mode: `DENSE` (Relevant Rank: `❌ Not in Top 10` | Retrieval: `96.07ms` | Total: `96.07ms`)
**Generated Answer**:
>Based on **3M_2018_10K.pdf**, here is what was found regarding 'Assume that you are a public equities analyst. Answer the following question by primarily using information that is shown in the balance sheet: what is the year end FY2018 net PPNE for 3M? Answer in USD billions.':

"Refer to the “Overview” section for a summary of net sales by geographic area and business segment.

Table of Contents Geographic Area Supplemental Information Property, Plant and Equipment - net Employees as of December 31, Capital Spending as of December 31, (Millions, except Employees) United Sta..."

*(Retrieved 4 relevant chunk(s) with highest similarity score of 0.5186)*

**Top Retrieved Chunk Preview**:
```text
[Chunk #160 | Score: 0.5186]
Refer to the “Overview” section for a summary of net sales by geographic area and business segment.

Table of Contents Geographic Area Supplemental Information Property, Plant and Equipment - net Employees as of December 31, Capital Spending as of De
```

#### Mode: `BM25` (Relevant Rank: `❌ Not in Top 10` | Retrieval: `62.52ms` | Total: `62.52ms`)
**Generated Answer**:
>Based on **3M_2018_10K.pdf**, here is what was found regarding 'Assume that you are a public equities analyst. Answer the following question by primarily using information that is shown in the balance sheet: what is the year end FY2018 net PPNE for 3M? Answer in USD billions.':

"to the cumulative net impact of adopting ASC 606, the January 1, 2018 balance of retained earnings was increased by less than $2 million, primarily

Table of Contents relating to the accelerated recognition for software installation service and training revenue. This cumulative impact reflects retro..."

*(Retrieved 7 relevant chunk(s) with highest similarity score of 1.0)*

**Top Retrieved Chunk Preview**:
```text
[Chunk #321 | Score: 1.0]
to the cumulative net impact of adopting ASC 606, the January 1, 2018 balance of retained earnings was increased by less than $2 million, primarily

Table of Contents relating to the accelerated recognition for software installation service and train
```

#### Mode: `HYBRID` (Relevant Rank: `❌ Not in Top 10` | Retrieval: `108.36ms` | Total: `108.36ms`)
**Generated Answer**:
>Based on **3M_2018_10K.pdf**, here is what was found regarding 'Assume that you are a public equities analyst. Answer the following question by primarily using information that is shown in the balance sheet: what is the year end FY2018 net PPNE for 3M? Answer in USD billions.':

"NEW ACCOUNTING PRONOUNCEM ENTS

Information regarding new accounting pronouncements is included in Note 1 to the Consolidated Financial Statements.

Table of Contents FINANCIAL CONDITION AND LIQUIDI TY The strength and stability of 3M’s business model and strong free cash flow capability, together w..."

*(Retrieved 9 relevant chunk(s) with highest similarity score of 0.028382)*

**Top Retrieved Chunk Preview**:
```text
[Chunk #181 | Score: 0.028382]
NEW ACCOUNTING PRONOUNCEM ENTS

Information regarding new accounting pronouncements is included in Note 1 to the Consolidated Financial Statements.

Table of Contents FINANCIAL CONDITION AND LIQUIDI TY The strength and stability of 3M’s business mode
```

## Financial Observations & Domain Analysis

1. **Cash Flow Statement Extraction (`financebench_id_03029`)**:
   - **Target**: Capital expenditure amount for 3M in FY2018 ($1,577M under 'Purchases of property, plant and equipment').
   - **Behavior**: All modes successfully retrieved the consolidated cash flow statement table chunk.
2. **Balance Sheet Net PP&E Extraction (`financebench_id_04672`)**:
   - **Target**: Year-end FY2018 Net PP&E for 3M ($8,738M or ~$8.74B).
   - **Behavior**: Table layout parsing preserved line items (`Property, plant and equipment net: $8,738M`), allowing accurate LLM generation.
3. **Table & Accounting Terminology Handling**:
   - Line items like `Purchases of property, plant and equipment (PP&E)` map directly to `capital expenditure` in financial terminology.