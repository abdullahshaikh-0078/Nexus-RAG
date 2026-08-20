# FinanceBench Test 2 — Multi-Document 8-Question Diagnostic Report

> **Note**: Diagnostic evaluation across 8 FinanceBench questions (3 SEC 10-K/10-Q reports). **Frozen V1 baseline remains untouched.**

## Target Details

- **Evaluated Documents**: `ACTIVISIONBLIZZARD_2019_10K, 3M_2023Q2_10Q, 3M_2022_10K`
- **Total Questions**: `8`
- **Question IDs**: `financebench_id_00499, financebench_id_01226, financebench_id_01865, financebench_id_00807, financebench_id_00941, financebench_id_01858, financebench_id_02987, financebench_id_07966`
- **Execution Timestamp**: `2026-08-19T11:38:50.283371+00:00`

## Mode-by-Mode Benchmark Performance Comparison

| Mode | Recall @ 1 | Recall @ 5 | MRR @ 10 | NDCG @ 10 | Avg Retrieval Latency | Avg Generation Latency | Avg Total Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **V1 — Dense** | 0.0% | 0.0% | 0.0000 | 0.0000 | 192.38 ms | 0.00 ms | 192.38 ms |
| **V2.1 — BM25** | 0.0% | 0.0% | 0.0000 | 0.0000 | 97.71 ms | 0.00 ms | 97.71 ms |
| **V2.2 — Hybrid RRF** | 0.0% | 0.0% | 0.0000 | 0.0000 | 214.17 ms | 0.00 ms | 214.17 ms |

## Failure Mode Taxonomy & Domain Breakdown

1. **`terminology_mismatch`**: Query financial terms (e.g., 'quick ratio', 'capital-intensive') do not literally match line item phrasing in 10-K tables.
2. **`multi_year_calculation`**: Financial metrics requiring 3-year historical average calculation across separate table columns.
3. **`table_fragmentation`**: Fixed 1000-character chunking splits multi-column balance sheets across chunk boundaries.
4. **`footnote_reference`**: Small font footnote text embedded below financial tables.

## Per-Question Detailed Results

### Question `financebench_id_00499` (`3M_2022_10K`)
**Question**: `Is 3M a capital-intensive business based on FY2022 data?`

**Ground-Truth Answer**: `No, the company is managing its CAPEX and Fixed Assets pretty efficiently, which is evident from below key metrics:
CAPEX/Revenue Ratio: 5.1%
Fixed assets/Total Assets: 20%
Return on Assets= 12.4%`

**Failure Category**: `terminology_mismatch`

#### Mode: `DENSE` (Relevant Rank: `❌ Not in Top 10` | Latency: `215.73ms`)
**Generated Answer**:
>Based on **3M_2022_10K.pdf**, here is what was found regarding 'Is 3M a capital-intensive business based on FY2022 data?':

"modernization • Robotics and automation • Automotive/mobility • Electronic materials • Semiconductor • Graphic and architectural films • Wound care

• Healthcare IT
 • Biopharma filtration • Home improvement • Consumer safety & well-being • Package protection & shipping • Appearance auto care Distri..."

*(Retrieved 7 relevant chunk(s) with highest similarity score of 0.6581)*

#### Mode: `BM25` (Relevant Rank: `❌ Not in Top 10` | Latency: `112.63ms`)
**Generated Answer**:
>Based on **3M_2022_10K.pdf**, here is what was found regarding 'Is 3M a capital-intensive business based on FY2022 data?':

"The Company places consistent emphasis on

environmental responsibility. 3M has made, and plans to continue making, necessary expenditures for compliance with applicable laws and regulations. 3M is also involved in
 remediation actions relating to environmental matters from past operations at certai..."

*(Retrieved 7 relevant chunk(s) with highest similarity score of 1.0)*

#### Mode: `HYBRID` (Relevant Rank: `❌ Not in Top 10` | Latency: `219.25ms`)
**Generated Answer**:
>Based on **3M_2022_10K.pdf**, here is what was found regarding 'Is 3M a capital-intensive business based on FY2022 data?':

"The Company places consistent emphasis on

environmental responsibility. 3M has made, and plans to continue making, necessary expenditures for compliance with applicable laws and regulations. 3M is also involved in
 remediation actions relating to environmental matters from past operations at certai..."

*(Retrieved 6 relevant chunk(s) with highest similarity score of 0.032258)*

### Question `financebench_id_01226` (`3M_2022_10K`)
**Question**: `What drove operating margin change as of FY2022 for 3M? If operating margin is not a useful metric for a company like this, then please state that and explain why.`

**Ground-Truth Answer**: `Operating Margin for 3M in FY2022 has decreased by 1.7% primarily due to: 
-Decrease in gross Margin
-mostly one-off charges including Combat Arms Earplugs litigation, impairment related to exiting PFAS manufacturing, costs related to exiting Russia and divestiture-related restructuring
charges`

**Failure Category**: `table_fragmentation`

#### Mode: `DENSE` (Relevant Rank: `❌ Not in Top 10` | Latency: `189.05ms`)
**Generated Answer**:
>Based on **3M_2022_10K.pdf**, here is what was found regarding 'What drove operating margin change as of FY2022 for 3M? If operating margin is not a useful metric for a company like this, then please state that and explain why.':

"Item 6. [Reserved].

Table of Contents Item 7. Management’s Discussion and Analysis of Financial Condition and Results of Operations. Management’s Discussion and Analysis of Financial Condition and Results of Operations (MD&A) is designed to provide a reader of 3M’s financial statements with a narra..."

*(Retrieved 7 relevant chunk(s) with highest similarity score of 0.6466)*

#### Mode: `BM25` (Relevant Rank: `❌ Not in Top 10` | Latency: `98.94ms`)
**Generated Answer**:
>Based on **3M_2022_10K.pdf**, here is what was found regarding 'What drove operating margin change as of FY2022 for 3M? If operating margin is not a useful metric for a company like this, then please state that and explain why.':

"See further discussion in Note 16.

3M is also impacted by the Russia-Ukraine conflict. In light of a number of factors, 3M suspended operations of its subsidiaries in Russia in March 2022, the net sales of which
 were less than one percent of 3M’s consolidated net sales for 2021. Further, in Septem..."

*(Retrieved 7 relevant chunk(s) with highest similarity score of 1.0)*

#### Mode: `HYBRID` (Relevant Rank: `❌ Not in Top 10` | Latency: `217.62ms`)
**Generated Answer**:
>Based on **3M_2022_10K.pdf**, here is what was found regarding 'What drove operating margin change as of FY2022 for 3M? If operating margin is not a useful metric for a company like this, then please state that and explain why.':

"Item 6. [Reserved].

Table of Contents Item 7. Management’s Discussion and Analysis of Financial Condition and Results of Operations. Management’s Discussion and Analysis of Financial Condition and Results of Operations (MD&A) is designed to provide a reader of 3M’s financial statements with a narra..."

*(Retrieved 7 relevant chunk(s) with highest similarity score of 0.029851)*

### Question `financebench_id_01865` (`3M_2022_10K`)
**Question**: `If we exclude the impact of M&A, which segment has dragged down 3M's overall growth in 2022?`

**Ground-Truth Answer**: `The consumer segment shrunk by 0.9% organically.`

**Failure Category**: `table_fragmentation`

#### Mode: `DENSE` (Relevant Rank: `❌ Not in Top 10` | Latency: `188.89ms`)
**Generated Answer**:
>Based on **3M_2022_10K.pdf**, here is what was found regarding 'If we exclude the impact of M&A, which segment has dragged down 3M's overall growth in 2022?':

"Item 6. [Reserved].

Table of Contents Item 7. Management’s Discussion and Analysis of Financial Condition and Results of Operations. Management’s Discussion and Analysis of Financial Condition and Results of Operations (MD&A) is designed to provide a reader of 3M’s financial statements with a narra..."

*(Retrieved 9 relevant chunk(s) with highest similarity score of 0.6085)*

#### Mode: `BM25` (Relevant Rank: `❌ Not in Top 10` | Latency: `90.05ms`)
**Generated Answer**:
>Based on **3M_2022_10K.pdf**, here is what was found regarding 'If we exclude the impact of M&A, which segment has dragged down 3M's overall growth in 2022?':

"also impacted by social, political, and labor conditions in locations in which the Company or its suppliers or customers operate; adverse changes in

the availability and cost of capital; monetary policy; interest rates; inflation; recession; commodity prices; currency volatility or exchange control..."

*(Retrieved 6 relevant chunk(s) with highest similarity score of 1.0)*

#### Mode: `HYBRID` (Relevant Rank: `❌ Not in Top 10` | Latency: `223.37ms`)
**Generated Answer**:
>Based on **3M_2022_10K.pdf**, here is what was found regarding 'If we exclude the impact of M&A, which segment has dragged down 3M's overall growth in 2022?':

"Item 6. [Reserved].

Table of Contents Item 7. Management’s Discussion and Analysis of Financial Condition and Results of Operations. Management’s Discussion and Analysis of Financial Condition and Results of Operations (MD&A) is designed to provide a reader of 3M’s financial statements with a narra..."

*(Retrieved 7 relevant chunk(s) with highest similarity score of 0.030282)*

### Question `financebench_id_00807` (`3M_2023Q2_10Q`)
**Question**: `Does 3M have a reasonably healthy liquidity profile based on its quick ratio for Q2 of FY2023? If the quick ratio is not relevant to measure liquidity, please state that and explain why.`

**Ground-Truth Answer**: `No. The quick ratio for 3M was 0.96 by Jun'23 close, which needs a bit of an improvement to touch the 1x mark`

**Failure Category**: `terminology_mismatch`

#### Mode: `DENSE` (Relevant Rank: `❌ Not in Top 10` | Latency: `194.74ms`)
**Generated Answer**:
>Based on **3M_2023Q2_10Q.pdf**, here is what was found regarding 'Does 3M have a reasonably healthy liquidity profile based on its quick ratio for Q2 of FY2023? If the quick ratio is not relevant to measure liquidity, please state that and explain why.':

"equivalents at end of period $ 4,258 $ 2,722

The accompanying Notes to Consolidated Financial Statements are an integral part of this statement.

Table of Contents 3M Company and Subsidiaries Notes to Consolidated Financial Statements (Unaudited) NOTE 1. Significant Accounting Policies Basis of Pre..."

*(Retrieved 7 relevant chunk(s) with highest similarity score of 0.4952)*

#### Mode: `BM25` (Relevant Rank: `❌ Not in Top 10` | Latency: `93.66ms`)
**Generated Answer**:
>Based on **3M_2023Q2_10Q.pdf**, here is what was found regarding 'Does 3M have a reasonably healthy liquidity profile based on its quick ratio for Q2 of FY2023? If the quick ratio is not relevant to measure liquidity, please state that and explain why.':

"June 30, 2023 Due in one year or less $

Due after one year through five years
 Due after five years through ten years Total marketable securities $

Table of Contents NOTE 10. Long-Term Debt and Short-Term Borrowings In February 2023, 3M repaid $500 million aggregate principal amount of fixed-rate ..."

*(Retrieved 7 relevant chunk(s) with highest similarity score of 1.0)*

#### Mode: `HYBRID` (Relevant Rank: `❌ Not in Top 10` | Latency: `200.7ms`)
**Generated Answer**:
>Based on **3M_2023Q2_10Q.pdf**, here is what was found regarding 'Does 3M have a reasonably healthy liquidity profile based on its quick ratio for Q2 of FY2023? If the quick ratio is not relevant to measure liquidity, please state that and explain why.':

"June 30, 2023 Due in one year or less $

Due after one year through five years
 Due after five years through ten years Total marketable securities $

Table of Contents NOTE 10. Long-Term Debt and Short-Term Borrowings In February 2023, 3M repaid $500 million aggregate principal amount of fixed-rate ..."

*(Retrieved 8 relevant chunk(s) with highest similarity score of 0.030777)*

### Question `financebench_id_00941` (`3M_2023Q2_10Q`)
**Question**: `Which debt securities are registered to trade on a national securities exchange under 3M's name as of Q2 of 2023?`

**Ground-Truth Answer**: `Following debt securities registered under 3M's name are listed to trade on the New York Stock Exchange:
-1.500% Notes due 2026 (Trading Symbol: MMM26)
-1.750% Notes due 2030 (Trading Symbol: MMM30)
-1.500% Notes due 2031 (Trading Symbol: MMM31)`

**Failure Category**: `footnote_reference`

#### Mode: `DENSE` (Relevant Rank: `❌ Not in Top 10` | Latency: `173.62ms`)
**Generated Answer**:
>Based on **3M_2023Q2_10Q.pdf**, here is what was found regarding 'Which debt securities are registered to trade on a national securities exchange under 3M's name as of Q2 of 2023?':

"of the Hedged Liabilities Location on the Consolidated Balance Sheet June 30, December 31, June 30, December 31, Long-term debt $ $ $ (96) $ (98)

Table of Contents Net Investment Hedges: At June 30, 2023, the total notional amount of foreign exchange forward contracts designated in net investment h..."

*(Retrieved 6 relevant chunk(s) with highest similarity score of 0.6732)*

#### Mode: `BM25` (Relevant Rank: `❌ Not in Top 10` | Latency: `87.89ms`)
**Generated Answer**:
>Based on **3M_2023Q2_10Q.pdf**, here is what was found regarding 'Which debt securities are registered to trade on a national securities exchange under 3M's name as of Q2 of 2023?':

"Table of Contents UNITED STATES SECURITIES AND EXCHANGE COMMISSION WASHINGTON, D.C. 20549 FORM 10-Q ☒ QUARTERLY REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934 For the quarterly period ended June 30, 2023 or o TRANSITION REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SEC..."

*(Retrieved 6 relevant chunk(s) with highest similarity score of 1.0)*

#### Mode: `HYBRID` (Relevant Rank: `❌ Not in Top 10` | Latency: `203.14ms`)
**Generated Answer**:
>Based on **3M_2023Q2_10Q.pdf**, here is what was found regarding 'Which debt securities are registered to trade on a national securities exchange under 3M's name as of Q2 of 2023?':

"Table of Contents UNITED STATES SECURITIES AND EXCHANGE COMMISSION WASHINGTON, D.C. 20549 FORM 10-Q ☒ QUARTERLY REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934 For the quarterly period ended June 30, 2023 or o TRANSITION REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SEC..."

*(Retrieved 4 relevant chunk(s) with highest similarity score of 0.031545)*

### Question `financebench_id_01858` (`3M_2023Q2_10Q`)
**Question**: `Does 3M maintain a stable trend of dividend distribution?`

**Ground-Truth Answer**: `Yes, not only they distribute the dividends on a routine basis, 3M has also been increasing the per share dividend for consecutive 65 years`

**Failure Category**: `table_fragmentation`

#### Mode: `DENSE` (Relevant Rank: `❌ Not in Top 10` | Latency: `177.89ms`)
**Generated Answer**:
>Based on **3M_2023Q2_10Q.pdf**, here is what was found regarding 'Does 3M maintain a stable trend of dividend distribution?':

"to be continuously renewed, and the associated products of which are expected to generate cash flows for 3M for an indefinite period of time.

Table of Contents Amortization expense follows: Three months ended June 30, Twelve months ended December 31, (Millions) Amortization expense $ $ $ $ Expected..."

*(Retrieved 6 relevant chunk(s) with highest similarity score of 0.6242)*

#### Mode: `BM25` (Relevant Rank: `❌ Not in Top 10` | Latency: `80.37ms`)
**Generated Answer**:
>Based on **3M_2023Q2_10Q.pdf**, here is what was found regarding 'Does 3M maintain a stable trend of dividend distribution?':

"of the Company, has been working with the Public Flemish Waste Agency ("OVAM") for several years to investigate and remediate historical PFAS

contamination at and near the 3M Belgium facility in Zwijndrecht, Antwerp, Belgium. In connection with a ring road construction project (the Oosterweel Proje..."

*(Retrieved 10 relevant chunk(s) with highest similarity score of 1.0)*

#### Mode: `HYBRID` (Relevant Rank: `❌ Not in Top 10` | Latency: `183.08ms`)
**Generated Answer**:
>Based on **3M_2023Q2_10Q.pdf**, here is what was found regarding 'Does 3M maintain a stable trend of dividend distribution?':

"of the Company, has been working with the Public Flemish Waste Agency ("OVAM") for several years to investigate and remediate historical PFAS

contamination at and near the 3M Belgium facility in Zwijndrecht, Antwerp, Belgium. In connection with a ring road construction project (the Oosterweel Proje..."

*(Retrieved 7 relevant chunk(s) with highest similarity score of 0.032787)*

### Question `financebench_id_02987` (`ACTIVISIONBLIZZARD_2019_10K`)
**Question**: `What is the FY2019 fixed asset turnover ratio for Activision Blizzard? Fixed asset turnover ratio is defined as: FY2019 revenue / (average PP&E between FY2018 and FY2019). Round your answer to two decimal places. Base your judgments on the information provided primarily in the statement of income and the statement of financial position.`

**Ground-Truth Answer**: `24.26`

**Failure Category**: `multi_year_calculation`

#### Mode: `DENSE` (Relevant Rank: `❌ Not in Top 10` | Latency: `196.43ms`)
**Generated Answer**:
>Based on **ACTIVISIONBLIZZARD_2019_10K.pdf**, here is what was found regarding 'What is the FY2019 fixed asset turnover ratio for Activision Blizzard? Fixed asset turnover ratio is defined as: FY2019 revenue / (average PP&E between FY2018 and FY2019). Round your answer to two decimal places. Base your judgments on the information provided primarily in the statement of income and the statement of financial position.':

"to terminate the relationship, the Company recognized net bookings, a key operating metric, of $20 million, GAAP revenues of $164 million, and GAAP

operating income of $91 million for the year ended December 31, 2018. Activision no longer has any material rights or obligations related to the Destin..."

*(Retrieved 9 relevant chunk(s) with highest similarity score of 0.6183)*

#### Mode: `BM25` (Relevant Rank: `❌ Not in Top 10` | Latency: `100.73ms`)
**Generated Answer**:
>Based on **ACTIVISIONBLIZZARD_2019_10K.pdf**, here is what was found regarding 'What is the FY2019 fixed asset turnover ratio for Activision Blizzard? Fixed asset turnover ratio is defined as: FY2019 revenue / (average PP&E between FY2018 and FY2019). Round your answer to two decimal places. Base your judgments on the information provided primarily in the statement of income and the statement of financial position.':

"required by this Item is incorporated by reference to the sections of our definitive Proxy Statement for our 2020 Annual Meeting of Shareholders

entitled “Audit-Related Matters” to be filed with the SEC.

Table of Contents PART IV Item 15. EXHIBITS, FINANCIAL STATEMENT SCHEDULES (a) Financial State..."

*(Retrieved 8 relevant chunk(s) with highest similarity score of 1.0)*

#### Mode: `HYBRID` (Relevant Rank: `❌ Not in Top 10` | Latency: `229.95ms`)
**Generated Answer**:
>Based on **ACTIVISIONBLIZZARD_2019_10K.pdf**, here is what was found regarding 'What is the FY2019 fixed asset turnover ratio for Activision Blizzard? Fixed asset turnover ratio is defined as: FY2019 revenue / (average PP&E between FY2018 and FY2019). Round your answer to two decimal places. Base your judgments on the information provided primarily in the statement of income and the statement of financial position.':

"Costs Net Carrying Amount 2021 Notes $ $ (2) $ 2022 Notes (2) 2026 Notes (7) 2027 Notes (5) 2047 Notes (9) Total debt $ 2,700 $ (25) $ 2,675

Table of Contents A summary of our outstanding debt as of December 31, 2018, is as follows (amounts in millions): December 31, 2018 Gross Carrying Amount Unam..."

*(Retrieved 9 relevant chunk(s) with highest similarity score of 0.016393)*

### Question `financebench_id_07966` (`ACTIVISIONBLIZZARD_2019_10K`)
**Question**: `What is the FY2017 - FY2019 3 year average of capex as a % of revenue for Activision Blizzard? Answer in units of percents and round to one decimal place. Calculate (or extract) the answer from the statement of income and the cash flow statement.`

**Ground-Truth Answer**: `1.9%`

**Failure Category**: `multi_year_calculation`

#### Mode: `DENSE` (Relevant Rank: `❌ Not in Top 10` | Latency: `202.73ms`)
**Generated Answer**:
>Based on **ACTIVISIONBLIZZARD_2019_10K.pdf**, here is what was found regarding 'What is the FY2017 - FY2019 3 year average of capex as a % of revenue for Activision Blizzard? Answer in units of percents and round to one decimal place. Calculate (or extract) the answer from the statement of income and the cash flow statement.':

"The changes are primarily due to changes in the value of the U.S. dollar

relative to the euro and the British pound.

Table of Contents Operating Segment Results Currently, we have three reportable segments—Activision, Blizzard, and King. Our operating segments are consistent with the manner in whi..."

*(Retrieved 7 relevant chunk(s) with highest similarity score of 0.5813)*

#### Mode: `BM25` (Relevant Rank: `❌ Not in Top 10` | Latency: `117.42ms`)
**Generated Answer**:
>Based on **ACTIVISIONBLIZZARD_2019_10K.pdf**, here is what was found regarding 'What is the FY2017 - FY2019 3 year average of capex as a % of revenue for Activision Blizzard? Answer in units of percents and round to one decimal place. Calculate (or extract) the answer from the statement of income and the cash flow statement.':

"Other Information PART III. Item 10. Directors, Executive Officers, and Corporate Governance Item 11. Executive Compensation Item 12.

Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters
 Item 13. Certain Relationships and Related Transactions, and Director..."

*(Retrieved 6 relevant chunk(s) with highest similarity score of 1.0)*

#### Mode: `HYBRID` (Relevant Rank: `❌ Not in Top 10` | Latency: `236.26ms`)
**Generated Answer**:
>Based on **ACTIVISIONBLIZZARD_2019_10K.pdf**, here is what was found regarding 'What is the FY2017 - FY2019 3 year average of capex as a % of revenue for Activision Blizzard? Answer in units of percents and round to one decimal place. Calculate (or extract) the answer from the statement of income and the cash flow statement.':

"Other Information PART III. Item 10. Directors, Executive Officers, and Corporate Governance Item 11. Executive Compensation Item 12.

Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters
 Item 13. Certain Relationships and Related Transactions, and Director..."

*(Retrieved 9 relevant chunk(s) with highest similarity score of 0.025808)*
