import pytest
from app.v3.query_expansion.config import QueryExpansionConfig
from app.v3.query_expansion.dictionary import FINANCIAL_TERMINOLOGY_DICTIONARY
from app.v3.query_expansion.expander import FinancialQueryExpander
from app.v3.query_expansion.rewriter import QueryRewriter


def test_initial_financial_dictionary_completeness():
    """Verifies all required 20+ financial terms exist in dictionary."""
    required_terms = [
        "PP&E", "PPE", "CAPEX", "DPO", "DIO", "DSO", "ROA", "ROE",
        "EBITDA", "EBIT", "D&A", "GAAP", "EPS", "FCF", "CFO", "CFI",
        "CFF", "NWC", "AR", "AP", "COGS", "YOY", "QOQ"
    ]
    for term in required_terms:
        assert term in FINANCIAL_TERMINOLOGY_DICTIONARY
        assert len(FINANCIAL_TERMINOLOGY_DICTIONARY[term]) > 0


def test_ppe_expansion_and_original_preservation():
    expander = FinancialQueryExpander()
    trace = expander.expand_query("What was net PP&E in 2018?")

    assert "PP&E" in trace.detected_terms
    assert trace.original_query == "What was net PP&E in 2018?"
    assert trace.expansion_count >= 1
    assert "property, plant and equipment" in trace.expansions["PP&E"]


def test_capex_and_multiple_terms_expansion():
    expander = FinancialQueryExpander()
    query = "Calculate CAPEX and EBITDA for 3M in FY2022"
    trace = expander.expand_query(query)

    assert "CAPEX" in trace.detected_terms
    assert "EBITDA" in trace.detected_terms
    assert len(trace.detected_terms) >= 2
    assert "capital expenditures" in trace.expansions["CAPEX"]
    assert "earnings before interest, taxes, depreciation and amortization" in trace.expansions["EBITDA"]


def test_case_insensitivity_and_punctuation():
    expander = FinancialQueryExpander()
    trace1 = expander.expand_query("what was the capex for Adobe?")
    trace2 = expander.expand_query("What was the CapEx for Adobe?")
    trace3 = expander.expand_query("What was the CAPEX?")

    assert len(trace1.detected_terms) > 0
    assert len(trace2.detected_terms) > 0
    assert len(trace3.detected_terms) > 0


def test_non_financial_query_preservation():
    expander = FinancialQueryExpander()
    query = "Who is the chief executive officer of 3M Company?"
    trace = expander.expand_query(query)

    assert len(trace.detected_terms) == 0
    assert trace.expansion_count == 0
    assert trace.rewritten_query == query


def test_expansion_limits_and_configuration():
    cfg = QueryExpansionConfig(max_expansions_per_term=2)
    expander = FinancialQueryExpander(config=cfg)
    trace = expander.expand_query("Check DPO and DSO values.")

    assert len(trace.expansions["DPO"]) <= 2
    assert len(trace.expansions["DSO"]) <= 2


def test_query_rewriter_bm25_and_vector():
    rewriter = QueryRewriter()
    query = "What is the net FCF for 2021?"

    bm25_str, trace_bm25 = rewriter.rewrite_for_bm25(query)
    vec_str, trace_vec = rewriter.rewrite_for_vector(query)

    assert "What is the net FCF for 2021?" in bm25_str
    assert "free cash flow" in bm25_str
    assert "Also search for" in vec_str
