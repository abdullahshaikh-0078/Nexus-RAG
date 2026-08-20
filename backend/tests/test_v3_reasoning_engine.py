import os
import json
import pytest
from app.v3.reasoning.schemas import FinancialFact, CalculationResult
from app.v3.reasoning.metrics import metric_registry, FinancialMetricRegistry
from app.v3.reasoning.query_classifier import query_classifier
from app.v3.reasoning.fact_extractor import fact_extractor
from app.v3.reasoning.calculator import calculation_engine
from app.v3.reasoning import formulas
from app.evaluation.runner import RESULTS_DIR


def test_metric_registry_definitions():
    """Verifies all 9 supported metrics exist in Metric Registry with correct definitions."""
    metrics = metric_registry.list_metrics()
    m_ids = [m.metric_id for m in metrics]

    expected = [
        "ROA", "ROE", "DPO", "INVENTORY_TURNOVER", "OPERATING_MARGIN",
        "CAPEX_RATIO", "YOY_GROWTH", "MULTI_YEAR_AVERAGE", "FCF_CONVERSION"
    ]
    for exp in expected:
        assert exp in m_ids, f"Metric '{exp}' missing from registry"


def test_query_classification():
    """Verifies query intent classification."""
    # Direct
    c1 = query_classifier.classify("What was revenue in 2018?")
    assert c1.query_type == "DIRECT_RETRIEVAL"
    assert c1.is_calculation_required is False

    # Derived
    c2 = query_classifier.classify("What was ROA in 2018?")
    assert c2.query_type == "DERIVED_CALCULATION"
    assert c2.is_calculation_required is True
    assert c2.target_metric_id == "ROA"
    assert 2018 in c2.target_years

    # Comparative
    c3 = query_classifier.classify("How much did revenue grow from 2017 to 2018?")
    assert c3.query_type == "COMPARATIVE"
    assert c3.is_calculation_required is True

    # Multi-step
    c4 = query_classifier.classify("What was the average ROA across 2016, 2017, and 2018?")
    assert c4.query_type == "MULTI_STEP"
    assert c4.is_calculation_required is True


def test_fact_extraction_number_and_sign_parsing():
    """Verifies number parsing: (1,577) -> -1577.0, -1577 -> -1577.0, 3,848 -> 3848.0."""
    val1, _ = fact_extractor.parse_number_and_sign("(1,577)")
    assert val1 == -1577.0

    val2, _ = fact_extractor.parse_number_and_sign("( 3,469.5 )")
    assert val2 == -3469.5

    val3, _ = fact_extractor.parse_number_and_sign("-1,577")
    assert val3 == -1577.0

    val4, _ = fact_extractor.parse_number_and_sign("$3,848")
    assert val4 == 3848.0


def test_fact_extraction_scale_detection():
    """Verifies scale detection for thousands, millions, and billions."""
    assert fact_extractor.detect_scale("Amounts in millions of USD") == 1e6
    assert fact_extractor.detect_scale("Operating income ($ in billions)") == 1e9
    assert fact_extractor.detect_scale("Figures in thousands") == 1e3
    assert fact_extractor.detect_scale("Direct raw amounts") == 1.0


def test_roa_calculation():
    """Deterministic test for ROA: Net Income / Average Total Assets."""
    res, steps = formulas.calculate_roa(net_income=3848.0, beg_assets=32100.0, end_assets=35000.0)
    expected_avg = (32100.0 + 35000.0) / 2.0  # 33550.0
    expected_roa = 3848.0 / 33550.0  # 0.11469448584202683

    assert abs(res - expected_roa) < 1e-6
    assert len(steps) == 2
    assert steps[0].intermediate_value == expected_avg


def test_roe_calculation():
    """Deterministic test for ROE: Net Income / Average Shareholders' Equity."""
    res, steps = formulas.calculate_roe(net_income=3848.0, beg_equity=10000.0, end_equity=12000.0)
    expected_roe = 3848.0 / 11000.0
    assert abs(res - expected_roe) < 1e-6


def test_dpo_calculation():
    """Deterministic test for DPO: (Average AP / COGS) * Days."""
    res, steps = formulas.calculate_dpo(accounts_payable=2000.0, cogs=10000.0, days=365.0, prev_accounts_payable=1800.0)
    # Avg AP = 1900.0
    # DPO = (1900 / 10000) * 365 = 69.35 days
    assert abs(res - 69.35) < 1e-2


def test_inventory_turnover_calculation():
    """Deterministic test for Inventory Turnover: COGS / Average Inventory."""
    res, steps = formulas.calculate_inventory_turnover(cogs=12000.0, curr_inventory=3000.0, prev_inventory=2500.0)
    # Avg Inv = 2750.0
    # Turnover = 12000 / 2750 = 4.3636x
    assert abs(res - (12000.0 / 2750.0)) < 1e-6


def test_operating_margin_calculation():
    """Deterministic test for Operating Margin: Operating Income / Revenue."""
    res, steps = formulas.calculate_operating_margin(operating_income=5000.0, revenue=20000.0)
    assert res == 0.25  # 25%


def test_capex_ratio_calculation():
    """Deterministic test for CAPEX / Revenue."""
    res, steps = formulas.calculate_capex_ratio(capex=1000.0, revenue=20000.0)
    assert res == 0.05  # 5%


def test_yoy_growth_calculation():
    """Deterministic test for YoY Growth: (Current - Previous) / Previous."""
    res, steps = formulas.calculate_yoy_growth(curr_value=12000.0, prev_value=10000.0)
    assert res == 0.20  # 20%


def test_multi_year_average_calculation():
    """Deterministic test for Multi-Year Average."""
    res, steps = formulas.calculate_multi_year_average([10.0, 12.0, 14.0])
    assert res == 12.0


def test_fcf_conversion_calculation():
    """Deterministic test for FCF Conversion: FCF / Net Income."""
    res, steps = formulas.calculate_fcf_conversion(fcf=4000.0, net_income=3848.0)
    assert abs(res - (4000.0 / 3848.0)) < 1e-6


def test_zero_division_handling():
    """Verifies that zero denominator returns ZERO_DIVISION validation status."""
    f_ni = FinancialFact(
        fact_id="f1", concept="net_income", value=100.0, raw_text="100",
        source_document="doc.pdf", page_number=1, fiscal_year=2018, period="2018"
    )
    f_assets = FinancialFact(
        fact_id="f2", concept="total_assets", value=0.0, raw_text="0",
        source_document="doc.pdf", page_number=1, fiscal_year=2018, period="2018"
    )
    res = calculation_engine.calculate("ROA", [f_ni, f_assets], [2018])
    assert res.validation_status == "ZERO_DIVISION"


def test_insufficient_evidence_handling():
    """Verifies missing facts return INSUFFICIENT_EVIDENCE validation status."""
    f_ni = FinancialFact(
        fact_id="f1", concept="net_income", value=100.0, raw_text="100",
        source_document="doc.pdf", page_number=1, fiscal_year=2018, period="2018"
    )
    res = calculation_engine.calculate("ROA", [f_ni], [2018])
    assert res.validation_status == "INSUFFICIENT_EVIDENCE"
    assert "total_assets" in res.missing_facts


def test_provenance_preservation():
    """Verifies that facts preserve provenance fields."""
    f = FinancialFact(
        fact_id="f100", concept="net_income", value=3848.0, raw_text="3,848",
        period="2018", unit="USD", scale=1e6, source_document="3M_2018_10K.pdf", page_number=62,
        section="Item 8", chunk_id="chk-123", table_id="tbl-456"
    )
    assert f.source_document == "3M_2018_10K.pdf"
    assert f.page_number == 62
    assert f.chunk_id == "chk-123"
    assert f.normalized_value == 3848000000.0


def test_baseline_preservation():
    """Verifies V1 baseline json files are untouched."""
    v1_latest_path = os.path.join(RESULTS_DIR, "latest.json")
    if os.path.exists(v1_latest_path):
        with open(v1_latest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("aggregate_recall_at_1") == 0.9167
