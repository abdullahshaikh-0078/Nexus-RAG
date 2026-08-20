from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field


class FinancialFact(BaseModel):
    """
    Structured Financial Fact extracted from retrieved document context or tables.
    Preserves period, unit, scale, sign, and exact provenance.
    """
    fact_id: str
    concept: str  # e.g., "net_income", "total_assets", "operating_income", "revenue", "capex", "accounts_payable", "cogs", "inventory", "free_cash_flow", "shareholders_equity"
    value: float  # Unscaled value after sign parsing (e.g. -1577.0)
    raw_text: str  # Raw extracted text, e.g., "(1,577)"
    unit: str = "USD"  # "USD", "EUR", "%", "ratio", "days"
    scale: float = 1.0  # 1.0 = raw, 1e3 = thousands, 1e6 = millions, 1e9 = billions
    currency: str = "USD"
    period: str  # e.g., "2018", "FY2018", "Q1 2023"
    period_type: str = "FY"  # "FY", "Q1", "Q2", "Q3", "Q4", "TTM"
    fiscal_year: Optional[int] = None  # e.g. 2018
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    source_document: str
    page_number: int
    section: Optional[str] = None
    chunk_id: Optional[str] = None
    table_id: Optional[str] = None
    bbox: Optional[List[float]] = None
    confidence: float = 1.0

    @property
    def normalized_value(self) -> float:
        """Returns value fully scaled to base units (e.g. 500 * 1e6 = 500,000,000.0)."""
        return self.value * self.scale


class MetricDefinition(BaseModel):
    """
    Definition for a supported financial metric in the central registry.
    """
    metric_id: str
    display_name: str
    aliases: List[str]
    formula_str: str
    required_facts: List[str]
    required_periods: List[str]  # e.g. ["current"], ["current", "previous"]
    unit: str = "%"  # "%", "ratio", "days", "USD"
    calculation_type: str = "percentage"  # "percentage", "ratio", "days", "currency", "growth"


class QueryClassification(BaseModel):
    """
    Classification of user query intent.
    """
    query_type: str = "DIRECT_RETRIEVAL"  # "DIRECT_RETRIEVAL", "DERIVED_CALCULATION", "MULTI_STEP", "COMPARATIVE"
    is_calculation_required: bool = False
    target_metric_id: Optional[str] = None
    target_years: List[int] = Field(default_factory=list)
    target_company: Optional[str] = None
    explanation: str = "Direct factual lookup"


class CalculationInput(BaseModel):
    """
    Single input variable bound to an extracted FinancialFact.
    """
    variable: str
    concept: str
    fiscal_year: Optional[int] = None
    value: float
    scaled_value: float
    unit: str
    source_fact: FinancialFact


class CalculationStep(BaseModel):
    """
    Step-by-step calculation trace.
    """
    step_number: int
    description: str
    expression: str
    intermediate_value: float


class CalculationResult(BaseModel):
    """
    Structured outcome of deterministic calculation execution.
    """
    metric_id: str
    metric_display_name: str
    formula: str
    inputs: List[CalculationInput] = Field(default_factory=list)
    intermediate_values: Dict[str, float] = Field(default_factory=dict)
    result: Optional[float] = None
    display_result: str = "N/A"
    unit: str = "%"
    period: str = "N/A"
    steps: List[CalculationStep] = Field(default_factory=list)
    source_facts: List[FinancialFact] = Field(default_factory=list)
    validation_status: str = "VALIDATED"  # "VALIDATED", "INSUFFICIENT_EVIDENCE", "ZERO_DIVISION", "UNIT_MISMATCH"
    missing_facts: List[str] = Field(default_factory=list)
    missing_periods: List[str] = Field(default_factory=list)
