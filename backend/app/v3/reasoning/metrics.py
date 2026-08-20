from typing import Dict, List, Optional
from app.v3.reasoning.schemas import MetricDefinition


class FinancialMetricRegistry:
    """
    Central Registry for supported financial metrics in V3.4.
    """

    def __init__(self):
        self._metrics: Dict[str, MetricDefinition] = {
            "ROA": MetricDefinition(
                metric_id="ROA",
                display_name="Return on Assets (ROA)",
                aliases=["roa", "return on assets", "return on total assets"],
                formula_str="Net Income / Average Total Assets",
                required_facts=["net_income", "total_assets"],
                required_periods=["current", "previous"],
                unit="%",
                calculation_type="percentage",
            ),
            "ROE": MetricDefinition(
                metric_id="ROE",
                display_name="Return on Equity (ROE)",
                aliases=["roe", "return on equity", "return on shareholders equity", "return on stockholders equity"],
                formula_str="Net Income / Average Shareholders Equity",
                required_facts=["net_income", "shareholders_equity"],
                required_periods=["current", "previous"],
                unit="%",
                calculation_type="percentage",
            ),
            "DPO": MetricDefinition(
                metric_id="DPO",
                display_name="Days Payable Outstanding (DPO)",
                aliases=["dpo", "days payable outstanding", "days payables outstanding", "days payables"],
                formula_str="(Average Accounts Payable / Cost of Goods Sold) * Days",
                required_facts=["accounts_payable", "cogs"],
                required_periods=["current", "previous"],
                unit="days",
                calculation_type="days",
            ),
            "INVENTORY_TURNOVER": MetricDefinition(
                metric_id="INVENTORY_TURNOVER",
                display_name="Inventory Turnover",
                aliases=["inventory turnover", "inventory turnover ratio", "stock turnover"],
                formula_str="Cost of Goods Sold / Average Inventory",
                required_facts=["cogs", "inventory"],
                required_periods=["current", "previous"],
                unit="x",
                calculation_type="ratio",
            ),
            "OPERATING_MARGIN": MetricDefinition(
                metric_id="OPERATING_MARGIN",
                display_name="Operating Margin",
                aliases=["operating margin", "operating income margin", "operating profit margin", "op margin"],
                formula_str="Operating Income / Revenue",
                required_facts=["operating_income", "revenue"],
                required_periods=["current"],
                unit="%",
                calculation_type="percentage",
            ),
            "CAPEX_RATIO": MetricDefinition(
                metric_id="CAPEX_RATIO",
                display_name="CAPEX to Revenue Ratio",
                aliases=["capex to revenue", "capex ratio", "capital expenditures to revenue", "capex / revenue", "capex percent of revenue"],
                formula_str="CAPEX / Revenue",
                required_facts=["capex", "revenue"],
                required_periods=["current"],
                unit="%",
                calculation_type="percentage",
            ),
            "YOY_GROWTH": MetricDefinition(
                metric_id="YOY_GROWTH",
                display_name="Year-over-Year Growth",
                aliases=["yoy growth", "year over year growth", "growth rate", "annual growth"],
                formula_str="(Current Value - Previous Value) / Previous Value",
                required_facts=["target_concept"],
                required_periods=["current", "previous"],
                unit="%",
                calculation_type="growth",
            ),
            "MULTI_YEAR_AVERAGE": MetricDefinition(
                metric_id="MULTI_YEAR_AVERAGE",
                display_name="Multi-Year Average",
                aliases=["multi year average", "average over years", "average annual"],
                formula_str="Sum of Period Values / Number of Periods",
                required_facts=["target_concept"],
                required_periods=["multi_year"],
                unit="USD",
                calculation_type="currency",
            ),
            "FCF_CONVERSION": MetricDefinition(
                metric_id="FCF_CONVERSION",
                display_name="Free Cash Flow Conversion",
                aliases=["fcf conversion", "free cash flow conversion", "fcf / net income"],
                formula_str="Free Cash Flow / Net Income",
                required_facts=["free_cash_flow", "net_income"],
                required_periods=["current"],
                unit="%",
                calculation_type="percentage",
            ),
        }

    def list_metrics(self) -> List[MetricDefinition]:
        return list(self._metrics.values())

    def get_metric(self, metric_id: str) -> Optional[MetricDefinition]:
        if not metric_id:
            return None
        return self._metrics.get(metric_id.upper().strip())

    def find_metric_by_query(self, query: str) -> Optional[MetricDefinition]:
        """Matches query keywords against metric aliases."""
        if not query:
            return None
        q_clean = query.lower()

        # Prioritize exact multi-word match
        for metric in self._metrics.values():
            for alias in metric.aliases:
                if alias in q_clean:
                    return metric
        return None


metric_registry = FinancialMetricRegistry()
