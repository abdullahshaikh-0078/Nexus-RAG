from typing import List, Dict, Optional, Tuple
from app.v3.reasoning.schemas import (
    FinancialFact,
    MetricDefinition,
    CalculationResult,
    CalculationInput,
    CalculationStep,
)
from app.v3.reasoning.metrics import metric_registry
from app.v3.reasoning import formulas
from app.v3.reasoning.validation import validator


class CalculationEngine:
    """
    Deterministic Calculation Engine for V3.4 Financial Reasoning.
    Executes pure Python formula functions with full floating point precision and scale normalization.
    """

    def calculate(
        self,
        metric_id: str,
        facts: List[FinancialFact],
        target_years: List[int],
        days_in_period: float = 365.0,
    ) -> CalculationResult:
        metric = metric_registry.get_metric(metric_id)
        if not metric:
            return CalculationResult(
                metric_id=metric_id,
                metric_display_name=metric_id,
                formula="Unknown Metric",
                validation_status="INSUFFICIENT_EVIDENCE",
                missing_facts=[metric_id],
            )

        # Validate facts
        is_valid, status, missing_concepts, missing_periods = validator.validate_facts_for_metric(
            metric=metric, facts=facts, target_years=target_years
        )

        if not is_valid:
            return CalculationResult(
                metric_id=metric.metric_id,
                metric_display_name=metric.display_name,
                formula=metric.formula_str,
                validation_status=status,
                missing_facts=missing_concepts,
                missing_periods=missing_periods,
                source_facts=facts,
            )

        # Map facts by concept and year
        fact_map: Dict[Tuple[str, Optional[int]], FinancialFact] = {}
        for f in facts:
            fact_map[(f.concept, f.fiscal_year)] = f
            # Also store without year as fallback
            if (f.concept, None) not in fact_map:
                fact_map[(f.concept, None)] = f

        target_yr = target_years[0] if target_years else None
        prev_yr = (target_yr - 1) if target_yr else None

        inputs: List[CalculationInput] = []
        steps: List[CalculationStep] = []
        result_val: Optional[float] = None
        display_res = "N/A"

        try:
            if metric.metric_id == "ROA":
                # Net Income (target_yr), Assets (target_yr), Assets (prev_yr)
                f_ni = fact_map.get(("net_income", target_yr)) or fact_map.get(("net_income", None))
                f_end = fact_map.get(("total_assets", target_yr)) or fact_map.get(("total_assets", None))
                f_beg = fact_map.get(("total_assets", prev_yr)) or f_end

                if not f_ni or not f_end or not f_beg:
                    return self._missing_result(metric, facts, ["net_income", "total_assets"])

                inputs = [
                    self._make_input("Net Income", f_ni),
                    self._make_input("Beginning Assets", f_beg),
                    self._make_input("Ending Assets", f_end),
                ]
                result_val, steps = formulas.calculate_roa(
                    net_income=f_ni.normalized_value,
                    beg_assets=f_beg.normalized_value,
                    end_assets=f_end.normalized_value,
                )
                display_res = f"{result_val * 100.0:.2f}%"

            elif metric.metric_id == "ROE":
                f_ni = fact_map.get(("net_income", target_yr)) or fact_map.get(("net_income", None))
                f_end = fact_map.get(("shareholders_equity", target_yr)) or fact_map.get(("shareholders_equity", None))
                f_beg = fact_map.get(("shareholders_equity", prev_yr)) or f_end

                if not f_ni or not f_end or not f_beg:
                    return self._missing_result(metric, facts, ["net_income", "shareholders_equity"])

                inputs = [
                    self._make_input("Net Income", f_ni),
                    self._make_input("Beginning Equity", f_beg),
                    self._make_input("Ending Equity", f_end),
                ]
                result_val, steps = formulas.calculate_roe(
                    net_income=f_ni.normalized_value,
                    beg_equity=f_beg.normalized_value,
                    end_equity=f_end.normalized_value,
                )
                display_res = f"{result_val * 100.0:.2f}%"

            elif metric.metric_id == "DPO":
                f_ap = fact_map.get(("accounts_payable", target_yr)) or fact_map.get(("accounts_payable", None))
                f_prev_ap = fact_map.get(("accounts_payable", prev_yr))
                f_cogs = fact_map.get(("cogs", target_yr)) or fact_map.get(("cogs", None))

                if not f_ap or not f_cogs:
                    return self._missing_result(metric, facts, ["accounts_payable", "cogs"])

                inputs = [self._make_input("Accounts Payable", f_ap), self._make_input("COGS", f_cogs)]
                if f_prev_ap:
                    inputs.append(self._make_input("Beginning Accounts Payable", f_prev_ap))

                result_val, steps = formulas.calculate_dpo(
                    accounts_payable=f_ap.normalized_value,
                    cogs=f_cogs.normalized_value,
                    days=days_in_period,
                    prev_accounts_payable=f_prev_ap.normalized_value if f_prev_ap else None,
                )
                display_res = f"{result_val:.2f} days"

            elif metric.metric_id == "INVENTORY_TURNOVER":
                f_cogs = fact_map.get(("cogs", target_yr)) or fact_map.get(("cogs", None))
                f_inv = fact_map.get(("inventory", target_yr)) or fact_map.get(("inventory", None))
                f_prev_inv = fact_map.get(("inventory", prev_yr))

                if not f_cogs or not f_inv:
                    return self._missing_result(metric, facts, ["cogs", "inventory"])

                inputs = [self._make_input("COGS", f_cogs), self._make_input("Ending Inventory", f_inv)]
                if f_prev_inv:
                    inputs.append(self._make_input("Beginning Inventory", f_prev_inv))

                result_val, steps = formulas.calculate_inventory_turnover(
                    cogs=f_cogs.normalized_value,
                    curr_inventory=f_inv.normalized_value,
                    prev_inventory=f_prev_inv.normalized_value if f_prev_inv else None,
                )
                display_res = f"{result_val:.2f}x"

            elif metric.metric_id == "OPERATING_MARGIN":
                f_op = fact_map.get(("operating_income", target_yr)) or fact_map.get(("operating_income", None))
                f_rev = fact_map.get(("revenue", target_yr)) or fact_map.get(("revenue", None))

                if not f_op or not f_rev:
                    return self._missing_result(metric, facts, ["operating_income", "revenue"])

                inputs = [self._make_input("Operating Income", f_op), self._make_input("Revenue", f_rev)]
                result_val, steps = formulas.calculate_operating_margin(
                    operating_income=f_op.normalized_value, revenue=f_rev.normalized_value
                )
                display_res = f"{result_val * 100.0:.2f}%"

            elif metric.metric_id == "CAPEX_RATIO":
                f_capex = fact_map.get(("capex", target_yr)) or fact_map.get(("capex", None))
                f_rev = fact_map.get(("revenue", target_yr)) or fact_map.get(("revenue", None))

                if not f_capex or not f_rev:
                    return self._missing_result(metric, facts, ["capex", "revenue"])

                inputs = [self._make_input("CAPEX", f_capex), self._make_input("Revenue", f_rev)]
                result_val, steps = formulas.calculate_capex_ratio(
                    capex=abs(f_capex.normalized_value), revenue=f_rev.normalized_value
                )
                display_res = f"{result_val * 100.0:.2f}%"

            elif metric.metric_id == "YOY_GROWTH":
                # Find concept present for both target_yr and prev_yr
                f_curr = None
                f_prev = None
                for concept in ["revenue", "net_income", "operating_income", "total_assets"]:
                    f_c = fact_map.get((concept, target_yr))
                    f_p = fact_map.get((concept, prev_yr))
                    if f_c and f_p:
                        f_curr, f_prev = f_c, f_p
                        break

                if not f_curr or not f_prev:
                    return self._missing_result(metric, facts, ["current_value", "previous_value"])

                inputs = [self._make_input("Current Period Value", f_curr), self._make_input("Previous Period Value", f_prev)]
                result_val, steps = formulas.calculate_yoy_growth(
                    curr_value=f_curr.normalized_value, prev_value=f_prev.normalized_value
                )
                display_res = f"{result_val * 100.0:.2f}%"

            elif metric.metric_id == "MULTI_YEAR_AVERAGE":
                vals = [f.normalized_value for f in facts if f.normalized_value is not None]
                if not vals:
                    return self._missing_result(metric, facts, ["target_values"])

                inputs = [self._make_input(f"Fact ({f.period})", f) for f in facts]
                result_val, steps = formulas.calculate_multi_year_average(vals)
                display_res = f"${result_val:,.2f}"

            elif metric.metric_id == "FCF_CONVERSION":
                f_fcf = fact_map.get(("free_cash_flow", target_yr)) or fact_map.get(("free_cash_flow", None))
                f_ni = fact_map.get(("net_income", target_yr)) or fact_map.get(("net_income", None))

                if not f_fcf or not f_ni:
                    return self._missing_result(metric, facts, ["free_cash_flow", "net_income"])

                inputs = [self._make_input("Free Cash Flow", f_fcf), self._make_input("Net Income", f_ni)]
                result_val, steps = formulas.calculate_fcf_conversion(
                    fcf=f_fcf.normalized_value, net_income=f_ni.normalized_value
                )
                display_res = f"{result_val * 100.0:.2f}%"

            return CalculationResult(
                metric_id=metric.metric_id,
                metric_display_name=metric.display_name,
                formula=metric.formula_str,
                inputs=inputs,
                result=result_val,
                display_result=display_res,
                unit=metric.unit,
                period=str(target_yr) if target_yr else "N/A",
                steps=steps,
                source_facts=facts,
                validation_status="VALIDATED",
            )

        except ZeroDivisionError as e:
            return CalculationResult(
                metric_id=metric.metric_id,
                metric_display_name=metric.display_name,
                formula=metric.formula_str,
                inputs=inputs,
                validation_status="ZERO_DIVISION",
                source_facts=facts,
            )

    def _make_input(self, variable_name: str, fact: FinancialFact) -> CalculationInput:
        return CalculationInput(
            variable=variable_name,
            concept=fact.concept,
            fiscal_year=fact.fiscal_year,
            value=fact.value,
            scaled_value=fact.normalized_value,
            unit=fact.unit,
            source_fact=fact,
        )

    def _missing_result(self, metric: MetricDefinition, facts: List[FinancialFact], missing: List[str]) -> CalculationResult:
        return CalculationResult(
            metric_id=metric.metric_id,
            metric_display_name=metric.display_name,
            formula=metric.formula_str,
            validation_status="INSUFFICIENT_EVIDENCE",
            missing_facts=missing,
            source_facts=facts,
        )


calculation_engine = CalculationEngine()
