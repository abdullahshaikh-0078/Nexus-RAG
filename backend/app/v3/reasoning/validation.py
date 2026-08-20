from typing import List, Dict, Tuple, Optional
from app.v3.reasoning.schemas import FinancialFact, MetricDefinition


class FinancialValidationEngine:
    """
    Validates fact completeness, period alignment, unit scaling, and provenance before calculation.
    """

    def validate_facts_for_metric(
        self,
        metric: MetricDefinition,
        facts: List[FinancialFact],
        target_years: List[int],
    ) -> Tuple[bool, str, List[str], List[str]]:
        """
        Validates whether extracted facts contain all required concepts and target periods for a metric.
        Returns (is_valid, validation_status, missing_concepts, missing_periods).
        """
        if not facts:
            return (
                False,
                "INSUFFICIENT_EVIDENCE",
                metric.required_facts,
                [str(y) for y in target_years] if target_years else ["current"],
            )

        found_concepts = set(f.concept for f in facts)
        missing_concepts = [c for c in metric.required_facts if c not in found_concepts and c != "target_concept"]

        found_years = set(f.fiscal_year for f in facts if f.fiscal_year is not None)
        missing_periods = []

        if target_years:
            for yr in target_years:
                if yr not in found_years:
                    missing_periods.append(str(yr))

        if missing_concepts or missing_periods:
            return (
                False,
                "INSUFFICIENT_EVIDENCE",
                missing_concepts,
                missing_periods,
            )

        # Check unit / scale compatibility
        currencies = set(f.currency for f in facts if f.currency)
        if len(currencies) > 1:
            return (
                False,
                "UNIT_MISMATCH",
                [],
                [],
            )

        return (True, "VALIDATED", [], [])


validator = FinancialValidationEngine()
