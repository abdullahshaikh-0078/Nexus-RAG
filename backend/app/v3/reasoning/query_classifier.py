import re
from typing import List, Optional
from app.v3.reasoning.schemas import QueryClassification
from app.v3.reasoning.metrics import metric_registry


class FinancialQueryClassifier:
    """
    Classifies queries into DIRECT_RETRIEVAL, DERIVED_CALCULATION, MULTI_STEP, or COMPARATIVE.
    """

    def classify(self, query: str) -> QueryClassification:
        if not query or not query.strip():
            return QueryClassification(
                query_type="DIRECT_RETRIEVAL",
                is_calculation_required=False,
                explanation="Empty query",
            )

        q_clean = query.strip()
        q_lower = q_clean.lower()

        # 1. Extract Target Fiscal Years (e.g., 2018, 2017, FY2023, etc.)
        year_matches = re.findall(r"\b(20\d{2}|19\d{2})\b", q_clean)
        target_years = sorted(list(set(int(y) for y in year_matches)))

        # 2. Check for metric in Metric Registry
        metric_def = metric_registry.find_metric_by_query(q_clean)

        # 3. Check for Comparative / YoY keywords
        is_comparative = any(kw in q_lower for kw in [
            "grow", "growth", "change", "increase", "decrease", "compared to", "versus", "vs"
        ])

        # 4. Check for Multi-year / Average keywords
        is_multi_step = any(kw in q_lower for kw in [
            "average", "avg", "multi-year", "across years", "over 3 years", "3-year"
        ]) or (len(target_years) >= 3 and ("average" in q_lower or "trend" in q_lower))

        if is_multi_step and metric_def:
            return QueryClassification(
                query_type="MULTI_STEP",
                is_calculation_required=True,
                target_metric_id=metric_def.metric_id,
                target_years=target_years,
                explanation=f"Multi-step calculation for {metric_def.display_name} across years {target_years}",
            )

        if is_comparative:
            metric_id = metric_def.metric_id if metric_def else "YOY_GROWTH"
            return QueryClassification(
                query_type="COMPARATIVE",
                is_calculation_required=True,
                target_metric_id=metric_id,
                target_years=target_years,
                explanation=f"Comparative growth calculation for {metric_id} across {target_years}",
            )

        if metric_def:
            return QueryClassification(
                query_type="DERIVED_CALCULATION",
                is_calculation_required=True,
                target_metric_id=metric_def.metric_id,
                target_years=target_years,
                explanation=f"Derived calculation required for metric '{metric_def.display_name}'",
            )

        # Default: Direct Factual Retrieval
        return QueryClassification(
            query_type="DIRECT_RETRIEVAL",
            is_calculation_required=False,
            target_metric_id=None,
            target_years=target_years,
            explanation="Direct factual lookup query",
        )


query_classifier = FinancialQueryClassifier()
