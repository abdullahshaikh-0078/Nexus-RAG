import re
import uuid
from typing import List, Dict, Any, Optional, Tuple
from app.v3.reasoning.schemas import FinancialFact
from app.models.schemas import SourceCitation


CONCEPT_ALIASES: Dict[str, List[str]] = {
    "net_income": [
        "net income", "net earnings", "net profit", "income from continuing operations", "net income attributable to"
    ],
    "total_assets": [
        "total assets", "assets, total", "assets"
    ],
    "shareholders_equity": [
        "shareholders' equity", "shareholders equity", "stockholders' equity", "stockholders equity",
        "total shareholders' equity", "total stockholders' equity", "total equity"
    ],
    "accounts_payable": [
        "accounts payable", "trade payables", "payables", "accounts payable and accrued liabilities"
    ],
    "cogs": [
        "cost of goods sold", "cost of sales", "cogs", "cost of products sold", "cost of revenues"
    ],
    "inventory": [
        "inventories", "inventory", "total inventory", "inventories, net"
    ],
    "operating_income": [
        "operating income", "operating profit", "operating earnings", "operating income (loss)", "income from operations"
    ],
    "revenue": [
        "revenue", "total revenue", "net sales", "sales", "total net sales", "net revenues"
    ],
    "capex": [
        "capital expenditures", "capex", "additions to property, plant and equipment",
        "purchases of property, plant and equipment", "capital additions"
    ],
    "free_cash_flow": [
        "free cash flow", "fcf", "free cash flow (fcf)"
    ],
}


class FinancialFactExtractor:
    """
    Extracts structured FinancialFact objects from retrieved citations, text chunks, and tables.
    """

    def parse_number_and_sign(self, raw_str: str) -> Tuple[Optional[float], str]:
        """
        Parses financial number string, correctly converting '(1,577)' or '-1577' into -1577.0.
        Returns (parsed_float, raw_str).
        """
        if not raw_str or not raw_str.strip():
            return None, raw_str

        clean = raw_str.strip()

        # Parentheses negative: (1,577) or ( 1,577.5 )
        paren_match = re.match(r"^\(\s*\$?\s*([\d,]+(?:\.\d+)?)\s*\)$", clean)
        if paren_match:
            num_part = paren_match.group(1).replace(",", "")
            try:
                return -float(num_part), clean
            except ValueError:
                return None, clean

        # Standard negative or positive number
        std_match = re.match(r"^[\-\–\—]?\s*\$?\s*([\d,]+(?:\.\d+)?)$", clean)
        if std_match:
            num_part = std_match.group(1).replace(",", "")
            try:
                val = float(num_part)
                if clean.startswith("-") or clean.startswith("–") or clean.startswith("—"):
                    val = -val
                return val, clean
            except ValueError:
                return None, clean

        return None, clean

    def detect_scale(self, text: str) -> float:
        """Detects financial scale (thousands, millions, billions) from context."""
        t_lower = text.lower()
        if "in billions" in t_lower or "billions" in t_lower:
            return 1e9
        if "in millions" in t_lower or "millions" in t_lower or "($ in millions)" in t_lower or "(in millions)" in t_lower:
            return 1e6
        if "in thousands" in t_lower or "thousands" in t_lower or "(in thousands)" in t_lower:
            return 1e3
        return 1.0

    def extract_facts_from_citations(
        self,
        citations: List[SourceCitation],
        target_concepts: Optional[List[str]] = None,
        target_years: Optional[List[int]] = None,
    ) -> List[FinancialFact]:
        """
        Scans retrieved SourceCitations to extract structured FinancialFacts.
        """
        facts: List[FinancialFact] = []

        concepts_to_check = target_concepts if target_concepts else list(CONCEPT_ALIASES.keys())

        for cite in citations:
            text = getattr(cite, "content", "")
            doc_name = getattr(cite, "document_name", getattr(cite, "document_id", "unknown.pdf"))
            page_num = getattr(cite, "page_number", getattr(cite, "page_start", 1))
            section = getattr(cite, "section", None)
            chunk_id = str(getattr(cite, "chunk_id", "")) if getattr(cite, "chunk_id", None) else None
            scale = self.detect_scale(text)

            # Extract year context from chunk
            years_in_chunk = [int(y) for y in re.findall(r"\b(20\d{2}|19\d{2})\b", text)]

            # Process line by line or pipe-delimited table rows
            lines = text.split("\n")
            for line in lines:
                l_lower = line.lower()

                # Check each concept
                for concept in concepts_to_check:
                    aliases = CONCEPT_ALIASES.get(concept, [concept])
                    if not any(alias in l_lower for alias in aliases):
                        continue

                    # Search numbers in line
                    # Match numbers like $3,848, (1,577), 4,858, 32,100
                    num_tokens = re.findall(r"(\(\s*\$?\s*[\d,]+(?:\.\d+)?\s*\)|[\-\–\—]?\s*\$?\s*[\d,]+(?:\.\d+)?)", line)

                    parsed_nums = []
                    for token in num_tokens:
                        val, raw = self.parse_number_and_sign(token)
                        if val is not None and abs(val) > 0.01:
                            # Exclude 4-digit years from number values unless surrounded by currency
                            if val in [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025] and "$" not in raw:
                                continue
                            parsed_nums.append((val, raw))

                    if not parsed_nums:
                        continue

                    # If multiple numbers exist on line and multiple years exist, associate them in order
                    if len(parsed_nums) == len(years_in_chunk) and len(years_in_chunk) > 0:
                        for (val, raw), yr in zip(parsed_nums, years_in_chunk):
                            fact = FinancialFact(
                                fact_id=f"fact-{uuid.uuid4().hex[:8]}",
                                concept=concept,
                                value=val,
                                raw_text=raw,
                                unit="USD",
                                scale=scale,
                                currency="USD",
                                period=str(yr),
                                period_type="FY",
                                fiscal_year=yr,
                                source_document=doc_name,
                                page_number=page_num,
                                section=section,
                                chunk_id=chunk_id,
                            )
                            facts.append(fact)
                    else:
                        # Otherwise associate first valid number with first detected year (or default year)
                        val, raw = parsed_nums[0]
                        yr = years_in_chunk[0] if years_in_chunk else (target_years[0] if target_years else None)
                        fact = FinancialFact(
                            fact_id=f"fact-{uuid.uuid4().hex[:8]}",
                            concept=concept,
                            value=val,
                            raw_text=raw,
                            unit="USD",
                            scale=scale,
                            currency="USD",
                            period=str(yr) if yr else "N/A",
                            period_type="FY",
                            fiscal_year=yr,
                            source_document=doc_name,
                            page_number=page_num,
                            section=section,
                            chunk_id=chunk_id,
                        )
                        facts.append(fact)

        return facts


fact_extractor = FinancialFactExtractor()
