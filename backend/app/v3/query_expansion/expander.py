import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field

from app.v3.query_expansion.config import QueryExpansionConfig
from app.v3.query_expansion.dictionary import FINANCIAL_TERMINOLOGY_DICTIONARY

logger = logging.getLogger(__name__)


class QueryExpansionTrace(BaseModel):
    original_query: str
    detected_terms: List[str] = Field(default_factory=list)
    expansions: Dict[str, List[str]] = Field(default_factory=dict)
    rewritten_query: str
    expansion_count: int = 0


class FinancialQueryExpander:
    """
    V3 Financial Query Expander.
    Detects financial abbreviations/terms in query strings and produces
    structured expansion metadata for BM25 and vector search.
    """

    def __init__(self, config: Optional[QueryExpansionConfig] = None):
        self.config = config or QueryExpansionConfig()
        self.dictionary = FINANCIAL_TERMINOLOGY_DICTIONARY

    def expand_query(self, query: str) -> QueryExpansionTrace:
        if not query or not self.config.enabled:
            return QueryExpansionTrace(
                original_query=query,
                detected_terms=[],
                expansions={},
                rewritten_query=query,
                expansion_count=0,
            )

        detected_terms: List[str] = []
        expansions_map: Dict[str, List[str]] = {}
        total_expansion_count = 0

        # Sort dictionary keys by length descending to match multi-word terms first
        sorted_keys = sorted(self.dictionary.keys(), key=lambda k: len(k), reverse=True)

        for key in sorted_keys:
            # Word boundary regex pattern handling & and hyphen characters
            escaped_key = re.escape(key)
            pattern = re.compile(rf"\b{escaped_key}\b", re.IGNORECASE)

            if pattern.search(query):
                detected_terms.append(key)
                raw_syns = self.dictionary[key]
                limited_syns = raw_syns[: self.config.max_expansions_per_term]
                expansions_map[key] = limited_syns
                total_expansion_count += len(limited_syns)

        # Build search-ready rewritten query string
        expanded_parts = [query]
        for term, syns in expansions_map.items():
            for syn in syns:
                if syn.lower() not in query.lower() and syn not in expanded_parts:
                    expanded_parts.append(syn)

        rewritten_str = " OR ".join(expanded_parts) if len(expanded_parts) > 1 else query

        trace = QueryExpansionTrace(
            original_query=query,
            detected_terms=detected_terms,
            expansions=expansions_map,
            rewritten_query=rewritten_str,
            expansion_count=total_expansion_count,
        )

        logger.info(
            f"V3 Query Expander: Query '{query}' -> Detected {detected_terms}, Total Expansions: {total_expansion_count}."
        )
        return trace


v3_query_expander = FinancialQueryExpander()
