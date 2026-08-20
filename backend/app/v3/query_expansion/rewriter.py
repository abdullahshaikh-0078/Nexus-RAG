from typing import List, Optional, Tuple
from app.v3.query_expansion.expander import FinancialQueryExpander, QueryExpansionTrace
from app.v3.query_expansion.config import QueryExpansionConfig

Tuple_Query_Rewritten = Tuple[str, QueryExpansionTrace]


class QueryRewriter:
    """
    Structured V3 Query Rewriter.
    Constructs search-optimized query representations for BM25 and vector retrieval pipelines.
    """

    def __init__(self, expander: Optional[FinancialQueryExpander] = None):
        self.expander = expander or FinancialQueryExpander()

    def rewrite_for_bm25(self, query: str) -> Tuple_Query_Rewritten:
        """
        Builds a lexical search string preserving original query terms alongside financial synonyms.
        """
        trace = self.expander.expand_query(query)
        bm25_text = trace.original_query
        syn_terms = []

        for term, syns in trace.expansions.items():
            for syn in syns:
                if syn.lower() not in bm25_text.lower():
                    syn_terms.append(syn)

        if syn_terms:
            bm25_text = f"{trace.original_query} {' '.join(syn_terms)}"

        return bm25_text, trace

    def rewrite_for_vector(self, query: str) -> Tuple_Query_Rewritten:
        """
        Builds a semantic vector text prompt incorporating accounting context.
        """
        trace = self.expander.expand_query(query)
        vec_text = trace.original_query

        syn_list = []
        for term, syns in trace.expansions.items():
            syn_list.extend(syns)

        if syn_list:
            vec_text = f"{trace.original_query} (Also search for: {', '.join(syn_list)})"

        return vec_text, trace


v3_query_rewriter = QueryRewriter()
