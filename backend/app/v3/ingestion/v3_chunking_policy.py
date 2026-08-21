import logging
from typing import Optional, Dict, Any
from app.v3.schemas.document_ir import V3DocumentIR

logger = logging.getLogger(__name__)


class V3ChunkingPolicy:
    """
    Backend V3 Chunking Policy Engine.
    Inspects Document Intermediate Representation (IR) and document metadata
    to select the optimal V3 chunking strategy automatically.
    """

    SUPPORTED_STRATEGIES = [
        "table_aware",
        "section_aware",
        "parent_child",
        "semantic",
        "recursive",
        "sliding_window",
        "hierarchical",
        "fixed",
    ]

    def evaluate_policy(
        self,
        doc_ir: Optional[V3DocumentIR] = None,
        document_name: Optional[str] = None,
        requested_strategy: Optional[str] = None,
        profile_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates structural features and returns selected strategy, deterministic reason,
        feature metrics, and strategy scores.
        """
        if requested_strategy and requested_strategy not in ["auto", "none", "None"] and requested_strategy in self.SUPPORTED_STRATEGIES:
            return {
                "strategy": requested_strategy,
                "reason": f"Explicitly requested strategy '{requested_strategy}'",
                "features": {"requested_strategy": requested_strategy},
                "scores": {requested_strategy: 10.0},
            }

        doc_title = document_name or (profile_dict.get("document_name") if profile_dict else "") or (doc_ir.document_name if doc_ir else "")
        doc_lower = doc_title.lower()

        is_financial = profile_dict.get("is_financial") if profile_dict is not None else any(
            term in doc_lower
            for term in ["10k", "10-k", "financial", "report", "statement", "annual", "balance sheet", "rag test doc"]
        )

        if profile_dict:
            total_pages = profile_dict.get("page_count", 1)
            table_candidate_count = profile_dict.get("table_candidate_count", 0)
            table_density = table_candidate_count / max(total_pages, 1)
            heading_count = profile_dict.get("heading_count_est", 0)
            paragraph_count = profile_dict.get("paragraph_count_est", 0)
            paragraph_density = paragraph_count / max(total_pages, 1)
        else:
            total_pages = doc_ir.total_pages if doc_ir else 1
            total_tables = sum(len(p.tables) for p in doc_ir.pages) if doc_ir else 0
            table_density = total_tables / max(total_pages, 1)

            headings = (
                [para for p in doc_ir.pages for para in p.paragraphs if para.is_heading or para.heading_level > 0]
                if doc_ir
                else []
            )
            heading_count = len(headings)

            sections = (
                set(para.section_title for p in doc_ir.pages for para in p.paragraphs if para.section_title)
                if doc_ir
                else set()
            )
            section_count = len(sections)

            paragraph_count = sum(len(p.paragraphs) for p in doc_ir.pages) if doc_ir else 0
            paragraph_density = paragraph_count / max(total_pages, 1)

        features = {
            "page_count": total_pages,
            "table_density": round(table_density, 3),
            "heading_count": heading_count,
            "paragraph_count": paragraph_count,
            "paragraph_density": round(paragraph_density, 2),
            "is_financial": is_financial,
        }

        # Calculate strategy scores based on structural metrics
        table_score = table_density * 2.0 + (5.0 if is_financial else 0.0)
        
        prose_weight = (paragraph_density * 0.4) if not is_financial else 0.0
        section_score = (
            (heading_count / max(total_pages, 1)) * 3.0
            + prose_weight
            + (3.0 if not is_financial and total_pages >= 20 else 0.0)
        )
        semantic_score = 3.0 if table_density < 0.2 and heading_count < 5 else 0.5

        scores = {
            "table_aware": round(table_score, 2),
            "section_aware": round(section_score, 2),
            "semantic": round(semantic_score, 2),
        }

        if is_financial or (table_score > section_score and table_density >= 3.0):
            strategy = "table_aware"
            reason = f"Financial report or high table density ({table_density:.2f} tables/page)"
        elif section_score >= table_score and section_score >= semantic_score:
            strategy = "section_aware"
            reason = f"High paragraph/section hierarchy ({paragraph_density:.1f} paras/page, {total_pages} pages) favoring section-aware textbook chunking"
        elif table_score >= semantic_score:
            strategy = "table_aware"
            reason = f"Table density ({table_density:.2f} tables/page) favors table-aware chunking"
        else:
            strategy = "semantic"
            reason = "Continuous prose structure with low table density and minimal section hierarchy"

        logger.info(f"[V3][POLICY] Selected '{strategy}' for '{doc_title}': {reason}")
        return {
            "strategy": strategy,
            "reason": reason,
            "features": features,
            "scores": scores,
        }

    def select_strategy(
        self,
        doc_ir: Optional[V3DocumentIR] = None,
        document_name: Optional[str] = None,
        requested_strategy: Optional[str] = None,
        profile_dict: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Determines optimal V3 chunking strategy using evaluate_policy."""
        res = self.evaluate_policy(
            doc_ir=doc_ir,
            document_name=document_name,
            requested_strategy=requested_strategy,
            profile_dict=profile_dict,
        )
        return res["strategy"]


v3_chunking_policy = V3ChunkingPolicy()
