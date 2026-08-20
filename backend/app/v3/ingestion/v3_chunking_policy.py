import logging
from typing import Optional
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

    def select_strategy(
        self,
        doc_ir: Optional[V3DocumentIR] = None,
        document_name: Optional[str] = None,
        requested_strategy: Optional[str] = None,
    ) -> str:
        """
        Determines optimal V3 chunking strategy.
        If requested_strategy is explicitly specified and valid, it is honored (for benchmarking/tests).
        Otherwise, backend policy inspects the IR structure and metadata.
        """
        if requested_strategy and requested_strategy in self.SUPPORTED_STRATEGIES:
            logger.info(f"[V3][POLICY] Explicit strategy requested: '{requested_strategy}'")
            return requested_strategy

        # Default fallback for V3 Financial RAG
        selected = "table_aware"

        if doc_ir:
            total_tables = doc_ir.metadata.get("total_tables", 0) if doc_ir.metadata else 0
            if total_tables == 0:
                total_tables = sum(len(p.tables) for p in doc_ir.pages)

            total_sections = len(doc_ir.sections) if hasattr(doc_ir, "sections") else 0

            # Policy logic:
            if total_tables > 0:
                selected = "table_aware"
                logger.info(f"[V3][POLICY] Document '{document_name or doc_ir.document_name}' contains {total_tables} tables. Policy selected: 'table_aware'")
            elif total_sections > 3:
                selected = "section_aware"
                logger.info(f"[V3][POLICY] Document '{document_name or doc_ir.document_name}' contains {total_sections} sections. Policy selected: 'section_aware'")
            else:
                selected = "table_aware"
                logger.info(f"[V3][POLICY] Policy default selected: 'table_aware'")
        else:
            doc_lower = (document_name or "").lower()
            if any(term in doc_lower for term in ["10k", "10-k", "financial", "report", "statement", "annual"]):
                selected = "table_aware"
            else:
                selected = "table_aware"
            logger.info(f"[V3][POLICY] Document name heuristic for '{document_name}'. Selected: '{selected}'")

        return selected


v3_chunking_policy = V3ChunkingPolicy()
