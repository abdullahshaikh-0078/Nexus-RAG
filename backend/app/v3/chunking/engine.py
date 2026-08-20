import logging
from typing import Dict, List, Type, Optional

from app.v3.schemas.document_ir import V3DocumentIR
from app.v3.schemas.chunk_schema import V3Chunk, ChunkingConfig
from app.v3.chunking.base import BaseChunkingStrategy
from app.v3.chunking.fixed import FixedChunkingStrategy
from app.v3.chunking.recursive import RecursiveChunkingStrategy
from app.v3.chunking.semantic import SemanticChunkingStrategy
from app.v3.chunking.section_aware import SectionAwareChunkingStrategy
from app.v3.chunking.table_aware import TableAwareChunkingStrategy
from app.v3.chunking.parent_child import ParentChildChunkingStrategy
from app.v3.chunking.sliding_window import SlidingWindowChunkingStrategy
from app.v3.chunking.hierarchical import HierarchicalChunkingStrategy
from app.v3.chunking.validator import V3ChunkValidator

logger = logging.getLogger(__name__)


class V3ChunkingEngine:
    """
    Central V3 Chunking Engine & Strategy Registry.
    Registers and executes all 8 V3 chunking strategies against V3DocumentIR.
    """

    def __init__(self):
        self._strategies: Dict[str, Type[BaseChunkingStrategy]] = {}
        self._register_default_strategies()

    def _register_default_strategies(self):
        self.register_strategy("fixed", FixedChunkingStrategy)
        self.register_strategy("recursive", RecursiveChunkingStrategy)
        self.register_strategy("semantic", SemanticChunkingStrategy)
        self.register_strategy("section_aware", SectionAwareChunkingStrategy)
        self.register_strategy("table_aware", TableAwareChunkingStrategy)
        self.register_strategy("parent_child", ParentChildChunkingStrategy)
        self.register_strategy("sliding_window", SlidingWindowChunkingStrategy)
        self.register_strategy("hierarchical", HierarchicalChunkingStrategy)

    def register_strategy(self, name: str, strategy_cls: Type[BaseChunkingStrategy]):
        self._strategies[name.lower()] = strategy_cls

    def get_registered_strategies(self) -> List[str]:
        return sorted(list(self._strategies.keys()))

    def chunk_document(
        self,
        doc_ir: V3DocumentIR,
        config: Optional[ChunkingConfig] = None,
    ) -> List[V3Chunk]:
        cfg = config or ChunkingConfig()
        strat_name = cfg.strategy.lower()

        if strat_name not in self._strategies:
            raise ValueError(
                f"Unknown chunking strategy '{strat_name}'. Available: {self.get_registered_strategies()}"
            )

        strategy_cls = self._strategies[strat_name]
        strategy_inst = strategy_cls(config=cfg)

        logger.info(f"V3 Engine: Chunking '{doc_ir.document_name}' using '{strat_name}' strategy...")

        chunks = strategy_inst.chunk(doc_ir)

        # Validate generated chunks
        is_valid, errors = V3ChunkValidator.validate_chunks(chunks)
        if not is_valid:
            logger.warning(f"V3 Engine: Strategy '{strat_name}' generated {len(errors)} validation warnings.")

        logger.info(f"V3 Engine: Strategy '{strat_name}' generated {len(chunks)} valid chunks.")
        return chunks


v3_chunking_engine = V3ChunkingEngine()
