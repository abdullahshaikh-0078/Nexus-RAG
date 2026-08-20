from abc import ABC, abstractmethod
from typing import List
from app.v3.schemas.document_ir import V3DocumentIR
from app.v3.schemas.chunk_schema import V3Chunk, ChunkingConfig


class BaseChunkingStrategy(ABC):
    """
    Abstract Base Class for all V3 Chunking Strategies.
    Consumes V3DocumentIR and produces a List of standardized V3Chunk objects.
    """

    def __init__(self, config: ChunkingConfig):
        self.config = config

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        pass

    @abstractmethod
    def chunk(self, doc_ir: V3DocumentIR) -> List[V3Chunk]:
        pass
