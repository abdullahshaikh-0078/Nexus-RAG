from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from app.v3.schemas.document_ir import BoundingBox


class V3Chunk(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    source_path: str
    page_number: int
    page_start: int
    page_end: int
    section: Optional[str] = None
    subsection: Optional[str] = None
    chunk_type: str  # paragraph, table, heading, footnote, parent, child, section, hierarchical
    content: str
    bbox: Optional[BoundingBox] = None
    table_id: Optional[str] = None
    table_title: Optional[str] = None
    row_range: Optional[List[int]] = None
    column_range: Optional[List[int]] = None
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = Field(default_factory=list)
    strategy: str


class ChunkingConfig(BaseModel):
    strategy: str = "table_aware"  # fixed, recursive, semantic, section_aware, table_aware, parent_child, sliding_window, hierarchical
    chunk_size: int = 1000
    overlap: int = 150
    max_table_rows_per_chunk: int = 10
    semantic_similarity_threshold: float = 0.75
    window_size: int = 3
    stride: int = 1
