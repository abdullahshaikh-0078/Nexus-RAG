from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float
    page_number: int


class TableCellIR(BaseModel):
    row_index: int
    col_index: int
    row_span: int = 1
    col_span: int = 1
    content: str
    is_header: bool = False
    bbox: Optional[BoundingBox] = None


class TableIR(BaseModel):
    table_id: str
    page_number: int
    title: Optional[str] = None
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    cells: List[TableCellIR] = Field(default_factory=list)
    markdown_content: str
    bbox: Optional[BoundingBox] = None


class ParagraphIR(BaseModel):
    paragraph_id: str
    page_number: int
    section_title: Optional[str] = None
    text: str
    is_heading: bool = False
    heading_level: int = 0
    is_footnote: bool = False
    is_header_footer: bool = False
    bbox: Optional[BoundingBox] = None


class FootnoteIR(BaseModel):
    footnote_id: str
    page_number: int
    marker: str
    text: str
    referenced_element_id: Optional[str] = None
    bbox: Optional[BoundingBox] = None


class FigureIR(BaseModel):
    figure_id: str
    page_number: int
    caption: Optional[str] = None
    bbox: Optional[BoundingBox] = None


class PageIR(BaseModel):
    page_number: int
    paragraphs: List[ParagraphIR] = Field(default_factory=list)
    tables: List[TableIR] = Field(default_factory=list)
    footnotes: List[FootnoteIR] = Field(default_factory=list)
    figures: List[FigureIR] = Field(default_factory=list)
    header_text: Optional[str] = None
    footer_text: Optional[str] = None


class V3DocumentIR(BaseModel):
    document_id: str
    document_name: str
    source_path: str
    total_pages: int
    pages: List[PageIR] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_chunk_candidates(self) -> List[Dict[str, Any]]:
        """
        Converts the document IR into a list of chunk-ready candidates supporting:
        document_id, document_name, source_path, page_number, page_start, page_end,
        section, subsection, chunk_id, chunk_type, content, bbox, table_id, table_title,
        row_range, column_range.
        """
        candidates = []

        for page in self.pages:
            p_num = page.page_number

            # Tables as first-class chunks
            for t_idx, table in enumerate(page.tables, 1):
                candidates.append({
                    "document_id": self.document_id,
                    "document_name": self.document_name,
                    "source_path": self.source_path,
                    "page_number": p_num,
                    "page_start": p_num,
                    "page_end": p_num,
                    "section": table.title or f"Page {p_num} Table {t_idx}",
                    "subsection": None,
                    "chunk_id": f"{self.document_id}_p{p_num}_table_{t_idx}",
                    "chunk_type": "table",
                    "content": table.markdown_content,
                    "bbox": table.bbox.model_dump() if table.bbox else None,
                    "table_id": table.table_id,
                    "table_title": table.title,
                    "row_range": [0, len(table.rows)],
                    "column_range": [0, len(table.headers) if table.headers else (len(table.rows[0]) if table.rows else 0)],
                })

            # Paragraphs
            for p_idx, para in enumerate(page.paragraphs, 1):
                if para.is_header_footer or not para.text.strip():
                    continue

                candidates.append({
                    "document_id": self.document_id,
                    "document_name": self.document_name,
                    "source_path": self.source_path,
                    "page_number": p_num,
                    "page_start": p_num,
                    "page_end": p_num,
                    "section": para.section_title or f"Page {p_num}",
                    "subsection": None,
                    "chunk_id": f"{self.document_id}_p{p_num}_para_{p_idx}",
                    "chunk_type": "heading" if para.is_heading else ("footnote" if para.is_footnote else "paragraph"),
                    "content": para.text,
                    "bbox": para.bbox.model_dump() if para.bbox else None,
                    "table_id": None,
                    "table_title": None,
                    "row_range": None,
                    "column_range": None,
                })

            # Footnotes
            for fn_idx, fn in enumerate(page.footnotes, 1):
                candidates.append({
                    "document_id": self.document_id,
                    "document_name": self.document_name,
                    "source_path": self.source_path,
                    "page_number": p_num,
                    "page_start": p_num,
                    "page_end": p_num,
                    "section": f"Page {p_num} Footnote",
                    "subsection": None,
                    "chunk_id": f"{self.document_id}_p{p_num}_fn_{fn_idx}",
                    "chunk_type": "footnote",
                    "content": f"{fn.marker} {fn.text}",
                    "bbox": fn.bbox.model_dump() if fn.bbox else None,
                    "table_id": None,
                    "table_title": None,
                    "row_range": None,
                    "column_range": None,
                })

        return candidates
