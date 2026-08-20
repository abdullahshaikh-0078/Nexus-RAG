from typing import List, Dict
from app.v3.schemas.document_ir import V3DocumentIR, ParagraphIR
from app.v3.schemas.chunk_schema import V3Chunk, ChunkingConfig
from app.v3.chunking.base import BaseChunkingStrategy


class SectionAwareChunkingStrategy(BaseChunkingStrategy):
    """
    Section-Aware Chunking Strategy grouping paragraphs and tables strictly by Section Title.
    """

    @property
    def strategy_name(self) -> str:
        return "section_aware"

    def chunk(self, doc_ir: V3DocumentIR) -> List[V3Chunk]:
        chunks: List[V3Chunk] = []
        c_idx = 1

        for page in doc_ir.pages:
            p_num = page.page_number

            # Group paragraphs by section_title
            section_groups: Dict[str, List[ParagraphIR]] = {}
            for p in page.paragraphs:
                if p.is_header_footer or not p.text.strip():
                    continue
                sec = p.section_title or f"Page {p_num} Section"
                section_groups.setdefault(sec, []).append(p)

            for sec_title, p_list in section_groups.items():
                content_text = "\n\n".join(p.text for p in p_list)
                chunks.append(
                    V3Chunk(
                        chunk_id=f"{doc_ir.document_id}_sec_{c_idx}",
                        document_id=doc_ir.document_id,
                        document_name=doc_ir.document_name,
                        source_path=doc_ir.source_path,
                        page_number=p_num,
                        page_start=p_num,
                        page_end=p_num,
                        section=sec_title,
                        chunk_type="section",
                        content=content_text,
                        bbox=p_list[0].bbox if p_list else None,
                        strategy=self.strategy_name,
                    )
                )
                c_idx += 1

            # Tables in section
            for table in page.tables:
                chunks.append(
                    V3Chunk(
                        chunk_id=f"{doc_ir.document_id}_sec_{c_idx}",
                        document_id=doc_ir.document_id,
                        document_name=doc_ir.document_name,
                        source_path=doc_ir.source_path,
                        page_number=p_num,
                        page_start=p_num,
                        page_end=p_num,
                        section=table.title or f"Page {p_num} Table Section",
                        chunk_type="table",
                        content=table.markdown_content,
                        bbox=table.bbox,
                        table_id=table.table_id,
                        table_title=table.title,
                        row_range=[0, len(table.rows)],
                        column_range=[0, len(table.headers) if table.headers else (len(table.rows[0]) if table.rows else 0)],
                        strategy=self.strategy_name,
                    )
                )
                c_idx += 1

        return chunks
