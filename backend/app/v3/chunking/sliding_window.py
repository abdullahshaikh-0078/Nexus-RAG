from typing import List
from app.v3.schemas.document_ir import V3DocumentIR
from app.v3.schemas.chunk_schema import V3Chunk, ChunkingConfig
from app.v3.chunking.base import BaseChunkingStrategy


class SlidingWindowChunkingStrategy(BaseChunkingStrategy):
    """
    Sliding-Window Chunking Strategy sliding a window of paragraphs across pages.
    Protects tables as intact units.
    """

    @property
    def strategy_name(self) -> str:
        return "sliding_window"

    def chunk(self, doc_ir: V3DocumentIR) -> List[V3Chunk]:
        chunks: List[V3Chunk] = []
        c_idx = 1
        window_size = self.config.window_size
        stride = max(1, self.config.stride)

        for page in doc_ir.pages:
            p_num = page.page_number

            # Tables protected
            for table in page.tables:
                chunks.append(
                    V3Chunk(
                        chunk_id=f"{doc_ir.document_id}_sw_tbl_{c_idx}",
                        document_id=doc_ir.document_id,
                        document_name=doc_ir.document_name,
                        source_path=doc_ir.source_path,
                        page_number=p_num,
                        page_start=p_num,
                        page_end=p_num,
                        section=table.title or f"Page {p_num} Table",
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

            paras = [p for p in page.paragraphs if not p.is_header_footer and p.text.strip()]
            if not paras:
                continue

            i = 0
            while i < len(paras):
                window_paras = paras[i : i + window_size]
                window_text = "\n\n".join(p.text for p in window_paras)
                sec_title = window_paras[0].section_title or f"Page {p_num}"

                chunks.append(
                    V3Chunk(
                        chunk_id=f"{doc_ir.document_id}_sw_{c_idx}",
                        document_id=doc_ir.document_id,
                        document_name=doc_ir.document_name,
                        source_path=doc_ir.source_path,
                        page_number=p_num,
                        page_start=p_num,
                        page_end=p_num,
                        section=sec_title,
                        chunk_type="paragraph",
                        content=window_text,
                        bbox=window_paras[0].bbox,
                        strategy=self.strategy_name,
                    )
                )
                c_idx += 1
                i += stride

        return chunks
