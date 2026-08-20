from typing import List
from app.v3.schemas.document_ir import V3DocumentIR
from app.v3.schemas.chunk_schema import V3Chunk, ChunkingConfig
from app.v3.chunking.base import BaseChunkingStrategy


class RecursiveChunkingStrategy(BaseChunkingStrategy):
    """
    Structural recursive splitting along hierarchy:
    document -> page -> section -> paragraph -> sentence
    Protects tables as intact units.
    """

    @property
    def strategy_name(self) -> str:
        return "recursive"

    def chunk(self, doc_ir: V3DocumentIR) -> List[V3Chunk]:
        chunks: List[V3Chunk] = []
        c_idx = 1

        for page in doc_ir.pages:
            p_num = page.page_number

            # Tables protected as intact chunks
            for table in page.tables:
                chunks.append(
                    V3Chunk(
                        chunk_id=f"{doc_ir.document_id}_rec_{c_idx}",
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

            # Paragraphs split recursively if length exceeds chunk_size
            for para in page.paragraphs:
                if para.is_header_footer or not para.text.strip():
                    continue

                text = para.text
                if len(text) <= self.config.chunk_size:
                    sub_texts = [text]
                else:
                    # Recursive split into sentences
                    sub_texts = [s.strip() for s in text.split(". ") if s.strip()]

                for sub in sub_texts:
                    chunks.append(
                        V3Chunk(
                            chunk_id=f"{doc_ir.document_id}_rec_{c_idx}",
                            document_id=doc_ir.document_id,
                            document_name=doc_ir.document_name,
                            source_path=doc_ir.source_path,
                            page_number=p_num,
                            page_start=p_num,
                            page_end=p_num,
                            section=para.section_title or f"Page {p_num}",
                            chunk_type="heading" if para.is_heading else "paragraph",
                            content=sub,
                            bbox=para.bbox,
                            strategy=self.strategy_name,
                        )
                    )
                    c_idx += 1

        return chunks
