from typing import List
from app.v3.schemas.document_ir import V3DocumentIR
from app.v3.schemas.chunk_schema import V3Chunk, ChunkingConfig
from app.v3.chunking.base import BaseChunkingStrategy


class HierarchicalChunkingStrategy(BaseChunkingStrategy):
    """
    Hierarchical Chunking Strategy representing document structure explicitly:
    Document -> Page -> Section -> Paragraph/Table -> Chunk
    """

    @property
    def strategy_name(self) -> str:
        return "hierarchical"

    def chunk(self, doc_ir: V3DocumentIR) -> List[V3Chunk]:
        chunks: List[V3Chunk] = []
        c_idx = 1

        # Document Root Chunk
        doc_root_id = f"{doc_ir.document_id}_root"
        doc_summary = f"Document: {doc_ir.document_name} ({doc_ir.total_pages} pages)"
        page_chunk_ids = []

        for page in doc_ir.pages:
            p_num = page.page_number
            page_chunk_id = f"{doc_ir.document_id}_page_{p_num}"
            page_chunk_ids.append(page_chunk_id)

            element_chunk_ids = []

            # Page Tables
            for table in page.tables:
                tbl_id = f"{doc_ir.document_id}_h_tbl_{c_idx}"
                element_chunk_ids.append(tbl_id)
                chunks.append(
                    V3Chunk(
                        chunk_id=tbl_id,
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
                        parent_chunk_id=page_chunk_id,
                        strategy=self.strategy_name,
                    )
                )
                c_idx += 1

            # Page Paragraphs
            for para in page.paragraphs:
                if para.is_header_footer or not para.text.strip():
                    continue

                p_id = f"{doc_ir.document_id}_h_p_{c_idx}"
                element_chunk_ids.append(p_id)
                chunks.append(
                    V3Chunk(
                        chunk_id=p_id,
                        document_id=doc_ir.document_id,
                        document_name=doc_ir.document_name,
                        source_path=doc_ir.source_path,
                        page_number=p_num,
                        page_start=p_num,
                        page_end=p_num,
                        section=para.section_title or f"Page {p_num}",
                        chunk_type="hierarchical",
                        content=para.text,
                        bbox=para.bbox,
                        parent_chunk_id=page_chunk_id,
                        strategy=self.strategy_name,
                    )
                )
                c_idx += 1

            # Intermediate Page Node Chunk
            page_chunk = V3Chunk(
                chunk_id=page_chunk_id,
                document_id=doc_ir.document_id,
                document_name=doc_ir.document_name,
                source_path=doc_ir.source_path,
                page_number=p_num,
                page_start=p_num,
                page_end=p_num,
                section=f"Page {p_num} Hierarchy Node",
                chunk_type="parent",
                content=f"Page {p_num} of {doc_ir.document_name}",
                parent_chunk_id=doc_root_id,
                child_chunk_ids=element_chunk_ids,
                strategy=self.strategy_name,
            )
            chunks.append(page_chunk)

        # Document Root Node Chunk
        root_chunk = V3Chunk(
            chunk_id=doc_root_id,
            document_id=doc_ir.document_id,
            document_name=doc_ir.document_name,
            source_path=doc_ir.source_path,
            page_number=1,
            page_start=1,
            page_end=doc_ir.total_pages,
            section="Document Root",
            chunk_type="parent",
            content=doc_summary,
            child_chunk_ids=page_chunk_ids,
            strategy=self.strategy_name,
        )
        chunks.append(root_chunk)

        return chunks
