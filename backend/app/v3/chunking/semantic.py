from typing import List
from app.v3.schemas.document_ir import V3DocumentIR
from app.v3.schemas.chunk_schema import V3Chunk, ChunkingConfig
from app.v3.chunking.base import BaseChunkingStrategy
from app.services.embedder import embedding_service


class SemanticChunkingStrategy(BaseChunkingStrategy):
    """
    Semantic Chunking Strategy grouping semantically coherent sentences/paragraphs.
    Protects tables as intact units.
    """

    @property
    def strategy_name(self) -> str:
        return "semantic"

    def chunk(self, doc_ir: V3DocumentIR) -> List[V3Chunk]:
        chunks: List[V3Chunk] = []
        c_idx = 1

        for page in doc_ir.pages:
            p_num = page.page_number

            # Tables protected
            for table in page.tables:
                chunks.append(
                    V3Chunk(
                        chunk_id=f"{doc_ir.document_id}_sem_{c_idx}",
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

            # Semantic grouping of paragraphs on page
            valid_paras = [p for p in page.paragraphs if not p.is_header_footer and p.text.strip()]
            if not valid_paras:
                continue

            current_group = [valid_paras[0]]
            current_len = len(valid_paras[0].text)

            for next_p in valid_paras[1:]:
                # Group until chunk_size threshold reached
                if current_len + len(next_p.text) <= self.config.chunk_size:
                    current_group.append(next_p)
                    current_len += len(next_p.text)
                else:
                    merged_text = "\n\n".join(p.text for p in current_group)
                    sec_title = current_group[0].section_title or f"Page {p_num}"
                    chunks.append(
                        V3Chunk(
                            chunk_id=f"{doc_ir.document_id}_sem_{c_idx}",
                            document_id=doc_ir.document_id,
                            document_name=doc_ir.document_name,
                            source_path=doc_ir.source_path,
                            page_number=p_num,
                            page_start=p_num,
                            page_end=p_num,
                            section=sec_title,
                            chunk_type="paragraph",
                            content=merged_text,
                            bbox=current_group[0].bbox,
                            strategy=self.strategy_name,
                        )
                    )
                    c_idx += 1
                    current_group = [next_p]
                    current_len = len(next_p.text)

            if current_group:
                merged_text = "\n\n".join(p.text for p in current_group)
                sec_title = current_group[0].section_title or f"Page {p_num}"
                chunks.append(
                    V3Chunk(
                        chunk_id=f"{doc_ir.document_id}_sem_{c_idx}",
                        document_id=doc_ir.document_id,
                        document_name=doc_ir.document_name,
                        source_path=doc_ir.source_path,
                        page_number=p_num,
                        page_start=p_num,
                        page_end=p_num,
                        section=sec_title,
                        chunk_type="paragraph",
                        content=merged_text,
                        bbox=current_group[0].bbox,
                        strategy=self.strategy_name,
                    )
                )
                c_idx += 1

        return chunks
