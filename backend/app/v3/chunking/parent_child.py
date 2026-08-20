from typing import List, Dict
from app.v3.schemas.document_ir import V3DocumentIR
from app.v3.schemas.chunk_schema import V3Chunk, ChunkingConfig
from app.v3.chunking.base import BaseChunkingStrategy


class ParentChildChunkingStrategy(BaseChunkingStrategy):
    """
    Parent-Child Chunking Strategy creating section parent summary chunks
    linked to detailed child paragraph/table chunks via parent_chunk_id & child_chunk_ids.
    """

    @property
    def strategy_name(self) -> str:
        return "parent_child"

    def chunk(self, doc_ir: V3DocumentIR) -> List[V3Chunk]:
        chunks: List[V3Chunk] = []
        c_idx = 1

        for page in doc_ir.pages:
            p_num = page.page_number

            # Group page elements by section
            section_map: Dict[str, List[Any]] = {}
            for p in page.paragraphs:
                if p.is_header_footer or not p.text.strip():
                    continue
                sec = p.section_title or f"Page {p_num} Section"
                section_map.setdefault(sec, []).append(p)

            for sec_name, p_list in section_map.items():
                parent_id = f"{doc_ir.document_id}_parent_{c_idx}"
                c_idx += 1

                child_ids = []
                child_chunks: List[V3Chunk] = []

                for p_item in p_list:
                    ch_id = f"{doc_ir.document_id}_child_{c_idx}"
                    child_ids.append(ch_id)
                    child_chunks.append(
                        V3Chunk(
                            chunk_id=ch_id,
                            document_id=doc_ir.document_id,
                            document_name=doc_ir.document_name,
                            source_path=doc_ir.source_path,
                            page_number=p_num,
                            page_start=p_num,
                            page_end=p_num,
                            section=sec_name,
                            chunk_type="child",
                            content=p_item.text,
                            bbox=p_item.bbox,
                            parent_chunk_id=parent_id,
                            strategy=self.strategy_name,
                        )
                    )
                    c_idx += 1

                # Parent chunk summary
                parent_text = f"Section Overview: {sec_name}\n\n" + "\n".join(p.text[:150] for p in p_list[:3])
                parent_chunk = V3Chunk(
                    chunk_id=parent_id,
                    document_id=doc_ir.document_id,
                    document_name=doc_ir.document_name,
                    source_path=doc_ir.source_path,
                    page_number=p_num,
                    page_start=p_num,
                    page_end=p_num,
                    section=sec_name,
                    chunk_type="parent",
                    content=parent_text,
                    bbox=p_list[0].bbox if p_list else None,
                    child_chunk_ids=child_ids,
                    strategy=self.strategy_name,
                )

                chunks.append(parent_chunk)
                chunks.extend(child_chunks)

            # Process Tables as standalone chunks
            for table in page.tables:
                chunks.append(
                    V3Chunk(
                        chunk_id=f"{doc_ir.document_id}_pc_tbl_{c_idx}",
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

        return chunks
