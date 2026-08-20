from typing import List, Optional
from app.v3.schemas.document_ir import V3DocumentIR, TableIR
from app.v3.schemas.chunk_schema import V3Chunk, ChunkingConfig
from app.v3.chunking.base import BaseChunkingStrategy


class TableAwareChunkingStrategy(BaseChunkingStrategy):
    """
    HIGHEST PRIORITY STRATEGY:
    Structure-preserving Table-Aware Chunking Strategy.
    - Treats tables as first-class structured retrieval units.
    - Repeats column headers and table title in EVERY table chunk.
    - Preserves row ranges, column ranges, bounding boxes, and multi-page table fragments.
    """

    @property
    def strategy_name(self) -> str:
        return "table_aware"

    def chunk(self, doc_ir: V3DocumentIR) -> List[V3Chunk]:
        chunks: List[V3Chunk] = []
        c_idx = 1

        max_rows = self.config.max_table_rows_per_chunk

        for page in doc_ir.pages:
            p_num = page.page_number

            # 1. Process Tables with Mandatory Header Repetition
            for t_idx, table in enumerate(page.tables, 1):
                headers = table.headers
                rows = table.rows
                num_rows = len(rows)
                num_cols = len(headers) if headers else (len(rows[0]) if rows else 0)

                # If small table, emit single intact table chunk
                if num_rows <= max_rows or num_rows == 0:
                    chunks.append(
                        V3Chunk(
                            chunk_id=f"{doc_ir.document_id}_tbl_{c_idx}",
                            document_id=doc_ir.document_id,
                            document_name=doc_ir.document_name,
                            source_path=doc_ir.source_path,
                            page_number=p_num,
                            page_start=p_num,
                            page_end=p_num,
                            section=table.title or f"Page {p_num} Table {t_idx}",
                            chunk_type="table",
                            content=table.markdown_content,
                            bbox=table.bbox,
                            table_id=table.table_id,
                            table_title=table.title or f"Page {p_num} Table {t_idx}",
                            row_range=[0, num_rows],
                            column_range=[0, num_cols],
                            strategy=self.strategy_name,
                        )
                    )
                    c_idx += 1
                else:
                    # Large table: split into row groups while REPEATING HEADERS in every group
                    row_start = 0
                    group_num = 1

                    header_line = "| " + " | ".join(headers) + " |" if headers else ""
                    sep_line = "| " + " | ".join(["---"] * max(len(headers), 1)) + " |" if headers else ""

                    while row_start < num_rows:
                        row_end = min(row_start + max_rows, num_rows)
                        row_slice = rows[row_start:row_end]

                        # Build markdown chunk with repeated headers
                        md_lines = []
                        if table.title:
                            md_lines.append(f"### Table: {table.title} (Part {group_num})\n")
                        else:
                            md_lines.append(f"### Table: Page {p_num} Table {t_idx} (Part {group_num})\n")

                        if header_line and sep_line:
                            md_lines.append(header_line)
                            md_lines.append(sep_line)

                        for r in row_slice:
                            md_lines.append("| " + " | ".join(r) + " |")

                        chunk_content = "\n".join(md_lines)

                        chunks.append(
                            V3Chunk(
                                chunk_id=f"{doc_ir.document_id}_tbl_{c_idx}",
                                document_id=doc_ir.document_id,
                                document_name=doc_ir.document_name,
                                source_path=doc_ir.source_path,
                                page_number=p_num,
                                page_start=p_num,
                                page_end=p_num,
                                section=table.title or f"Page {p_num} Table {t_idx}",
                                chunk_type="table",
                                content=chunk_content,
                                bbox=table.bbox,
                                table_id=f"{table.table_id}_part{group_num}",
                                table_title=table.title or f"Page {p_num} Table {t_idx}",
                                row_range=[row_start, row_end],
                                column_range=[0, num_cols],
                                strategy=self.strategy_name,
                            )
                        )
                        c_idx += 1
                        group_num += 1
                        row_start = row_end

            # 2. Process Paragraphs as Narrative Chunks
            for para in page.paragraphs:
                if para.is_header_footer or not para.text.strip():
                    continue

                chunks.append(
                    V3Chunk(
                        chunk_id=f"{doc_ir.document_id}_ta_p_{c_idx}",
                        document_id=doc_ir.document_id,
                        document_name=doc_ir.document_name,
                        source_path=doc_ir.source_path,
                        page_number=p_num,
                        page_start=p_num,
                        page_end=p_num,
                        section=para.section_title or f"Page {p_num}",
                        chunk_type="heading" if para.is_heading else "paragraph",
                        content=para.text,
                        bbox=para.bbox,
                        strategy=self.strategy_name,
                    )
                )
                c_idx += 1

        return chunks
