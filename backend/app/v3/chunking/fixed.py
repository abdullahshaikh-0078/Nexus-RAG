from typing import List
from app.v3.schemas.document_ir import V3DocumentIR
from app.v3.schemas.chunk_schema import V3Chunk, ChunkingConfig
from app.v3.chunking.base import BaseChunkingStrategy


class FixedChunkingStrategy(BaseChunkingStrategy):
    """
    Fixed-size character window chunking strategy preserving V3 document provenance.
    Used for baseline strategy comparison.
    """

    @property
    def strategy_name(self) -> str:
        return "fixed"

    def chunk(self, doc_ir: V3DocumentIR) -> List[V3Chunk]:
        chunks: List[V3Chunk] = []
        chunk_size = self.config.chunk_size
        overlap = self.config.overlap

        # Gather all candidates from V3DocumentIR
        candidates = doc_ir.to_chunk_candidates()
        full_text = "\n\n".join(c["content"] for c in candidates if c["content"])

        if not full_text:
            return chunks

        start = 0
        chunk_idx = 1

        while start < len(full_text):
            end = start + chunk_size
            segment = full_text[start:end]

            # Approximate page binding from full_text position
            approx_page = 1
            if doc_ir.pages:
                ratio = min(start / max(len(full_text), 1), 0.99)
                approx_page = max(1, int(ratio * doc_ir.total_pages) + 1)

            chunks.append(
                V3Chunk(
                    chunk_id=f"{doc_ir.document_id}_fixed_{chunk_idx}",
                    document_id=doc_ir.document_id,
                    document_name=doc_ir.document_name,
                    source_path=doc_ir.source_path,
                    page_number=approx_page,
                    page_start=approx_page,
                    page_end=approx_page,
                    section=f"Fixed Segment {chunk_idx}",
                    chunk_type="paragraph",
                    content=segment,
                    strategy=self.strategy_name,
                )
            )

            chunk_idx += 1
            start += (chunk_size - overlap)
            if start >= len(full_text):
                break

        return chunks
