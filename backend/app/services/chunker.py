import re
import uuid
from typing import List
from app.models.schemas import DocumentChunk


class RecursiveTextChunker:
    """
    Smart, heading-aware text chunker that preserves paragraph integrity 
    and keeps section titles bound to their following body text.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def is_heading(line: str) -> bool:
        """Determines if a line is likely a section heading."""
        text = line.strip()
        if not text or len(text) > 90:
            return False

        # Pattern matches: "1 Introduction", "3.2.3 Applications of Attention...", "Abstract", "A.1 Architecture"
        heading_patterns = [
            r"^(abstract|introduction|background|related work|conclusion|discussion|references|acknowledgments)$",
            r"^\d+(\.\d+)*\s+[A-Z]",
            r"^[A-Z][a-zA-Z0-9\s\-\,\:\(\)]{2,70}$",
        ]
        for pat in heading_patterns:
            if re.match(pat, text, re.IGNORECASE) and not text.endswith((".", "?", "!")):
                return True
        return False

    def chunk_document(self, text: str, document_id: str) -> List[DocumentChunk]:
        """Splits raw document text into overlapping, heading-bound chunks."""
        if not text or not text.strip():
            return []

        # 1. Group text into logical blocks (binding headings to following paragraphs)
        blocks = self._create_bound_blocks(text)

        # 2. Assemble blocks into chunks within size & overlap constraints
        raw_chunks = self._assemble_chunks(blocks)

        document_chunks: List[DocumentChunk] = []
        start_cursor = 0
        for index, chunk_text in enumerate(raw_chunks):
            start_pos = text.find(chunk_text[:40], start_cursor)
            if start_pos == -1:
                start_pos = start_cursor
            end_pos = start_pos + len(chunk_text)
            start_cursor = max(0, end_pos - self.chunk_overlap)

            chunk_id = f"{document_id}_chk_{index}_{uuid.uuid4().hex[:6]}"
            doc_chunk = DocumentChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                chunk_index=index,
                text=chunk_text,
                start_char=start_pos,
                end_char=end_pos,
                metadata={
                    "char_count": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                }
            )
            document_chunks.append(doc_chunk)

        return document_chunks

    def _create_bound_blocks(self, text: str) -> List[str]:
        """Splits text into paragraphs and binds isolated heading lines to the subsequent paragraph."""
        raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        bound_blocks: List[str] = []

        i = 0
        while i < len(raw_paragraphs):
            para = raw_paragraphs[i]
            # Check if this paragraph is purely a heading line or starts with a heading line
            lines = para.split("\n")
            if len(lines) == 1 and self.is_heading(lines[0]) and i + 1 < len(raw_paragraphs):
                # Bind heading with next paragraph
                next_para = raw_paragraphs[i + 1]
                bound_blocks.append(f"{para}\n{next_para}")
                i += 2
            else:
                bound_blocks.append(para)
                i += 1

        return bound_blocks

    def _assemble_chunks(self, blocks: List[str]) -> List[str]:
        """Combines structural blocks into chunks under chunk_size with overlap."""
        chunks: List[str] = []
        current_chunk = ""

        for block in blocks:
            # If a single block exceeds chunk_size, split by sentences/lines
            if len(block) > self.chunk_size:
                sub_parts = self._split_large_block(block)
                for part in sub_parts:
                    current_chunk = self._add_to_chunk(part, current_chunk, chunks)
            else:
                current_chunk = self._add_to_chunk(block, current_chunk, chunks)

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _add_to_chunk(self, block: str, current_chunk: str, chunks: List[str]) -> str:
        separator = "\n\n" if current_chunk else ""
        candidate = f"{current_chunk}{separator}{block}"

        if len(candidate) <= self.chunk_size:
            return candidate

        # Exceeds max_size: flush current chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Construct new chunk starting with overlap from flushed chunk
        if chunks and self.chunk_overlap > 0:
            prev = chunks[-1]
            overlap_text = self._get_sentence_overlap(prev, self.chunk_overlap)
            return f"{overlap_text}\n\n{block}" if overlap_text else block

        return block

    @staticmethod
    def _get_sentence_overlap(text: str, target_overlap: int) -> str:
        """Extracts complete sentences from end of text fitting target_overlap."""
        if len(text) <= target_overlap:
            return text
        
        sliced = text[-target_overlap:]
        # Find first sentence or space boundary to avoid slicing words mid-character
        first_period = sliced.find(". ")
        if first_period != -1 and first_period < len(sliced) - 10:
            return sliced[first_period + 2:].strip()
        
        first_space = sliced.find(" ")
        if first_space != -1:
            return sliced[first_space + 1:].strip()
            
        return sliced.strip()

    def _split_large_block(self, text: str) -> List[str]:
        """Splits an oversized paragraph on sentence boundaries."""
        sentences = re.split(r"(?<=\. )\s+|(?<=\n)\s*", text)
        parts = []
        curr = ""
        for s in sentences:
            if not s:
                continue
            if len(curr) + len(s) + 1 <= self.chunk_size:
                curr = f"{curr} {s}".strip()
            else:
                if curr:
                    parts.append(curr)
                curr = s
        if curr:
            parts.append(curr)
        return parts
