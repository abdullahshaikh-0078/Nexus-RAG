import logging
from typing import List, Tuple
from app.v3.schemas.chunk_schema import V3Chunk

logger = logging.getLogger(__name__)


class V3ChunkValidator:
    """
    Validation engine for V3 chunks ensuring data integrity, complete provenance,
    table header preservation, and valid parent-child hierarchy references.
    """

    @staticmethod
    def validate_chunks(chunks: List[V3Chunk]) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        valid_chunk_ids = {c.chunk_id for c in chunks}

        if not chunks:
            errors.append("Chunk list is empty.")
            return False, errors

        for idx, chunk in enumerate(chunks, 1):
            # 1. Provenance Checks
            if not chunk.chunk_id:
                errors.append(f"Chunk #{idx}: missing chunk_id.")
            if not chunk.document_id:
                errors.append(f"Chunk #{idx} ({chunk.chunk_id}): missing document_id.")
            if not chunk.document_name:
                errors.append(f"Chunk #{idx} ({chunk.chunk_id}): missing document_name.")
            if chunk.page_number < 1:
                errors.append(f"Chunk #{idx} ({chunk.chunk_id}): invalid page_number {chunk.page_number}.")

            # 2. Content Checks
            if not chunk.content or not chunk.content.strip():
                errors.append(f"Chunk #{idx} ({chunk.chunk_id}): content is empty.")

            # 3. Table Header Preservation Checks
            if chunk.chunk_type == "table":
                # Check if table content contains numerical values but no header indicators
                has_nums = any(c.isdigit() for c in chunk.content)
                has_pipe = "|" in chunk.content
                if has_nums and not has_pipe:
                    errors.append(f"Chunk #{idx} ({chunk.chunk_id}): table chunk contains numerical values without table headers/pipes.")

            # 4. Hierarchy Link Checks
            if chunk.parent_chunk_id and chunk.parent_chunk_id not in valid_chunk_ids:
                errors.append(f"Chunk #{idx} ({chunk.chunk_id}): parent_chunk_id '{chunk.parent_chunk_id}' not found in chunk list.")

            for child_id in chunk.child_chunk_ids:
                if child_id not in valid_chunk_ids:
                    errors.append(f"Chunk #{idx} ({chunk.chunk_id}): child_chunk_id '{child_id}' not found in chunk list.")

        is_valid = len(errors) == 0
        if not is_valid:
            logger.warning(f"V3ChunkValidator found {len(errors)} validation issues.")
        return is_valid, errors
