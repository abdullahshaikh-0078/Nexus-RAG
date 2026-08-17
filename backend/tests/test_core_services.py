import pytest
from app.services.chunker import RecursiveTextChunker
from app.services.document_parser import UnifiedDocumentParser


def test_chunker_basic():
    chunker = RecursiveTextChunker(chunk_size=1000, chunk_overlap=150)
    sample_text = (
        "NEXUS RAG is a production-oriented Retrieval Augmented Generation platform. "
        "It supports high throughput document ingestion, semantic search, and context synthesis. "
        "This unit test verifies that long text blocks are recursively split into smaller chunks properly."
    )
    chunks = chunker.chunk_document(sample_text, document_id="doc_test_1")
    assert len(chunks) > 0
    assert chunks[0].document_id == "doc_test_1"
    assert len(chunks[0].text) <= 1000


def test_document_parser_text():
    sample_bytes = b"Hello world! This is a test file for NEXUS RAG ingestion."
    text, file_type = UnifiedDocumentParser.extract_text(sample_bytes, "sample.txt")
    assert file_type == "txt"
    assert "NEXUS RAG" in text
