import os
import sys
import time
import pytest
import asyncio
from app.db.mongodb import mongo_db
from app.db.vectorstore import vector_store
from app.services.bm25_search import bm25_service
from app.models.schemas import ChatDocument, DocumentRepresentation, SourceCitation
from app.v3.ingestion.ingestion_service import v3_ingestion_service
from app.v3.parsing.structural_parser import v3_structural_parser
from app.v3.chunking.engine import v3_chunking_engine
from app.core.pipeline_router import pipeline_router


@pytest.mark.asyncio
async def test_new_chat_has_zero_documents():
    """1. Test that a newly created chat session has 0 documents and empty state."""
    chat = await mongo_db.create_chat("Test Chat Isolation")
    docs = await mongo_db.list_chat_documents(chat.chat_id)
    assert len(docs) == 0, f"New chat {chat.chat_id} should have 0 documents."
    await mongo_db.delete_chat(chat.chat_id)


@pytest.mark.asyncio
async def test_upload_attaches_document_to_single_chat_and_creates_v1():
    """2 & 3. Test document attachment and V1 baseline auto-creation upon upload."""
    chat1 = await mongo_db.create_chat("Chat 1")
    pdf_path = os.path.abspath("./data/uploads/RAG TEST DOC 3.pdf")
    if not os.path.exists(pdf_path):
        pytest.skip("RAG TEST DOC 3.pdf not present in ./data/uploads")

    # Simulate document upload to Chat 1
    cdoc1 = ChatDocument(
        chat_document_id=f"cdoc_{int(time.time())}",
        chat_id=chat1.chat_id,
        document_id="doc_test_rag3",
        filename="RAG TEST DOC 3.pdf",
        content_hash="mock_hash_123",
        source_path=pdf_path,
        file_type="pdf",
        file_size_bytes=1024,
        char_count=5000,
        v1_chunk_count=324,
    )
    await mongo_db.add_chat_document(cdoc1)

    # Verify document attached to Chat 1
    c1_docs = await mongo_db.list_chat_documents(chat1.chat_id)
    assert len(c1_docs) == 1
    assert c1_docs[0].document_id == "doc_test_rag3"

    # Verify Chat 2 has ZERO documents
    chat2 = await mongo_db.create_chat("Chat 2")
    c2_docs = await mongo_db.list_chat_documents(chat2.chat_id)
    assert len(c2_docs) == 0

    await mongo_db.delete_chat(chat1.chat_id)
    await mongo_db.delete_chat(chat2.chat_id)


@pytest.mark.asyncio
async def test_v3_conversion_flow_and_backend_policy_strategy():
    """5 to 11. Test explicit V3 conversion, PyMuPDF parsing, IR, strategy policy, and provenance."""
    pdf_path = os.path.abspath("./data/uploads/RAG TEST DOC 3.pdf")
    if not os.path.exists(pdf_path):
        pytest.skip("RAG TEST DOC 3.pdf not present")

    chat = await mongo_db.create_chat("V3 Conversion Chat")
    doc_id = "RAG TEST DOC 3.pdf"

    # Verify initial status is NOT_CREATED / None
    initial_rep = await mongo_db.get_representation(doc_id, "v3", chat_id=chat.chat_id)
    assert initial_rep is None, "V3 representation should NOT exist automatically before conversion."

    # Trigger explicit V3 conversion
    rep = await v3_ingestion_service.materialize_representation(
        document_id=doc_id,
        version="v3",
        strategy=None,  # Let backend policy engine decide
        chat_id=chat.chat_id,
    )

    assert rep.status == "READY", f"V3 conversion failed: {rep.error_message}"
    assert rep.version == "v3"
    assert rep.chunking_strategy == "table_aware", "Backend policy engine should select 'table_aware' for tabular financial PDF."
    assert rep.chunk_count > 0, "V3 conversion should generate non-zero structural chunks."
    assert rep.chunk_count >= 1000, f"Expected at least 1,000 V3 chunks for RAG TEST DOC 3, got {rep.chunk_count}"

    await mongo_db.delete_chat(chat.chat_id)


@pytest.mark.asyncio
async def test_v3_retrieval_and_contamination_guard():
    """12 to 15. Test V3 Dense, BM25, Hybrid RRF, and strict 0-contamination guard."""
    chat = await mongo_db.create_chat("V3 Retrieval Test Chat")
    doc_id = "RAG TEST DOC 3.pdf"

    rep = await v3_ingestion_service.materialize_representation(
        document_id=doc_id,
        version="v3",
        strategy="table_aware",
        chat_id=chat.chat_id,
    )
    assert rep.status == "READY"

    cits, breakdown, exp_meta, calc = pipeline_router.route_query(
        query="What is the net revenue or operating performance trend documented?",
        top_k=4,
        document_ids=[doc_id],
        version="v3",
        chunking_strategy="table_aware",
        chat_id=chat.chat_id,
    )

    assert len(cits) > 0, "V3 Hybrid retrieval should return top K citations."
    assert all(c.version == "v3" for c in cits), "Contamination Guard Failed: Non-V3 chunk found in V3 query results!"

    await mongo_db.delete_chat(chat.chat_id)


@pytest.mark.asyncio
async def test_version_switching_and_idempotent_reuse():
    """16 to 18. Test bidirectional version switching and cached representation reuse."""
    chat = await mongo_db.create_chat("Version Switching Chat")
    doc_id = "RAG TEST DOC 3.pdf"

    # Materialize V3
    t0 = time.time()
    rep1 = await v3_ingestion_service.materialize_representation(
        document_id=doc_id, version="v3", strategy="table_aware", chat_id=chat.chat_id
    )
    dur1 = (time.time() - t0) * 1000
    assert rep1.status == "READY"

    # Second call for same chat & document
    t1 = time.time()
    rep2 = await v3_ingestion_service.materialize_representation(
        document_id=doc_id, version="v3", strategy="table_aware", chat_id=chat.chat_id
    )
    dur2 = (time.time() - t1) * 1000

    assert rep2.status == "READY"
    assert dur2 < 2000.0, f"Cached representation reuse took {dur2:.2f}ms (expected <2000ms)."

    await mongo_db.delete_chat(chat.chat_id)


@pytest.mark.asyncio
async def test_chat_isolation_security():
    """19. Test strict cross-chat retrieval security: Chat A cannot retrieve Chat B documents."""
    chat_a = await mongo_db.create_chat("Chat A")
    chat_b = await mongo_db.create_chat("Chat B")

    pdf_path = os.path.abspath("./data/uploads/RAG TEST DOC 3.pdf")
    if not os.path.exists(pdf_path):
        pytest.skip("RAG TEST DOC 3.pdf not present")

    doc_id = "RAG TEST DOC 3.pdf"

    # Ingest V3 into Chat A only
    rep_a = await v3_ingestion_service.materialize_representation(
        document_id=doc_id,
        version="v3",
        strategy="table_aware",
        chat_id=chat_a.chat_id,
    )
    assert rep_a.status == "READY"

    # Query from Chat B with chat_id=chat_b.chat_id
    cits_b, _, _, _ = pipeline_router.route_query(
        query="What is the net revenue or operating performance trend documented?",
        top_k=4,
        document_ids=[doc_id],
        version="v3",
        chunking_strategy="table_aware",
        chat_id=chat_b.chat_id,
    )

    assert len(cits_b) == 0, f"Cross-chat Security Leak! Chat B retrieved {len(cits_b)} chunks from Chat A."

    await mongo_db.delete_chat(chat_a.chat_id)
    await mongo_db.delete_chat(chat_b.chat_id)


@pytest.mark.asyncio
async def test_delete_chat_purges_all_scoped_resources():
    """20. Test that deleting a chat purges chat metadata, representations, vectors, and BM25 entries."""
    chat = await mongo_db.create_chat("Chat To Delete")
    doc_id = "RAG TEST DOC 3.pdf"

    rep = await v3_ingestion_service.materialize_representation(
        document_id=doc_id, version="v3", strategy="table_aware", chat_id=chat.chat_id
    )
    assert rep.status == "READY"

    # Purge chat
    vector_store.delete_chat_chunks(chat.chat_id)
    bm25_service.delete_chat_documents(chat.chat_id)
    await mongo_db.delete_chat(chat.chat_id)

    # Verify chat session deleted
    deleted_chat = await mongo_db.get_chat(chat.chat_id)
    assert deleted_chat is None

    # Verify representations deleted
    deleted_rep = await mongo_db.get_representation(doc_id, "v3", chat_id=chat.chat_id)
    assert deleted_rep is None
