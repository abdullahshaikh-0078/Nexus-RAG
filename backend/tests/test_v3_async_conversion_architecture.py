import os
import sys
import time
import pytest
import asyncio
from datetime import datetime, timezone, timedelta

from app.db.mongodb import mongo_db
from app.db.vectorstore import vector_store
from app.services.bm25_search import bm25_service
from app.models.schemas import ChatDocument, DocumentRepresentation, ChatSession
from app.v3.ingestion.ingestion_service import v3_ingestion_service
from app.core.pipeline_router import pipeline_router
from app.v3.retrieval.v3_retriever import v3_retriever
from app.api.endpoints.chat import convert_document_to_v3


@pytest.mark.asyncio
async def test_missing_asyncio_import_regression():
    """1. Verifies that chat.py endpoint module imports asyncio and has asyncio available."""
    import app.api.endpoints.chat as chat_module
    assert hasattr(chat_module, "asyncio"), "chat.py module is missing asyncio import"
    assert chat_module.asyncio is not None


@pytest.mark.asyncio
async def test_hash_named_pdf_source_lookup():
    """2. Verifies PDF source lookup for chat-uploaded hash-named files ({content_hash}.pdf)."""
    upload_dir = os.path.abspath("./data/uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    # Create temporary dummy PDF file named by hash
    mock_hash = "11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff"
    hash_pdf_path = os.path.join(upload_dir, f"{mock_hash}.pdf")

    # Copy real PDF header/content if exists, else write simple valid PDF bytes
    rag3_path = os.path.abspath("./data/uploads/RAG TEST DOC 3.pdf")
    if os.path.exists(rag3_path):
        with open(rag3_path, "rb") as src, open(hash_pdf_path, "wb") as dst:
            dst.write(src.read())
    else:
        with open(hash_pdf_path, "wb") as dst:
            dst.write(b"%PDF-1.4 mock content for lookup test\n%%EOF\n")

    try:
        # Create chat session & ChatDocument referencing content_hash
        chat = await mongo_db.create_chat("Hash Lookup Chat")
        cdoc = ChatDocument(
            chat_document_id="cdoc_hash_test_1",
            chat_id=chat.chat_id,
            document_id="doc_hash_lookup_test",
            filename="Test_Report.pdf",
            content_hash=mock_hash,
            source_path=hash_pdf_path,
            file_type="pdf",
            file_size_bytes=1024,
            v1_chunk_count=10,
        )
        await mongo_db.add_chat_document(cdoc)

        # Lookup using find_pdf_path_async
        resolved_path = await v3_ingestion_service.find_pdf_path_async("doc_hash_lookup_test", chat_id=chat.chat_id)
        assert resolved_path is not None, "Failed to resolve hash-named PDF source path"
        assert os.path.exists(resolved_path)
        assert mock_hash in resolved_path or "RAG TEST DOC 3" in resolved_path

        # Lookup directly by content_hash
        resolved_by_hash = await v3_ingestion_service.find_pdf_path_async(mock_hash)
        assert resolved_by_hash is not None, "Failed to resolve PDF directly by content_hash"

        await mongo_db.delete_chat(chat.chat_id)
    finally:
        if os.path.exists(hash_pdf_path):
            try:
                os.remove(hash_pdf_path)
            except Exception:
                pass


@pytest.mark.asyncio
async def test_duplicate_concurrent_conversion_protection():
    """3. Verifies non-stale PROCESSING representation prevents duplicate job launch."""
    pdf_path = os.path.abspath("./data/uploads/RAG TEST DOC 3.pdf")
    if not os.path.exists(pdf_path):
        pytest.skip("RAG TEST DOC 3.pdf not found in ./data/uploads")

    chat = await mongo_db.create_chat("Concurrent Test Chat")
    doc_id = "doc_concurrent_test"

    cdoc = ChatDocument(
        chat_document_id="cdoc_concurrent_1",
        chat_id=chat.chat_id,
        document_id=doc_id,
        filename="RAG TEST DOC 3.pdf",
        content_hash="hash_concurrent_123",
        source_path=pdf_path,
        file_type="pdf",
        file_size_bytes=1000,
        v1_chunk_count=10,
    )
    await mongo_db.add_chat_document(cdoc)

    active_processing_rep = DocumentRepresentation(
        representation_id=f"{chat.chat_id}_{doc_id}_v3_table_aware",
        chat_id=chat.chat_id,
        document_id=doc_id,
        document_name="RAG TEST DOC 3.pdf",
        content_hash="hash_concurrent_123",
        version="v3",
        chunking_strategy="table_aware",
        status="PROCESSING",
        updated_at=datetime.now(timezone.utc),
    )
    await mongo_db.save_representation(active_processing_rep)

    # Call materialize_representation without force_reprocess
    res_rep = await v3_ingestion_service.materialize_representation(
        document_id=doc_id,
        version="v3",
        strategy="table_aware",
        chat_id=chat.chat_id,
        force_reprocess=False,
    )
    assert res_rep.status == "PROCESSING", f"Expected status='PROCESSING', got '{res_rep.status}'"
    assert res_rep.representation_id == active_processing_rep.representation_id

    await mongo_db.delete_chat(chat.chat_id)


@pytest.mark.asyncio
async def test_stale_processing_recovery():
    """4. Verifies stale PROCESSING representation (>300s old) triggers recovery."""
    chat = await mongo_db.create_chat("Stale Test Chat")
    doc_id = "doc_stale_test"

    # Create stale representation with updated_at = 10 minutes ago
    stale_time = datetime.now(timezone.utc) - timedelta(seconds=600)
    stale_rep = DocumentRepresentation(
        representation_id=f"{chat.chat_id}_{doc_id}_v3_table_aware",
        chat_id=chat.chat_id,
        document_id=doc_id,
        document_name="Stale.pdf",
        content_hash="hash_stale_123",
        version="v3",
        chunking_strategy="table_aware",
        status="PROCESSING",
        updated_at=stale_time,
    )
    await mongo_db.save_representation(stale_rep)

    # Call materialize_representation. Since PDF is missing, recovery will proceed to find_pdf_path
    # and safely return FAILED with "Original source PDF not found" instead of staying stuck in PROCESSING.
    res_rep = await v3_ingestion_service.materialize_representation(
        document_id=doc_id,
        version="v3",
        strategy="table_aware",
        chat_id=chat.chat_id,
        force_reprocess=False,
    )
    assert res_rep.status == "FAILED", f"Expected stale recovery to reach FAILED for missing PDF, got '{res_rep.status}'"
    assert "Original source PDF not found" in res_rep.error_message

    await mongo_db.delete_chat(chat.chat_id)


@pytest.mark.asyncio
async def test_failed_background_conversion_handling():
    """5. Verifies missing source PDF or conversion error produces FAILED status with error_message."""
    chat = await mongo_db.create_chat("Failed Conversion Chat")
    doc_id = "doc_non_existent_pdf"

    cdoc = ChatDocument(
        chat_document_id="cdoc_failed_test",
        chat_id=chat.chat_id,
        document_id=doc_id,
        filename="Missing.pdf",
        content_hash="missing_hash_999",
        source_path="./data/uploads/Missing.pdf",
        file_type="pdf",
        file_size_bytes=100,
        v1_chunk_count=5,
    )
    await mongo_db.add_chat_document(cdoc)

    # Call convert_document_to_v3 endpoint
    response = await convert_document_to_v3(chat.chat_id, doc_id)
    assert response.representation.status == "PROCESSING"

    # Wait 0.5s for background worker task to attempt materialization
    await asyncio.sleep(0.5)

    # Fetch updated representation from MongoDB
    updated_rep = await mongo_db.get_representation(doc_id, "v3", chat_id=chat.chat_id)
    assert updated_rep is not None, "Failed representation record should exist in MongoDB"
    assert updated_rep.status == "FAILED", f"Expected FAILED status, got '{updated_rep.status}'"
    assert updated_rep.error_message is not None

    await mongo_db.delete_chat(chat.chat_id)


@pytest.mark.asyncio
async def test_successful_v3_background_conversion_and_retrieval():
    """6 & 7. End-to-end test for V3 async background conversion and hybrid retrieval integrity."""
    pdf_path = os.path.abspath("./data/uploads/RAG TEST DOC 3.pdf")
    if not os.path.exists(pdf_path):
        pytest.skip("RAG TEST DOC 3.pdf not found in ./data/uploads")

    chat = await mongo_db.create_chat("V3 End-to-End Test Chat")
    doc_id = "doc_rag3_e2e_test"

    cdoc = ChatDocument(
        chat_document_id="cdoc_rag3_e2e",
        chat_id=chat.chat_id,
        document_id=doc_id,
        filename="RAG TEST DOC 3.pdf",
        content_hash="rag3_test_hash_e2e",
        source_path=pdf_path,
        file_type="pdf",
        file_size_bytes=os.path.getsize(pdf_path),
        v1_chunk_count=324,
    )
    await mongo_db.add_chat_document(cdoc)

    # Trigger V3 conversion via endpoint
    resp = await convert_document_to_v3(chat.chat_id, doc_id)
    assert resp.representation.status in ["PROCESSING", "READY"]

    # Poll until READY or FAILED (timeout 240s for CPU embedding generation of 1,376 chunks)
    start_time = time.time()
    final_rep = None
    while time.time() - start_time < 240:
        rep = await mongo_db.get_representation(doc_id, "v3", chat_id=chat.chat_id)
        if rep and rep.status in ["READY", "FAILED"]:
            final_rep = rep
            break
        await asyncio.sleep(1.0)

    assert final_rep is not None, "V3 conversion timed out in background test"
    assert final_rep.status == "READY", f"V3 conversion failed with error: {final_rep.error_message}"
    assert final_rep.chunk_count > 0

    # Verify retrieval integrity via pipeline_router
    raw_citations, latency, q_meta, calc = pipeline_router.route_query(
        query="net income financial performance",
        top_k=3,
        document_ids=[doc_id],
        version="v3",
        chat_id=chat.chat_id,
    )
    assert len(raw_citations) > 0, "V3 retrieval returned 0 citations for RAG TEST DOC 3"
    for c in raw_citations:
        assert c.version == "v3", f"Contaminated chunk found: version={c.version}"

    await mongo_db.delete_chat(chat.chat_id)
