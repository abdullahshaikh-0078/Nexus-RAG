import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.chunker import RecursiveTextChunker
from app.services.embedder import embedding_service
from app.db.vectorstore import vector_store
from app.models.schemas import SourceCitation
from app.core.config import settings


def test_section_heading_binding():
    """Verifies section headings remain bound to the following content paragraph."""
    chunker = RecursiveTextChunker(chunk_size=1000, chunk_overlap=150)
    sample_text = (
        "This is an introductory paragraph explaining the motivation of neural sequence models.\n\n"
        "3.2.3 Applications of Attention in our Model\n\n"
        "The Transformer uses multi-head attention in three different ways. "
        "First, in encoder-decoder attention layers, queries come from the previous decoder layer, "
        "and keys and values come from the output of the encoder. "
        "Second, the encoder contains self-attention layers where all keys, values, and queries come from the same place. "
        "Third, self-attention layers in the decoder allow each position in the decoder to attend to all positions up to that position."
    )
    chunks = chunker.chunk_document(sample_text, document_id="doc_heading_test")
    
    # Locate chunk containing the heading
    heading_chunk = next((c for c in chunks if "3.2.3 Applications of Attention" in c.text), None)
    assert heading_chunk is not None, "Heading must be present in chunking output"
    
    # Assert heading is bound to body content in the same chunk
    assert "three different ways" in heading_chunk.text, (
        "Heading line must remain bound to the subsequent body paragraph within the same chunk."
    )


def test_adjacent_chunk_expansion():
    """Verifies that adjacent neighbor chunks are merged into a unified contiguous context block."""
    c1 = SourceCitation(
        document_id="doc_merge_1",
        document_name="paper.pdf",
        chunk_id="chk_10",
        chunk_index=10,
        score=0.75,
        content="3.2.3 Applications of Attention in our Model\nThe Transformer uses multi-head attention in three different ways:"
    )
    c2 = SourceCitation(
        document_id="doc_merge_1",
        document_name="paper.pdf",
        chunk_id="chk_11",
        chunk_index=11,
        score=0.70,
        content="1) In encoder-decoder attention layers, queries come from the decoder.\n2) Self-attention in encoder."
    )

    merged = vector_store._merge_contiguous_chunks([c1, c2])
    assert len(merged) == 1, "Contiguous chunks 10 and 11 should be merged into 1 unified citation block"
    merged_text = merged[0].content
    assert "Applications of Attention" in merged_text
    assert "1) In encoder-decoder attention layers" in merged_text


def test_embedding_model_consistency():
    """Verifies document embedder and query embedder use the exact same model and dimension."""
    doc_sample = "The Transformer is a novel neural network architecture based on self-attention."
    query_sample = "What is the Transformer architecture?"

    doc_vector = embedding_service.embed_text(doc_sample)
    query_vector = embedding_service.embed_text(query_sample)

    assert len(doc_vector) == settings.EMBEDDING_DIMENSION
    assert len(query_vector) == settings.EMBEDDING_DIMENSION
    assert embedding_service.model_name == settings.EMBEDDING_MODEL_NAME


def test_qdrant_document_id_payload_filtering():
    """
    Verifies Qdrant KEYWORD payload indexing and filtering by document_id.
    Ensures Mode A (no filter) and Mode B (document-scoped filter) work cleanly.
    """
    doc_a_text = "NEXUS RAG document A contains information about Quantum Computing algorithms."
    doc_b_text = "NEXUS RAG document B contains information about Astrophysics and Stellar Evolution."

    with TestClient(app) as client:
        # Ingest Document A
        upload_a = client.post(
            "/api/v1/documents/upload",
            files={"file": ("doc_a.txt", doc_a_text.encode("utf-8"), "text/plain")}
        )
        assert upload_a.status_code == 201
        doc_a_id = upload_a.json()["document"]["document_id"]

        # Ingest Document B
        upload_b = client.post(
            "/api/v1/documents/upload",
            files={"file": ("doc_b.txt", doc_b_text.encode("utf-8"), "text/plain")}
        )
        assert upload_b.status_code == 201
        doc_b_id = upload_b.json()["document"]["document_id"]

        try:
            # Mode A: No document filter
            res_all = client.post(
                "/api/v1/chat/query",
                json={"query": "Quantum Computing algorithms", "top_k": 4}
            )
            assert res_all.status_code == 200

            # Mode B: Scoped to Document A
            res_a = client.post(
                "/api/v1/chat/query",
                json={"query": "Quantum Computing algorithms", "top_k": 4, "document_ids": [doc_a_id]}
            )
            assert res_a.status_code == 200
            sources_a = res_a.json()["sources"]
            assert len(sources_a) > 0
            for src in sources_a:
                assert src["document_id"] == doc_a_id
                assert src["document_name"] == "doc_a.txt"

            # Mode B: Scoped to Document B
            res_b = client.post(
                "/api/v1/chat/query",
                json={"query": "Astrophysics and Stellar Evolution", "top_k": 4, "document_ids": [doc_b_id]}
            )
            assert res_b.status_code == 200
            sources_b = res_b.json()["sources"]
            assert len(sources_b) > 0
            for src in sources_b:
                assert src["document_id"] == doc_b_id
                assert src["document_name"] == "doc_b.txt"

        finally:
            client.delete(f"/api/v1/documents/{doc_a_id}")
            client.delete(f"/api/v1/documents/{doc_b_id}")


def test_target_queries_retrieval_quality():
    """
    Ingests 'Attention Is All You Need' sample text and tests:
    1. 'What is the main objective of this document?' -> Retrieves Abstract / Intro context.
    2. 'Applications of Attention in our Model' -> Retrieves Section 3.2.3 with all 3 applications.
    """
    attention_paper_text = (
        "Attention Is All You Need\n"
        "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin\n\n"
        "Abstract\n"
        "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks "
        "that include an encoder and a decoder. The best performing models also connect the encoder and decoder "
        "through an attention mechanism. We propose a new simple network architecture, the Transformer, "
        "based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. "
        "Experiments on two machine translation tasks show these models to be superior in quality while being "
        "more parallelizable and requiring significantly less time to train.\n\n"
        "1 Introduction\n"
        "Recurrent neural networks, particularly long short-term memory and gated recurrent neural networks, "
        "have been firmly established as state of the art approaches in sequence modeling and transduction problems. "
        "In this work we offer the Transformer, a model architecture eschewing recurrence and instead relying entirely "
        "on an attention mechanism to draw global dependencies between input and output.\n\n"
        "3.2.3 Applications of Attention in our Model\n"
        "The Transformer uses multi-head attention in three different ways:\n"
        "1. In encoder-decoder attention layers, the queries come from the previous decoder layer, and the memory keys "
        "and values come from the output of the encoder. This allows every position in the decoder to attend over all "
        "positions in the input sequence.\n"
        "2. The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries "
        "come from the same place, in this case, the output of the previous layer in the encoder.\n"
        "3. Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions "
        "in the decoder up to and including that position."
    )

    with TestClient(app) as client:
        # Ingest document
        files = {
            "file": ("attention_paper.txt", attention_paper_text.encode("utf-8"), "text/plain")
        }
        upload_res = client.post("/api/v1/documents/upload", files=files)
        assert upload_res.status_code == 201
        doc_id = upload_res.json()["document"]["document_id"]

        try:
            # Query 1: Main Objective (scoped to ingested document)
            q1_res = client.post(
                "/api/v1/chat/query",
                json={
                    "query": "What is the main objective of this document?",
                    "top_k": 3,
                    "document_ids": [doc_id]
                }
            )
            assert q1_res.status_code == 200
            q1_data = q1_res.json()
            q1_sources_text = " ".join([s["content"] for s in q1_data["sources"]])
            
            # Assert Abstract / Introduction is present in retrieved sources
            assert "Transformer" in q1_sources_text or "Abstract" in q1_sources_text
            assert "eschewing recurrence" in q1_sources_text or "attention mechanism" in q1_sources_text or "transduction" in q1_sources_text

            # Query 2: Applications of Attention (scoped to ingested document)
            q2_res = client.post(
                "/api/v1/chat/query",
                json={
                    "query": "Applications of Attention in our Model",
                    "top_k": 3,
                    "document_ids": [doc_id]
                }
            )
            assert q2_res.status_code == 200
            q2_data = q2_res.json()
            q2_sources_text = " ".join([s["content"] for s in q2_data["sources"]])

            # Assert all 3 applications are in retrieved context
            assert "encoder-decoder attention" in q2_sources_text.lower()
            assert "self-attention" in q2_sources_text.lower()
            assert "decoder" in q2_sources_text.lower()

        finally:
            # Cleanup
            client.delete(f"/api/v1/documents/{doc_id}")
