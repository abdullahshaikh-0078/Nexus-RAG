import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.chunker import RecursiveTextChunker
from app.services.bm25_search import BM25IndexService, bm25_service
from app.models.schemas import DocumentChunk


def test_bm25_indexing_and_search():
    service = BM25IndexService()
    chunker = RecursiveTextChunker(chunk_size=500, chunk_overlap=50)

    doc1_text = "NEXUS RAG uses Qdrant vector database for dense vector retrieval."
    doc2_text = "BM25 is a term frequency-inverse document frequency ranking function for lexical search."

    chunks1 = chunker.chunk_document(doc1_text, document_id="doc_bm25_1")
    chunks2 = chunker.chunk_document(doc2_text, document_id="doc_bm25_2")

    service.index_chunks(chunks1, filename="doc1.txt")
    service.index_chunks(chunks2, filename="doc2.txt")

    # Search exact keyword "lexical"
    hits = service.search(query="lexical search", top_k=2)
    assert len(hits) > 0
    assert hits[0].document_id == "doc_bm25_2"
    assert "BM25" in hits[0].content or "lexical" in hits[0].content

    # Search vector keyword "Qdrant"
    hits_vector = service.search(query="Qdrant vector database", top_k=2)
    assert len(hits_vector) > 0
    assert hits_vector[0].document_id == "doc_bm25_1"


def test_bm25_document_deletion():
    service = BM25IndexService()
    c1 = DocumentChunk(
        document_id="doc_del_1",
        chunk_id="chk_del_1",
        chunk_index=0,
        text="Quantum Mechanics wave-particle duality physics.",
        start_char=0,
        end_char=45,
    )
    service.index_chunks([c1], filename="quantum.txt")
    assert len(service.search("Quantum Mechanics", top_k=5)) == 1

    service.delete_document("doc_del_1")
    assert len(service.search("Quantum Mechanics", top_k=5)) == 0


def test_bm25_scoped_search():
    service = BM25IndexService()
    c1 = DocumentChunk(
        document_id="doc_alpha",
        chunk_id="chk_alpha_1",
        chunk_index=0,
        text="Transformer multi-head attention mechanism",
        start_char=0,
        end_char=40,
    )
    c2 = DocumentChunk(
        document_id="doc_beta",
        chunk_id="chk_beta_1",
        chunk_index=0,
        text="Transformer positional encoding positional embedding",
        start_char=0,
        end_char=50,
    )
    service.index_chunks([c1], filename="alpha.txt")
    service.index_chunks([c2], filename="beta.txt")

    # Scope to doc_alpha only
    hits_alpha = service.search("Transformer", top_k=5, document_ids=["doc_alpha"])
    assert len(hits_alpha) == 1
    assert hits_alpha[0].document_id == "doc_alpha"


def test_bm25_entity_reference_retrieval():
    """Regression test proving Mitchell P. Marcus entity reference is retrievable by BM25."""
    service = BM25IndexService()
    chunker = RecursiveTextChunker(chunk_size=1000, chunk_overlap=150)

    paper_references_text = (
        "References\n"
        "[1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit. Attention is all you need. NIPS, 2017.\n"
        "[2] Mitchell P. Marcus, Mary Ann Marcinkiewicz, Beatrice Santorini. "
        "Building a large annotated corpus of English: The Penn Treebank. Computational Linguistics, 19(2):313-330, 1993."
    )

    chunks = chunker.chunk_document(paper_references_text, document_id="doc_mitchell_marcus_test")
    service.index_chunks(chunks, filename="attention_paper_references.pdf")

    query = "Who is Mitchell P. Marcus?"
    hits = service.search(query=query, top_k=5)

    assert len(hits) > 0
    assert hits[0].document_id == "doc_mitchell_marcus_test"
    assert "Mitchell P. Marcus" in hits[0].content


def test_api_retrieval_mode_selection():
    with TestClient(app) as client:
        # Ingest document
        doc_text = "Section 3.2.3 Applications of Multi-Head Attention in the Transformer model."
        upload_res = client.post(
            "/api/v1/documents/upload",
            files={"file": ("spec.txt", doc_text.encode("utf-8"), "text/plain")}
        )
        assert upload_res.status_code == 201
        doc_id = upload_res.json()["document"]["document_id"]

        try:
            # Query with retrieval_mode="dense" (default)
            res_dense = client.post(
                "/api/v1/chat/query",
                json={"query": "Applications of Multi-Head Attention", "top_k": 3, "retrieval_mode": "dense"}
            )
            assert res_dense.status_code == 200
            assert res_dense.json()["retrieval_mode"] == "dense"

            # Query with retrieval_mode="bm25"
            res_bm25 = client.post(
                "/api/v1/chat/query",
                json={"query": "Multi-Head Attention", "top_k": 3, "retrieval_mode": "bm25"}
            )
            assert res_bm25.status_code == 200
            assert res_bm25.json()["retrieval_mode"] == "bm25"
            assert len(res_bm25.json()["sources"]) > 0

        finally:
            client.delete(f"/api/v1/documents/{doc_id}")
