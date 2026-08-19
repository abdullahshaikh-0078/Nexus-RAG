import os
import pytest
from app.services.bm25_search import BM25IndexService, bm25_service
from app.models.schemas import DocumentChunk


def test_bm25_indexing_and_search():
    service = BM25IndexService()

    chunks = [
        DocumentChunk(
            document_id="doc_bm25_1",
            chunk_id="chk_bm25_1",
            chunk_index=0,
            text="Attention is all you need for transformer networks and multi-head attention.",
            start_char=0,
            end_char=75,
        ),
        DocumentChunk(
            document_id="doc_bm25_1",
            chunk_id="chk_bm25_2",
            chunk_index=1,
            text="Convolutional neural networks process image data with 2D spatial kernels.",
            start_char=76,
            end_char=148,
        ),
    ]

    service.index_chunks(chunks, filename="test_doc.pdf")

    hits = service.search("multi-head attention", top_k=5, document_ids=["doc_bm25_1"])
    assert len(hits) > 0
    assert hits[0].document_id == "doc_bm25_1"
    assert hits[0].score > 0.0

    hits_vector = service.search("transformer networks", top_k=5, document_ids=["doc_bm25_1"])
    assert len(hits_vector) > 0
    assert hits_vector[0].document_id == "doc_bm25_1"


def test_bm25_document_deletion():
    service = BM25IndexService()
    c1 = DocumentChunk(
        document_id="doc_del_1",
        chunk_id="chk_del_1",
        chunk_index=0,
        text="UniqueQuantumPhysicsTerm wave-particle duality physics.",
        start_char=0,
        end_char=55,
    )
    service.index_chunks([c1], filename="quantum.txt")
    assert len(service.search("UniqueQuantumPhysicsTerm", top_k=5)) == 1

    service.delete_document("doc_del_1")
    assert len(service.search("UniqueQuantumPhysicsTerm", top_k=5)) == 0


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

    hits_alpha = service.search("Transformer", top_k=5, document_ids=["doc_alpha"])
    assert len(hits_alpha) == 1
    assert hits_alpha[0].document_id == "doc_alpha"

    hits_beta = service.search("Transformer", top_k=5, document_ids=["doc_beta"])
    assert len(hits_beta) == 1
    assert hits_beta[0].document_id == "doc_beta"


def test_bm25_entity_reference_retrieval():
    service = BM25IndexService()
    c1 = DocumentChunk(
        document_id="doc_ref_1",
        chunk_id="chk_ref_1",
        chunk_index=0,
        text="References: Mitchell P. Marcus, Beatrice Santorini, and Mary Ann Marcinkiewicz.",
        start_char=0,
        end_char=80,
    )
    service.index_chunks([c1], filename="references.pdf")

    hits = service.search("Mitchell P. Marcus", top_k=5, document_ids=["doc_ref_1"])
    assert len(hits) > 0
    assert hits[0].document_id == "doc_ref_1"
    assert "Mitchell" in hits[0].content


def test_api_retrieval_mode_selection():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    response = client.post(
        "/api/v1/chat/query",
        json={"query": "test query", "retrieval_mode": "bm25"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["retrieval_mode"] == "bm25"
