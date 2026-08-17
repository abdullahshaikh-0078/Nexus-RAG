import pytest
from fastapi.testclient import TestClient
from app.main import app


def test_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["app"] == "NEXUS RAG"


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "services" in data
        assert "mongodb" in data["services"]
        assert "qdrant" in data["services"]
        assert "embeddings" in data["services"]
        assert "llm" in data["services"]


def test_document_ingestion_and_chat_query():
    with TestClient(app) as client:
        # 1. Upload sample text document
        file_content = (
            "Project NEXUS RAG Version 1.0 specifications.\n"
            "NEXUS RAG uses Qdrant for vector storage and Hugging Face sentence transformers for embeddings.\n"
            "The architecture features FastAPI on the backend and Next.js 15 on the frontend."
        )
        files = {
            "file": ("nexus_specs.txt", file_content.encode("utf-8"), "text/plain")
        }
        upload_res = client.post("/api/v1/documents/upload", files=files)
        assert upload_res.status_code == 201
        upload_data = upload_res.json()
        assert upload_data["success"] is True
        doc_id = upload_data["document"]["document_id"]
        assert doc_id.startswith("doc_")

        # 2. List documents
        list_res = client.get("/api/v1/documents/")
        assert list_res.status_code == 200
        list_data = list_res.json()
        assert list_data["total"] >= 1

        # 3. Query RAG
        query_payload = {
            "query": "What vector database does NEXUS RAG use?",
            "top_k": 2
        }
        chat_res = client.post("/api/v1/chat/query", json=query_payload)
        assert chat_res.status_code == 200
        chat_data = chat_res.json()
        assert "answer" in chat_data
        assert len(chat_data["sources"]) > 0
        assert chat_data["sources"][0]["document_name"] == "nexus_specs.txt"

        # 4. Clean up document
        del_res = client.delete(f"/api/v1/documents/{doc_id}")
        assert del_res.status_code == 200
