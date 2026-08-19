from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# --- Document Schemas ---

class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    token_count: Optional[int] = None
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    char_count: int
    chunk_count: int
    upload_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "processed"  # processing, processed, failed
    error_message: Optional[str] = None


class DocumentUploadResponse(BaseModel):
    success: bool
    message: str
    document: Optional[DocumentMetadata] = None


class DocumentListResponse(BaseModel):
    total: int
    documents: List[DocumentMetadata]


# --- Search & Chat Schemas ---

class SourceCitation(BaseModel):
    document_id: str
    document_name: str
    chunk_id: str
    chunk_index: int
    score: float
    content: str
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    rrf_score: Optional[float] = None


class ChatQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's query string.")
    top_k: int = Field(default=4, ge=1, le=20, description="Number of context chunks to retrieve.")
    document_ids: Optional[List[str]] = Field(default=None, description="Filter search to specific documents.")
    retrieval_mode: Optional[str] = Field(
        default="hybrid", description="Retrieval mode: 'dense', 'bm25', or 'hybrid' (Dense + BM25 + RRF)."
    )


class ChatQueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceCitation]
    retrieval_mode: str = "hybrid"
    llm_provider: str
    model_name: str
    processing_time_seconds: float
    evaluation_id: Optional[str] = None
    latency_breakdown: Optional[Dict[str, float]] = None


# --- System Status Schemas ---

class ServiceHealth(BaseModel):
    status: str  # healthy, degraded, unhealthy
    details: Optional[str] = None


class SystemHealthResponse(BaseModel):
    status: str
    app_name: str
    environment: str
    services: Dict[str, ServiceHealth]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
