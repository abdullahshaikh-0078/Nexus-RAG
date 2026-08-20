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


# --- Chat & Document Management Schemas ---

class ChatSession(BaseModel):
    chat_id: str
    title: str = "New Chat"
    active_document_id: Optional[str] = None
    active_version: str = "v1"  # Default pipeline version is V1 upon document upload
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatDocument(BaseModel):
    chat_document_id: str  # e.g. cdoc_12345
    chat_id: str
    document_id: str
    filename: str
    content_hash: str  # SHA-256 hash of original PDF source binary
    source_path: str
    file_type: str = "pdf"
    file_size_bytes: int = 0
    char_count: int = 0
    v1_chunk_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DocumentRepresentation(BaseModel):
    representation_id: str  # e.g. chat123_doc123_v3_table_aware
    chat_id: Optional[str] = None
    document_id: str
    document_name: str
    content_hash: str  # SHA-256 hash of original source document
    version: str  # v1, v2.1, v2.2, v3
    chunking_strategy: Optional[str] = None  # table_aware, section_aware, semantic, etc.
    status: str = "NOT_CREATED"  # NOT_CREATED, PROCESSING, READY, FAILED
    chunk_count: int = 0
    parser_version: str = "PyMuPDF_TableFinder_V3"
    chunker_version: str = "V3ChunkingEngine"
    index_status: str = "NOT_INDEXED"  # INDEXED, NOT_INDEXED, FAILED
    qdrant_collection: str = "nexus_chunks"
    bm25_index: str = "nexus_bm25"
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatCreateRequest(BaseModel):
    title: Optional[str] = Field(default=None, description="Optional chat title.")


class ChatListResponse(BaseModel):
    total: int
    chats: List[ChatSession]


class ChatDetailResponse(BaseModel):
    chat: ChatSession
    documents: List[ChatDocument]
    messages: List[Dict[str, Any]] = Field(default_factory=list)


class RepresentationListResponse(BaseModel):
    document_id: str
    document_name: str
    representations: List[DocumentRepresentation]


class MaterializeRepresentationRequest(BaseModel):
    version: str = Field(default="v3", description="Retrieval version ('v1', 'v2.1', 'v2.2', 'v3').")
    chunking_strategy: Optional[str] = Field(
        default=None, description="V3 Chunking strategy. If omitted, backend policy engine selects optimal strategy."
    )


class MaterializeRepresentationResponse(BaseModel):
    success: bool
    message: str
    representation: DocumentRepresentation


# --- Search & Chat Schemas ---

class SourceCitation(BaseModel):
    document_id: str
    document_name: str
    chunk_id: str
    chunk_index: int
    score: float
    content: str
    chat_id: Optional[str] = None
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    rrf_score: Optional[float] = None
    page_number: Optional[int] = 1
    section: Optional[str] = None
    content_type: Optional[str] = None
    table_id: Optional[str] = None
    table_title: Optional[str] = None
    row_range: Optional[List[int]] = None
    column_range: Optional[List[int]] = None
    bbox: Optional[Dict[str, Any]] = None
    strategy: Optional[str] = None
    version: Optional[str] = None


class ChatQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's query string.")
    chat_id: Optional[str] = Field(default=None, description="Optional chat session ID.")
    top_k: int = Field(default=4, ge=1, le=20, description="Number of context chunks to retrieve.")
    document_ids: Optional[List[str]] = Field(default=None, description="Filter search to specific documents.")
    version: Optional[str] = Field(
        default="v2.2", description="System version: 'v1', 'v2.1', 'v2.2', or 'v3'."
    )
    chunking_strategy: Optional[str] = Field(
        default="table_aware", description="V3 Chunking strategy ('fixed', 'recursive', 'semantic', 'section_aware', 'table_aware', 'parent_child', 'sliding_window', 'hierarchical')."
    )
    retrieval_mode: Optional[str] = Field(
        default="hybrid", description="Legacy retrieval mode filter for backward compatibility."
    )


class ChatQueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceCitation]
    retrieval_mode: str = "hybrid"
    version: str = "v2.2"
    chunking_strategy: Optional[str] = None
    query_expansion_meta: Optional[Dict[str, Any]] = None
    calculation: Optional[Dict[str, Any]] = None
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
