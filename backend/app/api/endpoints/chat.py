import os
import time
import uuid
import hashlib
import logging
import asyncio
from datetime import datetime, timezone
from typing import Optional, List
from fastapi import APIRouter, HTTPException, UploadFile, File, status

from app.core.config import settings
from app.services.embedder import embedding_service
from app.services.bm25_search import bm25_service
from app.services.hybrid_retriever import hybrid_retriever
from app.db.vectorstore import vector_store
from app.db.mongodb import mongo_db
from app.services.llm_service import llm_service
from app.services.query_eval_service import query_eval_service
from app.services.document_parser import UnifiedDocumentParser
from app.services.chunker import RecursiveTextChunker
from app.models.schemas import (
    ChatSession,
    ChatDocument,
    ChatCreateRequest,
    ChatListResponse,
    ChatDetailResponse,
    DocumentRepresentation,
    RepresentationListResponse,
    MaterializeRepresentationResponse,
    ChatQueryRequest,
    ChatQueryResponse,
)
from app.core.pipeline_router import pipeline_router

router = APIRouter()
logger = logging.getLogger(__name__)


# --- Chat Lifecycle Endpoints ---

@router.post("", response_model=ChatSession, status_code=status.HTTP_201_CREATED)
@router.post("/", response_model=ChatSession, status_code=status.HTTP_201_CREATED)
async def create_chat(req: Optional[ChatCreateRequest] = None):
    """Creates a new empty chat session (ChatGPT model: 0 documents, 0 messages)."""
    title = req.title if req else None
    chat = await mongo_db.create_chat(title=title)
    return chat


@router.get("", response_model=ChatListResponse)
@router.get("/", response_model=ChatListResponse)
async def list_chats():
    """Lists all chat sessions."""
    chats = await mongo_db.list_chats()
    return ChatListResponse(total=len(chats), chats=chats)


@router.get("/{chat_id}", response_model=ChatDetailResponse)
async def get_chat_detail(chat_id: str):
    """Retrieves chat metadata and attached chat documents."""
    chat = await mongo_db.get_chat(chat_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session '{chat_id}' not found.",
        )
    documents = await mongo_db.list_chat_documents(chat_id)
    return ChatDetailResponse(chat=chat, documents=documents, messages=[])


@router.delete("/{chat_id}")
async def delete_chat(chat_id: str):
    """
    Deletes chat session and all chat-scoped resources:
    - Chat documents metadata & representations
    - Qdrant vector points matching chat_id
    - BM25 lexical entries matching chat_id
    - Physical source PDF file if reference count reaches 0.
    """
    chat = await mongo_db.get_chat(chat_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session '{chat_id}' not found.",
        )

    # Fetch attached documents before deleting records for reference counting
    chat_docs = await mongo_db.list_chat_documents(chat_id)

    # 1. Delete Qdrant vector points and BM25 entries scoped to chat_id
    vector_store.delete_chat_chunks(chat_id)
    bm25_service.delete_chat_documents(chat_id)

    # 2. Delete chat record & document metadata from MongoDB
    await mongo_db.delete_chat(chat_id)

    # 3. Reference counting check for physical PDF files
    upload_dir = os.path.abspath("./data/uploads")
    for cdoc in chat_docs:
        c_hash = cdoc.content_hash
        if c_hash:
            ref_count = await mongo_db.count_chat_documents_for_hash(c_hash)
            if ref_count == 0:
                file_path = os.path.join(upload_dir, f"{c_hash}.pdf")
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        logger.info(f"Deleted unreferenced physical PDF file: {file_path}")
                    except Exception as fe:
                        logger.warning(f"Failed to remove physical file '{file_path}': {str(fe)}")

    return {"success": True, "message": f"Chat session '{chat_id}' and all chat-scoped resources purged successfully."}


# --- Chat-Scoped Document Upload & Representation Endpoints ---

@router.post("/{chat_id}/documents", response_model=ChatDocument, status_code=status.HTTP_201_CREATED)
async def upload_chat_document(chat_id: str, file: UploadFile = File(...)):
    """
    Uploads a PDF into a specific chat session:
    1. Computes SHA-256 content_hash.
    2. Stores immutable raw source PDF in ./data/uploads/{content_hash}.pdf.
    3. Runs V1 ingestion (UnifiedDocumentParser -> RecursiveTextChunker -> embeddings -> Qdrant/BM25).
    4. Sets default version = V1.
    5. Returns ChatDocument object.
    """
    chat = await mongo_db.get_chat(chat_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session '{chat_id}' not found.",
        )

    filename = file.filename or "unnamed_document.pdf"
    content_bytes = await file.read()
    if not content_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    content_hash = hashlib.sha256(content_bytes).hexdigest()
    upload_dir = os.path.abspath("./data/uploads")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, f"{content_hash}.pdf")

    # Store immutable physical PDF if not already existing
    if not os.path.exists(file_path):
        with open(file_path, "wb") as f_out:
            f_out.write(content_bytes)

    # Extract text & generate V1 chunks
    document_id = f"doc_{uuid.uuid4().hex[:12]}"
    text, file_type = UnifiedDocumentParser.extract_text(content_bytes, filename)
    chunker = RecursiveTextChunker(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    chunks = chunker.chunk_document(text, document_id=document_id)

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Document contained no valid text chunks.",
        )

    # Embed & Index V1 chunks with chat_id scope
    chunk_texts = [c.text for c in chunks]
    embeddings = embedding_service.embed_batch(chunk_texts)

    vector_store.upsert_chunks(
        chunks=chunks,
        embeddings=embeddings,
        filename=filename,
        chat_id=chat_id,
    )
    bm25_service.index_chunks(chunks=chunks, filename=filename, chat_id=chat_id)

    chat_doc = ChatDocument(
        chat_document_id=f"cdoc_{uuid.uuid4().hex[:12]}",
        chat_id=chat_id,
        document_id=document_id,
        filename=filename,
        content_hash=content_hash,
        source_path=file_path,
        file_type=file_type,
        file_size_bytes=len(content_bytes),
        char_count=len(text),
        v1_chunk_count=len(chunks),
    )
    await mongo_db.add_chat_document(chat_doc)

    # Save READY V1 representation
    v1_rep = DocumentRepresentation(
        representation_id=f"{chat_id}_{document_id}_v1",
        chat_id=chat_id,
        document_id=document_id,
        document_name=filename,
        content_hash=content_hash,
        version="v1",
        status="READY",
        chunk_count=len(chunks),
        parser_version="UnifiedDocumentParser",
        chunker_version="RecursiveTextChunker",
        index_status="INDEXED",
    )
    await mongo_db.save_representation(v1_rep)

    return chat_doc


@router.get("/{chat_id}/documents", response_model=List[ChatDocument])
async def list_chat_documents(chat_id: str):
    """Lists all documents attached to a specific chat."""
    chat = await mongo_db.get_chat(chat_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session '{chat_id}' not found.",
        )
    return await mongo_db.list_chat_documents(chat_id)


@router.get("/{chat_id}/documents/{document_id}/representations", response_model=RepresentationListResponse)
async def get_chat_document_representations(chat_id: str, document_id: str):
    """Lists all representations for a chat-scoped document."""
    from app.v3.ingestion.ingestion_service import v3_ingestion_service

    cdoc = await mongo_db.get_chat_document(chat_id, document_id)
    if not cdoc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found in chat '{chat_id}'.",
        )

    reps = await v3_ingestion_service.list_representations(document_id, chat_id=chat_id)

    # Check V3 status
    v3_rep = await mongo_db.get_representation(document_id, "v3", chat_id=chat_id)
    if not v3_rep:
        # Include NOT_CREATED V3 representation placeholder for UI convert action banner
        v3_placeholder = DocumentRepresentation(
            representation_id=f"{chat_id}_{document_id}_v3_not_created",
            chat_id=chat_id,
            document_id=document_id,
            document_name=cdoc.filename,
            content_hash=cdoc.content_hash,
            version="v3",
            status="NOT_CREATED",
            chunk_count=0,
            parser_version="PyMuPDF_TableFinder_V3",
            chunker_version="V3ChunkingEngine",
            index_status="NOT_INDEXED",
        )
        reps.append(v3_placeholder)

    return RepresentationListResponse(
        document_id=document_id,
        document_name=cdoc.filename,
        representations=reps,
    )


@router.post("/{chat_id}/documents/{document_id}/representations/v3/convert", response_model=MaterializeRepresentationResponse)
async def convert_document_to_v3(chat_id: str, document_id: str, force_reprocess: bool = False):
    """
    Explicit V3 conversion endpoint with background job lifecycle:
    1. If V3 representation is READY and not force_reprocess, returns immediately.
    2. If V3 representation is active PROCESSING (age <= 300s) and not force_reprocess, returns current status for UI polling.
    3. If V3 representation is stale PROCESSING (> 300s), force_reprocess=True, or missing/FAILED, launches background worker.
    """
    from app.v3.ingestion.ingestion_service import v3_ingestion_service

    cdoc = await mongo_db.get_chat_document(chat_id, document_id)
    if not cdoc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found in chat '{chat_id}'.",
        )

    # Check if representation exists and is READY or active PROCESSING
    existing_v3 = await mongo_db.get_representation(document_id, "v3", chat_id=chat_id)
    if existing_v3 and existing_v3.status == "READY" and not force_reprocess:
        await mongo_db.update_chat_active_state(chat_id, active_document_id=document_id, active_version="v3")
        return MaterializeRepresentationResponse(
            success=True,
            message=f"V3 conversion for '{cdoc.filename}' is READY ({existing_v3.chunk_count} structural chunks).",
            representation=existing_v3,
        )

    if existing_v3 and existing_v3.status == "PROCESSING" and not force_reprocess:
        is_stale = False
        if existing_v3.updated_at:
            dt = existing_v3.updated_at
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - dt).total_seconds()
            if age > 60:  # 60s threshold
                is_stale = True
        
        if not is_stale:
            return MaterializeRepresentationResponse(
                success=False,
                message=f"V3 conversion for '{cdoc.filename}' is currently PROCESSING.",
                representation=existing_v3,
            )
        else:
            logger.warning(f"Found stale PROCESSING representation '{existing_v3.representation_id}' for '{document_id}' (age > 60s). Re-triggering conversion.")

    # Canonical representation identity for UI response
    canonical_rep_id = existing_v3.representation_id if existing_v3 else f"{chat_id}_{document_id}_v3"

    # Async background convert worker
    async def _async_background_convert():
        try:
            rep = await v3_ingestion_service.materialize_representation(
                document_id=document_id,
                version="v3",
                strategy=None,
                chat_id=chat_id,
                force_reprocess=True,
            )
            if rep and rep.status == "READY":
                await mongo_db.update_chat_active_state(chat_id, active_document_id=document_id, active_version="v3")
        except Exception as bg_err:
            logger.exception(f"Background V3 conversion failed for document '{document_id}': {str(bg_err)}")
            failed_rep = DocumentRepresentation(
                representation_id=canonical_rep_id,
                chat_id=chat_id,
                document_id=document_id,
                document_name=cdoc.filename,
                content_hash=cdoc.content_hash,
                version="v3",
                status="FAILED",
                chunk_count=0,
                index_status="FAILED",
                error_message=str(bg_err),
                updated_at=datetime.now(timezone.utc),
            )
            await mongo_db.save_representation(failed_rep)

    # Schedule background worker task using asyncio
    asyncio.create_task(_async_background_convert())

    rep_processing = DocumentRepresentation(
        representation_id=canonical_rep_id,
        chat_id=chat_id,
        document_id=document_id,
        document_name=cdoc.filename,
        content_hash=cdoc.content_hash,
        version="v3",
        status="PROCESSING",
        parser_version="PyMuPDF_TableFinder_V3",
        chunker_version="V3ChunkingEngine",
        index_status="NOT_INDEXED",
        updated_at=datetime.now(timezone.utc),
    )

    return MaterializeRepresentationResponse(
        success=False,
        message=f"V3 conversion for '{cdoc.filename}' initiated in background.",
        representation=rep_processing,
    )


# --- Chat-Scoped RAG Query Endpoint ---

@router.post("/query", response_model=ChatQueryResponse)
@router.post("/{chat_id}/query", response_model=ChatQueryResponse)
async def query_rag(request: ChatQueryRequest, chat_id: Optional[str] = None):
    """
    Chat-Scoped RAG Query Flow:
    1. Validates chat_id, active document, and active pipeline version.
    2. Strictly asserts chat retrieval isolation.
    3. Rejects V3 retrieval if V3 representation is NOT READY (no silent fallbacks).
    4. Routes query cleanly through requested version pipeline.
    """
    start_time = time.time()
    effective_chat_id = chat_id or request.chat_id

    query_str = request.query.strip()
    if not query_str:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty.",
        )

    version_str = request.version or request.retrieval_mode or "v1"
    doc_ids = request.document_ids

    if effective_chat_id:
        chat = await mongo_db.get_chat(effective_chat_id)
        if chat:
            # If doc_ids not explicitly supplied in request, default to chat active document
            if not doc_ids and chat.active_document_id:
                doc_ids = [chat.active_document_id]

            # Update chat active version if changed from UI
            if version_str != chat.active_version:
                await mongo_db.update_chat_active_state(
                    effective_chat_id,
                    active_document_id=doc_ids[0] if doc_ids else chat.active_document_id,
                    active_version=version_str,
                )

    # Explicit V3 Readiness Guard (No Silent Fallback)
    if version_str == "v3" and doc_ids:
        for d_id in doc_ids:
            rep = await mongo_db.get_representation(d_id, "v3", chat_id=effective_chat_id)
            if not rep or rep.status != "READY":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"V3 representation is not ready for document '{d_id}' (Status: {rep.status if rep else 'NOT_CREATED'}). Please click 'Convert PDF to V3' first.",
                )

    try:
        raw_citations, latency_breakdown, query_exp_meta, calc_dict = pipeline_router.route_query(
            query=query_str,
            top_k=request.top_k,
            document_ids=doc_ids,
            version=version_str,
            chunking_strategy=request.chunking_strategy or "table_aware",
            chat_id=effective_chat_id,
        )

        # Context Expansion
        t0_exp = time.time()
        citations = vector_store.expand_adjacent_context(raw_citations, window=1)
        latency_breakdown["context_expansion_ms"] = round((time.time() - t0_exp) * 1000, 2)

        # LLM Synthesis
        t0_gen = time.time()
        answer, provider_used, model_used = llm_service.generate_answer(
            query=query_str, citations=citations
        )
        latency_breakdown["llm_generation_ms"] = round((time.time() - t0_gen) * 1000, 2)

        if calc_dict and calc_dict.get("validation_status") == "VALIDATED" and calc_dict.get("display_result") != "N/A":
            metric_title = calc_dict.get("metric_display_name", "Calculation")
            disp_res = calc_dict.get("display_result")
            formula_used = calc_dict.get("formula")
            calc_summary = f"**{metric_title}**: `{disp_res}`\n*Formula*: `{formula_used}`\n\n"
            answer = calc_summary + answer

        elapsed = round(time.time() - start_time, 3)
        latency_breakdown["total_request_ms"] = round(elapsed * 1000, 2)

        # Audit log in MongoDB
        citation_dicts = [c.model_dump() for c in citations]
        try:
            await mongo_db.log_chat_interaction(
                query=query_str,
                answer=answer,
                sources=citation_dicts,
                provider=provider_used,
            )
        except Exception as mongo_err:
            logger.warning(f"MongoDB chat interaction logging skipped: {str(mongo_err)}")

        ret_mode_str = (request.retrieval_mode or ("dense" if version_str == "v1" else ("bm25" if version_str == "v2.1" else "hybrid"))).lower()

        eval_record = query_eval_service.record_evaluation(
            query=query_str,
            answer=answer,
            retrieval_mode=ret_mode_str,
            citations=citation_dicts,
            latency_breakdown=latency_breakdown,
            document_ids=doc_ids,
        )

        return ChatQueryResponse(
            query=query_str,
            answer=answer,
            sources=citations,
            retrieval_mode=ret_mode_str,
            version=version_str,
            chunking_strategy=request.chunking_strategy if version_str == "v3" else None,
            query_expansion_meta=query_exp_meta,
            calculation=calc_dict,
            llm_provider=provider_used,
            model_name=model_used,
            processing_time_seconds=elapsed,
            evaluation_id=eval_record.evaluation_id,
            latency_breakdown=latency_breakdown,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error executing RAG query '{query_str}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process RAG query: {str(e)}",
        )
