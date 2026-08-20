import os
import uuid
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.core.config import settings
from app.services.document_parser import UnifiedDocumentParser, DocumentExtractionError
from app.services.chunker import RecursiveTextChunker
from app.services.embedder import embedding_service
from app.services.bm25_search import bm25_service
from app.db.vectorstore import vector_store
from app.db.mongodb import mongo_db
from app.models.schemas import (
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentMetadata,
    DocumentRepresentation,
    RepresentationListResponse,
    MaterializeRepresentationRequest,
    MaterializeRepresentationResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    """
    Ingests a document (PDF, DOCX, TXT, MD):
    1. Extracts text content.
    2. Splits into overlapping chunks.
    3. Generates vector embeddings.
    4. Stores vectors + payload in Qdrant.
    5. Indexes text in BM25 lexical index.
    6. Stores metadata record in MongoDB.
    """
    filename = file.filename or "unnamed_document"
    logger.info(f"Receiving document upload: {filename}")

    try:
        content_bytes = await file.read()
        if not content_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        # Save raw file to disk for V3 layout parsing
        upload_dir = os.path.abspath("./data/uploads")
        os.makedirs(upload_dir, exist_ok=True)
        raw_file_path = os.path.join(upload_dir, filename)
        with open(raw_file_path, "wb") as f_out:
            f_out.write(content_bytes)

        # 1. Parse text (Legacy V1/V2 path)
        text, file_type = UnifiedDocumentParser.extract_text(content_bytes, filename)

        document_id = f"doc_{uuid.uuid4().hex[:12]}"
        char_count = len(text)

        # 2. Chunk text (Legacy V1/V2 path)
        chunker = RecursiveTextChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        chunks = chunker.chunk_document(text, document_id=document_id)

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Document contained no valid text chunks to process.",
            )

        # 3. Generate embeddings
        chunk_texts = [c.text for c in chunks]
        embeddings = embedding_service.embed_batch(chunk_texts)

        # 4. Save to Qdrant (Legacy V1/V2)
        vector_store.upsert_chunks(
            chunks=chunks,
            embeddings=embeddings,
            filename=filename,
        )

        # 5. Index in BM25 search service (Legacy V1/V2)
        bm25_service.index_chunks(chunks=chunks, filename=filename)

        # 6. Trigger V3 Structural PDF Parsing & Chunking for PDFs
        if filename.lower().endswith(".pdf"):
            try:
                from app.v3.ingestion.ingestion_service import v3_ingestion_service
                v3_ingestion_service.ingest_pdf(
                    pdf_path=raw_file_path,
                    document_id=document_id,
                    document_name=filename,
                    strategy="table_aware",
                )
            except Exception as v3_err:
                logger.warning(f"V3 structural ingestion deferred/failed for '{filename}': {str(v3_err)}")

        # 7. Save metadata to MongoDB
        doc_metadata = DocumentMetadata(
            document_id=document_id,
            filename=filename,
            file_type=file_type,
            file_size_bytes=len(content_bytes),
            char_count=char_count,
            chunk_count=len(chunks),
            status="processed",
        )
        await mongo_db.save_document_metadata(doc_metadata)

        return DocumentUploadResponse(
            success=True,
            message=f"Document '{filename}' successfully ingested into NEXUS RAG.",
            document=doc_metadata,
        )

    except DocumentExtractionError as err:
        logger.error(f"Extraction error processing '{filename}': {str(err)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error during document upload of '{filename}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during ingestion: {str(e)}",
        )


@router.get("/", response_model=DocumentListResponse)
async def list_documents():
    """Returns metadata for all ingested documents."""
    docs = await mongo_db.list_documents()
    return DocumentListResponse(total=len(docs), documents=docs)


@router.get("/{document_id}/representations", response_model=RepresentationListResponse)
async def get_document_representations(document_id: str):
    """Retrieves all representations (V1, V2, V3 strategies) for a specific document."""
    from app.v3.ingestion.ingestion_service import v3_ingestion_service

    doc = await mongo_db.get_document(document_id)
    doc_name = doc.filename if doc else document_id

    reps = await v3_ingestion_service.list_representations(document_id)

    # Ensure baseline representation info is populated if not yet recorded
    existing_versions = {r.version: r for r in reps}
    if "v1" not in existing_versions:
        v1_rep = await v3_ingestion_service.materialize_representation(document_id, version="v1")
        reps.append(v1_rep)
    if "v2.2" not in existing_versions:
        v2_rep = await v3_ingestion_service.materialize_representation(document_id, version="v2.2")
        reps.append(v2_rep)

    return RepresentationListResponse(
        document_id=document_id,
        document_name=doc_name,
        representations=reps,
    )


@router.post("/{document_id}/representations/materialize", response_model=MaterializeRepresentationResponse)
async def materialize_document_representation(
    document_id: str,
    req: MaterializeRepresentationRequest,
):
    """
    Lazy materialization endpoint:
    Checks if requested representation exists and is READY; if not, triggers V3 layout parse & strategy chunking.
    """
    from app.v3.ingestion.ingestion_service import v3_ingestion_service

    rep = await v3_ingestion_service.materialize_representation(
        document_id=document_id,
        version=req.version,
        strategy=req.chunking_strategy,
    )

    msg = f"Representation '{rep.representation_id}' is {rep.status} ({rep.chunk_count} chunks)."
    return MaterializeRepresentationResponse(
        success=(rep.status == "READY"),
        message=msg,
        representation=rep,
    )


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Deletes document metadata from MongoDB, vectors from Qdrant, and payload from BM25 index."""
    doc = await mongo_db.get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )

    vector_store.delete_document_chunks(document_id)
    bm25_service.delete_document(document_id)
    await mongo_db.delete_document_metadata(document_id)

    return {
        "success": True,
        "message": f"Document '{doc.filename}' ({document_id}) deleted successfully.",
    }
