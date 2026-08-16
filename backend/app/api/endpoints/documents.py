import uuid
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.core.config import settings
from app.services.document_parser import UnifiedDocumentParser, DocumentExtractionError
from app.services.chunker import RecursiveTextChunker
from app.services.embedder import embedding_service
from app.db.vectorstore import vector_store
from app.db.mongodb import mongo_db
from app.models.schemas import (
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentMetadata,
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
    5. Stores metadata record in MongoDB.
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

        # 1. Parse text
        text, file_type = UnifiedDocumentParser.extract_text(content_bytes, filename)

        document_id = f"doc_{uuid.uuid4().hex[:12]}"
        char_count = len(text)

        # 2. Chunk text
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

        # 4. Save to Qdrant
        vector_store.upsert_chunks(
            chunks=chunks,
            embeddings=embeddings,
            filename=filename,
        )

        # 5. Save metadata to MongoDB
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


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Deletes document metadata from MongoDB and vectors from Qdrant."""
    doc = await mongo_db.get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )

    vector_store.delete_document_chunks(document_id)
    await mongo_db.delete_document_metadata(document_id)

    return {
        "success": True,
        "message": f"Document '{doc.filename}' ({document_id}) deleted successfully.",
    }
