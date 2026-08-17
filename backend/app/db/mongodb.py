import asyncio
import logging
from typing import List, Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings
from app.models.schemas import DocumentMetadata

logger = logging.getLogger(__name__)


class MongoDBManager:
    """Manages async MongoDB operations for document metadata and chat logs."""

    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        self._is_connected = False
        # In-memory storage fallback if MongoDB service is offline
        self._fallback_docs: Dict[str, dict] = {}
        self._fallback_chats: List[dict] = []

    async def connect(self):
        """Connects to MongoDB cluster or local instance on current event loop."""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if self._is_connected and self.client is not None:
            # Check if motor client's loop is still open and matches current running loop
            try:
                if self.client.get_io_loop() == current_loop and not current_loop.is_closed():
                    return
            except Exception:
                pass

        try:
            self.client = AsyncIOMotorClient(
                settings.MONGODB_URI,
                serverSelectionTimeoutMS=1500
            )
            await self.client.admin.command('ping')
            self.db = self.client[settings.MONGODB_DB_NAME]
            self._is_connected = True
            logger.info(f"Connected to MongoDB database: {settings.MONGODB_DB_NAME}")
        except Exception as e:
            logger.warning(f"MongoDB connection offline ({str(e)}). Using in-memory fallback storage.")
            self._is_connected = False
            self.client = None
            self.db = None

    async def save_document_metadata(self, doc: DocumentMetadata) -> bool:
        """Stores document metadata."""
        await self.connect()
        doc_dict = doc.model_dump()
        doc_dict["_id"] = doc.document_id

        if self._is_connected and self.db is not None:
            try:
                await self.db.documents.replace_one(
                    {"_id": doc.document_id}, doc_dict, upsert=True
                )
                return True
            except Exception as e:
                logger.warning(f"MongoDB save failed ({str(e)}). Falling back to in-memory store.")
                self._fallback_docs[doc.document_id] = doc_dict
        else:
            self._fallback_docs[doc.document_id] = doc_dict
        return True

    async def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        """Retrieves metadata for a specific document."""
        await self.connect()
        if self._is_connected and self.db is not None:
            try:
                res = await self.db.documents.find_one({"_id": document_id})
                if res:
                    res.pop("_id", None)
                    return DocumentMetadata(**res)
            except Exception as e:
                logger.warning(f"MongoDB query failed ({str(e)}). Checking fallback store.")

        if document_id in self._fallback_docs:
            data = dict(self._fallback_docs[document_id])
            data.pop("_id", None)
            return DocumentMetadata(**data)
        return None

    async def list_documents(self) -> List[DocumentMetadata]:
        """Lists metadata for all uploaded documents."""
        await self.connect()
        documents = []

        if self._is_connected and self.db is not None:
            try:
                cursor = self.db.documents.find().sort("upload_timestamp", -1)
                async for doc in cursor:
                    doc.pop("_id", None)
                    documents.append(DocumentMetadata(**doc))
                return documents
            except Exception as e:
                logger.warning(f"MongoDB list failed ({str(e)}). Falling back to in-memory store.")

        sorted_docs = sorted(
            self._fallback_docs.values(),
            key=lambda x: str(x.get("upload_timestamp")),
            reverse=True
        )
        for doc in sorted_docs:
            data = dict(doc)
            data.pop("_id", None)
            documents.append(DocumentMetadata(**data))

        return documents

    async def delete_document_metadata(self, document_id: str) -> bool:
        """Deletes metadata for a specific document."""
        await self.connect()
        if self._is_connected and self.db is not None:
            try:
                await self.db.documents.delete_one({"_id": document_id})
            except Exception as e:
                logger.warning(f"MongoDB delete failed ({str(e)}).")

        self._fallback_docs.pop(document_id, None)
        return True

    async def log_chat_interaction(
        self, query: str, answer: str, sources: List[dict], provider: str
    ):
        """Logs chat query transaction for audit and history."""
        await self.connect()
        log_entry = {
            "query": query,
            "answer": answer,
            "sources": sources,
            "provider": provider,
        }
        if self._is_connected and self.db is not None:
            try:
                await self.db.chat_history.insert_one(log_entry)
            except Exception:
                self._fallback_chats.append(log_entry)
        else:
            self._fallback_chats.append(log_entry)

    async def check_health(self) -> Dict[str, Any]:
        """Checks MongoDB service connectivity."""
        try:
            await self.connect()
            if self._is_connected:
                return {"status": "healthy", "details": "Connected to MongoDB instance"}
            return {"status": "degraded", "details": "Running with in-memory fallback store"}
        except Exception as e:
            return {"status": "degraded", "details": str(e)}


mongo_db = MongoDBManager()
