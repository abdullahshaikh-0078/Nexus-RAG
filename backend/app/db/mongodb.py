import uuid
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings
from app.models.schemas import DocumentMetadata, DocumentRepresentation, ChatSession, ChatDocument

logger = logging.getLogger(__name__)


class MongoDBManager:
    """Manages async MongoDB operations for chats, chat-scoped documents, document representations, and chat logs."""

    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        self._is_connected = False
        # In-memory storage fallback if MongoDB service is offline
        self._fallback_docs: Dict[str, dict] = {}
        self._fallback_reps: Dict[str, dict] = {}
        self._fallback_chats_dict: Dict[str, dict] = {}
        self._fallback_chat_docs: Dict[str, dict] = {}
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

    # --- Chat Management ---

    async def create_chat(self, title: Optional[str] = None) -> ChatSession:
        """Creates a new empty chat session."""
        await self.connect()
        chat_id = f"chat_{uuid.uuid4().hex[:12]}"
        chat_title = title.strip() if title and title.strip() else "New Chat"
        chat = ChatSession(
            chat_id=chat_id,
            title=chat_title,
            active_document_id=None,
            active_version="v1",
        )
        chat_dict = chat.model_dump()
        chat_dict["_id"] = chat_id

        if self._is_connected and self.db is not None:
            try:
                await self.db.chats.replace_one({"_id": chat_id}, chat_dict, upsert=True)
            except Exception as e:
                logger.warning(f"MongoDB chat create failed ({str(e)}). Using in-memory fallback.")

        self._fallback_chats_dict[chat_id] = chat_dict
        return chat

    async def list_chats(self) -> List[ChatSession]:
        """Lists all chat sessions ordered by updated_at descending."""
        await self.connect()
        chats = []

        if self._is_connected and self.db is not None:
            try:
                cursor = self.db.chats.find({}).sort("updated_at", -1)
                async for c in cursor:
                    c.pop("_id", None)
                    chats.append(ChatSession(**c))
                return chats
            except Exception as e:
                logger.warning(f"MongoDB list chats failed ({str(e)}). Using fallback store.")

        for c_data in sorted(
            self._fallback_chats_dict.values(),
            key=lambda x: str(x.get("updated_at", "")),
            reverse=True,
        ):
            data = dict(c_data)
            data.pop("_id", None)
            chats.append(ChatSession(**data))

        return chats

    async def get_chat(self, chat_id: str) -> Optional[ChatSession]:
        """Retrieves a chat session by chat_id."""
        await self.connect()
        if self._is_connected and self.db is not None:
            try:
                res = await self.db.chats.find_one({"_id": chat_id})
                if res:
                    res.pop("_id", None)
                    return ChatSession(**res)
            except Exception as e:
                logger.warning(f"MongoDB get chat failed ({str(e)}).")

        if chat_id in self._fallback_chats_dict:
            data = dict(self._fallback_chats_dict[chat_id])
            data.pop("_id", None)
            return ChatSession(**data)

        return None

    async def update_chat_active_state(
        self, chat_id: str, active_document_id: Optional[str], active_version: str
    ) -> bool:
        """Updates active document and pipeline version for a chat."""
        await self.connect()
        now = datetime.now(timezone.utc)
        update_data = {
            "active_document_id": active_document_id,
            "active_version": active_version,
            "updated_at": now,
        }

        if self._is_connected and self.db is not None:
            try:
                await self.db.chats.update_one({"_id": chat_id}, {"$set": update_data})
            except Exception as e:
                logger.warning(f"MongoDB update chat state failed ({str(e)}).")

        if chat_id in self._fallback_chats_dict:
            self._fallback_chats_dict[chat_id].update(update_data)

        return True

    async def delete_chat(self, chat_id: str) -> bool:
        """Deletes chat session record and associated chat documents."""
        await self.connect()
        if self._is_connected and self.db is not None:
            try:
                await self.db.chats.delete_one({"_id": chat_id})
                await self.db.chat_documents.delete_many({"chat_id": chat_id})
                await self.db.document_representations.delete_many({"chat_id": chat_id})
            except Exception as e:
                logger.warning(f"MongoDB delete chat failed ({str(e)}).")

        self._fallback_chats_dict.pop(chat_id, None)

        # Clean fallback chat documents
        keys_to_del_cdoc = [k for k, v in self._fallback_chat_docs.items() if v.get("chat_id") == chat_id]
        for k in keys_to_del_cdoc:
            self._fallback_chat_docs.pop(k, None)

        # Clean fallback representations
        keys_to_del_rep = [k for k, v in self._fallback_reps.items() if v.get("chat_id") == chat_id]
        for k in keys_to_del_rep:
            self._fallback_reps.pop(k, None)

        return True

    # --- Chat-Scoped Document Operations ---

    async def add_chat_document(self, chat_doc: ChatDocument) -> bool:
        """Associates an uploaded document with a specific chat."""
        await self.connect()
        doc_dict = chat_doc.model_dump()
        doc_dict["_id"] = chat_doc.chat_document_id

        if self._is_connected and self.db is not None:
            try:
                await self.db.chat_documents.replace_one(
                    {"_id": chat_doc.chat_document_id}, doc_dict, upsert=True
                )
            except Exception as e:
                logger.warning(f"MongoDB add chat document failed ({str(e)}). Using fallback.")

        self._fallback_chat_docs[chat_doc.chat_document_id] = doc_dict
        await self.update_chat_active_state(
            chat_id=chat_doc.chat_id,
            active_document_id=chat_doc.document_id,
            active_version="v1",
        )
        return True

    async def list_chat_documents(self, chat_id: str) -> List[ChatDocument]:
        """Lists all documents attached to a specific chat."""
        await self.connect()
        docs = []

        if self._is_connected and self.db is not None:
            try:
                cursor = self.db.chat_documents.find({"chat_id": chat_id})
                async for d in cursor:
                    d.pop("_id", None)
                    docs.append(ChatDocument(**d))
                return docs
            except Exception as e:
                logger.warning(f"MongoDB list chat documents failed ({str(e)}).")

        for d_data in self._fallback_chat_docs.values():
            if d_data.get("chat_id") == chat_id:
                data = dict(d_data)
                data.pop("_id", None)
                docs.append(ChatDocument(**data))

        return docs

    async def get_chat_document(self, chat_id: str, document_id: str) -> Optional[ChatDocument]:
        """Retrieves a specific document associated with chat_id."""
        await self.connect()
        if self._is_connected and self.db is not None:
            try:
                res = await self.db.chat_documents.find_one(
                    {"chat_id": chat_id, "document_id": document_id}
                )
                if res:
                    res.pop("_id", None)
                    return ChatDocument(**res)
            except Exception as e:
                logger.warning(f"MongoDB get chat document failed ({str(e)}).")

        for d_data in self._fallback_chat_docs.values():
            if d_data.get("chat_id") == chat_id and (
                d_data.get("document_id") == document_id or d_data.get("filename") == document_id
            ):
                data = dict(d_data)
                data.pop("_id", None)
                return ChatDocument(**data)

        return None

    async def get_chat_document_by_doc_id(self, document_id: str) -> Optional[ChatDocument]:
        """Retrieves a ChatDocument by document_id across any chat session."""
        await self.connect()
        if self._is_connected and self.db is not None:
            try:
                res = await self.db.chat_documents.find_one({"document_id": document_id})
                if not res:
                    res = await self.db.chat_documents.find_one({"filename": document_id})
                if res:
                    res.pop("_id", None)
                    return ChatDocument(**res)
            except Exception as e:
                logger.warning(f"MongoDB get chat document by doc_id failed ({str(e)}).")

        for d_data in self._fallback_chat_docs.values():
            if d_data.get("document_id") == document_id or d_data.get("filename") == document_id:
                data = dict(d_data)
                data.pop("_id", None)
                return ChatDocument(**data)

        return None

    async def count_chat_documents_for_hash(self, content_hash: str) -> int:
        """Reference count: counts how many chat documents across all chats share content_hash."""
        await self.connect()
        count = 0
        if self._is_connected and self.db is not None:
            try:
                count = await self.db.chat_documents.count_documents({"content_hash": content_hash})
                return count
            except Exception as e:
                logger.warning(f"MongoDB count content_hash failed ({str(e)}).")

        for d_data in self._fallback_chat_docs.values():
            if d_data.get("content_hash") == content_hash:
                count += 1
        return count

    async def save_representation(self, rep: DocumentRepresentation) -> bool:
        """Stores or updates a document representation record."""
        await self.connect()
        rep_dict = rep.model_dump()
        rep_dict["_id"] = rep.representation_id

        if self._is_connected and self.db is not None:
            try:
                await self.db.document_representations.replace_one(
                    {"_id": rep.representation_id}, rep_dict, upsert=True
                )
                return True
            except Exception as e:
                logger.warning(f"MongoDB representation save failed ({str(e)}). Falling back to in-memory store.")
                self._fallback_reps[rep.representation_id] = rep_dict
        else:
            self._fallback_reps[rep.representation_id] = rep_dict
        return True

    async def get_representation(
        self, document_id: str, version: str, strategy: Optional[str] = None, chat_id: Optional[str] = None
    ) -> Optional[DocumentRepresentation]:
        """Retrieves a specific document representation by document_id, version, strategy, and chat_id."""
        await self.connect()
        query: Dict[str, Any] = {"document_id": document_id, "version": version}
        if chat_id:
            query["chat_id"] = chat_id
        if version == "v3" and strategy:
            query["chunking_strategy"] = strategy

        if self._is_connected and self.db is not None:
            try:
                res = await self.db.document_representations.find_one(query)
                if res:
                    res.pop("_id", None)
                    return DocumentRepresentation(**res)
            except Exception as e:
                logger.warning(f"MongoDB query failed ({str(e)}). Checking fallback representation store.")

        # Search fallback store
        for rep_data in self._fallback_reps.values():
            if rep_data.get("document_id") == document_id and rep_data.get("version") == version:
                if chat_id and rep_data.get("chat_id") != chat_id:
                    continue
                if version == "v3" and strategy:
                    if rep_data.get("chunking_strategy") != strategy:
                        continue
                data = dict(rep_data)
                data.pop("_id", None)
                return DocumentRepresentation(**data)

        return None

    async def list_representations(self, document_id: str, chat_id: Optional[str] = None) -> List[DocumentRepresentation]:
        """Lists all representations associated with a document_id and chat_id."""
        await self.connect()
        reps = []
        query: Dict[str, Any] = {"document_id": document_id}
        if chat_id:
            query["chat_id"] = chat_id

        if self._is_connected and self.db is not None:
            try:
                cursor = self.db.document_representations.find(query)
                async for rep in cursor:
                    rep.pop("_id", None)
                    reps.append(DocumentRepresentation(**rep))
                return reps
            except Exception as e:
                logger.warning(f"MongoDB list representations failed ({str(e)}). Checking fallback store.")

        for rep_data in self._fallback_reps.values():
            if rep_data.get("document_id") == document_id:
                if chat_id and rep_data.get("chat_id") != chat_id:
                    continue
                data = dict(rep_data)
                data.pop("_id", None)
                reps.append(DocumentRepresentation(**data))

        return reps

    async def update_representation_status(
        self, representation_id: str, status: str, chunk_count: int = 0, error_message: Optional[str] = None
    ) -> bool:
        """Updates representation status, chunk_count, and error_message."""
        await self.connect()
        update_fields: Dict[str, Any] = {"status": status}
        if chunk_count > 0:
            update_fields["chunk_count"] = chunk_count
            update_fields["index_status"] = "INDEXED" if status == "READY" else "FAILED"
        if error_message is not None:
            update_fields["error_message"] = error_message

        if self._is_connected and self.db is not None:
            try:
                await self.db.document_representations.update_one(
                    {"_id": representation_id}, {"$set": update_fields}
                )
            except Exception as e:
                logger.warning(f"MongoDB update representation status failed ({str(e)}).")

        if representation_id in self._fallback_reps:
            self._fallback_reps[representation_id].update(update_fields)

        return True

    async def delete_representations_for_document(self, document_id: str) -> bool:
        """Deletes all representations associated with document_id."""
        await self.connect()
        if self._is_connected and self.db is not None:
            try:
                await self.db.document_representations.delete_many({"document_id": document_id})
            except Exception as e:
                logger.warning(f"MongoDB delete representations failed ({str(e)}).")

        keys_to_del = [k for k, v in self._fallback_reps.items() if v.get("document_id") == document_id]
        for k in keys_to_del:
            self._fallback_reps.pop(k, None)
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
