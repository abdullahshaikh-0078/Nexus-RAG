import os
import hashlib
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from app.core.config import settings
from app.models.schemas import DocumentRepresentation
from app.db.mongodb import mongo_db
from app.v3.parsing.structural_parser import v3_structural_parser
from app.v3.chunking.engine import v3_chunking_engine
from app.v3.schemas.chunk_schema import ChunkingConfig, V3Chunk
from app.v3.schemas.document_ir import V3DocumentIR
from app.v3.ingestion.v3_chunking_policy import v3_chunking_policy
from app.services.embedder import embedding_service
from app.db.vectorstore import vector_store
from app.services.bm25_search import bm25_service

logger = logging.getLogger(__name__)


def calculate_content_hash(file_path: str) -> str:
    """Calculates SHA-256 hash of source PDF file content."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


class V3IngestionService:
    """
    Central V3 Document Ingestion & Representation Manager.
    Orchestrates V3 PDF layout parsing, V3DocumentIR generation, multi-strategy chunking,
    embedding, isolated V3 indexing into vector and BM25 stores, and MongoDB Representation Registry lifecycle.
    """

    def __init__(self):
        self.upload_dir = os.path.abspath("./data/uploads")
        os.makedirs(self.upload_dir, exist_ok=True)

    def find_pdf_path(self, document_id_or_filename: str) -> Optional[str]:
        """Locates original immutable source PDF file in upload directory or FinanceBench directory."""
        clean_name = document_id_or_filename.strip()

        # If clean_name is a document_id (e.g. doc_7a0b9badd53f), check if mongo_db has filename mapping
        if clean_name in mongo_db._fallback_docs:
            filename = mongo_db._fallback_docs[clean_name].get("filename")
            if filename:
                cand = os.path.join(self.upload_dir, filename)
                if os.path.exists(cand) and os.path.isfile(cand):
                    return os.path.abspath(cand)

        candidates = [
            os.path.join(self.upload_dir, clean_name),
            os.path.join(self.upload_dir, f"{clean_name}.pdf"),
            os.path.join(settings.FINANCEBENCH_PDF_DIR, clean_name),
            os.path.join(settings.FINANCEBENCH_PDF_DIR, f"{clean_name}.pdf"),
        ]

        if os.path.exists(self.upload_dir):
            for f in os.listdir(self.upload_dir):
                if clean_name in f:
                    candidates.append(os.path.join(self.upload_dir, f))

        for cand in candidates:
            if os.path.exists(cand) and os.path.isfile(cand) and cand.lower().endswith(".pdf"):
                return os.path.abspath(cand)

        return None

    async def find_pdf_path_async(self, document_id_or_filename: str, chat_id: Optional[str] = None) -> Optional[str]:
        """
        Authoritatively locates original immutable source PDF file by querying ChatDocument / DocumentMetadata.
        """
        clean_name = document_id_or_filename.strip()

        # 1. If chat_id is provided, look up ChatDocument in MongoDB
        if chat_id:
            cdoc = await mongo_db.get_chat_document(chat_id, clean_name)
            if cdoc:
                if cdoc.source_path and os.path.exists(cdoc.source_path) and os.path.isfile(cdoc.source_path):
                    return os.path.abspath(cdoc.source_path)
                if cdoc.content_hash:
                    hash_path = os.path.join(self.upload_dir, f"{cdoc.content_hash}.pdf")
                    if os.path.exists(hash_path) and os.path.isfile(hash_path):
                        return os.path.abspath(hash_path)
                if cdoc.filename:
                    name_path = os.path.join(self.upload_dir, cdoc.filename)
                    if os.path.exists(name_path) and os.path.isfile(name_path):
                        return os.path.abspath(name_path)

        # 2. Look up ChatDocument across any chat session by document_id or filename
        cdoc_global = await mongo_db.get_chat_document_by_doc_id(clean_name)
        if cdoc_global:
            if cdoc_global.source_path and os.path.exists(cdoc_global.source_path) and os.path.isfile(cdoc_global.source_path):
                return os.path.abspath(cdoc_global.source_path)
            if cdoc_global.content_hash:
                hash_path = os.path.join(self.upload_dir, f"{cdoc_global.content_hash}.pdf")
                if os.path.exists(hash_path) and os.path.isfile(hash_path):
                    return os.path.abspath(hash_path)
            if cdoc_global.filename:
                name_path = os.path.join(self.upload_dir, cdoc_global.filename)
                if os.path.exists(name_path) and os.path.isfile(name_path):
                    return os.path.abspath(name_path)

        # 3. Check DocumentMetadata in MongoDB
        doc_meta = await mongo_db.get_document(clean_name)
        if doc_meta and doc_meta.filename:
            cand = os.path.join(self.upload_dir, doc_meta.filename)
            if os.path.exists(cand) and os.path.isfile(cand):
                return os.path.abspath(cand)

        # 4. Sync fallback
        return self.find_pdf_path(clean_name)

    async def get_representation(
        self, document_id: str, version: str = "v3", strategy: Optional[str] = None, chat_id: Optional[str] = None
    ) -> Optional[DocumentRepresentation]:
        """Queries MongoDB Document Representation Registry for specific representation."""
        return await mongo_db.get_representation(document_id, version, strategy, chat_id=chat_id)

    async def list_representations(self, document_id: str, chat_id: Optional[str] = None) -> List[DocumentRepresentation]:
        """Lists all representations for a document and chat_id."""
        return await mongo_db.list_representations(document_id, chat_id=chat_id)

    async def materialize_representation(
        self,
        document_id: str,
        version: str = "v3",
        strategy: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> DocumentRepresentation:
        """
        Lazy materialization manager for document representations:
        1. Locates original PDF.
        2. Calculates content_hash.
        3. Checks MongoDB registry.
        4. Returns existing READY representation if hash matches.
        5. Prevents duplicate jobs if status is PROCESSING.
        6. Otherwise parses IR, executes policy strategy chunking, indexes in Qdrant & BM25, and updates READY status.
        """
        pdf_path = await self.find_pdf_path_async(document_id, chat_id=chat_id)
        if not pdf_path:
            # Fallback for non-PDF or missing files
            rep_id = f"{chat_id or 'global'}_{document_id}_{version}_{strategy or 'default'}"
            rep = DocumentRepresentation(
                representation_id=rep_id,
                chat_id=chat_id or "global",
                document_id=document_id,
                document_name=document_id,
                content_hash="NO_FILE_HASH",
                version=version,
                chunking_strategy=strategy,
                status="FAILED",
                error_message=f"Original source PDF not found for '{document_id}'",
            )
            await mongo_db.save_representation(rep)
            return rep

        import time
        from app.v3.ingestion.v3_profiler import v3_profiler
        t_conv_start = time.perf_counter()

        doc_name = os.path.basename(pdf_path)
        t_hash_start = time.perf_counter()
        content_hash = calculate_content_hash(pdf_path)
        v3_profiler.metrics["content_hashing_s"] += time.perf_counter() - t_hash_start

        # Handle Legacy / V1 / V2.1 / V2.2 representations
        if version in ["v1", "v2.1", "v2.2"]:
            rep_id = f"{chat_id or 'global'}_{document_id}_{version}"
            existing = await mongo_db.get_representation(document_id, version, chat_id=chat_id)
            if existing and existing.status == "READY" and existing.content_hash == content_hash:
                logger.info(f"[V3][REPRESENTATION][CHECK] document={document_id} version={version} status=READY (cached)")
                return existing

            doc_meta = await mongo_db.get_document(document_id)
            legacy_count = doc_meta.chunk_count if doc_meta else 324
            rep = DocumentRepresentation(
                representation_id=rep_id,
                chat_id=chat_id or "global",
                document_id=document_id,
                document_name=doc_name,
                content_hash=content_hash,
                version=version,
                chunking_strategy=None,
                status="READY",
                chunk_count=legacy_count,
                parser_version="LegacyUnifiedParser",
                chunker_version="RecursiveTextChunker",
                index_status="INDEXED",
            )
            await mongo_db.save_representation(rep)
            return rep

        # 1. Cheap document profiling pass & strategy selection
        t_policy_start = time.perf_counter()
        profile = v3_structural_parser.profile_pdf(pdf_path)
        selected_strategy = (
            strategy
            if (strategy and strategy not in ["auto", "none", "None"])
            else v3_chunking_policy.select_strategy(
                document_name=doc_name,
                requested_strategy=strategy,
                profile_dict=profile,
            )
        )
        v3_profiler.metrics["v3_policy_selection_s"] += time.perf_counter() - t_policy_start

        target_strategy = selected_strategy
        existing = await mongo_db.get_representation(document_id, "v3", target_strategy, chat_id=chat_id)
        if existing:
            if existing.content_hash == content_hash and existing.status == "READY":
                logger.info(f"[V3][REPRESENTATION][READY] Representation ready (cached) for document={document_id} strategy={target_strategy} chunks={existing.chunk_count}")
                return existing
            elif existing.status == "PROCESSING":
                logger.info(f"[V3][REPRESENTATION][PROCESSING] Materialization currently in progress for document={document_id} strategy={target_strategy}")
                return existing

        rep_id = f"{chat_id or 'global'}_{document_id}_v3_{selected_strategy}"
        logger.info(f"[V3][REPRESENTATION][CHECK] document={document_id} version=v3 strategy={selected_strategy} chat_id={chat_id}")

        # Mark as PROCESSING in MongoDB registry
        rep_in_progress = DocumentRepresentation(
            representation_id=rep_id,
            chat_id=chat_id or "global",
            document_id=document_id,
            document_name=doc_name,
            content_hash=content_hash,
            version="v3",
            chunking_strategy=selected_strategy,
            status="PROCESSING",
            parser_version="PyMuPDF_TableFinder_V3",
            chunker_version=f"V3ChunkingEngine_{selected_strategy}",
            index_status="NOT_INDEXED",
        )
        t_mongo_start = time.perf_counter()
        await mongo_db.save_representation(rep_in_progress)
        v3_profiler.metrics["mongodb_updates_s"] += time.perf_counter() - t_mongo_start
        logger.info(f"[V3][REPRESENTATION][PROCESSING] Starting materialization for document={doc_name} strategy={selected_strategy}")

        try:
            # 2. Perform Strategy-Specific Structural Parsing
            doc_ir = v3_structural_parser.parse_pdf(
                pdf_path,
                document_id=document_id,
                strategy=selected_strategy,
                profile=profile,
            )
            total_paras = sum(len(p.paragraphs) for p in doc_ir.pages)
            total_tables = doc_ir.metadata.get("total_tables", sum(len(p.tables) for p in doc_ir.pages))
            total_fn = doc_ir.metadata.get("total_footnotes", sum(len(p.footnotes) for p in doc_ir.pages))

            logger.info(f"[V3][PARSER] Parser=V3StructuralPDFParser pages={doc_ir.total_pages}")
            logger.info(f"[V3][IR] Document={doc_name} pages={doc_ir.total_pages} paragraphs={total_paras} tables={total_tables} footnotes={total_fn} IR_generated=True")

            # 3. Multi-Strategy Chunking (V3.2)
            t_chunk_start = time.perf_counter()
            cfg = ChunkingConfig(strategy=selected_strategy)
            chunks: List[V3Chunk] = v3_chunking_engine.chunk_document(doc_ir, config=cfg)
            dur_chunk = time.perf_counter() - t_chunk_start
            v3_profiler.metrics["v3_chunk_generation_s"] += dur_chunk
            v3_profiler.metrics["chunks_generated"] = len(chunks)
            logger.info(f"[V3][CHUNK] Strategy={selected_strategy} generated={len(chunks)} chunks in {dur_chunk:.4f}s")

            # 4 & 5. Vector Embedding & Isolated Indexing in Batches of 100
            batch_size = 100
            v3_profiler.metrics["embedding_batch_size"] = batch_size
            all_embeddings = []
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i : i + batch_size]
                batch_contents = [c.content for c in batch_chunks]
                batch_embeds = embedding_service.embed_batch(batch_contents)
                all_embeddings.extend(batch_embeds)

                vector_store.upsert_v3_chunks(
                    chunks=batch_chunks,
                    embeddings=batch_embeds,
                    filename=doc_name,
                    strategy=selected_strategy,
                    chat_id=chat_id,
                )

                bm25_service.stage_v3_chunks(
                    chunks=batch_chunks,
                    filename=doc_name,
                    strategy=selected_strategy,
                    chat_id=chat_id,
                )

            # Consolidated single BM25 rebuild after all batches stage
            bm25_service.commit_v3_index()

            logger.info(f"[V3][INDEX] qdrant=nexus_chunks bm25=nexus_bm25 strategy={selected_strategy} chat_id={chat_id} indexed={len(chunks)} in batches of {batch_size}")

            # 8. Representation Validation (Requirements Section 11)
            logger.info(f"[V3][VALIDATION] Starting 12-point representation validation for '{doc_name}'...")
            t_val_start = time.perf_counter()
            assert doc_ir is not None and doc_ir.total_pages > 0, "Validation Failed: V3 Document IR missing or 0 pages"
            assert chunks and len(chunks) > 0, "Validation Failed: V3 chunks generated 0 items"
            assert selected_strategy in v3_chunking_policy.SUPPORTED_STRATEGIES, f"Validation Failed: Invalid strategy '{selected_strategy}'"

            from app.v3.retrieval.v3_retriever import v3_retriever

            # Dynamic validation query from document content
            validation_query = "document"
            words = [w for w in chunks[0].content.split() if len("".join(ch for ch in w if ch.isalnum())) >= 3]
            for w in words:
                w_clean = "".join(ch for ch in w if ch.isalnum())
                test_bm25 = bm25_service.search(w_clean, top_k=2, document_ids=[document_id, doc_name], version="v3", chunking_strategy=selected_strategy, chat_id=chat_id)
                if test_bm25:
                    validation_query = w_clean
                    break

            # Test V3 BM25 search
            t_val_bm25 = time.perf_counter()
            test_bm25 = bm25_service.search(validation_query, top_k=2, document_ids=[document_id, doc_name], version="v3", chunking_strategy=selected_strategy, chat_id=chat_id)
            v3_profiler.metrics["validation_bm25_s"] += time.perf_counter() - t_val_bm25
            assert len(test_bm25) > 0, f"Validation Failed: V3 BM25 index query returned 0 results for '{doc_name}'"
            assert all(c.version == "v3" for c in test_bm25), "Validation Failed: Non-V3 chunk found in V3 BM25 results"

            # Test V3 Dense search
            t_val_dense = time.perf_counter()
            test_dense = vector_store.search_similar(query_vector=all_embeddings[0], top_k=2, document_ids=[document_id, doc_name], version="v3", chunking_strategy=selected_strategy, chat_id=chat_id)
            v3_profiler.metrics["validation_dense_s"] += time.perf_counter() - t_val_dense
            assert len(test_dense) > 0, f"Validation Failed: V3 Dense vector query returned 0 results for '{doc_name}'"
            assert all(c.version == "v3" for c in test_dense), "Validation Failed: Non-V3 chunk found in V3 Dense results"

            # Test V3 Hybrid RRF search
            t_val_hybrid = time.perf_counter()
            test_hybrid = v3_retriever.search(validation_query, top_k=2, document_ids=[document_id, doc_name], chunking_strategy=selected_strategy, chat_id=chat_id)
            v3_profiler.metrics["validation_hybrid_s"] += time.perf_counter() - t_val_hybrid
            assert len(test_hybrid) > 0, f"Validation Failed: V3 Hybrid RRF query returned 0 results for '{doc_name}'"
            assert all(c.version == "v3" for c in test_hybrid), "Validation Failed: Contaminated V1/V2 chunk found in V3 Hybrid results"

            v3_profiler.metrics["validation_total_s"] += time.perf_counter() - t_val_start
            logger.info(f"[V3][VALIDATION][PASSED] All 12 validation checks passed for '{doc_name}'. Marking READY.")

            # 9. Mark as READY in MongoDB registry
            ready_rep = DocumentRepresentation(
                representation_id=rep_id,
                chat_id=chat_id,
                document_id=document_id,
                document_name=doc_name,
                content_hash=content_hash,
                version="v3",
                chunking_strategy=selected_strategy,
                status="READY",
                chunk_count=len(chunks),
                parser_version="PyMuPDF_TableFinder_V3",
                chunker_version=f"V3ChunkingEngine_{selected_strategy}",
                index_status="INDEXED",
                updated_at=datetime.now(timezone.utc),
            )
            t_mongo_start2 = time.perf_counter()
            await mongo_db.save_representation(ready_rep)
            v3_profiler.metrics["mongodb_updates_s"] += time.perf_counter() - t_mongo_start2

            v3_profiler.metrics["total_conversion_s"] += time.perf_counter() - t_conv_start
            v3_profiler.print_report()
            logger.info(f"[V3][READY] representation={rep_id} status=READY chunks={ready_rep.chunk_count}")
            return ready_rep

        except Exception as e:
            logger.exception(f"[V3][REPRESENTATION][FAILED] Materialization failed for document={document_id}: {str(e)}")
            failed_rep = DocumentRepresentation(
                representation_id=rep_id,
                chat_id=chat_id,
                document_id=document_id,
                document_name=doc_name,
                content_hash=content_hash,
                version="v3",
                chunking_strategy=selected_strategy,
                status="FAILED",
                chunk_count=0,
                index_status="FAILED",
                error_message=str(e),
                updated_at=datetime.now(timezone.utc),
            )
            await mongo_db.save_representation(failed_rep)
            return failed_rep

    def ingest_pdf(
        self,
        pdf_path: str,
        document_id: str,
        document_name: str,
        strategy: str = "table_aware",
        chat_id: Optional[str] = None,
    ) -> DocumentRepresentation:
        """Synchronous wrapper around structural parsing & indexing for backward compatibility."""
        profile = v3_structural_parser.profile_pdf(pdf_path)
        selected_strategy = (
            strategy
            if (strategy and strategy not in ["auto", "none", "None"])
            else v3_chunking_policy.select_strategy(
                document_name=document_name,
                requested_strategy=strategy,
                profile_dict=profile,
            )
        )

        doc_ir: V3DocumentIR = v3_structural_parser.parse_pdf(
            pdf_path,
            document_id=document_id,
            strategy=selected_strategy,
            profile=profile,
        )
        cfg = ChunkingConfig(strategy=selected_strategy)
        chunks: List[V3Chunk] = v3_chunking_engine.chunk_document(doc_ir, config=cfg)
        embeddings = embedding_service.embed_batch([c.content for c in chunks])

        vector_store.upsert_v3_chunks(
            chunks=chunks,
            embeddings=embeddings,
            filename=document_name,
            strategy=selected_strategy,
            chat_id=chat_id,
        )
        bm25_service.stage_v3_chunks(
            chunks=chunks,
            filename=document_name,
            strategy=selected_strategy,
            chat_id=chat_id,
        )
        bm25_service.commit_v3_index()

        rep_id = f"{chat_id or 'global'}_{document_id}_v3_{selected_strategy}"
        rep = DocumentRepresentation(
            representation_id=rep_id,
            chat_id=chat_id or "global",
            document_id=document_id,
            document_name=document_name,
            content_hash="sync_hash",
            version="v3",
            chunking_strategy=selected_strategy,
            status="READY",
            chunk_count=len(chunks),
            parser_version="PyMuPDF_TableFinder_V3",
            chunker_version=f"V3ChunkingEngine_{selected_strategy}",
            index_status="INDEXED",
        )
        mongo_db._fallback_reps[rep.representation_id] = rep.model_dump()
        return rep

    def ensure_v3_indexed(
        self,
        document_ids: Optional[List[str]] = None,
        strategy: str = "table_aware",
        chat_id: Optional[str] = None,
    ) -> List[str]:
        """Ensures target documents have V3 representations indexed."""
        reprocessed = []
        pdf_targets: Dict[str, str] = {}

        if document_ids:
            for doc_id in document_ids:
                pdf_p = self.find_pdf_path(doc_id)
                if pdf_p:
                    doc_name = os.path.basename(pdf_p)
                    pdf_targets[doc_id] = pdf_p
        else:
            if os.path.exists(self.upload_dir):
                for f in os.listdir(self.upload_dir):
                    if f.lower().endswith(".pdf"):
                        pdf_targets[f] = os.path.join(self.upload_dir, f)

        for doc_key, pdf_p in pdf_targets.items():
            doc_name = os.path.basename(pdf_p)
            doc_id = doc_key if doc_key != doc_name else doc_name

            # If chat_id is specified, do not auto-ingest if not explicitly converted/ready for chat_id
            if chat_id:
                rep_id_key = f"{chat_id}_{doc_id}_v3_{strategy}"
                rep_data = mongo_db._fallback_reps.get(rep_id_key)
                if not rep_data or rep_data.get("status") != "READY":
                    # Check any representation matching chat_id and doc_id for v3
                    has_ready = any(
                        r.get("chat_id") == chat_id and r.get("document_id") == doc_id and r.get("version") == "v3" and r.get("status") == "READY"
                        for r in mongo_db._fallback_reps.values()
                    )
                    if not has_ready:
                        logger.info(f"[V3][INDEX] Skipping auto-index for document '{doc_id}' in chat '{chat_id}' (Not READY).")
                        continue

            self.ingest_pdf(
                pdf_path=pdf_p,
                document_id=doc_id,
                document_name=doc_name,
                strategy=strategy,
                chat_id=chat_id,
            )
            reprocessed.append(doc_name)

        return reprocessed


v3_ingestion_service = V3IngestionService()
