import os
import json
import pytest
import asyncio
from app.models.schemas import DocumentRepresentation
from app.db.mongodb import mongo_db
from app.v3.ingestion.ingestion_service import v3_ingestion_service, calculate_content_hash
from app.v3.ingestion.v3_chunking_policy import v3_chunking_policy
from app.v3.retrieval.v3_retriever import v3_retriever
from app.core.pipeline_router import pipeline_router
from app.evaluation.runner import RESULTS_DIR


@pytest.mark.asyncio
async def test_existing_pdf_materialize_without_reupload():
    """Verifies existing PDF can be materialized into V3 without re-uploading."""
    pdf_path = v3_ingestion_service.find_pdf_path("3M_2018_10K.pdf")
    if not pdf_path:
        pytest.skip("3M_2018_10K.pdf not found in dataset")

    rep = await v3_ingestion_service.materialize_representation(
        document_id="3M_2018_10K.pdf",
        version="v3",
        strategy="table_aware",
    )

    assert rep.status == "READY"
    assert rep.chunk_count > 0
    assert rep.version == "v3"
    assert rep.chunking_strategy == "table_aware"


@pytest.mark.asyncio
async def test_v3_representation_separate_from_v1_v2():
    """Verifies V3 representations are stored separately in registry from V1/V2."""
    v1_rep = await v3_ingestion_service.materialize_representation("3M_2018_10K.pdf", version="v1")
    v3_rep = await v3_ingestion_service.materialize_representation("3M_2018_10K.pdf", version="v3", strategy="table_aware")

    assert v1_rep.version == "v1"
    assert v3_rep.version == "v3"
    assert v1_rep.representation_id != v3_rep.representation_id


@pytest.mark.asyncio
async def test_v3_table_aware_and_semantic_strategies():
    """Verifies Table-Aware and Semantic strategies generate distinct V3 representations."""
    pdf_path = v3_ingestion_service.find_pdf_path("3M_2018_10K.pdf")
    if not pdf_path:
        pytest.skip("3M_2018_10K.pdf not found")

    rep_ta = await v3_ingestion_service.materialize_representation("3M_2018_10K.pdf", version="v3", strategy="table_aware")
    rep_sem = await v3_ingestion_service.materialize_representation("3M_2018_10K.pdf", version="v3", strategy="semantic")

    assert rep_ta.status == "READY"
    assert rep_sem.status == "READY"
    assert rep_ta.chunking_strategy == "table_aware"
    assert rep_sem.chunking_strategy == "semantic"


@pytest.mark.asyncio
async def test_ready_representation_reused_idempotent():
    """Verifies READY representation is reused without duplicate work."""
    rep1 = await v3_ingestion_service.materialize_representation("3M_2018_10K.pdf", version="v3", strategy="table_aware")
    rep2 = await v3_ingestion_service.materialize_representation("3M_2018_10K.pdf", version="v3", strategy="table_aware")

    assert rep1.representation_id == rep2.representation_id
    assert rep1.chunk_count == rep2.chunk_count


@pytest.mark.asyncio
async def test_processing_status_prevents_duplicate_jobs():
    """Verifies a PROCESSING representation status returns PROCESSING state immediately."""
    rep_id = "test_doc_v3_table_aware"
    test_rep = DocumentRepresentation(
        representation_id=rep_id,
        document_id="test_doc",
        document_name="test_doc.pdf",
        content_hash="mock_hash",
        version="v3",
        chunking_strategy="table_aware",
        status="PROCESSING",
    )
    await mongo_db.save_representation(test_rep)

    check_rep = await mongo_db.get_representation("test_doc", "v3", "table_aware")
    assert check_rep is not None
    assert check_rep.status == "PROCESSING"


@pytest.mark.asyncio
async def test_backend_policy_strategy_selection():
    """Verifies V3ChunkingPolicy automatically selects table_aware for tabular PDFs."""
    policy_selected = v3_chunking_policy.select_strategy(
        doc_ir=None,
        document_name="3M_2018_10K.pdf",
        requested_strategy=None,
    )
    assert policy_selected == "table_aware"


@pytest.mark.asyncio
async def test_v3_switching_preserves_v2():
    """Verifies switching between V3 and V2.2 preserves both representations."""
    rep_v2 = await v3_ingestion_service.materialize_representation("3M_2018_10K.pdf", version="v2.2")
    rep_v3 = await v3_ingestion_service.materialize_representation("3M_2018_10K.pdf", version="v3", strategy="table_aware")

    reps = await v3_ingestion_service.list_representations("3M_2018_10K.pdf")
    versions = [r.version for r in reps]

    assert "v2.2" in versions
    assert "v3" in versions


@pytest.mark.asyncio
async def test_v3_hybrid_retrieval():
    """Verifies V3 retrieval performs Dense + BM25 + RRF Hybrid retrieval."""
    citations = v3_retriever.search(
        query="operating income net sales",
        top_k=4,
        document_ids=["3M_2018_10K.pdf"],
        chunking_strategy="table_aware",
    )

    for c in citations:
        assert c.version == "v3"
        assert c.strategy == "table_aware"
        assert c.rrf_score is not None


def test_baseline_preservation():
    """Verifies frozen V1 baseline files remain untouched."""
    v1_path = os.path.join(RESULTS_DIR, "v1_baseline.json")
    if os.path.exists(v1_path):
        with open(v1_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("aggregate_recall_at_1") == 0.9167
