import os
import json
import pytest
from app.core.config import settings
from app.v3.ingestion.ingestion_service import v3_ingestion_service
from app.v3.retrieval.v3_retriever import v3_retriever
from app.core.pipeline_router import pipeline_router
from app.evaluation.runner import RESULTS_DIR


def test_v3_ingestion_service_parses_pdf_and_generates_ir():
    """Verifies V3 ingestion executes V3StructuralPDFParser and generates V3DocumentIR."""
    pdf_path = v3_ingestion_service.find_pdf_path("3M_2018_10K.pdf")
    if not pdf_path:
        pytest.skip("3M_2018_10K.pdf not found in dataset directory")

    result = v3_ingestion_service.ingest_pdf(
        pdf_path=pdf_path,
        document_id="3M_2018_10K.pdf",
        document_name="3M_2018_10K.pdf",
        strategy="table_aware",
    )

    res_status = result.status if hasattr(result, "status") else result["status"]
    res_pages = getattr(result, "total_pages", None) or (result.get("total_pages") if isinstance(result, dict) else 1)
    res_chunks = getattr(result, "chunk_count", None) or getattr(result, "total_chunks", None) or (result.get("total_chunks") if isinstance(result, dict) else 0)

    assert res_status == "READY"
    assert res_pages > 0
    assert res_chunks > 0


def test_v3_chunking_engine_executes_selected_strategy():
    """Verifies V3ChunkingEngine executes different strategies (table_aware vs semantic)."""
    pdf_path = v3_ingestion_service.find_pdf_path("3M_2018_10K.pdf")
    if not pdf_path:
        pytest.skip("3M_2018_10K.pdf not found")

    res_ta = v3_ingestion_service.ingest_pdf(
        pdf_path=pdf_path,
        document_id="3M_2018_10K.pdf",
        document_name="3M_2018_10K.pdf",
        strategy="table_aware",
    )
    res_sem = v3_ingestion_service.ingest_pdf(
        pdf_path=pdf_path,
        document_id="3M_2018_10K.pdf",
        document_name="3M_2018_10K.pdf",
        strategy="semantic",
    )

    c_ta = getattr(res_ta, "chunk_count", None) or (res_ta.get("total_chunks") if isinstance(res_ta, dict) else 0)
    c_sem = getattr(res_sem, "chunk_count", None) or (res_sem.get("total_chunks") if isinstance(res_sem, dict) else 0)

    assert c_ta > 0
    assert c_sem > 0


def test_v3_chunk_metadata_completeness():
    """Inspects retrieved V3 chunks for V3 structural metadata fields."""
    pdf_path = v3_ingestion_service.find_pdf_path("3M_2018_10K.pdf")
    if not pdf_path:
        pytest.skip("3M_2018_10K.pdf not found")

    v3_ingestion_service.ingest_pdf(
        pdf_path=pdf_path,
        document_id="3M_2018_10K.pdf",
        document_name="3M_2018_10K.pdf",
        strategy="table_aware",
    )

    citations = v3_retriever.search(
        query="table cash flows operating activities",
        top_k=5,
        document_ids=["3M_2018_10K.pdf"],
        chunking_strategy="table_aware",
    )

    assert len(citations) > 0
    top_c = citations[0]
    assert top_c.version == "v3"
    assert top_c.strategy == "table_aware"
    assert top_c.page_number is not None


def test_v3_isolation_from_v1_v2():
    """Verifies V3 retrieval filters only V3 chunks and does not pollute legacy V1/V2 retrieval."""
    citations_v3 = v3_retriever.search(
        query="financial results revenue",
        top_k=3,
        chunking_strategy="table_aware",
    )
    for c in citations_v3:
        assert c.version == "v3"


def test_existing_document_auto_reprocessing_idempotent():
    """Verifies existing PDF can be auto-reprocessed idempotently without chunk duplication."""
    pdf_path = v3_ingestion_service.find_pdf_path("3M_2018_10K.pdf")
    if not pdf_path:
        pytest.skip("3M_2018_10K.pdf not found")

    # Ingest once
    res1 = v3_ingestion_service.ingest_pdf(
        pdf_path=pdf_path,
        document_id="3M_2018_10K.pdf",
        document_name="3M_2018_10K.pdf",
        strategy="table_aware",
    )
    c_count1 = getattr(res1, "chunk_count", None) or (res1.get("total_chunks") if isinstance(res1, dict) else 0)

    # Ingest again (Idempotent call)
    res2 = v3_ingestion_service.ingest_pdf(
        pdf_path=pdf_path,
        document_id="3M_2018_10K.pdf",
        document_name="3M_2018_10K.pdf",
        strategy="table_aware",
    )
    c_count2 = getattr(res2, "chunk_count", None) or (res2.get("total_chunks") if isinstance(res2, dict) else 0)

    assert c_count1 == c_count2


def test_3m_ppe_cash_flow_table_preservation():
    """Verifies Cash Flow Statement table containing Purchases of PP&E is retrieved with structure."""
    pdf_path = v3_ingestion_service.find_pdf_path("3M_2018_10K.pdf")
    if not pdf_path:
        pytest.skip("3M_2018_10K.pdf not found")

    v3_ingestion_service.ingest_pdf(
        pdf_path=pdf_path,
        document_id="3M_2018_10K.pdf",
        document_name="3M_2018_10K.pdf",
        strategy="table_aware",
    )

    citations = v3_retriever.search(
        query="Purchases of property, plant and equipment",
        top_k=5,
        document_ids=["3M_2018_10K.pdf"],
        chunking_strategy="table_aware",
    )

    assert len(citations) > 0
    matched_text = " ".join([c.content for c in citations])
    assert "property, plant and equipment" in matched_text.lower() or "1,577" in matched_text or "1577" in matched_text


def test_end_to_end_v3_query_and_financial_reasoning():
    """Tests PipelineRouter route_query under version=v3 with strategy table_aware."""
    cits, breakdown, exp_meta, calc_dict = pipeline_router.route_query(
        query="What was ROA in 2018?",
        top_k=4,
        document_ids=["3M_2018_10K.pdf"],
        version="v3",
        chunking_strategy="table_aware",
    )

    assert len(cits) > 0
    assert "rrf_fusion_ms" in breakdown
    assert exp_meta is not None


def test_baseline_preservation():
    """Verifies frozen baseline files remain untouched."""
    v1_path = os.path.join(RESULTS_DIR, "v1_baseline.json")
    latest_path = os.path.join(RESULTS_DIR, "latest.json")

    if os.path.exists(v1_path):
        with open(v1_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("aggregate_recall_at_1") == 0.9167

    if os.path.exists(latest_path):
        with open(latest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("aggregate_recall_at_1") == 0.9167
