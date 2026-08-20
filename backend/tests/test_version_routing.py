import os
import json
import pytest
from app.core.pipeline_router import pipeline_router, PipelineRouter
from app.models.schemas import ChatQueryRequest
from app.evaluation.runner import RESULTS_DIR


def test_pipeline_router_version_registry():
    router = PipelineRouter()
    versions = router.list_versions()
    v_ids = [v.version_id for v in versions]

    assert "v1" in v_ids
    assert "v2.1" in v_ids
    assert "v2.2" in v_ids
    assert "v3" in v_ids


def test_version_resolution_aliases():
    router = PipelineRouter()
    assert router.get_version("v1").version_id == "v1"
    assert router.get_version("dense").version_id == "v1"
    assert router.get_version("v2.1").version_id == "v2.1"
    assert router.get_version("bm25").version_id == "v2.1"
    assert router.get_version("v2.2").version_id == "v2.2"
    assert router.get_version("hybrid").version_id == "v2.2"
    assert router.get_version("v3").version_id == "v3"


def test_invalid_version_rejection():
    router = PipelineRouter()
    with pytest.raises(ValueError, match="Unknown system version"):
        router.route_query(query="test", version="v99.9")


def test_v3_route_execution():
    """Verifies V3 pipeline route execution returns citations, breakdown, query_exp_meta, and calculation."""
    router = PipelineRouter()
    citations, breakdown, exp_meta, calc_dict = router.route_query(
        query="What was net PP&E in 2018?",
        top_k=2,
        version="v3",
        chunking_strategy="table_aware",
    )

    assert isinstance(citations, list)
    assert "total_request_ms" in breakdown
    assert "query_expansion_ms" in breakdown
    assert exp_meta is not None
    assert "detected_terms" in exp_meta
    assert "PP&E" in exp_meta["detected_terms"]


def test_chat_query_request_backward_compatibility():
    # Legacy Request
    req_legacy = ChatQueryRequest(query="What is revenue?", retrieval_mode="hybrid")
    assert req_legacy.version == "v2.2"
    assert req_legacy.retrieval_mode == "hybrid"

    # New Version Request
    req_v3 = ChatQueryRequest(query="What is CAPEX?", version="v3", chunking_strategy="semantic")
    assert req_v3.version == "v3"
    assert req_v3.chunking_strategy == "semantic"


def test_version_routing_preserves_frozen_v1_baseline():
    v1_latest_path = os.path.join(RESULTS_DIR, "latest.json")
    if os.path.exists(v1_latest_path):
        with open(v1_latest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("aggregate_recall_at_1") == 0.9167
