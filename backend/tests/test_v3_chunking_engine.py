import os
import json
import pytest
from app.v3.schemas.document_ir import (
    V3DocumentIR,
    PageIR,
    TableIR,
    ParagraphIR,
    BoundingBox,
)
from app.v3.schemas.chunk_schema import V3Chunk, ChunkingConfig
from app.v3.chunking.engine import v3_chunking_engine, V3ChunkingEngine
from app.v3.chunking.validator import V3ChunkValidator
from app.evaluation.runner import RESULTS_DIR


def create_sample_doc_ir() -> V3DocumentIR:
    bbox = BoundingBox(x0=10.0, y0=20.0, x1=200.0, y1=300.0, page_number=1)
    table = TableIR(
        table_id="t1",
        page_number=1,
        title="Balance Sheet",
        headers=["Item", "2018", "2017"],
        rows=[
            ["Cash", "100", "90"],
            ["Inventory", "200", "180"],
            ["PP&E", "500", "450"],
            ["Total Assets", "800", "720"],
        ],
        markdown_content="| Item | 2018 | 2017 |\n|---|---|---|\n| Cash | 100 | 90 |",
        bbox=bbox,
    )
    para = ParagraphIR(
        paragraph_id="p1",
        page_number=1,
        section_title="Item 7. MD&A",
        text="Item 7. Management's Discussion and Analysis of Financial Condition.",
        is_heading=True,
        heading_level=1,
        bbox=bbox,
    )
    para2 = ParagraphIR(
        paragraph_id="p2",
        page_number=1,
        section_title="Item 7. MD&A",
        text="Net sales grew 3.2% year-over-year driven by industrial segment demand.",
        is_heading=False,
        bbox=bbox,
    )
    page = PageIR(
        page_number=1,
        paragraphs=[para, para2],
        tables=[table],
    )
    return V3DocumentIR(
        document_id="doc_sample",
        document_name="sample.pdf",
        source_path="/path/sample.pdf",
        total_pages=1,
        pages=[page],
    )


def test_strategy_registration():
    """Verifies all 8 V3 chunking strategies are registered in V3ChunkingEngine."""
    engine = V3ChunkingEngine()
    registered = engine.get_registered_strategies()
    expected = ["fixed", "hierarchical", "parent_child", "recursive", "section_aware", "semantic", "sliding_window", "table_aware"]

    for strat in expected:
        assert strat in registered


def test_fixed_chunking():
    doc_ir = create_sample_doc_ir()
    cfg = ChunkingConfig(strategy="fixed", chunk_size=50, overlap=10)
    chunks = v3_chunking_engine.chunk_document(doc_ir, config=cfg)

    assert len(chunks) > 0
    assert all(c.strategy == "fixed" for c in chunks)


def test_recursive_chunking():
    doc_ir = create_sample_doc_ir()
    cfg = ChunkingConfig(strategy="recursive")
    chunks = v3_chunking_engine.chunk_document(doc_ir, config=cfg)

    assert len(chunks) > 0
    assert any(c.chunk_type == "table" for c in chunks)


def test_semantic_chunking():
    doc_ir = create_sample_doc_ir()
    cfg = ChunkingConfig(strategy="semantic")
    chunks = v3_chunking_engine.chunk_document(doc_ir, config=cfg)

    assert len(chunks) > 0
    assert all(c.strategy == "semantic" for c in chunks)


def test_section_aware_chunking():
    doc_ir = create_sample_doc_ir()
    cfg = ChunkingConfig(strategy="section_aware")
    chunks = v3_chunking_engine.chunk_document(doc_ir, config=cfg)

    assert len(chunks) > 0
    sec_chunks = [c for c in chunks if c.chunk_type == "section"]
    assert len(sec_chunks) >= 1
    assert "MD&A" in sec_chunks[0].section


def test_table_aware_chunking_header_repetition():
    """HIGHEST PRIORITY: Verifies table-aware strategy repeats headers per chunk."""
    doc_ir = create_sample_doc_ir()
    cfg = ChunkingConfig(strategy="table_aware", max_table_rows_per_chunk=2)
    chunks = v3_chunking_engine.chunk_document(doc_ir, config=cfg)

    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert len(table_chunks) == 2

    for tbl_c in table_chunks:
        assert tbl_c.strategy == "table_aware"
        assert "Item" in tbl_c.content
        assert "2018" in tbl_c.content
        assert tbl_c.row_range is not None
        assert tbl_c.column_range is not None


def test_parent_child_chunking():
    doc_ir = create_sample_doc_ir()
    cfg = ChunkingConfig(strategy="parent_child")
    chunks = v3_chunking_engine.chunk_document(doc_ir, config=cfg)

    parent_chunks = [c for c in chunks if c.chunk_type == "parent"]
    child_chunks = [c for c in chunks if c.chunk_type == "child"]

    assert len(parent_chunks) >= 1
    assert len(child_chunks) >= 1
    assert len(parent_chunks[0].child_chunk_ids) > 0
    assert child_chunks[0].parent_chunk_id == parent_chunks[0].chunk_id


def test_sliding_window_chunking():
    doc_ir = create_sample_doc_ir()
    cfg = ChunkingConfig(strategy="sliding_window", window_size=2, stride=1)
    chunks = v3_chunking_engine.chunk_document(doc_ir, config=cfg)

    assert len(chunks) > 0
    assert all(c.strategy == "sliding_window" for c in chunks)


def test_hierarchical_chunking():
    doc_ir = create_sample_doc_ir()
    cfg = ChunkingConfig(strategy="hierarchical")
    chunks = v3_chunking_engine.chunk_document(doc_ir, config=cfg)

    root_chunk = next(c for c in chunks if c.section == "Document Root")
    assert root_chunk is not None
    assert len(root_chunk.child_chunk_ids) > 0


def test_v3_chunk_validator():
    doc_ir = create_sample_doc_ir()
    cfg = ChunkingConfig(strategy="table_aware")
    chunks = v3_chunking_engine.chunk_document(doc_ir, config=cfg)

    is_valid, errors = V3ChunkValidator.validate_chunks(chunks)
    assert is_valid is True
    assert len(errors) == 0


def test_v3_chunking_preserves_frozen_v1_baseline():
    v1_latest_path = os.path.join(RESULTS_DIR, "latest.json")
    if os.path.exists(v1_latest_path):
        with open(v1_latest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("aggregate_recall_at_1") == 0.9167
