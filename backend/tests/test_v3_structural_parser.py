import os
import json
import pytest
from app.v3.schemas.document_ir import (
    V3DocumentIR,
    PageIR,
    TableIR,
    ParagraphIR,
    FootnoteIR,
    BoundingBox,
)
from app.v3.parsing.structural_parser import V3StructuralPDFParser, v3_structural_parser
from app.evaluation.runner import RESULTS_DIR


FINANCEBENCH_PDF_PATH = r"C:\Abdullah files\datasets\financebench\pdfs\3M_2018_10K.pdf"


def test_v3_ir_schemas_and_chunk_candidates():
    """Verifies V3 Intermediate Representation models and chunk candidate generation."""
    bbox = BoundingBox(x0=10.0, y0=20.0, x1=100.0, y1=200.0, page_number=1)
    table = TableIR(
        table_id="tab_1",
        page_number=1,
        title="Consolidated Statements of Cash Flows",
        headers=["Item", "2018", "2017", "2016"],
        rows=[["Purchases of PP&E", "(1,577)", "(1,373)", "(1,420)"]],
        markdown_content="| Item | 2018 |\n|---|---|\n| Purchases of PP&E | (1,577) |",
        bbox=bbox,
    )

    page = PageIR(
        page_number=1,
        paragraphs=[
            ParagraphIR(
                paragraph_id="p_1",
                page_number=1,
                section_title="Item 7",
                text="Item 7. Management's Discussion and Analysis",
                is_heading=True,
                heading_level=1,
                bbox=bbox,
            )
        ],
        tables=[table],
        footnotes=[
            FootnoteIR(
                footnote_id="fn_1",
                page_number=1,
                marker="*",
                text="* Non-GAAP measure",
                bbox=bbox,
            )
        ],
    )

    doc_ir = V3DocumentIR(
        document_id="doc_test",
        document_name="test.pdf",
        source_path="/path/test.pdf",
        total_pages=1,
        pages=[page],
    )

    assert doc_ir.total_pages == 1
    assert len(doc_ir.pages[0].tables) == 1
    assert doc_ir.pages[0].tables[0].headers == ["Item", "2018", "2017", "2016"]

    candidates = doc_ir.to_chunk_candidates()
    assert len(candidates) >= 3

    # Verify Table Chunk Candidate
    tab_cand = next(c for c in candidates if c["chunk_type"] == "table")
    assert tab_cand["table_id"] == "tab_1"
    assert tab_cand["page_number"] == 1
    assert "Purchases of PP&E" in tab_cand["content"]
    assert tab_cand["row_range"] == [0, 1]
    assert tab_cand["column_range"] == [0, 4]


@pytest.mark.skipif(not os.path.exists(FINANCEBENCH_PDF_PATH), reason="FinanceBench source PDF not available")
def test_v3_structural_pdf_parser_on_financebench_pdf():
    """Validates real V3 PDF table extraction on FinanceBench 3M_2018_10K.pdf."""
    parser = V3StructuralPDFParser()
    doc_ir = parser.parse_pdf(FINANCEBENCH_PDF_PATH, document_id="3M_2018_10K")

    assert doc_ir.total_pages == 160
    assert len(doc_ir.pages) == 160
    assert doc_ir.metadata["total_tables"] > 0

    # Inspect Page 49 (Cash Flow / Free Cash Flow Table)
    page_49 = doc_ir.pages[48]  # 0-indexed index for page 49
    assert page_49.page_number == 49
    assert len(page_49.tables) >= 1

    # Locate Cash Flow Purchases of PP&E Table
    found_table = False
    for tab in page_49.tables:
        md_text = tab.markdown_content
        if "Purchases of property, plant and equipment" in md_text:
            found_table = True
            assert "(1,577)" in md_text or "1,577" in md_text
            assert tab.bbox is not None
            assert tab.bbox.page_number == 49
            assert tab.bbox.x0 >= 0
            break

    assert found_table is True, "Expected table containing Purchases of PP&E on Page 49 of 3M 2018 10-K"


def test_v3_structural_parser_preserves_frozen_v1_baseline():
    """Ensures Sprint 3.1 preserves frozen V1 baseline latest.json untouched."""
    v1_latest_path = os.path.join(RESULTS_DIR, "latest.json")
    if os.path.exists(v1_latest_path):
        with open(v1_latest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("aggregate_recall_at_1") == 0.9167
