import os
import pytest
from app.core.config import settings
from app.v3.parsing.structural_parser import v3_structural_parser
from app.v3.chunking.engine import v3_chunking_engine
from app.v3.schemas.chunk_schema import ChunkingConfig
from app.v3.reasoning.fact_extractor import fact_extractor
from app.v3.reasoning.calculator import calculation_engine
from app.v3.reasoning.schemas import FinancialFact
from app.models.schemas import SourceCitation


def test_v3_financebench_3m_integration():
    """
    Integration test processing FinanceBench PDF (3M_2018_10K.pdf) through
    V3.1 PDF Structural Parser -> V3.2 Chunking Engine -> V3.4 Fact Extractor & Calculator.
    """
    pdf_dir = settings.FINANCEBENCH_PDF_DIR
    target_pdf = os.path.join(pdf_dir, "3M_2018_10K.pdf")

    if not os.path.exists(target_pdf):
        pytest.skip(f"FinanceBench PDF not found at {target_pdf}")

    # 1. V3.1 Structural Parsing
    doc_ir = v3_structural_parser.parse_pdf(target_pdf, document_id="3M_2018_10K.pdf")
    assert doc_ir is not None
    assert doc_ir.total_pages > 0
    assert doc_ir.metadata.get("total_tables", 0) > 0

    # 2. V3.2 Chunking Engine (Table-Aware)
    chunks = v3_chunking_engine.chunk_document(
        doc_ir, config=ChunkingConfig(strategy="table_aware")
    )
    assert len(chunks) > 0

    # Convert V3 chunks into SourceCitations for fact extraction test
    citations = [
        SourceCitation(
            document_id=c.document_id,
            document_name=c.document_name,
            chunk_index=idx,
            content=c.content,
            score=1.0,
            page_number=c.page_start or 1,
            section=c.section or "Item 8",
            chunk_id=c.chunk_id,
        )
        for idx, c in enumerate(chunks[:50])
    ]

    # 3. V3.4 Fact Extraction
    facts = fact_extractor.extract_facts_from_citations(
        citations=citations,
        target_years=[2018, 2017],
    )
    assert isinstance(facts, list)

    # 4. Deterministic Calculation Test with Facts
    # Inject 3M 2018/2017 Grounded Facts
    fact_ni = FinancialFact(
        fact_id="f-3m-ni-2018",
        concept="net_income",
        value=3848.0,
        raw_text="3,848",
        unit="USD",
        scale=1e6,
        currency="USD",
        period="2018",
        fiscal_year=2018,
        source_document="3M_2018_10K.pdf",
        page_number=62,
    )
    fact_ast_2018 = FinancialFact(
        fact_id="f-3m-ast-2018",
        concept="total_assets",
        value=36500.0,
        raw_text="36,500",
        unit="USD",
        scale=1e6,
        currency="USD",
        period="2018",
        fiscal_year=2018,
        source_document="3M_2018_10K.pdf",
        page_number=63,
    )
    fact_ast_2017 = FinancialFact(
        fact_id="f-3m-ast-2017",
        concept="total_assets",
        value=35000.0,
        raw_text="35,000",
        unit="USD",
        scale=1e6,
        currency="USD",
        period="2017",
        fiscal_year=2017,
        source_document="3M_2018_10K.pdf",
        page_number=63,
    )

    calc_res = calculation_engine.calculate(
        metric_id="ROA",
        facts=[fact_ni, fact_ast_2018, fact_ast_2017],
        target_years=[2018],
    )

    assert calc_res.validation_status == "VALIDATED"
    assert calc_res.result is not None
    assert calc_res.display_result.endswith("%")
    assert len(calc_res.steps) == 2
    assert len(calc_res.source_facts) == 3
