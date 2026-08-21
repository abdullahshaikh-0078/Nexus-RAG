import os
import re
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple

import pymupdf  # PyMuPDF

from app.v3.schemas.document_ir import (
    V3DocumentIR,
    PageIR,
    TableIR,
    TableCellIR,
    ParagraphIR,
    FootnoteIR,
    FigureIR,
    BoundingBox,
)

logger = logging.getLogger(__name__)

# Heuristic regex patterns for Section Headings
HEADING_PATTERNS = [
    re.compile(r"^\s*Item\s+\d+[A-Z]?[\.\:]?\s+.*", re.IGNORECASE),
    re.compile(r"^\s*Part\s+[I|V|X]+[\.\:]?\s+.*", re.IGNORECASE),
    re.compile(r"^\s*Consolidated\s+Balance\s+Sheets?.*", re.IGNORECASE),
    re.compile(r"^\s*Consolidated\s+Statements?\s+of\s+.*", re.IGNORECASE),
    re.compile(r"^\s*Note\s+\d+[\.\:]?\s+.*", re.IGNORECASE),
]

FOOTNOTE_MARKERS = [
    re.compile(r"^\s*\(\d+\)\s+.*"),
    re.compile(r"^\s*\*\s+.*"),
    re.compile(r"^\s*Note:\s+.*", re.IGNORECASE),
    re.compile(r"^\s*\d+\s+[A-Z].*"),
]


class V3StructuralPDFParser:
    """
    V3 Structural PDF Parser converting PDFs into layout-aware V3DocumentIR.
    Preserves page boundaries, sections, table structures, cell bounding boxes, and footnotes.
    Does NOT modify V1/V2 ingestion pipeline.
    """

    def __init__(self):
        self._cache: Dict[str, V3DocumentIR] = {}

    def profile_pdf(self, source_path: str) -> Dict[str, Any]:
        """
        Lightweight document profiling pass running before expensive structural parsing.
        Collects cheap metrics (page count, text length, numeric density, table likelihood)
        without calling PyMuPDF find_tables().
        """
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"PDF file not found at: {source_path}")

        import time
        from app.v3.ingestion.v3_profiler import v3_profiler

        t_prof_start = time.perf_counter()
        doc = pymupdf.open(source_path)
        doc_name = os.path.basename(source_path)
        total_pages = len(doc)

        total_text_length = 0
        paragraph_count_est = 0
        heading_count_est = 0
        digit_count = 0
        char_count = 0
        table_candidate_pages: List[int] = []

        financial_keywords = [
            "balance sheet",
            "statement of operations",
            "cash flows",
            "consolidated",
            "10-k",
            "financial report",
            "income statement",
            "rag test doc",
        ]
        is_financial = any(kw in doc_name.lower() for kw in financial_keywords)

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            page = doc[page_idx]

            text = page.get_text("text") or ""
            text_len = len(text)
            total_text_length += text_len
            char_count += text_len

            if text_len > 0:
                digits = sum(1 for c in text if c.isdigit())
                digit_count += digits

                # Fast estimate without expensive line splits
                paragraph_count_est += max(1, text_len // 300)

                text_lower = text.lower()
                import re
                has_table_keyword = any(
                    kw in text_lower
                    for kw in [
                        "balance sheet",
                        "statement of operations",
                        "statement of cash",
                        "consolidated statements",
                        "net sales",
                        "total assets",
                        "in millions",
                        "in thousands",
                    ]
                ) or bool(re.search(r"\btable\s+\d+", text_lower))
                has_pipe = "|" in text

                if has_table_keyword or has_pipe:
                    table_candidate_pages.append(page_num)

        doc.close()

        avg_text_len = total_text_length / max(total_pages, 1)
        numeric_density = digit_count / max(char_count, 1)

        profile = {
            "document_name": doc_name,
            "page_count": total_pages,
            "total_text_length": total_text_length,
            "average_text_length_per_page": round(avg_text_len, 2),
            "paragraph_count_est": paragraph_count_est,
            "heading_count_est": heading_count_est,
            "numeric_density": round(numeric_density, 4),
            "table_candidate_pages": table_candidate_pages,
            "table_candidate_count": len(table_candidate_pages),
            "is_financial": is_financial,
        }

        t_prof = time.perf_counter() - t_prof_start
        v3_profiler.metrics["v3_policy_selection_s"] += t_prof
        logger.info(
            f"V3 Cheap Document Profiler for '{doc_name}': {total_pages} pages, "
            f"{len(table_candidate_pages)} table candidate pages in {t_prof:.4f}s"
        )
        return profile

    def parse_pdf(
        self,
        source_path: str,
        document_id: Optional[str] = None,
        strategy: Optional[str] = None,
        profile: Optional[Dict[str, Any]] = None,
    ) -> V3DocumentIR:
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"PDF file not found at: {source_path}")

        abs_path = os.path.abspath(source_path)
        if abs_path in self._cache:
            cached_ir = self._cache[abs_path]
            if document_id:
                cached_ir.document_id = document_id
            return cached_ir

        if len(self._cache) >= 2:
            self._cache.clear()

        doc_name = os.path.basename(source_path)
        doc_id = document_id or f"doc_{uuid.uuid4().hex[:10]}"

        import time
        from app.v3.ingestion.v3_profiler import v3_profiler

        t_load_start = time.perf_counter()
        doc = pymupdf.open(source_path)
        total_pages = len(doc)
        v3_profiler.metrics["source_pdf_loading_s"] += time.perf_counter() - t_load_start

        logger.info(f"V3 Structural Parser: Processing '{doc_name}' ({total_pages} pages) strategy={strategy}...")

        t_parse_start = time.perf_counter()
        pages_ir: List[PageIR] = []
        current_section = "Document Header"
        tables_passed_count = 0

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            page = doc[page_idx]

            page_ir, current_section, passed_table_scan = self._parse_single_page(
                page=page,
                page_num=page_num,
                doc_id=doc_id,
                current_section=current_section,
                strategy=strategy,
                profile=profile,
            )
            pages_ir.append(page_ir)
            if passed_table_scan:
                tables_passed_count += 1

        doc.close()
        v3_profiler.metrics["pymupdf_structural_parsing_s"] += time.perf_counter() - t_parse_start

        t_ir_start = time.perf_counter()
        v3_doc = V3DocumentIR(
            document_id=doc_id,
            document_name=doc_name,
            source_path=source_path,
            total_pages=total_pages,
            pages=pages_ir,
            metadata={
                "parser_engine": "PyMuPDF_TableFinder_V3",
                "total_tables": sum(len(p.tables) for p in pages_ir),
                "total_paragraphs": sum(len(p.paragraphs) for p in pages_ir),
                "total_footnotes": sum(len(p.footnotes) for p in pages_ir),
                "pages_passed_to_find_tables": tables_passed_count,
                "strategy": strategy or "auto",
            },
        )
        v3_profiler.metrics["v3_doc_ir_construction_s"] += time.perf_counter() - t_ir_start

        self._cache[abs_path] = v3_doc

        logger.info(
            f"V3 Parser Complete for '{doc_name}': {v3_doc.metadata['total_tables']} tables, "
            f"{v3_doc.metadata['total_paragraphs']} paragraphs ({tables_passed_count}/{total_pages} pages scanned for tables)."
        )
        return v3_doc

    def _parse_single_page(
        self,
        page: pymupdf.Page,
        page_num: int,
        doc_id: str,
        current_section: str,
        strategy: Optional[str] = None,
        profile: Optional[Dict[str, Any]] = None,
    ) -> Tuple[PageIR, str, bool]:
        page_rect = page.rect
        page_height = page_rect.height

        tables_ir: List[TableIR] = []
        table_bboxes: List[Tuple[float, float, float, float]] = []

        should_detect_tables = (
            (strategy in ["table_aware", None] and profile and page_num in profile.get("table_candidate_pages", []))
            or (not profile and (strategy == "table_aware" or not strategy))
        )

        passed_table_scan = False
        if should_detect_tables:
            passed_table_scan = True
            try:
                tab_finder = page.find_tables()
                for t_idx, tab in enumerate(tab_finder.tables, 1):
                    raw_bbox = tab.bbox
                    table_bboxes.append(raw_bbox)
                    bbox_obj = BoundingBox(
                        x0=float(raw_bbox[0]),
                        y0=float(raw_bbox[1]),
                        x1=float(raw_bbox[2]),
                        y1=float(raw_bbox[3]),
                        page_number=page_num,
                    )

                    tab_id = f"{doc_id}_p{page_num}_table_{t_idx}"
                    markdown_str = tab.to_markdown()

                    # Extract headers and rows
                    tab_extract = tab.extract()
                    headers = [str(cell or "").strip() for cell in tab_extract[0]] if tab_extract else []
                    rows = []
                    if len(tab_extract) > 1:
                        for r in tab_extract[1:]:
                            rows.append([str(cell or "").strip() for cell in r])

                    cells_ir: List[TableCellIR] = []
                    for r_idx, row in enumerate(tab_extract):
                        for c_idx, val in enumerate(row):
                            cells_ir.append(
                                TableCellIR(
                                    row_index=r_idx,
                                    col_index=c_idx,
                                    content=str(val or "").strip(),
                                    is_header=(r_idx == 0),
                                    bbox=bbox_obj,
                                )
                            )

                    tables_ir.append(
                        TableIR(
                            table_id=tab_id,
                            page_number=page_num,
                            title=f"Page {page_num} Table {t_idx}",
                            headers=headers,
                            rows=rows,
                            cells=cells_ir,
                            markdown_content=markdown_str,
                            bbox=bbox_obj,
                        )
                    )
            except Exception as e:
                logger.warning(f"Table detection warning on page {page_num}: {e}")

        # 2. Extract Text Blocks & Footnotes
        paragraphs_ir: List[ParagraphIR] = []
        footnotes_ir: List[FootnoteIR] = []
        header_text_lines = []
        footer_text_lines = []

        blocks = page.get_text("blocks")
        # Block format: (x0, y0, x1, y1, text, block_no, block_type)

        for b_idx, block in enumerate(blocks, 1):
            x0, y0, x1, y1, text_content, block_no, block_type = block
            text = text_content.strip()

            if not text:
                continue

            # Skip blocks that overlap inside extracted tables
            if self._is_inside_any_table((x0, y0, x1, y1), table_bboxes):
                continue

            bbox_obj = BoundingBox(
                x0=float(x0),
                y0=float(y0),
                x1=float(x1),
                y1=float(y1),
                page_number=page_num,
            )

            # Check top/bottom margin for page header/footer
            is_hdr_ftr = (y0 < page_height * 0.05) or (y1 > page_height * 0.95)
            if is_hdr_ftr:
                if y0 < page_height * 0.05:
                    header_text_lines.append(text)
                else:
                    footer_text_lines.append(text)

            # Check if section heading
            is_heading = False
            heading_lvl = 0
            for pat in HEADING_PATTERNS:
                if pat.match(text):
                    is_heading = True
                    heading_lvl = 1
                    current_section = text[:120].strip()
                    break

            # Check footnote
            is_footnote = False
            if y0 > page_height * 0.75:
                for fn_pat in FOOTNOTE_MARKERS:
                    if fn_pat.match(text):
                        is_footnote = True
                        footnotes_ir.append(
                            FootnoteIR(
                                footnote_id=f"{doc_id}_p{page_num}_fn_{len(footnotes_ir)+1}",
                                page_number=page_num,
                                marker=text.split()[0],
                                text=text,
                                bbox=bbox_obj,
                            )
                        )
                        break

            paragraphs_ir.append(
                ParagraphIR(
                    paragraph_id=f"{doc_id}_p{page_num}_p_{b_idx}",
                    page_number=page_num,
                    section_title=current_section,
                    text=text,
                    is_heading=is_heading,
                    heading_level=heading_lvl,
                    is_footnote=is_footnote,
                    is_header_footer=is_hdr_ftr,
                    bbox=bbox_obj,
                )
            )

        page_ir = PageIR(
            page_number=page_num,
            paragraphs=paragraphs_ir,
            tables=tables_ir,
            footnotes=footnotes_ir,
            figures=[],
            header_text=" ".join(header_text_lines) if header_text_lines else None,
            footer_text=" ".join(footer_text_lines) if footer_text_lines else None,
        )

        return page_ir, current_section, passed_table_scan

    def _is_inside_any_table(
        self,
        block_bbox: Tuple[float, float, float, float],
        table_bboxes: List[Tuple[float, float, float, float]],
    ) -> bool:
        bx0, by0, bx1, by1 = block_bbox
        for tx0, ty0, tx1, ty1 in table_bboxes:
            # Overlap threshold check
            if bx0 >= tx0 - 5 and bx1 <= tx1 + 5 and by0 >= ty0 - 5 and by1 <= ty1 + 5:
                return True
        return False


v3_structural_parser = V3StructuralPDFParser()
