import io
import re
from typing import Tuple
import fitz  # PyMuPDF
import docx


class DocumentExtractionError(Exception):
    """Raised when document parsing fails."""
    pass


class UnifiedDocumentParser:
    """Extracts and cleans text content from PDF, DOCX, and plain text files."""

    @staticmethod
    def extract_text(file_bytes: bytes, filename: str) -> Tuple[str, str]:
        """
        Parses document bytes and returns (extracted_text, file_type).
        """
        ext = filename.split(".")[-1].lower() if "." in filename else ""

        if ext == "pdf":
            return UnifiedDocumentParser._extract_pdf(file_bytes), "pdf"
        elif ext in ["docx", "doc"]:
            return UnifiedDocumentParser._extract_docx(file_bytes), "docx"
        elif ext in ["txt", "md", "markdown", "csv"]:
            return UnifiedDocumentParser._extract_plain_text(file_bytes), ext
        else:
            try:
                text = UnifiedDocumentParser._extract_plain_text(file_bytes)
                return text, ext or "unknown"
            except Exception as err:
                raise DocumentExtractionError(
                    f"Unsupported file format '.{ext}' or corrupt file: {str(err)}"
                )

    @staticmethod
    def _extract_pdf(file_bytes: bytes) -> str:
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            cleaned_pages = []
            for page in doc:
                raw_page_text = page.get_text("text")
                if raw_page_text:
                    cleaned_page = UnifiedDocumentParser._clean_pdf_page_text(raw_page_text)
                    if cleaned_page:
                        cleaned_pages.append(cleaned_page)
            doc.close()
            full_text = "\n\n".join(cleaned_pages).strip()
            if not full_text:
                raise DocumentExtractionError("No readable text found in PDF document.")
            return full_text
        except Exception as e:
            raise DocumentExtractionError(f"Failed to parse PDF document: {str(e)}")

    @staticmethod
    def _clean_pdf_page_text(text: str) -> str:
        """Strips running headers, page numbers, arXiv stamp noise, and normalizes spacing."""
        lines = text.split("\n")
        filtered_lines = []
        for line in lines:
            stripped = line.strip()
            # Filter arXiv stamps, standalone page numbers, and repetitive header noise
            if re.match(r"^arXiv:\d+\.\d+(v\d+)?\s*(\[[a-zA-Z\.\-]+\])?", stripped, re.IGNORECASE):
                continue
            if re.match(r"^\d+\s*$", stripped):  # Standalone page number
                continue
            filtered_lines.append(line)

        cleaned = "\n".join(filtered_lines)
        # Collapse 3+ newlines into 2
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    @staticmethod
    def _extract_docx(file_bytes: bytes) -> str:
        try:
            doc_file = io.BytesIO(file_bytes)
            doc = docx.Document(doc_file)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            full_text = "\n\n".join(paragraphs).strip()
            if not full_text:
                raise DocumentExtractionError("No readable text found in DOCX document.")
            return full_text
        except Exception as e:
            raise DocumentExtractionError(f"Failed to parse DOCX document: {str(e)}")

    @staticmethod
    def _extract_plain_text(file_bytes: bytes) -> str:
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                text = file_bytes.decode(encoding).strip()
                if text:
                    return text
            except UnicodeDecodeError:
                continue
        raise DocumentExtractionError("Unable to decode text file with standard encodings.")
