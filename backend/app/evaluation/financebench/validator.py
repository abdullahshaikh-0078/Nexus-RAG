import os
import glob
import logging
from typing import Dict, Any, List
from app.core.config import settings
from app.evaluation.financebench.loader import FinanceBenchLoader

logger = logging.getLogger(__name__)


def validate_financebench_environment() -> Dict[str, Any]:
    """
    Validates the FinanceBench environment, dataset file, PDF folder,
    question count, unique documents, and PDF file availability.
    """
    root_dir = settings.FINANCEBENCH_ROOT
    dataset_file = settings.FINANCEBENCH_DATASET
    pdf_dir = settings.FINANCEBENCH_PDF_DIR

    root_exists = os.path.exists(root_dir)
    dataset_exists = os.path.exists(dataset_file)
    pdf_dir_exists = os.path.exists(pdf_dir)

    if not root_exists:
        logger.error(f"FinanceBench root directory missing: {root_dir}")
    if not dataset_exists:
        logger.error(f"FinanceBench dataset file missing: {dataset_file}")
    if not pdf_dir_exists:
        logger.error(f"FinanceBench PDF directory missing: {pdf_dir}")

    total_questions = 0
    unique_docs: List[str] = []
    available_pdfs: List[str] = []
    missing_pdfs: List[str] = []
    extra_pdfs: List[str] = []

    if dataset_exists:
        loader = FinanceBenchLoader(dataset_path=dataset_file)
        questions = loader.load_dataset()
        total_questions = len(questions)
        unique_docs = loader.get_unique_doc_names()

    if pdf_dir_exists:
        pdf_paths = glob.glob(os.path.join(pdf_dir, "*.pdf"))
        all_pdf_filenames = [os.path.basename(p) for p in pdf_paths]
        all_pdf_stem_map = {os.path.splitext(f)[0]: f for f in all_pdf_filenames}

        req_doc_set = set(unique_docs)
        pdf_stem_set = set(all_pdf_stem_map.keys())

        matched = req_doc_set.intersection(pdf_stem_set)
        missing = req_doc_set - pdf_stem_set
        extra = pdf_stem_set - req_doc_set

        available_pdfs = sorted(list(matched))
        missing_pdfs = sorted(list(missing))
        extra_pdfs = sorted(list(extra))

    is_valid = (
        root_exists
        and dataset_exists
        and pdf_dir_exists
        and total_questions == 150
        and len(unique_docs) == 84
        and len(missing_pdfs) == 0
    )

    report = {
        "valid": is_valid,
        "root_directory": root_dir,
        "root_exists": root_exists,
        "dataset_file": dataset_file,
        "dataset_exists": dataset_exists,
        "pdf_directory": pdf_dir,
        "pdf_directory_exists": pdf_dir_exists,
        "total_questions": total_questions,
        "unique_documents_count": len(unique_docs),
        "available_matching_pdfs_count": len(available_pdfs),
        "total_pdf_files_in_folder": len(available_pdfs) + len(extra_pdfs),
        "missing_pdfs_count": len(missing_pdfs),
        "missing_pdfs": missing_pdfs,
        "extra_pdfs_count": len(extra_pdfs),
    }

    return report


def print_validation_report(report: Dict[str, Any]):
    print("=" * 60)
    print("NEXUS RAG - FINANCEBENCH ENVIRONMENT VALIDATION REPORT")
    print("=" * 60)
    print(f"Overall Status:            {'[VALID]' if report['valid'] else '[INVALID]'}")
    print(f"Root Directory:            {report['root_directory']} (Exists: {report['root_exists']})")
    print(f"Annotation File:           {report['dataset_file']} (Exists: {report['dataset_exists']})")
    print(f"PDF Directory:             {report['pdf_directory']} (Exists: {report['pdf_directory_exists']})")
    print(f"Total Questions Detected:  {report['total_questions']} (Expected: 150)")
    print(f"Unique Required Documents: {report['unique_documents_count']} (Expected: 84)")
    print(f"Matching PDFs Available:   {report['available_matching_pdfs_count']} / 84")
    print(f"Total PDF Files in Folder: {report['total_pdf_files_in_folder']}")
    print(f"Missing PDFs:              {report['missing_pdfs_count']}")
    if report['missing_pdfs']:
        print(f"  Missing List: {report['missing_pdfs']}")
    print(f"Extra PDFs in Directory:   {report['extra_pdfs_count']}")
    print("=" * 60)


if __name__ == "__main__":
    report = validate_financebench_environment()
    print_validation_report(report)
