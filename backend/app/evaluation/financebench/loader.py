import os
import json
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from app.core.config import settings

logger = logging.getLogger(__name__)


class FinanceBenchEvidence(BaseModel):
    doc_name: str
    evidence_text: Optional[str] = None
    evidence_page_num: Optional[int] = None
    evidence_text_full_page: Optional[str] = None


class FinanceBenchQuestion(BaseModel):
    financebench_id: str
    company: str
    doc_name: str
    question_type: Optional[str] = None
    question_reasoning: Optional[str] = None
    question: str
    answer: str
    justification: Optional[str] = None
    dataset_subset_label: Optional[str] = None
    evidence: List[FinanceBenchEvidence] = Field(default_factory=list)


class FinanceBenchLoader:
    """Parses and manages FinanceBench jsonl dataset items."""

    def __init__(self, dataset_path: Optional[str] = None):
        self.dataset_path = dataset_path or settings.FINANCEBENCH_DATASET

    def load_dataset(self) -> List[FinanceBenchQuestion]:
        """Loads and parses all records from the FinanceBench jsonl file."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"FinanceBench dataset file not found at: {self.dataset_path}")

        items: List[FinanceBenchQuestion] = []
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                    q_item = FinanceBenchQuestion(**data)
                    items.append(q_item)
                except Exception as e:
                    logger.warning(f"Error parsing line {line_idx} in FinanceBench dataset: {str(e)}")

        logger.info(f"Loaded {len(items)} FinanceBench questions from {self.dataset_path}")
        return items

    def get_questions_by_doc_name(self, doc_name: str) -> List[FinanceBenchQuestion]:
        """Filters dataset questions by specific target SEC 10-K doc_name."""
        all_q = self.load_dataset()
        return [q for q in all_q if q.doc_name == doc_name]

    def get_unique_doc_names(self) -> List[str]:
        """Returns sorted list of unique doc_name values in dataset."""
        all_q = self.load_dataset()
        return sorted(list(set(q.doc_name for q in all_q)))
