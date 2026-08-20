from typing import List, Optional
from pydantic import BaseModel, Field


class QueryExpansionConfig(BaseModel):
    enabled: bool = True
    max_expansions_per_term: int = 3
    preserve_original_query: bool = True
    exact_match_priority: bool = True
    enabled_categories: Optional[List[str]] = Field(
        default=None, description="Optional list of terminology categories to enable."
    )
