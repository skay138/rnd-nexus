from typing import List, Optional
from pydantic import BaseModel, Field

class Paper(BaseModel):
    paper_id: str
    title: str
    year: Optional[int] = None
    citations: int = 0
    journal: Optional[str] = None
    abstract: Optional[str] = None
    keywords: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
