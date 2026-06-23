from typing import Optional
from datetime import date
from pydantic import BaseModel

class Patent(BaseModel):
    patent_id: str
    title: str
    applicant: Optional[str] = None
    filing_date: Optional[date] = None
    country: Optional[str] = None
    keywords: Optional[str] = None
    abstract: Optional[str] = None
