from typing import Optional
from pydantic import BaseModel

class Project(BaseModel):
    project_id: str
    title: str
    organization: Optional[str] = None
    budget_billion_krw: float = 0.0
    year: Optional[int] = None
    status: Optional[str] = None
    keywords: Optional[str] = None
