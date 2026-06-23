from typing import Optional
from pydantic import BaseModel

class Researcher(BaseModel):
    researcher_id: str
    name: str
    affiliation: Optional[str] = None
    h_index: int = 0
    recent_papers: int = 0
    email_domain: Optional[str] = None
    specialty: Optional[str] = None
