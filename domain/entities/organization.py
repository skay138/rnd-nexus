from typing import Optional
from pydantic import BaseModel

class Organization(BaseModel):
    org_id: str
    name: str
    full_name: Optional[str] = None
    type: Optional[str] = None
    location: Optional[str] = None
