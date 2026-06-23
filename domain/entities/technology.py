from typing import List, Optional
from pydantic import BaseModel, Field

class Technology(BaseModel):
    tech_id: str
    name: str
    trl: int = 1
    market_growth_rate_percent: float = 0.0
    investment_priority: Optional[str] = None
    description: Optional[str] = None
    keywords: Optional[str] = None
    key_players: List[str] = Field(default_factory=list)
