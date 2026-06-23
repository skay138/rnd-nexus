from typing import Dict, Optional
from pydantic import BaseModel, Field

class BudgetTrend(BaseModel):
    domain: str
    years: int
    total_budget_billion_krw: float = 0.0
    cagr_percent: float = 0.0
    government_ratio: float = 0.0
    private_ratio: float = 0.0
    trend: Optional[str] = None
    yearly_breakdown: Dict[str, float] = Field(default_factory=dict)
    error: Optional[str] = None
