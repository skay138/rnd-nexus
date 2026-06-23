from typing import Protocol
from domain.entities.budget import BudgetTrend

class BudgetRepository(Protocol):
    def analyze_budget(self, domain: str, years: int = 5) -> BudgetTrend:
        ...
