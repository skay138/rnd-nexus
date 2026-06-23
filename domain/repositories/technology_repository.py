from typing import List, Protocol
from domain.entities.technology import Technology

class TechnologyRepository(Protocol):
    def search_technologies(self, query: str = "", trl_min: int = 0, top_k: int = 10) -> List[Technology]:
        ...
