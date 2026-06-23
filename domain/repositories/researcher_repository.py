from typing import List, Protocol
from domain.entities.researcher import Researcher

class ResearcherRepository(Protocol):
    def search_researchers(self, query: str = "", specialty: str = "", affiliation: str = "", top_k: int = 10) -> List[Researcher]:
        ...
