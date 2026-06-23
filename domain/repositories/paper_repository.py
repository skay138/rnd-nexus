from typing import List, Protocol
from domain.entities.paper import Paper

class PaperRepository(Protocol):
    def search_papers(self, query: str = "", year_from: int = 0, year_to: int = 0, author: str = "", limit: int = 10) -> List[Paper]:
        ...
