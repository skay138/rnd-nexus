from typing import List, Protocol
from domain.entities.patent import Patent

class PatentRepository(Protocol):
    def search_patents(self, query: str = "", country: str = "KR", year_from: int = 0, assignee: str = "", limit: int = 10) -> List[Patent]:
        ...

    def get_by_ids(self, ids: List[str]) -> List[Patent]:
        ...
