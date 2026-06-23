from typing import List, Protocol, Optional
from domain.entities.project import Project

class ProjectRepository(Protocol):
    def search_projects(self, keyword: str = "", institution: str = "", status: str = "", year_from: int = 0, limit: int = 10) -> List[Project]:
        ...
