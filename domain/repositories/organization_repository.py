from typing import List, Protocol, runtime_checkable
from domain.entities.organization import Organization

@runtime_checkable
class OrganizationRepository(Protocol):
    def get_by_ids(self, ids: List[str]) -> List[Organization]: ...
