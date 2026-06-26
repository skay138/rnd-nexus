from typing import List, Optional
from pathlib import Path

from domain.repositories.organization_repository import OrganizationRepository
from domain.entities.organization import Organization
from infrastructure.repositories.in_memory_utils import load_fixture


class InMemoryOrganizationRepository(OrganizationRepository):
    def __init__(self, fixtures_dir: Optional[Path] = None) -> None:
        self.organizations = load_fixture("organizations.json", fixtures_dir)

    def get_by_ids(self, ids: List[str]) -> List[Organization]:
        id_set = set(ids)
        return [Organization(**o) for o in self.organizations if o.get("org_id") in id_set]
