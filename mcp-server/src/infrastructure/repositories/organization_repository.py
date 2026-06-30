from typing import List, Optional
from pathlib import Path

from domain.repositories.organization_repository import OrganizationRepository
from domain.entities.organization import Organization
from infrastructure.repositories.in_memory_utils import load_fixture


class MariaDBOrganizationRepository(OrganizationRepository):
    def __init__(self, db_pool) -> None:
        self.db_pool = db_pool

    def get_by_ids(self, ids: List[str]) -> List[Organization]:
        if not ids:
            return []
        placeholders = ",".join(["%s"] * len(ids))
        sql = f"SELECT * FROM organizations WHERE org_id IN ({placeholders})"
        with self.db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, ids)
                rows = cur.fetchall()
        return [Organization(**row) for row in rows]


class InMemoryOrganizationRepository(OrganizationRepository):
    def __init__(self, fixtures_dir: Optional[Path] = None) -> None:
        self.organizations = load_fixture("organizations.json", fixtures_dir)

    def get_by_ids(self, ids: List[str]) -> List[Organization]:
        id_set = set(ids)
        return [Organization(**o) for o in self.organizations if o.get("org_id") in id_set]
