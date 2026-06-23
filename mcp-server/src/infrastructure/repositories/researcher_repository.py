import logging
from typing import List, Optional
from pathlib import Path

from domain.repositories.researcher_repository import ResearcherRepository
from domain.entities.researcher import Researcher
from infrastructure.repositories.in_memory_utils import load_fixture, keyword_score

logger = logging.getLogger(__name__)

class MariaDBResearcherRepository(ResearcherRepository):
    def __init__(self, db_pool) -> None:
        self.db_pool = db_pool

    def search_researchers(self, query: str = "", specialty: str = "", affiliation: str = "", top_k: int = 10) -> List[Researcher]:
        conditions = []
        params = []
        if query:
            like = f"%{query}%"
            conditions.append("(name LIKE %s OR specialty LIKE %s OR affiliation LIKE %s)")
            params.extend([like, like, like])
        if specialty:
            conditions.append("specialty LIKE %s")
            params.append(f"%{specialty}%")
        if affiliation:
            conditions.append("affiliation LIKE %s")
            params.append(f"%{affiliation}%")

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM researchers {where} ORDER BY h_index DESC LIMIT %s"
        params.append(top_k)

        with self.db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
            if not rows and (query or specialty or affiliation):
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM researchers ORDER BY h_index DESC LIMIT %s", (top_k,))
                    rows = cur.fetchall()
        logger.debug("[MariaDB] researcher search q='%s' spec='%s' aff='%s' → %d rows", query, specialty, affiliation, len(rows))
        return [Researcher(**row) for row in rows]


    def get_by_ids(self, ids: List[str]) -> List[Researcher]:
        if not ids:
            return []
        placeholders = ",".join(["%s"] * len(ids))
        sql = f"SELECT * FROM researchers WHERE researcher_id IN ({placeholders})"
        with self.db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, ids)
                rows = cur.fetchall()
        return [Researcher(**row) for row in rows]


class InMemoryResearcherRepository(ResearcherRepository):
    def __init__(self, fixtures_dir: Optional[Path] = None) -> None:
        self.researchers = load_fixture("researchers.json", fixtures_dir)

    def get_by_ids(self, ids: List[str]) -> List[Researcher]:
        id_set = set(ids)
        return [Researcher(**r) for r in self.researchers if r.get("researcher_id") in id_set]

    def search_researchers(self, query: str = "", specialty: str = "", affiliation: str = "", top_k: int = 10) -> List[Researcher]:
        spec_lower = specialty.lower()
        aff_lower = affiliation.lower()
        keywords = query.lower().split() if query else []

        scored = []
        for r in self.researchers:
            if spec_lower and spec_lower not in r.get("specialty", "").lower():
                continue
            if aff_lower and aff_lower not in r.get("affiliation", "").lower():
                continue
            score = keyword_score(
                f"{r.get('name','')} {r.get('specialty','')} {r.get('affiliation','')}",
                keywords,
            ) if keywords else 1
            scored.append((score, r))

        scored.sort(key=lambda x: (-x[0], -x[1].get("h_index", 0)))
        return [Researcher(**r) for _, r in scored[:top_k]]
