import logging
from typing import List, Optional
from pathlib import Path

from domain.repositories.project_repository import ProjectRepository
from domain.entities.project import Project
from infrastructure.repositories.in_memory_utils import load_fixture, keyword_score

logger = logging.getLogger(__name__)

class MariaDBProjectRepository(ProjectRepository):
    def __init__(self, db_pool) -> None:
        self.db_pool = db_pool

    def search_projects(self, keyword: str = "", institution: str = "", status: str = "", year_from: int = 0, limit: int = 10) -> List[Project]:
        conditions = []
        params = []
        if keyword:
            like = f"%{keyword}%"
            conditions.append("(title LIKE %s OR organization LIKE %s OR keywords LIKE %s)")
            params.extend([like, like, like])
        if institution:
            conditions.append("organization LIKE %s")
            params.append(f"%{institution}%")
        if status:
            conditions.append("status = %s")
            params.append(status)
        if year_from:
            conditions.append("year >= %s")
            params.append(year_from)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM projects {where} ORDER BY year DESC, budget_billion_krw DESC LIMIT %s"
        params.append(limit)

        with self.db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
            if not rows and keyword:
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM projects ORDER BY year DESC LIMIT %s", (limit,))
                    rows = cur.fetchall()
        logger.debug("[MariaDB] project search kw='%s' inst='%s' status='%s' year_from=%s → %d rows", keyword, institution, status, year_from or "*", len(rows))
        return [Project(**row) for row in rows]


    def get_by_ids(self, ids: List[str]) -> List[Project]:
        if not ids:
            return []
        placeholders = ",".join(["%s"] * len(ids))
        sql = f"SELECT * FROM projects WHERE project_id IN ({placeholders})"
        with self.db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, ids)
                rows = cur.fetchall()
        return [Project(**row) for row in rows]


class InMemoryProjectRepository(ProjectRepository):
    def __init__(self, fixtures_dir: Optional[Path] = None) -> None:
        self.projects = load_fixture("projects.json", fixtures_dir)

    def get_by_ids(self, ids: List[str]) -> List[Project]:
        id_set = set(ids)
        return [Project(**p) for p in self.projects if p.get("project_id") in id_set]

    def search_projects(self, keyword: str = "", institution: str = "", status: str = "", year_from: int = 0, limit: int = 10) -> List[Project]:
        keywords = keyword.lower().split() if keyword else []
        inst_lower = institution.lower()
        stat_lower = status.lower()
        results = []
        for p in self.projects:
            if year_from and p.get("year", 0) < year_from:
                continue
            if inst_lower and inst_lower not in p.get("organization", "").lower():
                continue
            if stat_lower and stat_lower not in p.get("status", "").lower():
                continue
            score = keyword_score(
                f"{p.get('title','')} {p.get('organization','')} {p.get('keywords','')}",
                keywords,
            ) if keywords else 1
            if score > 0 or not keywords:
                results.append((score, p))

        results.sort(key=lambda x: (-x[0], -x[1].get("year", 0)))
        if not results and not keywords:
            results = [(0, p) for p in self.projects]
            results.sort(key=lambda x: -x[1].get("year", 0))
        return [Project(**p) for _, p in results[:limit]]
