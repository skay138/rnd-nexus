import logging
from typing import List, Optional
from pathlib import Path

from domain.repositories.patent_repository import PatentRepository
from domain.entities.patent import Patent
from infrastructure.repositories.in_memory_utils import load_fixture, keyword_score

logger = logging.getLogger(__name__)

class MariaDBPatentRepository(PatentRepository):
    def __init__(self, db_pool) -> None:
        self.db_pool = db_pool

    def search_patents(self, query: str = "", country: str = "KR", year_from: int = 0, assignee: str = "", limit: int = 10) -> List[Patent]:
        conditions = []
        params = []
        if country != "ALL":
            conditions.append("country = %s")
            params.append(country)
        if query:
            like = f"%{query}%"
            conditions.append("(title LIKE %s OR abstract LIKE %s OR keywords LIKE %s OR applicant LIKE %s)")
            params.extend([like, like, like, like])
        if year_from:
            conditions.append("YEAR(filing_date) >= %s")
            params.append(year_from)
        if assignee:
            conditions.append("applicant LIKE %s")
            params.append(f"%{assignee}%")

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM patents {where} ORDER BY filing_date DESC LIMIT %s"
        params.append(limit)

        with self.db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        logger.debug("[MariaDB] patent search q='%s' country=%s year_from=%s assignee='%s' → %d rows", query, country, year_from or "*", assignee, len(rows))
        return [Patent(**row) for row in rows]


    def get_by_ids(self, ids: List[str]) -> List[Patent]:
        if not ids:
            return []
        placeholders = ",".join(["%s"] * len(ids))
        sql = f"SELECT * FROM patents WHERE patent_id IN ({placeholders})"
        with self.db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, ids)
                rows = cur.fetchall()
        return [Patent(**row) for row in rows]


class InMemoryPatentRepository(PatentRepository):
    def __init__(self, fixtures_dir: Optional[Path] = None) -> None:
        self.patents = load_fixture("patents.json", fixtures_dir)

    def get_by_ids(self, ids: List[str]) -> List[Patent]:
        id_set = set(ids)
        return [Patent(**p) for p in self.patents if p.get("patent_id") in id_set]

    def search_patents(self, query: str = "", country: str = "KR", year_from: int = 0, assignee: str = "", limit: int = 10) -> List[Patent]:
        keywords = query.lower().split() if query else []
        assignee_lower = assignee.lower()
        results = []
        for p in self.patents:
            if country != "ALL" and p.get("country") != country:
                continue
            if year_from:
                filing = str(p.get("filing_date", "0"))
                if filing[:4].isdigit() and int(filing[:4]) < year_from:
                    continue
            if assignee_lower and assignee_lower not in p.get("applicant", "").lower():
                continue
            score = keyword_score(
                f"{p.get('title','')} {p.get('abstract','')} {p.get('keywords','')} {p.get('applicant','')}",
                keywords,
            ) if keywords else 1
            if score > 0 or not keywords:
                results.append((score, p))

        results.sort(key=lambda x: -x[0])
        return [Patent(**p) for _, p in results[:limit]]
