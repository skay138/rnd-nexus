import logging
from typing import List, Optional
from pathlib import Path

from domain.repositories.paper_repository import PaperRepository
from domain.entities.paper import Paper
from infrastructure.repositories.in_memory_utils import load_fixture, keyword_score

logger = logging.getLogger(__name__)

class MariaDBPaperRepository(PaperRepository):
    def __init__(self, db_pool) -> None:
        self.db_pool = db_pool

    def search_papers(self, query: str = "", year_from: int = 0, year_to: int = 0, author: str = "", limit: int = 10) -> List[Paper]:
        conditions = []
        params = []
        if query:
            like = f"%{query}%"
            conditions.append("(p.title LIKE %s OR p.abstract LIKE %s OR p.keywords LIKE %s)")
            params.extend([like, like, like])
        if year_from:
            conditions.append("p.year >= %s")
            params.append(year_from)
        if year_to:
            conditions.append("p.year <= %s")
            params.append(year_to)
        if author:
            conditions.append("pa.author_name LIKE %s")
            params.append(f"%{author}%")

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"""
            SELECT p.paper_id, p.title, p.year, p.citations, p.journal, p.abstract,
                   GROUP_CONCAT(pa.author_id ORDER BY pa.display_order SEPARATOR ', ') AS authors
            FROM papers p
            LEFT JOIN paper_authors pa ON pa.paper_id = p.paper_id
            {where}
            GROUP BY p.paper_id
            ORDER BY p.citations DESC
            LIMIT %s
        """
        params.append(limit)
        with self.db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        for row in rows:
            row["authors"] = row["authors"].split(", ") if row.get("authors") else []
        logger.debug("[MariaDB] paper search q='%s' year=%s~%s author='%s' → %d rows", query, year_from or "*", year_to or "*", author, len(rows))
        return [Paper(**row) for row in rows]


    def get_by_ids(self, ids: List[str]) -> List[Paper]:
        if not ids:
            return []
        placeholders = ",".join(["%s"] * len(ids))
        sql = f"""
            SELECT p.paper_id, p.title, p.year, p.citations, p.journal, p.abstract, p.keywords,
                   GROUP_CONCAT(pa.author_id ORDER BY pa.display_order SEPARATOR ', ') AS authors
            FROM papers p
            LEFT JOIN paper_authors pa ON pa.paper_id = p.paper_id
            WHERE p.paper_id IN ({placeholders})
            GROUP BY p.paper_id
        """
        with self.db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, ids)
                rows = cur.fetchall()
        for row in rows:
            row["authors"] = row["authors"].split(", ") if row.get("authors") else []
        return [Paper(**row) for row in rows]


class InMemoryPaperRepository(PaperRepository):
    def __init__(self, fixtures_dir: Optional[Path] = None) -> None:
        self.papers = load_fixture("papers.json", fixtures_dir)

    def get_by_ids(self, ids: List[str]) -> List[Paper]:
        id_set = set(ids)
        return [Paper(**p) for p in self.papers if p.get("paper_id") in id_set]

    def search_papers(self, query: str = "", year_from: int = 0, year_to: int = 0, author: str = "", limit: int = 10) -> List[Paper]:
        keywords = query.lower().split() if query else []
        author_lower = author.lower()
        scored = []
        for p in self.papers:
            if year_from and p.get("year", 0) < year_from:
                continue
            if year_to and p.get("year", 0) > year_to:
                continue
            if author_lower and not any(author_lower in a.lower() for a in p.get("authors", [])):
                continue
            score = keyword_score(
                f"{p.get('title','')} {p.get('abstract','')} {p.get('keywords','')} "
                f"{' '.join(p.get('authors', []))}",
                keywords,
            ) if keywords else 1
            if score > 0 or not keywords:
                scored.append((score, p))

        scored.sort(key=lambda x: (-x[0], -x[1].get("citations", 0)))
        return [Paper(**p) for _, p in scored[:limit]]
