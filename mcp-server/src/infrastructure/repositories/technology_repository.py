import logging
from typing import List, Optional
from pathlib import Path

from domain.repositories.technology_repository import TechnologyRepository
from domain.entities.technology import Technology
from infrastructure.repositories.in_memory_utils import load_fixture, keyword_score

logger = logging.getLogger(__name__)

class MariaDBTechnologyRepository(TechnologyRepository):
    def __init__(self, db_pool) -> None:
        self.db_pool = db_pool

    def search_technologies(self, query: str = "", trl_min: int = 0, top_k: int = 10) -> List[Technology]:
        conditions = []
        params = []
        if query:
            like = f"%{query}%"
            conditions.append("(t.name LIKE %s OR t.description LIKE %s OR t.keywords LIKE %s)")
            params.extend([like, like, like])
        if trl_min:
            conditions.append("t.trl >= %s")
            params.append(trl_min)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"""
            SELECT t.*, GROUP_CONCAT(kp.player_name SEPARATOR ', ') AS key_players
            FROM technologies t
            LEFT JOIN tech_key_players kp ON kp.tech_id = t.tech_id
            {where}
            GROUP BY t.tech_id
            ORDER BY t.market_growth_rate_percent DESC, t.trl DESC
            LIMIT %s
        """
        params.append(top_k)
        with self.db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
            if not rows and query:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT t.*, GROUP_CONCAT(kp.player_name SEPARATOR ', ') AS key_players "
                        "FROM technologies t LEFT JOIN tech_key_players kp ON kp.tech_id = t.tech_id "
                        "GROUP BY t.tech_id ORDER BY t.market_growth_rate_percent DESC LIMIT %s",
                        (top_k,),
                    )
                    rows = cur.fetchall()
        for row in rows:
            row["key_players"] = row["key_players"].split(", ") if row.get("key_players") else []
        logger.debug("[MariaDB] tech search q='%s' trl_min=%d → %d rows", query, trl_min, len(rows))
        return [Technology(**row) for row in rows]


    def get_by_ids(self, ids: List[str]) -> List[Technology]:
        if not ids:
            return []
        placeholders = ",".join(["%s"] * len(ids))
        sql = f"""
            SELECT t.*, GROUP_CONCAT(kp.player_name SEPARATOR ', ') AS key_players
            FROM technologies t
            LEFT JOIN tech_key_players kp ON kp.tech_id = t.tech_id
            WHERE t.tech_id IN ({placeholders})
            GROUP BY t.tech_id
        """
        with self.db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, ids)
                rows = cur.fetchall()
        for row in rows:
            row["key_players"] = row["key_players"].split(", ") if row.get("key_players") else []
        return [Technology(**row) for row in rows]


class InMemoryTechnologyRepository(TechnologyRepository):
    def __init__(self, fixtures_dir: Optional[Path] = None) -> None:
        self.technologies = load_fixture("technologies.json", fixtures_dir)

    def get_by_ids(self, ids: List[str]) -> List[Technology]:
        id_set = set(ids)
        return [Technology(**t) for t in self.technologies if t.get("tech_id") in id_set]

    def search_technologies(self, query: str = "", trl_min: int = 0, top_k: int = 10) -> List[Technology]:
        keywords = query.lower().split() if query else []
        scored = []
        for t in self.technologies:
            if trl_min and t.get("trl", 0) < trl_min:
                continue
            score = keyword_score(
                f"{t.get('name','')} {t.get('description','')} {t.get('keywords','')}",
                keywords,
            ) if keywords else 1
            scored.append((score, t))

        scored.sort(key=lambda x: (-x[0], -x[1].get("market_growth_rate_percent", 0)))
        return [Technology(**t) for _, t in scored[:top_k]]
