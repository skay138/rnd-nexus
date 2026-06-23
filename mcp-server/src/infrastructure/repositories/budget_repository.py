import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

from domain.repositories.budget_repository import BudgetRepository
from domain.entities.budget import BudgetTrend
from infrastructure.repositories.in_memory_utils import load_fixture, keyword_score

logger = logging.getLogger(__name__)

class MariaDBBudgetRepository(BudgetRepository):
    def __init__(self, db_pool) -> None:
        self.db_pool = db_pool

    def analyze_budget(self, domain: str, years: int = 5) -> BudgetTrend:
        like = f"%{domain}%"
        with self.db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM budget_domains WHERE name LIKE %s OR keywords LIKE %s LIMIT 1",
                    (like, like),
                )
                row = cur.fetchone()

            if row is None:
                logger.warning("[MariaDB] budget domain '%s' not found — using aggregate fallback", domain)
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM budget_domains ORDER BY cagr_percent DESC LIMIT 1")
                    row = cur.fetchone()

            if row is None:
                return BudgetTrend(domain=domain, years=years, error="데이터 없음")

            domain_id = row["domain_id"]
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT year, amount_billion_krw FROM budget_yearly "
                    "WHERE domain_id = %s ORDER BY year DESC LIMIT %s",
                    (domain_id, years),
                )
                yearly_rows = sorted(cur.fetchall(), key=lambda r: r["year"])

        yearly_breakdown = {str(r["year"]): float(r["amount_billion_krw"]) for r in yearly_rows}
        total = sum(yearly_breakdown.values())

        return BudgetTrend(
            domain=domain,
            years=years,
            total_budget_billion_krw=round(total, 2),
            cagr_percent=float(row["cagr_percent"]),
            government_ratio=float(row["government_ratio"]),
            private_ratio=float(row["private_ratio"]),
            trend="증가" if row["cagr_percent"] > 0 else ("감소" if row["cagr_percent"] < 0 else "유지"),
            yearly_breakdown=yearly_breakdown,
        )


class InMemoryBudgetRepository(BudgetRepository):
    def __init__(self, fixtures_dir: Optional[Path] = None) -> None:
        raw = load_fixture("budget.json", fixtures_dir)
        self.domains: List[Dict[str, Any]] = raw.get("domains", [])
        self.default: Dict[str, Any] = raw.get("default", {})

    def analyze_budget(self, domain: str, years: int = 5) -> BudgetTrend:
        domain_lower = domain.lower()
        matched = None
        for d in self.domains:
            keywords = d.get("keywords", "")
            if domain_lower in d["name"].lower() or any(kw in domain_lower for kw in keywords.lower().split()):
                matched = d
                break

        data = matched or self.default
        yearly_entries: List[Dict[str, Any]] = data.get("yearly", [])
        yearly_slice = sorted(yearly_entries, key=lambda e: e["year"])[-years:]
        yearly_breakdown = {str(e["year"]): float(e["amount_billion_krw"]) for e in yearly_slice}
        total = sum(yearly_breakdown.values())

        best = data
        return BudgetTrend(
            domain=domain,
            years=years,
            total_budget_billion_krw=round(total, 2),
            cagr_percent=float(best["cagr_percent"]),
            government_ratio=float(best["government_ratio"]),
            private_ratio=float(best["private_ratio"]),
            trend="증가" if best["cagr_percent"] > 0 else ("감소" if best["cagr_percent"] < 0 else "유지"),
            yearly_breakdown=yearly_breakdown,
        )
