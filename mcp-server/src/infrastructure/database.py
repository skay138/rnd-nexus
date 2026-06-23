"""
Database Connection Manager & Schema Initialization
"""
from __future__ import annotations
import logging
import urllib.parse
import contextlib
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_DDL_STATEMENTS = [
    """CREATE TABLE IF NOT EXISTS papers (
        paper_id     VARCHAR(64)  NOT NULL,
        title        TEXT         NOT NULL,
        year         INT,
        citations    INT          DEFAULT 0,
        journal      VARCHAR(255),
        keywords     TEXT,
        abstract     TEXT,
        PRIMARY KEY (paper_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",

    """CREATE TABLE IF NOT EXISTS paper_authors (
        paper_id     VARCHAR(64)  NOT NULL,
        author_name  VARCHAR(255) NOT NULL,
        display_order INT         NOT NULL DEFAULT 0,
        PRIMARY KEY (paper_id, author_name),
        INDEX idx_paper_id (paper_id),
        FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",

    """CREATE TABLE IF NOT EXISTS patents (
        patent_id    VARCHAR(64)  NOT NULL,
        title        TEXT         NOT NULL,
        applicant    VARCHAR(255),
        filing_date  DATE,
        country      VARCHAR(8),
        keywords     TEXT,
        abstract     TEXT,
        PRIMARY KEY (patent_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",

    """CREATE TABLE IF NOT EXISTS projects (
        project_id           VARCHAR(64)   NOT NULL,
        title                TEXT          NOT NULL,
        organization         VARCHAR(255),
        budget_billion_krw   DECIMAL(10,2),
        year                 INT,
        status               VARCHAR(32),
        keywords             TEXT,
        PRIMARY KEY (project_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",

    """CREATE TABLE IF NOT EXISTS researchers (
        researcher_id  VARCHAR(64)  NOT NULL,
        name           VARCHAR(255) NOT NULL,
        affiliation    VARCHAR(512),
        h_index        INT          DEFAULT 0,
        recent_papers  INT          DEFAULT 0,
        email_domain   VARCHAR(255),
        specialty      TEXT,
        PRIMARY KEY (researcher_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",

    """CREATE TABLE IF NOT EXISTS technologies (
        tech_id                   VARCHAR(64)   NOT NULL,
        name                      VARCHAR(255)  NOT NULL,
        trl                       INT,
        market_growth_rate_percent DECIMAL(5,1),
        investment_priority       VARCHAR(16),
        description               TEXT,
        keywords                  TEXT,
        PRIMARY KEY (tech_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",

    """CREATE TABLE IF NOT EXISTS tech_key_players (
        tech_id     VARCHAR(64)  NOT NULL,
        player_name VARCHAR(255) NOT NULL,
        PRIMARY KEY (tech_id, player_name),
        FOREIGN KEY (tech_id) REFERENCES technologies(tech_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",

]

def _parse_url(mariadb_url: str) -> Dict[str, Any]:
    url = mariadb_url.strip()
    url = url.replace("mysql+pymysql://", "mysql://").replace("mariadb+pymysql://", "mysql://")
    parsed = urllib.parse.urlparse(url)
    return {
        "host":     parsed.hostname or "localhost",
        "port":     parsed.port or 3306,
        "user":     urllib.parse.unquote(parsed.username or "root"),
        "password": urllib.parse.unquote(parsed.password or ""),
        "database": parsed.path.lstrip("/") or "rnd_nexus",
        "charset":  "utf8mb4",
    }

class PyMySQLPool:
    """간단하고 안전한 스레드 세이프 커넥션 풀 구현"""
    def __init__(self, mariadb_url: str, pool_size: int = 10):
        import pymysql
        import pymysql.cursors
        import queue

        self.params = _parse_url(mariadb_url)
        self.params["cursorclass"] = pymysql.cursors.DictCursor
        self.pool: queue.Queue = queue.Queue(maxsize=pool_size)

        for _ in range(pool_size):
            self.pool.put(self._create_conn())

    def _create_conn(self):
        import pymysql
        return pymysql.connect(**self.params)

    def acquire(self):
        conn = self.pool.get()
        try:
            conn.ping(reconnect=True)
        except Exception:
            conn = self._create_conn()
        return conn

    def release(self, conn):
        self.pool.put(conn)

    @contextlib.contextmanager
    def get_connection(self):
        conn = self.acquire()
        try:
            yield conn
        finally:
            self.release(conn)

def ensure_schema(mariadb_url: str) -> None:
    import pymysql
    params = _parse_url(mariadb_url)
    with pymysql.connect(**params) as conn:
        with conn.cursor() as cur:
            for stmt in _DDL_STATEMENTS:
                cur.execute(stmt)
        conn.commit()
    logger.info("MariaDB schema ensured (rnd-nexus)")

def seed_from_fixtures(mariadb_url: str, fixtures_dir: str, clear: bool = False) -> None:
    import json
    import pymysql
    import pymysql.cursors
    from pathlib import Path

    params = _parse_url(mariadb_url)
    conn = pymysql.connect(**params, cursorclass=pymysql.cursors.DictCursor)
    base = Path(fixtures_dir)

    try:
        with conn.cursor() as cur:
            if clear:
                for tbl in ["paper_authors", "papers", "patents", "projects",
                             "researchers", "tech_key_players", "technologies",
                             "budget_yearly", "budget_domains"]:
                    cur.execute(f"DELETE FROM {tbl}")
                logger.info("기존 시드 데이터 삭제 완료")

            papers = json.loads((base / "papers.json").read_text(encoding="utf-8"))
            for p in papers:
                cur.execute(
                    "INSERT INTO papers (paper_id, title, year, citations, journal, keywords, abstract) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE "
                    "title=VALUES(title), citations=VALUES(citations), abstract=VALUES(abstract)",
                    (p["paper_id"], p["title"], p.get("year"), p.get("citations", 0),
                     p.get("journal"), p.get("keywords"), p.get("abstract")),
                )
                for i, author in enumerate(p.get("authors", [])):
                    cur.execute(
                        "INSERT INTO paper_authors (paper_id, author_name, display_order) "
                        "VALUES (%s,%s,%s) ON DUPLICATE KEY UPDATE display_order=VALUES(display_order)",
                        (p["paper_id"], author, i),
                    )

            patents = json.loads((base / "patents.json").read_text(encoding="utf-8"))
            for p in patents:
                cur.execute(
                    "INSERT INTO patents (patent_id, title, applicant, filing_date, country, keywords, abstract) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE "
                    "title=VALUES(title), applicant=VALUES(applicant)",
                    (p["patent_id"], p["title"], p.get("applicant"), p.get("filing_date"),
                     p.get("country"), p.get("keywords"), p.get("abstract")),
                )

            projects = json.loads((base / "projects.json").read_text(encoding="utf-8"))
            for p in projects:
                cur.execute(
                    "INSERT INTO projects (project_id, title, organization, budget_billion_krw, year, status, keywords) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE "
                    "title=VALUES(title), status=VALUES(status)",
                    (p["project_id"], p["title"], p.get("organization"), p.get("budget_billion_krw"),
                     p.get("year"), p.get("status"), p.get("keywords")),
                )

            researchers = json.loads((base / "researchers.json").read_text(encoding="utf-8"))
            for r in researchers:
                cur.execute(
                    "INSERT INTO researchers (researcher_id, name, affiliation, h_index, recent_papers, email_domain, specialty) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE "
                    "name=VALUES(name), h_index=VALUES(h_index), specialty=VALUES(specialty)",
                    (r["researcher_id"], r["name"], r.get("affiliation"), r.get("h_index", 0),
                     r.get("recent_papers", 0), r.get("email_domain"), r.get("specialty")),
                )

            technologies = json.loads((base / "technologies.json").read_text(encoding="utf-8"))
            for t in technologies:
                cur.execute(
                    "INSERT INTO technologies (tech_id, name, trl, market_growth_rate_percent, investment_priority, description, keywords) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE "
                    "name=VALUES(name), trl=VALUES(trl), description=VALUES(description)",
                    (t["tech_id"], t["name"], t.get("trl"), t.get("market_growth_rate_percent"),
                     t.get("investment_priority"), t.get("description"), t.get("keywords")),
                )
                for player in t.get("key_players", []):
                    cur.execute(
                        "INSERT IGNORE INTO tech_key_players (tech_id, player_name) VALUES (%s,%s)",
                        (t["tech_id"], player),
                    )

        conn.commit()
        logger.info("MariaDB 시드 완료")
    finally:
        conn.close()
