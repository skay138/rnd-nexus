"""MariaDB URL 파서 — config_repository가 사용."""
from __future__ import annotations
from typing import Any
from urllib.parse import urlparse


def parse_mariadb_url(url: str) -> dict[str, Any]:
    """mysql+pymysql://user:pass@host:port/db → pymysql.connect kwargs."""
    parsed = urlparse(url.replace("mysql+pymysql://", "mysql://"))
    return {
        "host":     parsed.hostname or "localhost",
        "port":     parsed.port or 3306,
        "user":     parsed.username or "root",
        "password": parsed.password or "",
        "database": (parsed.path or "/").lstrip("/"),
        "charset":  "utf8mb4",
    }
