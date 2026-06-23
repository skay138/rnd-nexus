"""
Config Repository (R&D Nexus)
- MemoryConfigRepository:  인메모리 (테스트 / MariaDB 미연결 fallback)
- MariaDBConfigRepository: system_config 테이블 기반 영속 저장소
  - 앱 재시작 후에도 설정 유지
  - 기본값은 INSERT IGNORE로 시드 → DB에 이미 있으면 유지
  - 값은 JSON 직렬화로 타입 보존 (int, float, str)
"""

from __future__ import annotations
import json
import logging
from typing import Any, Optional, Union

from common.config.query_config import CONFIG_DEFAULTS

logger = logging.getLogger(__name__)


class MemoryConfigRepository:
    """인메모리 설정 저장소 (테스트 / fallback)."""

    def __init__(self, overrides: Optional[dict[str, Any]] = None):
        self._store: dict[str, Any] = {**CONFIG_DEFAULTS, **(overrides or {})}

    def get(self, key: str) -> Any:
        return self._store.get(key)

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def all(self) -> dict[str, Any]:
        return dict(self._store)


class MariaDBConfigRepository:
    """
    MariaDB system_config 테이블 기반 설정 저장소.
    - 초기화 시 기본값을 INSERT IGNORE로 시드
    - 런타임 set()은 즉시 DB에 반영
    """

    _CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS system_config (
            `key`   VARCHAR(128) NOT NULL PRIMARY KEY,
            `value` TEXT         NOT NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """

    def __init__(self, mariadb_url: str, overrides: Optional[dict[str, Any]] = None):
        from infrastructure.mariadb import parse_mariadb_url
        self._params = parse_mariadb_url(mariadb_url)
        self._conn: Any = None

        seed = {**CONFIG_DEFAULTS, **(overrides or {})}
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(self._CREATE_TABLE)
            seeded = 0
            for key, value in seed.items():
                cur.execute(
                    "INSERT IGNORE INTO system_config (`key`, `value`) VALUES (%s, %s)",
                    (key, json.dumps(value)),
                )
                seeded += cur.rowcount
        conn.commit()
        logger.info("MariaDBConfigRepository initialized (%d keys seeded)", seeded)

    def _get_conn(self) -> Any:
        import pymysql
        import pymysql.cursors
        try:
            if self._conn is not None and self._conn.open:
                self._conn.ping(reconnect=True)
                return self._conn
        except Exception:
            pass
        self._conn = pymysql.connect(**self._params, cursorclass=pymysql.cursors.DictCursor)
        return self._conn

    def get(self, key: str) -> Any:
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT `value` FROM system_config WHERE `key` = %s", (key,))
            row = cur.fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            return row["value"]

    def set(self, key: str, value: Any) -> None:
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO system_config (`key`, `value`) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE `value` = VALUES(`value`)",
                (key, json.dumps(value)),
            )
        conn.commit()

    def all(self) -> dict[str, Any]:
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT `key`, `value` FROM system_config")
            rows: list[dict[str, Any]] = cur.fetchall()
        result: dict[str, Any] = {}
        for row in rows:
            try:
                result[row["key"]] = json.loads(row["value"])
            except (json.JSONDecodeError, TypeError):
                result[row["key"]] = row["value"]
        return result


def make_config_repo(
    mariadb_url: Optional[str] = None,
    overrides: Optional[dict[str, Any]] = None,
) -> Union[MemoryConfigRepository, MariaDBConfigRepository]:
    """MariaDB URL 있으면 DB 저장소, 없으면 인메모리 반환."""
    if mariadb_url:
        try:
            return MariaDBConfigRepository(mariadb_url, overrides)
        except Exception as e:
            logger.warning("MariaDB config repo 초기화 실패, 인메모리 fallback: %s", e)
    return MemoryConfigRepository(overrides)
