from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py 위치: mcp-server/src/config.py → parents[2] = 리포 루트
_REPO_ROOT = Path(__file__).parents[2]


class MCPSettings(BaseSettings):
    mariadb_url: Optional[str] = None
    rnd_log_level: str = "INFO"

    # Milvus (벡터 검색) — 미설정 시 비활성화
    milvus_host: Optional[str] = None
    milvus_port: int = 19530
    milvus_collection: str = "rnd_nodes"
    sentence_transformer_model: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"

    # Neo4j (그래프 탐색) — 미설정 시 비활성화
    neo4j_uri: Optional[str] = None
    neo4j_username: str = "neo4j"
    neo4j_password: str = ""

    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
    )


_settings: MCPSettings | None = None


def get_settings() -> MCPSettings:
    global _settings
    if _settings is None:
        _settings = MCPSettings()
        logging.basicConfig(level=_settings.rnd_log_level.upper())
    return _settings
