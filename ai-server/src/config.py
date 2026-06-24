from __future__ import annotations
import logging
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    llm_provider: str = "ollama"  # "ollama" 또는 "openai" (Triton/vLLM 지원)
    llm_api_key: Optional[str] = None
    llm_base_url: str = "http://localhost:11434"
    rnd_model: str = "qwen2.5:7b"  # 최초 기동 시 모든 역할의 시드값 — 이후엔 /settings(DB)에서 관리

    # Agent 제어
    rnd_max_iterations: int = 3
    rnd_log_level: str = "DEBUG"

    # Redis (LangGraph CheckpointSaver)
    redis_url: str = "redis://localhost:6379"

    # MCP Server URL (SSE 통신)
    mcp_server_url: str = "http://localhost:8000/sse"

    # MariaDB (system_config 테이블용 — 선택)
    mariadb_url: Optional[str] = None

    # Web API 서버
    api_host: str = "0.0.0.0"
    api_port: int = 8080

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        logging.basicConfig(level=_settings.rnd_log_level.upper())
    return _settings
