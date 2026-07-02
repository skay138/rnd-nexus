"""FastAPI 앱 팩토리 — lifespan에서 MCP·Redis·Graph 초기화."""
from __future__ import annotations
import io
import logging
import pathlib
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from config import get_settings
from agent.graph import build_graph
from memory.session import create_memory
from infrastructure.config_repository import make_config_repo
from api.routes import health, query, admin, stats

logger = logging.getLogger(__name__)

_STATIC = pathlib.Path(__file__).parent.parent / "static"

# ── 컬러 로그 포매터 ────────────────────────────────────────────────────────
_RST  = "\033[0m"
_DIM  = "\033[2m"
_BOLD = "\033[1m"

_LEVEL_CLR: dict[str, str] = {
    "DEBUG":    "\033[90m",   # dark gray
    "INFO":     "\033[94m",   # bright blue
    "WARNING":  "\033[93m",   # bright yellow
    "ERROR":    "\033[91m",   # bright red
    "CRITICAL": "\033[41m",   # red background
}
# 노드별 색상 (logger name prefix → ANSI color)
_NAME_CLR: list[tuple[str, str]] = [
    ("agent.nodes.orchestrator",     "\033[35m"),   # magenta
    ("agent.nodes.parallel_executor","\033[33m"),   # yellow
    ("agent.nodes.generate",         "\033[32m"),   # green
    ("agent.mcp_client",             "\033[90m"),   # dark gray
    ("memory.tool_cache",            "\033[34m"),   # blue
    ("agent.graph",                  "\033[36m"),   # cyan
    ("api",                          "\033[97m"),   # bright white
    ("infrastructure",               "\033[96m"),   # bright cyan
    ("mcp_server",                   "\033[96m"),   # bright cyan
]


class _ColorFmt(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        clr = next((c for p, c in _NAME_CLR if record.name.startswith(p)), "")
        lclr = _LEVEL_CLR.get(record.levelname, "")
        parts = record.name.split(".")
        short = f"{parts[-2]}.{parts[-1]}" if len(parts) > 2 else record.name
        ts = self.formatTime(record, "%H:%M:%S")
        line = (
            f"{_DIM}{ts}{_RST} "
            f"{lclr}{record.levelname:<7}{_RST} "
            f"{clr}[{short:<22}]{_RST} "
            f"{clr}{record.getMessage()}{_RST}"
        )
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def _configure_logging(log_level: int) -> None:
    # UTF-8 스트림 (Docker 기본 로케일이 C/POSIX 일 때 한글 깨짐 방지)
    if hasattr(sys.stdout, "buffer"):
        utf8_stream = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
    else:
        utf8_stream = sys.stdout

    root = logging.getLogger()
    root.setLevel(log_level)

    # 기존 핸들러(uvicorn이 추가한 것 포함) 포매터 교체
    if root.handlers:
        for h in root.handlers:
            h.setFormatter(_ColorFmt())
    else:
        _h = logging.StreamHandler(stream=utf8_stream)
        _h.setFormatter(_ColorFmt())
        root.addHandler(_h)

    # 노이즈 라이브러리 로거 억제 (앱 로그레벨과 무관하게 WARNING 고정)
    for pkg in (
        "openai._base_client",
        "openai.http_client",
        "httpx",
        "httpcore",
        "mcp.client.sse",
        "sse_starlette",
        "langchain",
        "langchain_core",
        "langchain_ollama",
        "uvicorn.access",
    ):
        logging.getLogger(pkg).setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    log_level = getattr(logging, settings.rnd_log_level.upper(), logging.INFO)
    _configure_logging(log_level)

    logger.info("R&D Nexus API 서버 시작 중...")

    model_defaults = {
        "orchestrator_model": settings.rnd_model,
        "worker_model":       settings.rnd_model,
        "generate_model":     settings.rnd_model,
        "compact_model":      settings.rnd_model,
    }
    app.state.config_repo     = make_config_repo(settings.mariadb_url, overrides=model_defaults)
    app.state.redis_connected = False

    try:
        async with create_memory() as memory:
            app.state.redis_connected = True
            app.state.graph = build_graph(memory)
            logger.info("R&D Nexus API 서버 준비 완료 — http://%s:%d",
                        settings.api_host, settings.api_port)
            yield

    except Exception:
        logger.exception("서버 초기화 실패")
        raise


def create_app() -> FastAPI:
    app = FastAPI(
        title="R&D Nexus API",
        description="R&D 전반 지원 멀티에이전트 AI 서비스",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(health.router, prefix="/api/v1")
    app.include_router(query.router,  prefix="/api/v1")
    app.include_router(admin.router,  prefix="/api/v1")
    app.include_router(stats.router,  prefix="/api/v1")

    if _STATIC.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

        @app.get("/")
        async def index() -> FileResponse:
            return FileResponse(str(_STATIC / "index.html"))

        @app.get("/settings")
        async def settings_page() -> FileResponse:
            return FileResponse(str(_STATIC / "settings.html"))

    return app
