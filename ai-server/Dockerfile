FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

# pyproject.toml을 먼저 복사해 의존성 레이어를 캐시
COPY ai-server/pyproject.toml /app/pyproject.toml

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-dev

COPY domain/ /app/domain/
COPY ai-server/src/ /app/src/
COPY .env.example /app/.env.example

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev

ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "-m", "main"]
