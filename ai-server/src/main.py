"""
R&D Nexus AI Server

실행:
    python -m main

Docker:
    docker compose -f infra/docker-compose.yml up ai-server
"""
import asyncio
import uvicorn
from api.app import create_app
from config import get_settings


async def serve() -> None:
    settings = get_settings()
    app = create_app()
    config = uvicorn.Config(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.rnd_log_level.lower(),
    )
    await uvicorn.Server(config).serve()


if __name__ == "__main__":
    asyncio.run(serve())
