from __future__ import annotations
import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from config import get_settings

router = APIRouter()


@router.get("/stats")
async def get_stats() -> JSONResponse:
    settings = get_settings()
    # mcp_server_url = "http://host:port/sse" → "http://host:port/stats"
    base_url = settings.mcp_server_url.rsplit("/", 1)[0]
    stats_url = f"{base_url}/stats"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(stats_url, timeout=5.0)
            resp.raise_for_status()
            return JSONResponse(resp.json())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=503)
