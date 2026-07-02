from __future__ import annotations
from fastapi import APIRouter, Request

router = APIRouter()


import httpx
from config import get_settings

@router.get("/health")
async def health(request: Request) -> dict:
    settings = get_settings()
    mcp_ok = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(settings.mcp_server_url)
            mcp_ok = resp.status_code in (200, 404, 405) # if sse endpoint, it might return 405 or 200
    except Exception:
        pass

    return {
        "status":          "ok",
        "mcp_connected":   mcp_ok,
        "redis_connected": getattr(request.app.state, "redis_connected", False),
    }
