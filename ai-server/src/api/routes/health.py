from __future__ import annotations
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict:
    return {
        "status":          "ok",
        "mcp_connected":   getattr(request.app.state, "mcp_connected",   False),
        "redis_connected": getattr(request.app.state, "redis_connected", False),
    }
