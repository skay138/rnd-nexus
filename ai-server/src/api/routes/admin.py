from __future__ import annotations
import logging
from typing import Any

from fastapi import APIRouter, Request
from api.schemas import ConfigPatchRequest

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/admin/config")
async def get_config(request: Request) -> dict[str, Any]:
    return request.app.state.config_repo.all()


@router.patch("/admin/config")
async def patch_config(body: ConfigPatchRequest, request: Request) -> dict[str, Any]:
    repo = request.app.state.config_repo
    for key, value in body.updates.items():
        repo.set(key, value)
    logger.info("[admin] 설정 업데이트: %s", list(body.updates.keys()))
    return repo.all()
