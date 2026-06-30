from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel


class ConfigOverride(BaseModel):
    generate_model:  Optional[str]   = None
    max_iterations:  Optional[int]   = None
    temperature:     Optional[float] = None
    semantic_top_k:  Optional[int]   = None
    keyword_weight:  Optional[float] = None


class QueryRequest(BaseModel):
    query:      str
    session_id: Optional[str]            = None
    config:     Optional[ConfigOverride] = None


class HealthResponse(BaseModel):
    status:          str
    mcp_connected:   bool
    redis_connected: bool


class ConfigPatchRequest(BaseModel):
    updates: dict[str, Any]

    model_config = {"arbitrary_types_allowed": True}
