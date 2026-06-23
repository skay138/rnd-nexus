"""
Config Service (R&D Nexus)
- QueryConfig:   요청별 설정값 (API 파라미터 또는 DB 기본값)
- RequestConfig: 현재 요청 설정 접근자 (ContextVar 기반)
                 우선순위: API 파라미터 > ConfigRepository(DB) > 내장 기본값
"""

from __future__ import annotations
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, ClassVar, Optional, Protocol, runtime_checkable

# 순수 알고리즘 기본값만 보관 — 모델명/replan은 config.py(env)가 단일 출처
CONFIG_DEFAULTS: dict[str, Any] = {
    "temperature":     0.0,
    "semantic_top_k":  20,
    "dense_weight":    0.3,
    "sparse_weight":   0.7,
}


@dataclass
class QueryConfig:
    """요청별 설정값. None 필드는 _resolve()가 채운다."""
    generate_model:  Optional[str]   = None  # generate 노드 모델명
    max_replan:      Optional[int]   = None  # reflection 루프 최대 횟수
    temperature:     Optional[float] = None  # LLM temperature
    semantic_top_k:  Optional[int]   = None  # Milvus 시맨틱 검색 top-k
    dense_weight:    Optional[float] = None  # hybrid search dense 가중치
    sparse_weight:   Optional[float] = None  # hybrid search sparse 가중치


@runtime_checkable
class ConfigRepository(Protocol):
    def get(self, key: str) -> Any: ...


class RequestConfig:
    """
    현재 요청 설정 접근자.
    API 파라미터 > ConfigRepository(DB) > 내장 기본값 순.

    요청 시작 시:
        resolved = RequestConfig._resolve(repo, api_override)
        RequestConfig.set_current(resolved, original_query=body.query)

    노드/도구에서:
        cfg = RequestConfig.current()
        cfg.generate_model   # str, 항상 non-None
        cfg.max_replan       # int, 항상 non-None
    """

    _DEFAULTS: ClassVar[dict[str, Any]] = CONFIG_DEFAULTS
    _ctx: ClassVar[ContextVar[Optional["RequestConfig"]]] = ContextVar(
        "_rnd_request_config_ctx", default=None
    )

    def __init__(
        self,
        resolved: Optional[QueryConfig] = None,
        original_query: str = "",
    ) -> None:
        self._resolved       = resolved
        self._original_query = original_query

    @staticmethod
    def _resolve(
        repo:     Optional[ConfigRepository],
        override: Optional[QueryConfig] = None,
    ) -> QueryConfig:
        """API 파라미터 > repo(DB) > config.py(env) > 내장 기본값 순으로 완전한 QueryConfig 반환."""
        from config import get_settings  # 순환 import 방지용 지연 import
        s = get_settings()

        o = override or QueryConfig()

        def _pick(key: str, api_val: Any) -> Any:
            if api_val is not None:
                return api_val
            if repo is not None:
                val = repo.get(key)
                if val is not None:
                    return val
            # config.py(env) 기반 fallback — 모델명/replan은 여기서 처리
            if key == "generate_model":
                return s.rnd_model_generate
            if key == "max_replan":
                return s.rnd_max_replan
            return RequestConfig._DEFAULTS.get(key)

        return QueryConfig(
            generate_model = _pick("generate_model", o.generate_model),
            max_replan     = _pick("max_replan",     o.max_replan),
            temperature    = _pick("temperature",    o.temperature),
            semantic_top_k = _pick("semantic_top_k", o.semantic_top_k),
            dense_weight   = _pick("dense_weight",   o.dense_weight),
            sparse_weight  = _pick("sparse_weight",  o.sparse_weight),
        )

    @classmethod
    def set_current(
        cls,
        resolved: Optional[QueryConfig],
        original_query: str = "",
    ) -> None:
        cls._ctx.set(cls(resolved, original_query))

    @classmethod
    def current(cls) -> "RequestConfig":
        inst = cls._ctx.get()
        return inst if inst is not None else cls()

    def get(self, key: str, fallback: Any = None) -> Any:
        if self._resolved is not None:
            val = getattr(self._resolved, key, None)
            if val is not None:
                return val
        default = self._DEFAULTS.get(key)
        return default if default is not None else fallback

    @property
    def generate_model(self) -> str:
        return self.get("generate_model")

    @property
    def max_replan(self) -> int:
        return self.get("max_replan")

    @property
    def temperature(self) -> float:
        return self.get("temperature")

    @property
    def semantic_top_k(self) -> int:
        return self.get("semantic_top_k")

    @property
    def dense_weight(self) -> float:
        return self.get("dense_weight")

    @property
    def sparse_weight(self) -> float:
        return self.get("sparse_weight")

    @property
    def original_query(self) -> str:
        return self._original_query
