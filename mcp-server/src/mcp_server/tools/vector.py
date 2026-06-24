"""MCP 도구: 시맨틱 벡터 검색 (Milvus Hybrid Search)"""
from __future__ import annotations
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


def register_vector_tools(mcp: FastMCP) -> None:
    from infrastructure.component_factory import repository_factory

    @mcp.tool()
    def semantic_search(
        query: str,
        node_type: str = "",
        top_k: int = 20,
        dense_weight: float = 0.3,
        sparse_weight: float = 0.7,
    ) -> list[dict[str, Any]]:
        """
        시맨틱 벡터 검색 (Milvus Hybrid: Dense COSINE + BM25 Sparse).
        자연어 질문의 의미를 이해하여 관련 R&D 엔티티(논문·특허·기술·연구자·과제)를 검색합니다.
        구체적인 키워드를 모를 때, 또는 복수 도메인에 걸쳐 탐색할 때 가장 먼저 호출하세요.

        Args:
            query:         검색 쿼리 (자연어. 예: 'AI 반도체 저전력 설계 연구')
            node_type:     노드 타입 필터 (Paper/Patent/Technology/Researcher/Project, 빈값=전체)
            top_k:         반환할 최대 결과 수
            dense_weight:  Dense 벡터 가중치 (0~1, 의미 유사도 중시 시 높게)
            sparse_weight: Sparse BM25 가중치 (0~1, 키워드 일치 중시 시 높게)

        Returns:
            [{id, node_type, name, score}, ...] — id를 get_entities에 전달하여 상세 조회 가능
        """
        search_fn = repository_factory.get_vector_search_fn()
        if search_fn is None:
            return [{"error": "Milvus 미설정 — MILVUS_HOST 환경변수를 확인하세요."}]

        _SCORE_THRESHOLD = 0.3
        results = search_fn(
            query=query,
            node_type=node_type,
            top_k=top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )
        return [
            {"id": nid, "node_type": ntype, "name": name, "score": round(score, 4)}
            for nid, score, ntype, name in results
            if score >= _SCORE_THRESHOLD
        ]
