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
        특정 타입의 엔티티 목록을 시맨틱 검색으로 가져올 때 사용합니다.
        동일 타입 엔티티(예: 논문 목록, 기술 목록)만 필요하거나, 관계 탐색 없이 ID를 수집할 때 호출하세요.
        다른 타입으로 연결된 엔티티가 필요하면(예: 기술→연구자, 논문→저자) semantic_graph_search를 사용하세요.

        Args:
            query:         검색 쿼리 (자연어. 예: 'AI 반도체 저전력 설계 연구')
            node_type:     노드 타입 필터 (Paper/Patent/Technology/Researcher/Project/Organization, 빈값=전체)
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
