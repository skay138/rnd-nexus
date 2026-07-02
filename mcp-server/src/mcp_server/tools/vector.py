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
        keyword_weight: float = 0.5,
        year_from: int = 0,
        year_to: int = 0,
    ) -> list[dict[str, Any]]:
        """
        <role>
        자연어 쿼리로 엔티티를 시맨틱 검색합니다 (Milvus Dense+BM25 하이브리드).
        동일 타입 엔티티(논문 목록, 기술 목록 등) ID 수집에 사용하세요.
        </role>

        <instructions>
        - 동일 타입 엔티티를 키워드/의미로 검색할 때 사용하세요
        - 다른 타입으로 연결된 엔티티(기술→연구자, 논문→저자)는 semantic_graph_search를 사용하세요
        - 반환된 id를 get_entities에 전달하면 상세 정보 조회 가능
        - name 필드만 필요한 경우 get_entities 생략 가능
        </instructions>

        <constraints>
        - Milvus 미설정 시 에러 반환 — 연도/기관 필터만 필요하면 filter_entities를 사용하세요
        - keyword_weight + (1 - keyword_weight) = 1.0 (합산 자동 보정)
        - score 0.3 미만 결과는 자동 제외 — 결과가 없으면 다른 키워드·표현으로 재시도하세요
        </constraints>

        Args:
            query:          검색 쿼리 (자연어. 예: 'AI 반도체 저전력 설계 연구')
            node_type:      노드 타입 필터 (Paper/Patent/Technology/Researcher/Project/Organization, 빈값=전체)
            top_k:          반환할 최대 결과 수
            keyword_weight: BM25 키워드 가중치 (0~1, 기본 0.5. 높을수록 키워드 일치 중시)
            year_from:      연도 하한 필터 (0=제한없음)
            year_to:        연도 상한 필터 (0=제한없음)

        Returns:
            [{id, node_type, name, score}, ...] — score 내림차순 정렬
            score는 내부 검색 유사도 (0~1)
        """
        search_fn = repository_factory.get_vector_search_fn()
        if search_fn is None:
            return [{"error": "Milvus 미설정 — MILVUS_HOST 환경변수를 확인하세요."}]

        _dense_weight  = 1.0 - keyword_weight
        _sparse_weight = keyword_weight

        _SCORE_THRESHOLD = 0.3
        results = search_fn(
            query=query,
            node_type=node_type,
            top_k=top_k,
            dense_weight=_dense_weight,
            sparse_weight=_sparse_weight,
            year_from=year_from,
            year_to=year_to,
        )
        return [
            {"id": nid, "node_type": ntype, "name": name, "score": round(score, 4)}
            for nid, score, ntype, name in results
            if score >= _SCORE_THRESHOLD
        ]
