"""MCP 도구: 그래프 탐색 (Neo4j)"""
from __future__ import annotations
import logging
import re
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


def register_graph_tools(mcp: FastMCP) -> None:
    from infrastructure.component_factory import repository_factory

    @mcp.tool()
    def get_researcher_network(researcher_name: str) -> list[dict[str, Any]]:
        """
        연구자 네트워크 조회 (Neo4j).
        연구자가 발표한 논문, 발명 특허, 소속 기관, 연구 기술 분야를 반환합니다.

        Args:
            researcher_name: 연구자 이름 (부분 일치 검색)
        """
        fn = repository_factory.get_researcher_network_fn()
        if fn is None:
            return [{"error": "Neo4j 미설정 — NEO4J_URI 환경변수를 확인하세요."}]
        return fn(researcher_name)

    @mcp.tool()
    def get_citation_graph(paper_title: str, depth: int = 2) -> list[dict[str, Any]]:
        """
        논문 인용 네트워크 조회 (Neo4j).
        특정 논문으로부터 depth 홉 이내의 인용 관계를 반환합니다.

        Args:
            paper_title: 논문 제목 (부분 일치)
            depth:       탐색 홉 수 (1~3)
        """
        fn = repository_factory.get_citation_graph_fn()
        if fn is None:
            return [{"error": "Neo4j 미설정 — NEO4J_URI 환경변수를 확인하세요."}]
        return fn(paper_title, min(depth, 3))

    @mcp.tool()
    def run_graph_query(cypher: str) -> list[dict[str, Any]]:
        """
        Cypher 쿼리 직접 실행 (Neo4j). MATCH/RETURN 기반 READ 전용.
        이미 확보한 엔티티 ID로 그래프를 탐색하거나 집계(COUNT/DISTINCT)가 필요할 때 사용합니다.
        semantic_graph_search로 표현하기 어려운 복잡한 패턴에 활용하세요.

        [관계 타입] AUTHORED · INVENTED · RESEARCHES · WORKS_AT · CITES · EMPLOYS · USES
        [노드 레이블] Researcher · Paper · Patent · Technology · Project · Organization
        [방향 예시]
          (Researcher)-[:AUTHORED]->(Paper)
          (Researcher)-[:RESEARCHES]->(Technology)
          (Project)-[:EMPLOYS]->(Researcher)
          (Project)-[:USES]->(Technology)

        Args:
            cypher: READ 전용 Cypher 쿼리 (MATCH/RETURN만 허용)
        """
        fn = repository_factory.get_graph_query_fn()
        if fn is None:
            return [{"error": "Neo4j 미설정 — NEO4J_URI 환경변수를 확인하세요."}]

        # 관계 타입 대소문자 자동 정규화: [:employs] → [:EMPLOYS]
        cypher = re.sub(r'\[:([A-Za-z_]+)\]', lambda m: f'[:{m.group(1).upper()}]', cypher)

        _WRITE_KEYWORDS = {"CREATE", "MERGE", "DELETE", "DETACH", "SET", "REMOVE", "DROP", "CALL"}
        if any(kw in cypher.upper() for kw in _WRITE_KEYWORDS):
            return [{"error": "쓰기 작업 차단. MATCH/RETURN 전용 쿼리만 허용됩니다."}]

        try:
            return fn(cypher)
        except Exception as e:
            logger.warning("[Neo4j] Cypher 실행 오류: %s", e)
            return [{"error": str(e)}]
