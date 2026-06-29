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
    def get_researcher_network(researcher_name: str = "", researcher_id: str = "") -> list[dict[str, Any]]:
        """
        연구자 네트워크 조회 (Neo4j).
        연구자가 발표한 논문, 발명 특허, 소속 기관, 연구 기술 분야를 반환합니다.

        동명이인 방지를 위해 researcher_id(예: 'R006')를 우선 사용하세요.
        ID를 모를 때만 researcher_name(부분 일치)으로 조회하세요.

        Args:
            researcher_name: 연구자 이름 (부분 일치, ID 미확보 시 사용)
            researcher_id:   연구자 ID (예: 'R006') — 정확 매칭, 동명이인 방지
        """
        fn = repository_factory.get_researcher_network_fn()
        if fn is None:
            return [{"error": "Neo4j 미설정 — NEO4J_URI 환경변수를 확인하세요."}]
        if not researcher_name and not researcher_id:
            return [{"error": "researcher_name 또는 researcher_id 중 하나는 필수입니다."}]
        return fn(researcher_name=researcher_name, researcher_id=researcher_id)

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
        관계 방향이 헷갈리면 semantic_graph_search를 사용하세요.

        [전체 관계 타입 — 방향 정확히 준수]
          (Researcher)-[:AUTHORED]->(Paper)
          (Researcher)-[:INVENTED]->(Patent)
          (Researcher)-[:RESEARCHES]->(Technology)
          (Researcher)-[:WORKS_AT]->(Organization)
          (Paper)-[:CITES]->(Paper)
          (Project)-[:EMPLOYS]->(Researcher)   ← EMPLOYED_BY 없음. 연구자→과제는 역방향으로
          (Project)-[:USES]->(Technology)

        [노드 ID 속성] 모든 노드에 .id 속성 사용
          Researcher.id='R001'  Paper.id='P001'  Patent.id='KR10-...'
          Technology.id='T001'  Project.id='RS-2024-...'  Organization.id='ORG001'

        [올바른 역방향 패턴 예시]
          # 연구자가 참여하는 과제: Project가 주어
          MATCH (p:Project)-[:EMPLOYS]->(r:Researcher {id: 'R007'}) RETURN p.id, p.title LIMIT 10
          # 기술을 활용하는 과제:
          MATCH (p:Project)-[:USES]->(t:Technology {id: 'T001'}) RETURN p.id, p.title LIMIT 10
          # 논문 저자 연구자:
          MATCH (r:Researcher)-[:AUTHORED]->(p:Paper {id: 'P001'}) RETURN r.id, r.name

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

        _INVERSE_HINTS = {
            "USED_BY":      "(p:Project)-[:USES]->(t:Technology) — Project가 주어",
            "EMPLOYED_BY":  "(p:Project)-[:EMPLOYS]->(r:Researcher) — Project가 주어",
            "AUTHORED_BY":  "(r:Researcher)-[:AUTHORED]->(p:Paper) — Researcher가 주어",
            "INVENTED_BY":  "(r:Researcher)-[:INVENTED]->(p:Patent) — Researcher가 주어",
            "CITED_BY":     "(p:Paper)-[:CITES]->(p2:Paper) — 인용하는 Paper가 주어",
            "WORKS_AT_BY":  "(r:Researcher)-[:WORKS_AT]->(o:Organization)",
        }
        cypher_upper = cypher.upper()
        for inv, hint in _INVERSE_HINTS.items():
            if inv in cypher_upper:
                return [{"error": f"존재하지 않는 관계 타입 '{inv}'. 올바른 패턴: {hint}"}]

        try:
            return fn(cypher)
        except Exception as e:
            logger.warning("[Neo4j] Cypher 실행 오류: %s", e)
            return [{"error": str(e)}]
