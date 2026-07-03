"""MCP 도구: 그래프 탐색 (Neo4j)"""
from __future__ import annotations
import logging
import re
from typing import Any

from mcp.server.fastmcp import FastMCP
from infrastructure.graph_compiler import _NEO4J_RELATIONS, SCHEMA_HINT

logger = logging.getLogger(__name__)


def register_graph_tools(mcp: FastMCP) -> None:
    from infrastructure.component_factory import repository_factory

    @mcp.tool()
    def get_researcher_network(researcher_name: str = "", researcher_id: str = "") -> list[dict[str, Any]]:
        """
        <role>
        특정 연구자의 논문·특허·소속 기관·연구 기술 분야를 한 번에 조회합니다.
        연구자 프로필이나 전문 분야 파악에 사용하세요.
        </role>

        <instructions>
        - 동명이인 방지를 위해 researcher_id('R006' 형태)를 우선 사용하세요
        - ID를 모를 때만 researcher_name(부분 일치)으로 조회하세요
        - researcher_id는 semantic_search → get_entities 또는 run_graph_query로 확보 가능합니다
        </instructions>

        <constraints>
        - researcher_name 또는 researcher_id 중 하나는 필수입니다
        </constraints>

        Args:
            researcher_name: 연구자 이름 (부분 일치, ID 미확보 시 사용)
            researcher_id:   연구자 ID (예: 'R006') — 정확 매칭, 동명이인 방지

        Returns:
            [{name, researcher_id, papers, patents, organization, technologies}, ...]
            researcher_id 매칭 시 1행, 이름 부분 일치 시 최대 10행 (동명이인 각 1행)
            papers/patents: [{id, title}] 최대 5건 — id·title만 포함하므로
            출판 연도·저자·초록 등 상세가 필요하면 이 id로 get_entities(Paper/Patent)를 호출하세요
            organization: 소속 기관명, technologies: 연구 기술명 목록
            연구자의 h_index·specialty 등 상세 필드는 get_entities(Researcher)로 조회하세요
        """
        fn = repository_factory.get_researcher_network_fn()
        if fn is None:
            return [{"error": "Neo4j 미설정 — NEO4J_URI 환경변수를 확인하세요."}]
        if not researcher_name and not researcher_id:
            return [{"error": "researcher_name 또는 researcher_id 중 하나는 필수입니다."}]
        return fn(researcher_name=researcher_name, researcher_id=researcher_id)

    @mcp.tool()
    def get_citation_graph(paper_title: str = "", paper_id: str = "", depth: int = 2) -> list[dict[str, Any]]:
        """
        <role>
        특정 논문으로부터 depth 홉 이내의 인용 관계를 조회합니다.
        논문의 영향력, 연구 계보 파악에 사용하세요.
        </role>

        <instructions>
        - paper_id(예: 'P001')가 있으면 우선 사용하세요 — 제목 부분 일치보다 정확합니다
        - paper_id는 semantic_search(node_type="Paper")로 수집 가능합니다
        - depth=1: 직접 인용, depth=2: 2단계 인용 체인 (기본값)
        </instructions>

        <constraints>
        - paper_title 또는 paper_id 중 하나는 필수입니다
        - depth 최대 3 (그 이상은 내부적으로 3으로 제한)
        </constraints>

        Args:
            paper_title: 논문 제목 (부분 일치, ID 미확보 시 사용)
            paper_id:    논문 ID (예: 'P001') — 정확 매칭 우선
            depth:       탐색 홉 수 (1~3, 기본 2)

        Returns:
            [{source, source_paper_id, target, target_paper_id, year, hops}, ...]
            source/target: 논문 제목
            source_paper_id/target_paper_id: 논문 ID
            hops: 출발 논문으로부터의 홉 수
        """
        fn = repository_factory.get_citation_graph_fn()
        if fn is None:
            return [{"error": "Neo4j 미설정 — NEO4J_URI 환경변수를 확인하세요."}]
        if not paper_title and not paper_id:
            return [{"error": "paper_title 또는 paper_id 중 하나는 필수입니다."}]
        return fn(paper_title=paper_title, paper_id=paper_id, depth=min(depth, 3))

    @mcp.tool()
    def run_graph_query(cypher: str) -> list[dict[str, Any]]:
        """
        <role>
        Cypher 쿼리를 직접 실행합니다 (Neo4j, MATCH/RETURN READ 전용).
        이미 확보한 ID로 커스텀 탐색이나 집계(COUNT/DISTINCT)가 필요할 때 사용하세요.
        </role>

        <instructions>
        - 이미 알고 있는 엔티티 ID로 관계 탐색: run_graph_query 적합
        - 키워드로 엔티티를 찾아야 할 때: semantic_search / semantic_graph_search 사용
        - 기관별 연구자 수, 기술분야별 논문 수 등 집계에도 사용 가능
        - 엔티티 반환 시 UI 출처 표기를 위해 반드시 노드 라벨을 `node_type`으로 함께 반환하세요 (예: `RETURN p.id AS id, p.title AS title, labels(p)[0] AS node_type`).
        </instructions>

        <constraints>
        - MATCH/RETURN 전용. CREATE/MERGE/DELETE/SET/REMOVE/DROP/CALL 차단
        - 반드시 LIMIT 절 포함 (예: LIMIT 20)
        - 관계 타입·방향은 아래 [스키마] 섹션을 따르세요

        [노드 ID 속성] 모든 노드에 .id 속성 사용
          Researcher.id='R001'  Paper.id='P001'  Patent.id='KR10-...'
          Technology.id='T001'  Project.id='RS-2024-...'  Organization.id='ORG001'
        </constraints>

        Args:
            cypher: READ 전용 Cypher 쿼리 (MATCH/RETURN만 허용, LIMIT 필수)

        Returns:
            [{...row fields...}, ...] — RETURN 절에 지정한 컬럼명으로 구성된 dict 목록
        """
        fn = repository_factory.get_graph_query_fn()
        if fn is None:
            return [{"error": "Neo4j 미설정 — NEO4J_URI 환경변수를 확인하세요."}]

        # 단어 경계 매칭 — 'Dataset'(SET), 'Recall'(CALL) 등 제목 문자열의 부분 일치 오차단 방지.
        # 따옴표 안 문자열의 단독 키워드는 여전히 차단될 수 있으나 READ 전용 보장을 우선한다.
        if re.search(r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|CALL)\b", cypher.upper()):
            return [{"error": "쓰기 작업 차단. MATCH/RETURN 전용 쿼리만 허용됩니다."}]

        unknown = [
            m for m in re.findall(r"\[:([A-Z_]+)\]", cypher)
            if m not in _NEO4J_RELATIONS
        ]
        if unknown:
            valid = ", ".join(sorted(_NEO4J_RELATIONS))
            return [{"error": f"알 수 없는 관계 타입: {unknown}. 유효: {valid}"}]

        # docstring의 'LIMIT 필수'를 코드로 보장 — 누락 시 자동 부여
        if not re.search(r"\bLIMIT\b", cypher, re.IGNORECASE):
            cypher = f"{cypher.rstrip().rstrip(';')} LIMIT 50"

        try:
            return fn(cypher)
        except Exception as e:
            logger.warning("[Neo4j] Cypher 실행 오류: %s", e)
            return [{"error": str(e)}]

    get_researcher_network.__doc__ = (get_researcher_network.__doc__ or "") + SCHEMA_HINT
    get_citation_graph.__doc__     = (get_citation_graph.__doc__     or "") + SCHEMA_HINT
    run_graph_query.__doc__        = (run_graph_query.__doc__        or "") + SCHEMA_HINT
