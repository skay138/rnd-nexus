"""
Neo4j Graph Database Adapter (R&D Nexus)

노드: Paper, Patent, Researcher, Technology, Project, Organization
관계:
  (Researcher)-[:AUTHORED]   -> (Paper)
  (Researcher)-[:INVENTED]   -> (Patent)
  (Paper)     -[:CITES]      -> (Paper)
  (Researcher)-[:WORKS_AT]   -> (Organization)
  (Researcher)-[:RESEARCHES] -> (Technology)
  (Project)   -[:EMPLOYS]    -> (Researcher)
  (Project)   -[:USES]       -> (Technology)
"""

from __future__ import annotations
from typing import Any
import logging
import time

logger = logging.getLogger(__name__)


def make_graph_query_fn(driver: Any) -> Any:
    """
    Neo4j Cypher 쿼리 콜백 생성.
    인터페이스: (cypher: str) -> list[dict[str, Any]]
    """
    def graph_query(cypher: str) -> list[dict[str, Any]]:
        t0 = time.perf_counter()
        with driver.session() as session:
            result = session.run(cypher)
            rows: list[dict[str, Any]] = [dict(r.data()) for r in result]
        logger.debug("[Neo4j] graph_query: %d rows  %.1f ms",
                     len(rows), (time.perf_counter() - t0) * 1000)
        return rows
    return graph_query


def make_fetch_researcher_network_fn(driver: Any) -> Any:
    """
    연구자 네트워크 조회 콜백.
    인터페이스: (researcher_name: str, researcher_id: str) -> list[dict[str, Any]]
    researcher_id 제공 시 ID 정확 매칭 (동명이인 방지), 미제공 시 이름 부분 일치.
    """
    def fetch_researcher_network(researcher_name: str = "", researcher_id: str = "") -> list[dict[str, Any]]:
        t0 = time.perf_counter()
        if researcher_id:
            where_clause = "WHERE r.id = $rid"
            params: dict[str, Any] = {"rid": researcher_id}
            limit = 1
        else:
            where_clause = "WHERE r.name CONTAINS $name"
            params = {"name": researcher_name}
            limit = 10
        # 패턴 컴프리헨션 — OPTIONAL MATCH 조합의 카티전 곱 없이 관계별 수집.
        # papers/patents는 후속 조회(get_entities 등)가 가능하도록 id와 node_type을 함께 반환.
        cypher = f"""
            MATCH (r:Researcher)
            {where_clause}
            RETURN
                r.name AS name,
                r.id   AS researcher_id,
                [(r)-[:AUTHORED]->(p:Paper)    | {{id: p.id, title: p.title, node_type: 'Paper'}}][..5]   AS papers,
                [(r)-[:INVENTED]->(pat:Patent) | {{id: pat.id, title: pat.title, node_type: 'Patent'}}][..5] AS patents,
                [(r)-[:WORKS_AT]->(org:Organization) | org.name][0] AS organization,
                [(r)-[:RESEARCHES]->(t:Technology)   | t.name][..10] AS technologies
            LIMIT {limit}
        """
        with driver.session() as session:
            result = session.run(cypher, **params)
            rows: list[dict[str, Any]] = [dict(r.data()) for r in result]
        logger.debug("[Neo4j] researcher_network id='%s' name='%s': %d rows  %.1f ms",
                     researcher_id, researcher_name, len(rows), (time.perf_counter() - t0) * 1000)
        return rows
    return fetch_researcher_network


def make_fetch_citation_graph_fn(driver: Any) -> Any:
    """
    논문 인용 네트워크 조회 콜백.
    인터페이스: (paper_title: str, paper_id: str, depth: int) -> list[dict[str, Any]]
    paper_id 제공 시 ID 정확 매칭 우선, 미제공 시 paper_title 부분 일치.
    """
    def fetch_citation_graph(paper_title: str, paper_id: str = "", depth: int = 2) -> list[dict[str, Any]]:
        t0 = time.perf_counter()
        safe_depth = min(max(depth, 1), 3)
        if paper_id:
            where = "WHERE p.id = $pid"
            params: dict[str, Any] = {"pid": paper_id}
        else:
            where = "WHERE p.title CONTAINS $title"
            params = {"title": paper_title}
        cypher = (
            f"MATCH path = (p:Paper)-[:CITES*1..{safe_depth}]->(cited:Paper) "
            f"{where} "
            "RETURN p.title AS source, p.id AS source_id, "
            "cited.title AS target, cited.id AS target_id, "
            "cited.year AS year, length(path) AS hops "
            "LIMIT 50"
        )
        with driver.session() as session:
            result = session.run(cypher, **params)
            rows: list[dict[str, Any]] = [dict(r.data()) for r in result]
        logger.debug("[Neo4j] citation_graph id='%s' title='%s': %d edges  %.1f ms",
                     paper_id, paper_title, len(rows), (time.perf_counter() - t0) * 1000)
        return rows
    return fetch_citation_graph
