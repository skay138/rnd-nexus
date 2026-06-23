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
    인터페이스: (researcher_name: str) -> list[dict[str, Any]]
    """
    def fetch_researcher_network(researcher_name: str) -> list[dict[str, Any]]:
        t0 = time.perf_counter()
        with driver.session() as session:
            result = session.run(
                """
                MATCH (r:Researcher)
                WHERE r.name CONTAINS $name
                OPTIONAL MATCH (r)-[:AUTHORED]->(p:Paper)
                OPTIONAL MATCH (r)-[:INVENTED]->(pat:Patent)
                OPTIONAL MATCH (r)-[:WORKS_AT]->(org:Organization)
                OPTIONAL MATCH (r)-[:RESEARCHES]->(t:Technology)
                RETURN
                    r.name   AS researcher,
                    r.id     AS researcher_id,
                    collect(DISTINCT p.title)[..5]   AS papers,
                    collect(DISTINCT pat.title)[..5] AS patents,
                    org.name AS organization,
                    collect(DISTINCT t.name)[..10]   AS technologies
                LIMIT 10
                """,
                name=researcher_name,
            )
            rows: list[dict[str, Any]] = [dict(r.data()) for r in result]
        logger.debug("[Neo4j] researcher_network '%s': %d rows  %.1f ms",
                     researcher_name, len(rows), (time.perf_counter() - t0) * 1000)
        return rows
    return fetch_researcher_network


def make_fetch_citation_graph_fn(driver: Any) -> Any:
    """
    논문 인용 네트워크 조회 콜백.
    인터페이스: (paper_title: str, depth: int) -> list[dict[str, Any]]
    """
    def fetch_citation_graph(paper_title: str, depth: int = 2) -> list[dict[str, Any]]:
        t0 = time.perf_counter()
        safe_depth = min(max(depth, 1), 3)
        cypher = (
            "MATCH path = (p:Paper)-[:CITES*1.." + str(safe_depth) + "]->(cited:Paper) "
            "WHERE p.title CONTAINS $title "
            "RETURN p.title AS source, cited.title AS target, "
            "cited.year AS year, length(path) AS hops "
            "LIMIT 50"
        )
        with driver.session() as session:
            result = session.run(cypher, title=paper_title)
            rows: list[dict[str, Any]] = [dict(r.data()) for r in result]
        logger.debug("[Neo4j] citation_graph '%s': %d edges  %.1f ms",
                     paper_title, len(rows), (time.perf_counter() - t0) * 1000)
        return rows
    return fetch_citation_graph
