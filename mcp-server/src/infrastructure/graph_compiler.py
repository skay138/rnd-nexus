"""
R&D Nexus Graph Compiler
Neo4j 관계 타입과 Cypher 쿼리 생성 유틸리티.
"""

from __future__ import annotations
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# 단일 소스 오브 트루스 — 실제 Neo4j 관계 타입 7개
_NEO4J_RELATIONS: dict[str, tuple[str, str]] = {
    "AUTHORED":   ("Researcher", "Paper"),
    "INVENTED":   ("Researcher", "Patent"),
    "RESEARCHES": ("Researcher", "Technology"),
    "WORKS_AT":   ("Researcher", "Organization"),
    "CITES":      ("Paper",      "Paper"),
    "EMPLOYS":    ("Project",    "Researcher"),
    "USES":       ("Project",    "Technology"),
}

SCHEMA_HINT = """
Neo4j 노드: Paper | Patent | Researcher | Technology | Project | Organization

관계 타입 (run_graph_query Cypher 및 semantic_graph_search hops 모두 동일):
  AUTHORED   (Researcher)-[:AUTHORED]->(Paper)
  INVENTED   (Researcher)-[:INVENTED]->(Patent)
  RESEARCHES (Researcher)-[:RESEARCHES]->(Technology)
  WORKS_AT   (Researcher)-[:WORKS_AT]->(Organization)
  CITES      (Paper)-[:CITES]->(Paper)
  EMPLOYS    (Project)-[:EMPLOYS]->(Researcher)
  USES       (Project)-[:USES]->(Technology)

hops direction (semantic_graph_search):
  direction="out" (기본): (현재노드)-[:REL]->(다음노드)
  direction="in":         (현재노드)<-[:REL]-(다음노드)

예시:
  Technology→Researcher: {"relation":"RESEARCHES","direction":"in","target_type":"Researcher"}
  Paper→Researcher:      {"relation":"AUTHORED","direction":"in","target_type":"Researcher"}
  Project→Researcher:    {"relation":"EMPLOYS","direction":"out","target_type":"Researcher"}
  Technology→Project:    {"relation":"USES","direction":"in","target_type":"Project"}
  Paper→인용Paper:        {"relation":"CITES","direction":"out","target_type":"Paper"}
"""

# WHERE IN 절에 넣을 최대 ID 수
_MAX_IDS = 200
# 노드 표시명 우선순위 (Paper/Patent/Project은 title, 나머지는 name)
_NAME_EXPR = "COALESCE(n2.name, n2.title, n2.id)"


def compile_hop(
    start_ids: list[str],
    relation: str,
    direction: str,
    target_type: str,
    from_type: str = "",
    limit: int = 100,
    exclude_ids: Optional[list[str]] = None,
) -> str:
    """
    단일 홉 READ-ONLY Cypher 쿼리 문자열 반환 (실행은 호출자가 담당).

    Args:
        start_ids:   시작 노드 ID 목록
        relation:    Neo4j 관계 타입 (예: "AUTHORED", "EMPLOYS")
        direction:   "out" = (start)-[:REL]->(target), "in" = (start)<-[:REL]-(target)
        target_type: 도착 노드 레이블 (예: "Researcher")
        from_type:   시작 노드 레이블 (선택, 인덱스 효율화용)
        limit:       반환 행 최대 수
        exclude_ids: 결과에서 제외할 ID 목록

    Returns:
        MATCH ... WHERE ... RETURN id, name, start_id LIMIT n 형태의 Cypher 문자열
    """
    rel = relation.upper()
    if rel not in _NEO4J_RELATIONS:
        valid = ", ".join(sorted(_NEO4J_RELATIONS))
        raise ValueError(f"알 수 없는 관계 타입: '{rel}'. 유효: {valid}")

    ids_literal = "[" + ", ".join(f'"{i}"' for i in start_ids[:_MAX_IDS]) + "]"

    exclude_clause = ""
    if exclude_ids:
        ex_literal = "[" + ", ".join(f'"{i}"' for i in exclude_ids[:_MAX_IDS]) + "]"
        exclude_clause = f" AND NOT n2.id IN {ex_literal}"

    n1_label = f":{from_type}" if from_type else ""
    if direction == "out":
        pattern = f"(n1{n1_label})-[:{rel}]->(n2:{target_type})"
    else:
        pattern = f"(n1{n1_label})<-[:{rel}]-(n2:{target_type})"

    return (
        f"MATCH {pattern} "
        f"WHERE n1.id IN {ids_literal}{exclude_clause} "
        f"RETURN n2.id AS id, {_NAME_EXPR} AS name, n1.id AS start_id "
        f"LIMIT {limit}"
    )
