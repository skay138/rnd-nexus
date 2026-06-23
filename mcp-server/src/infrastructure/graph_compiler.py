"""
R&D Nexus Graph Compiler
LLM의 추상 관계 개념(relation)을 Neo4j Cypher로 변환합니다.
LLM은 Cypher를 직접 작성하지 않고 관계명만 지정합니다.
"""

from __future__ import annotations
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# rnd-nexus Neo4j 스키마 기준 관계 매핑
# key : LLM이 지정하는 추상 개념 (소문자, 언더스코어)
# value: (Neo4j 관계 타입, 방향)  out=정방향  in=역방향
_RELATION_MAP: dict[str, tuple[str, str]] = {
    # Researcher → Paper
    "authored":          ("AUTHORED", "out"),
    "authored_by":       ("AUTHORED", "in"),
    "writes":            ("AUTHORED", "out"),
    "written_by":        ("AUTHORED", "in"),
    # Researcher → Patent
    "invented":          ("INVENTED", "out"),
    "invented_by":       ("INVENTED", "in"),
    # Researcher → Technology
    "researches":        ("RESEARCHES", "out"),
    "researched_by":     ("RESEARCHES", "in"),
    "in_field":          ("RESEARCHES", "in"),
    # Researcher → Organization
    "works_at":          ("WORKS_AT", "out"),
    "has_researcher":    ("WORKS_AT", "in"),
    # Paper → Paper
    "cites":             ("CITES", "out"),
    "cited_by":          ("CITES", "in"),
    # Project → Researcher
    "employs":           ("EMPLOYS", "out"),
    "employed_by":       ("EMPLOYS", "in"),
    # Project → Technology
    "uses":              ("USES", "out"),
    "used_by":           ("USES", "in"),
}

# Neo4j 노드에서 표시명으로 쓸 속성 우선순위 (Paper/Patent/Project은 title, 나머지는 name)
_NAME_EXPR = "COALESCE(n2.name, n2.title, n2.id)"

# WHERE IN 절에 넣을 최대 ID 수 (Cypher 쿼리 크기 제한)
_MAX_IDS = 200


def resolve_relation(relation: str) -> tuple[str, str]:
    """
    추상 관계명 → (Neo4j 관계 타입, 방향).
    알 수 없는 관계명은 대문자 변환 + outbound fallback.
    """
    key = relation.lower().replace(" ", "_")
    if key in _RELATION_MAP:
        return _RELATION_MAP[key]
    logger.warning("[Compiler] 알 수 없는 relation '%s' → fallback: %s out", relation, relation.upper())
    return relation.upper(), "out"


def compile_hop(
    from_type: str,
    relation: str,
    to_type: str,
    start_ids: list[str],
    limit: int = 100,
    exclude_ids: Optional[list[str]] = None,
) -> str:
    """
    단일 홉 READ-ONLY Cypher 생성.

    Returns:
        MATCH 패턴 WHERE n1.id IN [...] RETURN id, name, start_id LIMIT {limit}
    """
    rel_type, direction = resolve_relation(relation)

    ids_literal = "[" + ", ".join(f'"{i}"' for i in start_ids[:_MAX_IDS]) + "]"

    exclude_clause = ""
    if exclude_ids:
        ex_literal = "[" + ", ".join(f'"{i}"' for i in exclude_ids[:_MAX_IDS]) + "]"
        exclude_clause = f" AND NOT n2.id IN {ex_literal}"

    if direction == "out":
        pattern = f"(n1:{from_type})-[:{rel_type}]->(n2:{to_type})"
    elif direction == "in":
        pattern = f"(n1:{from_type})<-[:{rel_type}]-(n2:{to_type})"
    else:
        pattern = f"(n1:{from_type})-[:{rel_type}]-(n2:{to_type})"

    return (
        f"MATCH {pattern} "
        f"WHERE n1.id IN {ids_literal}{exclude_clause} "
        f"RETURN n2.id AS id, {_NAME_EXPR} AS name, n1.id AS start_id "
        f"LIMIT {limit}"
    )


def get_relation_schema() -> dict[str, tuple[str, str]]:
    """LLM 프롬프트용 — 사용 가능한 관계 목록 반환."""
    return dict(_RELATION_MAP)
