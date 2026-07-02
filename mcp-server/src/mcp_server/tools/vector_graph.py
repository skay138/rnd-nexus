"""MCP 도구: 시맨틱 벡터 진입 + 그래프 다중 홉 탐색"""
from __future__ import annotations
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP
from infrastructure.graph_compiler import compile_hop, SCHEMA_HINT

logger = logging.getLogger(__name__)


def register_vector_graph_tools(mcp: FastMCP) -> None:
    from infrastructure.component_factory import repository_factory

    @mcp.tool()
    def semantic_graph_search(
        query: str,
        entry_type: str,
        hops: list[dict],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        <role>
        관계를 통해 연결된 다른 타입의 엔티티를 찾을 때 사용합니다 (예: 기술→연구자, 과제→연구자, 논문→저자).
        동일 타입 엔티티만 필요하면 semantic_search를 사용하세요.
        </role>

        <instructions>
        - entry_type 엔티티를 시맨틱 검색으로 찾은 뒤, hops에 정의된 Neo4j 관계를 따라 target_type으로 이동합니다
        - 2~3홉 연결도 지원합니다 (예: 기술→연구자→기관)
        - 각 hop: {"relation": "ACTUAL_TYPE", "direction": "in"|"out", "target_type": "NodeLabel"}
          direction="out": (현재노드)-[:REL]->(target)  /  direction="in": (현재노드)<-[:REL]-(target)
          direction 생략 시 "out" 기본값

        [사용 예시]
        1. "AI 반도체 분야 핵심 연구자 추천"
           entry_type="Technology",
           hops=[{"relation":"RESEARCHES","direction":"in","target_type":"Researcher"}]

        2. "뉴로모픽 논문 저자들의 소속 기관"
           entry_type="Paper",
           hops=[{"relation":"AUTHORED",  "direction":"in", "target_type":"Researcher"},
                 {"relation":"WORKS_AT",  "direction":"out","target_type":"Organization"}]

        3. "PIM 기술을 활용한 국가 R&D 과제"
           entry_type="Technology",
           hops=[{"relation":"USES","direction":"in","target_type":"Project"}]

        4. "특정 논문을 인용한 후속 연구 논문"
           entry_type="Paper",
           hops=[{"relation":"CITES","direction":"out","target_type":"Paper"}]

        5. "AI 반도체 연구자가 참여하는 과제"
           entry_type="Researcher",
           hops=[{"relation":"EMPLOYS","direction":"in","target_type":"Project"}]
        </instructions>

        <constraints>
        - query에 엔티티 ID (예: 'P011', 'R002')를 입력하지 마세요.
        - 이미 정확한 엔티티 ID를 알고 있다면, `run_graph_query` 도구를 사용하여 직접 매칭하세요 (예: MATCH (p:Paper {id: 'P011'})<-[:AUTHORED]-(r:Researcher) RETURN ...).
        </constraints>

        Args:
            query:      시맨틱 검색 쿼리 (자연어. 예: "AI 반도체 저전력 설계")
            entry_type: 진입 노드 타입 (Paper / Patent / Technology / Researcher / Project / Organization)
            hops:       관계 홉 목록. 각 항목: {"relation": str, "direction": "in"|"out", "target_type": str}
                        최대 3홉 권장 (성능상)
            top_k:      최종 반환할 최대 결과 수

        Returns:
            [{id, name, node_type, score, path}, ...] — score 내림차순 정렬
            path: "semantic(Technology:'뉴로모픽') -[researched_by]-> Researcher('홍길동')" 형태
            score: 진입 엔티티의 유사도가 홉을 통해 전파된 값 (0~1)
        """
        search_fn = repository_factory.get_vector_search_fn()
        if search_fn is None:
            return [{"error": "Milvus 미설정 — MILVUS_HOST 환경변수를 확인하세요."}]

        # ── Step 1: Milvus 시맨틱 진입 검색 ──────────────────────────────────
        entry_results = search_fn(
            query=query,
            node_type=entry_type,
            top_k=top_k * 3,
        )
        if not entry_results:
            logger.info("[GraphSearch] entry '%s' (%s): 결과 없음", query, entry_type)
            return []

        # (id, score, node_type, name)
        scores:     dict[str, float] = {r[0]: r[1] for r in entry_results}
        names:      dict[str, str]   = {r[0]: r[3] for r in entry_results}
        provenance: dict[str, str]   = {
            r[0]: f"semantic({entry_type}:'{r[3]}')" for r in entry_results
        }
        current_ids  = list(scores.keys())
        current_type = entry_type
        visited: set[str] = set(current_ids)

        logger.info("[GraphSearch] entry '%s' (%s): %d건", query, entry_type, len(current_ids))

        # ── Step 2: Neo4j 그래프 홉 탐색 + 점수 전파 ────────────────────────
        graph_fn = repository_factory.get_graph_query_fn()

        if graph_fn is None or not hops:
            sorted_ids = sorted(current_ids, key=lambda x: scores[x], reverse=True)[:top_k]
            return [
                {
                    "id":        nid,
                    "name":      names.get(nid, ""),
                    "node_type": current_type,
                    "score":     round(scores[nid], 4),
                    "path":      provenance.get(nid, ""),
                }
                for nid in sorted_ids
            ]

        for hop_idx, hop in enumerate(hops):
            relation    = hop.get("relation", "")
            target_type = hop.get("target_type", "")
            direction   = hop.get("direction", "out")

            if not relation or not target_type:
                logger.warning("[GraphSearch] hop[%d] 형식 오류 (relation/target_type 필수): %s", hop_idx, hop)
                continue

            try:
                cypher = compile_hop(
                    start_ids=current_ids,
                    relation=relation,
                    direction=direction,
                    target_type=target_type,
                    from_type=current_type,
                    limit=top_k * 10,
                    exclude_ids=list(visited - set(current_ids)),
                )
            except ValueError as exc:
                logger.warning("[GraphSearch] hop[%d] compile_hop 오류: %s", hop_idx, exc)
                break
            logger.debug("[GraphSearch] hop[%d] cypher: %s", hop_idx, cypher)

            try:
                rows = graph_fn(cypher)
            except Exception as e:
                logger.warning("[GraphSearch] hop[%d] 실행 오류 (%s): %s", hop_idx, relation, e)
                break

            if not rows:
                logger.info("[GraphSearch] hop[%d] '%s': 결과 없음 — 탐색 중단", hop_idx, relation)
                break

            # 점수 전파: 자식 노드는 부모의 entry score를 상속
            hop_scores:     dict[str, float] = {}
            hop_names:      dict[str, str]   = {}
            hop_provenance: dict[str, str]   = {}

            for row in rows:
                nid    = str(row.get("id", ""))
                src_id = str(row.get("start_id", ""))
                name   = row.get("name") or nid

                parent_score = scores.get(src_id, 0.0) if src_id else 0.0

                if nid not in hop_scores or parent_score > hop_scores[nid]:
                    hop_scores[nid] = parent_score
                    hop_names[nid]  = name
                    parent_path = provenance.get(src_id, f"({current_type})")
                    hop_provenance[nid] = (
                        f"{parent_path} -[{relation}]-> {target_type}('{name}')"
                    )

            sorted_hop = sorted(hop_scores.items(), key=lambda x: x[1], reverse=True)
            sorted_hop = sorted_hop[: top_k * 2]

            logger.info(
                "[GraphSearch] hop[%d] '%s' → %s: raw=%d kept=%d",
                hop_idx, relation, target_type, len(hop_scores), len(sorted_hop),
            )

            current_ids = [nid for nid, _ in sorted_hop]
            visited.update(current_ids)
            scores.update(hop_scores)
            names.update(hop_names)
            provenance.update(hop_provenance)
            current_type = target_type

        # ── Step 3: score 임계값 필터 후 최종 정렬 반환 ──────────────────────
        _SCORE_THRESHOLD = 0.2  # 전파 점수(희석) 보정 — vector.py(0.3)보다 낮게 설정
        final_ids = sorted(
            [nid for nid in current_ids if scores.get(nid, 0.0) >= _SCORE_THRESHOLD],
            key=lambda x: scores.get(x, 0.0),
            reverse=True,
        )[:top_k]

        return [
            {
                "id":        nid,
                "name":      names.get(nid, ""),
                "node_type": current_type,
                "score":     round(scores.get(nid, 0.0), 4),
                "path":      provenance.get(nid, ""),
            }
            for nid in final_ids
        ]

    semantic_graph_search.__doc__ = (semantic_graph_search.__doc__ or "") + SCHEMA_HINT
