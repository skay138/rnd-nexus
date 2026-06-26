"""MCP 도구: 시맨틱 벡터 진입 + 그래프 다중 홉 탐색 (researcher-nexus ExecutionEngine 패턴)"""
from __future__ import annotations
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP
from infrastructure.graph_compiler import compile_hop

logger = logging.getLogger(__name__)


def register_graph_search_tools(mcp: FastMCP) -> None:
    from infrastructure.component_factory import repository_factory

    @mcp.tool()
    def semantic_graph_search(
        query: str,
        entry_type: str,
        hops: list[dict],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        관계를 통해 연결된 다른 타입의 엔티티를 찾을 때만 사용하세요 (예: 기술→연구자, 과제→연구자, 논문→저자).
        entry_type 엔티티를 시맨틱 검색으로 찾은 뒤, hops에 정의된 Neo4j 관계를 따라 target_type 엔티티로 이동합니다.
        동일 타입 엔티티만 필요하면 semantic_search를 사용하세요.

        [사용 가능한 relation 값]
        Researcher→Paper     : "authored"       / Paper→Researcher    : "authored_by"
        Researcher→Patent    : "invented"        / Patent→Researcher   : "invented_by"
        Researcher→Technology: "researches"      / Technology→Researcher: "researched_by"
        Researcher→Organization: "works_at"      / Organization→Researcher: "has_researcher"
        Paper→Paper          : "cites"           / Paper←Paper         : "cited_by"
        Project→Researcher   : "employs"         / Researcher→Project  : "employed_by"
        Project→Technology   : "uses"            / Technology→Project  : "used_by"

        [사용 예시]
        1. "AI 반도체 분야 핵심 연구자 추천"
           entry_type="Technology",
           hops=[{"relation": "researched_by", "target_type": "Researcher"}]

        2. "뉴로모픽 논문 저자들의 소속 기관"
           entry_type="Paper",
           hops=[{"relation": "authored_by",  "target_type": "Researcher"},
                 {"relation": "works_at",      "target_type": "Organization"}]

        3. "PIM 기술을 활용한 국가 R&D 과제"
           entry_type="Technology",
           hops=[{"relation": "used_by", "target_type": "Project"}]

        4. "특정 논문을 인용한 후속 연구 논문"
           entry_type="Paper",
           hops=[{"relation": "cited_by", "target_type": "Paper"}]

        Args:
            query:      시맨틱 검색 쿼리 (자연어. 예: "AI 반도체 저전력 설계")
            entry_type: 진입 노드 타입 (Paper / Patent / Technology / Researcher / Project)
            hops:       관계 홉 목록. 각 항목: {"relation": str, "target_type": str}
                        최대 3홉 권장 (성능상)
            top_k:      최종 반환할 최대 결과 수

        Returns:
            [{id, name, node_type, score, path}, ...] — score 내림차순 정렬
            path 필드는 "semantic(Technology:'뉴로모픽') -[researched_by]-> Researcher('홍길동')" 형태
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

            if not relation or not target_type:
                logger.warning("[GraphSearch] hop[%d] 형식 오류 (relation/target_type 필수): %s", hop_idx, hop)
                continue

            cypher = compile_hop(
                from_type=current_type,
                relation=relation,
                to_type=target_type,
                start_ids=current_ids,
                limit=top_k * 10,
                exclude_ids=list(visited - set(current_ids)),
            )
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

                # 동일 자식에 여러 부모가 있으면 최고 점수 선택
                if nid not in hop_scores or parent_score > hop_scores[nid]:
                    hop_scores[nid] = parent_score
                    hop_names[nid]  = name
                    parent_path = provenance.get(src_id, f"({current_type})")
                    hop_provenance[nid] = (
                        f"{parent_path} -[{relation}]-> {target_type}('{name}')"
                    )

            # 다음 홉 입력 크기를 top_k * 2로 제한 (성능)
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
        _SCORE_THRESHOLD = 0.2
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
