"""MCP 도구: ID 기반 엔티티 상세 조회"""
from typing import Any
from mcp.server.fastmcp import FastMCP


_ENTITY_CONFIG: dict[str, dict[str, Any]] = {
    "Researcher":   {"get_repo": "get_researcher_repository"},
    "Paper":        {"get_repo": "get_paper_repository"},
    "Patent":       {"get_repo": "get_patent_repository"},
    "Technology":   {"get_repo": "get_technology_repository"},
    "Project":      {"get_repo": "get_project_repository"},
    "Organization": {"get_repo": "get_organization_repository"},
}


def register_entity_tools(mcp: FastMCP) -> None:
    from infrastructure.component_factory import repository_factory

    @mcp.tool()
    def get_entities(entity_type: str, ids: list[str]) -> list[dict]:
        """
        <role>
        ID 목록으로 엔티티 상세 정보를 조회합니다.
        semantic_search, semantic_graph_search, run_graph_query로 수집한 ID의 전체 필드가 필요할 때 사용하세요.
        이름(name)만 필요한 경우 semantic_search 결과를 그대로 활용하고 이 도구는 생략 가능합니다.
        </role>

        <instructions>
        - semantic_search / semantic_graph_search 결과의 id 필드를 그대로 전달하세요
        - 여러 타입이 섞여 있으면 entity_type별로 나눠 호출하세요
        - 최대 50개 ID를 한 번에 전달 가능합니다
        </instructions>

        <constraints>
        - entity_type은 Researcher / Paper / Patent / Technology / Project / Organization 중 하나
        - ids가 빈 목록이면 빈 목록 반환
        </constraints>

        Args:
            entity_type: 엔티티 타입 — Researcher | Paper | Patent | Technology | Project | Organization
            ids:         조회할 ID 목록 (예: ["R001", "R007"])

        Returns:
            [{...entity fields...}, ...] — entity_type에 따라 필드 다름
            Researcher: researcher_id, name, affiliation, h_index, specialty
            Paper:      paper_id, title, year, citations, journal, abstract, authors
            Project:    project_id, title, year, status, organization, keywords
            Patent:     patent_id, title, year, assignee, abstract
            Technology: tech_id, name, description, trl
            Organization: org_id, name, type
        """
        cfg = _ENTITY_CONFIG.get(entity_type)
        if cfg is None:
            valid = ", ".join(_ENTITY_CONFIG)
            return [{"error": f"알 수 없는 entity_type: '{entity_type}'. 유효값: {valid}"}]
        if not ids:
            return []

        repo = getattr(repository_factory, cfg["get_repo"])()
        results = repo.get_by_ids(ids)
        return [r.model_dump(mode="json") for r in results]
