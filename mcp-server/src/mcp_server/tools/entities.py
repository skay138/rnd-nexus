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
        태스크에 ID가 이미 명시되어 있거나, 검색·그래프 도구 결과로 ID를 확보했다면 이 도구로 전체 필드를 조회하세요.
        ID를 알고 있다면 검색 없이 이 도구를 첫 호출로 사용해도 됩니다.
        </role>

        <instructions>
        - 검색·그래프 결과의 id 필드를 그대로 전달하세요 — get_researcher_network의 papers/patents 안에 중첩된 id도 조회 대상입니다
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
