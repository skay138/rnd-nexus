"""MCP 도구: ID 기반 엔티티 상세 조회"""
import json
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
    def get_entities(entity_type: str, ids: list[str]) -> str:
        """
        ID 목록으로 엔티티 상세 정보를 조회합니다.
        semantic_search 또는 graph 도구로 발견한 ID를 이 도구에 전달하세요.

        Args:
            entity_type: 엔티티 타입 — Researcher | Paper | Patent | Technology | Project | Organization
            ids:         조회할 ID 목록 (예: ["R001", "R007"])

        Returns:
            JSON 형태의 엔티티 상세 정보 목록
        """
        cfg = _ENTITY_CONFIG.get(entity_type)
        if cfg is None:
            valid = ", ".join(_ENTITY_CONFIG)
            return json.dumps({"error": f"알 수 없는 entity_type: '{entity_type}'. 유효값: {valid}"})
        if not ids:
            return json.dumps([])

        repo = getattr(repository_factory, cfg["get_repo"])()
        results = repo.get_by_ids(ids)
        return json.dumps([r.model_dump(mode="json") for r in results], ensure_ascii=False, indent=2)
