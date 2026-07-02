"""MCP 도구: 연도·기관·상태 기반 필터 검색"""
from __future__ import annotations
import logging
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


def register_filter_tools(mcp: FastMCP) -> None:

    @mcp.tool()
    def filter_entities(
        entity_type: str,
        year_from: int = 0,
        year_to: int = 0,
        organization: str = "",
        status: str = "",
        limit: int = 20,
    ) -> list[dict]:
        """
        <role>
        연도·기관·상태 조건으로 엔티티를 필터링합니다.
        시간 범위나 상태 조건이 명확한 경우 semantic_search 대신 또는 병행하여 사용하세요.
        </role>

        <instructions>
        - "최근 3년 AI 반도체 과제": entity_type="Project", year_from=2022, year_to=2024
        - "ETRI 소속 연구자 목록": entity_type="Researcher", organization="ETRI"
        - "진행 중인 과제만": entity_type="Project", status="진행중"
        </instructions>

        <constraints>
        - entity_type: Paper / Patent / Researcher / Technology / Project / Organization
        - 타입별 지원 필터 (지원하지 않는 파라미터는 무시됨):
          Paper=연도 / Patent=연도+organization(출원인) / Project=연도+organization+status
          Researcher=organization(소속 기관) / Organization=organization(기관명) / Technology=필터 없음(전체 목록)
        - year_from=0이면 하한 없음, year_to=0이면 상한 없음
        - organization은 부분 문자열 매칭 — 기관명만 가능, 사람 이름 검색 불가(연구자 이름은 semantic_search 사용)
        </constraints>

        Args:
            entity_type:  조회할 엔티티 타입
            year_from:    시작 연도 (0=제한없음)
            year_to:      종료 연도 (0=제한없음)
            organization: 기관명 부분 일치 필터 (빈값=전체)
            status:       과제 상태 필터 — Project 전용 (빈값=전체)
            limit:        최대 반환 수

        Returns:
            [{...entity fields...}, ...] — entity_type에 따라 필드 다름
        """
        from infrastructure.component_factory import repository_factory

        try:
            if entity_type == "Paper":
                repo = repository_factory.get_paper_repository()
                items = repo.search_papers(year_from=year_from, year_to=year_to, limit=limit)

            elif entity_type == "Patent":
                repo = repository_factory.get_patent_repository()
                items = repo.search_patents(year_from=year_from, assignee=organization, limit=limit)

            elif entity_type == "Project":
                repo = repository_factory.get_project_repository()
                items = repo.search_projects(
                    institution=organization, status=status, year_from=year_from, limit=limit
                )

            elif entity_type == "Researcher":
                repo = repository_factory.get_researcher_repository()
                items = repo.search_researchers(affiliation=organization, top_k=limit)

            elif entity_type == "Technology":
                repo = repository_factory.get_technology_repository()
                items = repo.search_technologies(top_k=limit)

            elif entity_type == "Organization":
                repo = repository_factory.get_organization_repository()
                items = repo.get_all(name=organization, limit=limit)

            else:
                valid = "Paper, Patent, Researcher, Technology, Project, Organization"
                return [{"error": f"알 수 없는 entity_type: '{entity_type}'. 유효값: {valid}"}]

        except Exception as e:
            logger.warning("[filter] %s 조회 실패: %s", entity_type, e)
            return [{"error": f"조회 실패: {e}"}]

        results = [item.model_dump(mode="json") for item in items]

        # year_to 필터: search_papers만 지원 — 나머지는 Python-level 필터
        if year_to and entity_type != "Paper":
            results = [r for r in results if not r.get("year") or r["year"] <= year_to]

        logger.debug("[filter] %s year=%s~%s org='%s' status='%s' → %d건",
                     entity_type, year_from or "*", year_to or "*", organization, status, len(results))
        return results
