import json
from mcp.server.fastmcp import FastMCP
from infrastructure.component_factory import repository_factory

def register_project_tools(mcp: FastMCP):
    ntis_repo = repository_factory.get_project_repository()

    @mcp.tool()
    def search_projects(keyword: str = "", institution: str = "", status: str = "", year_from: int = 0, limit: int = 10) -> str:
        """
        국가 R&D 과제(NTIS) 검색 도구.
        키워드·수행 기관·과제 상태·시작 연도로 국가 R&D 과제를 검색합니다.

        Args:
            keyword:     검색어 (과제명, 키워드. 예: 'PIM', 'AI 반도체')
            institution: 수행 기관 필터 (예: 'KAIST', 'ETRI', '서울대')
            status:      과제 상태 필터 (예: '진행중', '완료', '계획')
            year_from:   검색 시작 연도 (예: 2020)
            limit:       반환할 최대 과제 수

        Returns:
            JSON 문자열 형태의 과제 리스트 (title, organization, year, budget_billion_krw, status 등 포함)
        """
        results = ntis_repo.search_projects(keyword, institution, status, year_from, limit)
        return json.dumps([p.model_dump(mode='json') for p in results], ensure_ascii=False, indent=2)
