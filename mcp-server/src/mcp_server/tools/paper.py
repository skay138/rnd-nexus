import json
from mcp.server.fastmcp import FastMCP
from infrastructure.component_factory import repository_factory

def register_paper_tools(mcp: FastMCP):
    paper_repo = repository_factory.get_paper_repository()

    @mcp.tool()
    def search_papers(query: str = "", year_from: int = 0, year_to: int = 0, author: str = "", limit: int = 10) -> str:
        """
        연구 논문 검색 도구.
        키워드·연도 범위·저자명으로 논문을 검색하고 피인용수 기준으로 정렬합니다.

        Args:
            query:     검색어 (제목, 초록, 키워드. 예: 'neuromorphic computing')
            year_from: 게재 연도 하한 (예: 2020)
            year_to:   게재 연도 상한 (예: 2024)
            author:    저자명 필터 (부분 일치. 예: '김철수', 'Kim')
            limit:     반환할 최대 논문 수

        Returns:
            JSON 문자열 형태의 논문 리스트 (title, authors, year, citations, journal, abstract 포함)
        """
        results = paper_repo.search_papers(query, year_from, year_to, author, limit)
        return json.dumps([p.model_dump(mode='json') for p in results], ensure_ascii=False, indent=2)
