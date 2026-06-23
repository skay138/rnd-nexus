import json
from mcp.server.fastmcp import FastMCP
from infrastructure.component_factory import repository_factory

def register_patent_tools(mcp: FastMCP):
    patent_repo = repository_factory.get_patent_repository()

    @mcp.tool()
    def search_patents(query: str = "", country: str = "KR", year_from: int = 0, assignee: str = "", limit: int = 10) -> str:
        """
        특허 검색 도구.
        키워드·국가·출원 연도·출원인으로 특허를 검색합니다.

        Args:
            query:    검색어 (제목, 초록, 키워드. 예: 'AI semiconductor', 'PIM')
            country:  국가 코드 (KR, US, JP, CN, ALL. 기본값: KR)
            year_from: 출원 연도 하한 (예: 2020)
            assignee: 출원인(기관) 필터 (부분 일치. 예: '삼성', 'KAIST')
            limit:    반환할 최대 특허 수

        Returns:
            JSON 문자열 형태의 특허 리스트 (title, applicant, country, filing_date, abstract 포함)
        """
        results = patent_repo.search_patents(query, country, year_from, assignee, limit)
        return json.dumps([p.model_dump(mode='json') for p in results], ensure_ascii=False, indent=2)
