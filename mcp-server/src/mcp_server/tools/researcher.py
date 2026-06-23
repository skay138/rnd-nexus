import json
from mcp.server.fastmcp import FastMCP
from infrastructure.component_factory import repository_factory

def register_researcher_tools(mcp: FastMCP):
    researcher_repo = repository_factory.get_researcher_repository()

    @mcp.tool()
    def search_researchers(query: str = "", specialty: str = "", affiliation: str = "", top_k: int = 10) -> str:
        """
        연구자 검색 및 추천 도구.
        키워드·전문분야·소속 기관으로 연구자를 검색하고 h-index 기준으로 정렬합니다.

        Args:
            query:       검색어 (연구자 이름, 연구 주제 등. 예: 'neuromorphic computing')
            specialty:   전문 분야 필터 (예: 'AI semiconductor', '뉴로모픽')
            affiliation: 소속 기관 필터 (예: 'KAIST', 'ETRI', '삼성')
            top_k:       반환할 최대 연구자 수

        Returns:
            JSON 문자열 형태의 연구자 리스트 (name, specialty, affiliation, h_index 등 포함)
        """
        results = researcher_repo.search_researchers(query, specialty, affiliation, top_k)
        return json.dumps([p.model_dump(mode='json') for p in results], ensure_ascii=False, indent=2)
