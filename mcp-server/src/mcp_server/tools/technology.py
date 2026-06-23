import json
from mcp.server.fastmcp import FastMCP
from infrastructure.component_factory import repository_factory

def register_technology_tools(mcp: FastMCP):
    tech_repo = repository_factory.get_technology_repository()

    @mcp.tool()
    def search_technologies(query: str = "", trl_min: int = 0, top_k: int = 10) -> str:
        """
        기술 검색 및 유망 기술 추천 도구.
        키워드와 TRL(기술성숙도) 조건으로 R&D 기술을 검색합니다.

        Args:
            query:   검색어 (기술명, 설명 등. 예: 'PIM 메모리', 'neuromorphic')
            trl_min: 최소 TRL 레벨 (1~9). 상용화 근접 기술 필터링에 사용.
                     예: trl_min=7 → 시제품 검증 이상 기술만 반환
            top_k:   반환할 최대 기술 수

        Returns:
            JSON 문자열 형태의 기술 리스트 (name, trl, market_growth_rate_percent, key_players 등 포함)
        """
        results = tech_repo.search_technologies(query, trl_min, top_k)
        return json.dumps([p.model_dump(mode='json') for p in results], ensure_ascii=False, indent=2)
