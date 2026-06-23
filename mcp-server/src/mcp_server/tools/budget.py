import json
from mcp.server.fastmcp import FastMCP
from infrastructure.component_factory import repository_factory

def register_budget_tools(mcp: FastMCP):
    budget_repo = repository_factory.get_budget_repository()

    @mcp.tool()
    def analyze_budget_trend(domain: str, years: int = 5) -> str:
        """
        특정 분야의 예산 동향 분석 도구.

        Args:
            domain: 연구 분야 (예: 'AI 반도체')
            years: 분석 기간 (년)

        Returns:
            JSON 문자열 형태의 예산 트렌드 데이터
        """
        result = budget_repo.analyze_budget(domain, years)
        return result.model_dump_json(indent=2)
