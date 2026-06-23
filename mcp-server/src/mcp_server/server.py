"""
R&D Nexus MCP Server
독립된 프로세스로 실행되며, 데이터 접근(Repository)을 래핑하여 MCP 도구로 노출합니다.
"""

from __future__ import annotations
from mcp.server.fastmcp import FastMCP

from config import get_settings
import logging

settings = get_settings()
# uvicorn 환경에서 infrastructure 패키지의 로그 레벨이 무시되지 않도록 강제 설정
logging.getLogger("infrastructure").setLevel(settings.rnd_log_level.upper())
logging.getLogger("mcp_server").setLevel(settings.rnd_log_level.upper())

from mcp_server.tools.paper import register_paper_tools
from mcp_server.tools.patent import register_patent_tools
from mcp_server.tools.project import register_project_tools
from mcp_server.tools.researcher import register_researcher_tools
from mcp_server.tools.technology import register_technology_tools
from mcp_server.tools.vector import register_vector_tools
from mcp_server.tools.graph import register_graph_tools
from mcp_server.tools.graph_search import register_graph_search_tools

mcp = FastMCP("RndNexusServer", host="0.0.0.0", port=8000)

register_paper_tools(mcp)
register_patent_tools(mcp)
register_project_tools(mcp)
register_researcher_tools(mcp)
register_technology_tools(mcp)
register_vector_tools(mcp)
register_graph_tools(mcp)
register_graph_search_tools(mcp)

if __name__ == "__main__":
    mcp.run(transport='sse')
