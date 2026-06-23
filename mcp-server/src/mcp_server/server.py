"""
R&D Nexus MCP Server
독립된 프로세스로 실행되며, 데이터 접근(Repository)을 래핑하여 MCP 도구로 노출합니다.
"""

from __future__ import annotations
from mcp.server.fastmcp import FastMCP

from config import get_settings
import logging

settings = get_settings()
logging.getLogger("infrastructure").setLevel(settings.rnd_log_level.upper())
logging.getLogger("mcp_server").setLevel(settings.rnd_log_level.upper())

from mcp_server.tools.entities import register_entity_tools
from mcp_server.tools.vector import register_vector_tools
from mcp_server.tools.graph import register_graph_tools
from mcp_server.tools.graph_search import register_graph_search_tools

mcp = FastMCP("RndNexusServer", host="0.0.0.0", port=8000)

register_vector_tools(mcp)        # semantic_search
register_graph_search_tools(mcp)  # semantic_graph_search
register_entity_tools(mcp)        # get_entities (ID 기반 상세 조회)
register_graph_tools(mcp)         # get_researcher_network, get_citation_graph, run_graph_query

if __name__ == "__main__":
    mcp.run(transport='sse')
