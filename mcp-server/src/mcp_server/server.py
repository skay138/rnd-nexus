"""
R&D Nexus MCP Server
독립된 프로세스로 실행되며, 데이터 접근(Repository)을 래핑하여 MCP 도구로 노출합니다.
"""

from __future__ import annotations
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from config import get_settings
import logging

settings = get_settings()
logging.getLogger("infrastructure").setLevel(settings.rnd_log_level.upper())
logging.getLogger("mcp_server").setLevel(settings.rnd_log_level.upper())

from mcp_server.tools.entities import register_entity_tools
from mcp_server.tools.vector import register_vector_tools
from mcp_server.tools.graph import register_graph_tools
from mcp_server.tools.vector_graph import register_vector_graph_tools
from mcp_server.tools.filter import register_filter_tools

mcp = FastMCP("RndNexusServer", host="0.0.0.0", port=8000)

register_vector_tools(mcp)        # semantic_search
register_vector_graph_tools(mcp)  # semantic_graph_search
register_entity_tools(mcp)        # get_entities (ID 기반 상세 조회)
register_graph_tools(mcp)         # get_researcher_network, get_citation_graph, run_graph_query
register_filter_tools(mcp)        # filter_entities (연도·기관·상태 필터)


_FIXTURE_META = [
    ("papers",        "papers.json",        "paper_id",      "title",  "authors"),
    ("patents",       "patents.json",        "patent_id",     "title",  "assignee"),
    ("researchers",   "researchers.json",    "researcher_id", "name",   "affiliation"),
    ("technologies",  "technologies.json",   "tech_id",       "name",   "trl"),
    ("projects",      "projects.json",       "project_id",    "title",  "lead_organization"),
    ("organizations", "organizations.json",  "org_id",        "name",   "type"),
]


@mcp.custom_route("/stats", methods=["GET"])
async def stats_handler(request: Request) -> JSONResponse:
    from infrastructure.repositories.in_memory_utils import load_fixture
    result: dict = {}
    for key, filename, id_field, name_field, sub_field in _FIXTURE_META:
        try:
            data = load_fixture(filename)
            items = []
            for d in data:
                name = d.get(name_field) or d.get("title") or d.get("name") or ""
                sub  = d.get(sub_field)
                if isinstance(sub, list):
                    sub = ", ".join(str(s) for s in sub[:2])
                items.append({"id": d.get(id_field, ""), "name": name, "sub": sub or ""})
            result[key] = {"count": len(data), "items": items}
        except Exception as e:
            result[key] = {"count": 0, "items": [], "error": str(e)}
    return JSONResponse(result)


if __name__ == "__main__":
    mcp.run(transport='sse')
