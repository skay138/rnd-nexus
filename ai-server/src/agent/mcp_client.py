"""
MCP Client Session 관리 및 LangChain Tool 동적 래핑
"""

from contextlib import asynccontextmanager

from mcp import ClientSession
from mcp.client.sse import sse_client
from langchain_mcp_adapters.tools import load_mcp_tools
import logging

logger = logging.getLogger(__name__)

@asynccontextmanager
async def mcp_server_session():
    """
    원격 MCP 서버(SSE)에 연결하고 통신 세션을 반환합니다.
    """
    from config import get_settings
    settings = get_settings()

    logger.info("MCP 서버(SSE) 연결 중... %s", settings.mcp_server_url)

    async with sse_client(settings.mcp_server_url, timeout=10.0, sse_read_timeout=300.0) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            logger.info("MCP 서버 초기화 완료")
            yield session

async def get_llm_and_tools(session: ClientSession) -> dict:
    """MCP 서버에서 도구 목록을 로드하여 {name: tool} 딕셔너리로 반환합니다."""
    mcp_tools = await load_mcp_tools(session)
    tools_by_name = {t.name: t for t in mcp_tools}
    logger.info("MCP 도구 %d개 로드 완료: %s", len(tools_by_name), list(tools_by_name.keys()))
    return tools_by_name
