"""
MCP Client Session 관리 및 LangChain Tool 동적 래핑
"""

from contextlib import asynccontextmanager

from langchain_ollama import ChatOllama
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

async def get_llm_and_tools(session: ClientSession):
    """
    설정된 LLM에 동적으로 불러온 MCP 도구들을 바인딩하여 반환합니다.
    """
    from config import get_settings
    settings = get_settings()

    llm = ChatOllama(model=settings.rnd_model, base_url=settings.ollama_base_url)

    from agent.local_tools import read_redis_data
    mcp_tools = await load_mcp_tools(session)
    tools = mcp_tools + [read_redis_data]
    tools_by_name = {t.name: t for t in tools}

    llm_with_tools = llm.bind_tools(tools)

    return llm_with_tools, tools_by_name
