from langchain_core.tools import tool
from memory.tool_cache import get_tool_result

@tool
async def read_redis_data(redis_key: str) -> str:
    """
    미리보기로 제공된 데이터의 원본 전체 내용이 필요할 때 호출합니다.
    이전 도구 호출 결과에서 제공된 redis_key를 입력하세요.
    """
    return await get_tool_result(redis_key)
