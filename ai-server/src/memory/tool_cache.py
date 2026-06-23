import uuid
import logging
import redis.asyncio as aioredis
from config import get_settings

logger = logging.getLogger(__name__)

_redis_pool = None

async def get_redis():
    global _redis_pool
    if _redis_pool is None:
        settings = get_settings()
        _redis_pool = aioredis.from_url(settings.redis_url, decode_responses=True)
    return _redis_pool

async def save_tool_result(content: str, ttl_seconds: int = 3600) -> str:
    """
    긴 텍스트를 Redis에 임시 저장하고 참조키(redis_key)를 반환합니다.
    """
    redis_client = await get_redis()
    redis_key = f"tool_res:{uuid.uuid4()}"
    try:
        await redis_client.set(redis_key, content, ex=ttl_seconds)
        logger.debug("[tool_cache] Saved data to redis with key %s (len: %d)", redis_key, len(content))
        return redis_key
    except Exception as e:
        logger.error("[tool_cache] Failed to save to redis: %s", e)
        return ""

async def get_tool_result(redis_key: str) -> str:
    """참조키를 통해 Redis에서 원본 텍스트를 가져옵니다."""
    redis_client = await get_redis()
    try:
        result = await redis_client.get(redis_key)
        if result is None:
            return f"[ERROR] 캐시된 데이터를 찾을 수 없거나 만료되었습니다: {redis_key}"
        return result
    except Exception as e:
        logger.error("[tool_cache] Failed to get from redis: %s", e)
        return f"[ERROR] Redis 조회 실패: {e}"


async def delete_tool_result(redis_key: str) -> None:
    """Redis 캐시 항목을 삭제합니다."""
    redis_client = await get_redis()
    try:
        await redis_client.delete(redis_key)
        logger.debug("[tool_cache] Deleted redis key: %s", redis_key)
    except Exception as e:
        logger.error("[tool_cache] Failed to delete from redis: %s", e)
