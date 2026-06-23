from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from config import get_settings
from contextlib import asynccontextmanager

@asynccontextmanager
async def create_memory():
    settings = get_settings()
    async with AsyncRedisSaver.from_conn_string(settings.redis_url) as memory:
        yield memory

async def load_session_context(thread_id: str, memory: AsyncRedisSaver) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    checkpoint = await memory.aget(config)
    if checkpoint is None:
        return {}
    return checkpoint.get("channel_values", {})
