from typing import Any
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage

COMPACTION_THRESHOLD = 80_000
COMPACTION_TARGET    = 40_000


def should_compact(messages: list[Any], token_count: int) -> bool:
    return token_count > COMPACTION_THRESHOLD


async def compact_messages(messages: list[Any], llm: Any) -> list[Any]:
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    non_system  = [m for m in messages if not isinstance(m, SystemMessage)]

    summary_prompt = f"""다음 대화 내용을 압축하세요.
반드시 보존해야 할 내용:
- 사용자의 원래 R&D 질문
- 각 MCP 도구 호출 결과 핵심 데이터
- Planner가 수립한 계획
- 완료된 단계 목록

압축 대상 대화:
{_format_messages(non_system)}
"""
    summary = await llm.ainvoke([HumanMessage(content=summary_prompt)])

    recent_messages = non_system[-10:]
    return system_msgs + [
        HumanMessage(content=f"[압축된 이전 맥락]\n{summary.content}")
    ] + recent_messages


def _format_messages(messages: list[Any]) -> str:
    lines = []
    for m in messages:
        role = type(m).__name__.replace("Message", "")
        content = m.content if isinstance(m.content, str) else str(m.content)
        lines.append(f"{role}: {content[:500]}")
    return "\n".join(lines)
