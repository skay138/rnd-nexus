from typing import Any
from langchain_core.messages import HumanMessage

COMPACTION_THRESHOLD = 80_000


def should_compact(messages: list[Any], token_count: int) -> bool:
    return token_count > COMPACTION_THRESHOLD


async def compact_messages(messages: list[Any], llm: Any) -> list[Any]:
    # 최소 2개의 최근 메시지는 남기고, 나머지를 압축 대상으로 삼음 (전체 메시지가 적어도 압축이 진행되도록 보장)
    keep_count = min(10, max(2, len(messages) - 2)) if len(messages) > 2 else 1
    dropped_messages = messages[:-keep_count]
    recent_messages = messages[-keep_count:]

    summary_prompt = f"""다음 R&D 에이전트 대화를 압축하세요.

반드시 보존해야 할 내용:
- 사용자의 원래 질문과 요청 의도
- Orchestrator가 계획한 태스크 목록
- 각 도구 호출로 수집된 핵심 데이터 (연구자명, 논문/특허 제목, 기관, 수치 등)
- 완료된 태스크와 아직 수집되지 않은 정보

압축 대상 과거 대화:
{_format_messages(dropped_messages)}
"""
    summary = await llm.ainvoke([HumanMessage(content=summary_prompt)])

    return [HumanMessage(content=f"[압축된 이전 맥락]\n{summary.content}")] + recent_messages


def _format_messages(messages: list[Any]) -> str:
    lines = []
    for m in messages:
        role = type(m).__name__.replace("Message", "")
        content = m.content if isinstance(m.content, str) else str(m.content)
        lines.append(f"{role}({getattr(m, 'name', '') or role}):\n{content}")
    return "\n\n---\n\n".join(lines)
