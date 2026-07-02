import ast
import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage

COMPACTION_THRESHOLD = 24_000  # 추정 토큰 기준


def estimate_tokens(messages: list[Any]) -> int:
    """한국어 위주 텍스트의 보수적 토큰 추정 (약 2자/토큰).

    영문 기준 4자/토큰 추정은 한국어에서 실제 토큰 수를 크게 과소평가해
    컨텍스트 오버플로 전에 압축이 트리거되지 않는 문제가 있다.
    """
    return sum(len(str(m.content)) for m in messages) // 2


def should_compact(messages: list[Any]) -> bool:
    return estimate_tokens(messages) > COMPACTION_THRESHOLD


async def apply_compaction(messages: list[Any], llm: Any) -> tuple[list[Any], list[Any]]:
    """임계 초과 시 압축을 수행한다.

    반환: (이번 LLM 호출에 사용할 메시지 목록,
           state["messages"]에 반영할 연산 목록 — RemoveMessage들 + 압축 요약.
           압축 불필요 시 빈 목록)
    """
    if not should_compact(messages):
        return messages, []
    compacted = await compact_messages(messages, llm)
    kept_ids = {m.id for m in compacted if getattr(m, "id", None)}
    ops: list[Any] = [
        RemoveMessage(id=m.id)
        for m in messages
        if getattr(m, "id", None) and m.id not in kept_ids
    ]
    ops.append(compacted[0])
    return compacted, ops


async def compact_messages(messages: list[Any], llm: Any) -> list[Any]:
    """메시지 압축 전략:

    1. final_answer 경계가 있으면 → 완성된 이전 턴을 요약, 현재 턴(이후 메시지)은 raw 보존
    2. 단일 턴 진행 중 → 전체를 요약 (to_keep 없음)
    """
    final_answer_indices = [
        i for i, m in enumerate(messages)
        if getattr(m, "name", None) == "final_answer"
    ]

    if final_answer_indices:
        # 멀티턴: 마지막 final_answer까지 압축, 이후(현재 턴)는 raw 보존
        cut         = final_answer_indices[-1] + 1
        to_compress = messages[:cut]
        to_keep     = messages[cut:]
    else:
        # 단일 턴: 전체 압축 — 요약 LLM이 핵심 사실(ID·엔티티명) 보존
        to_compress = messages
        to_keep     = []

    if not to_compress:
        return messages

    summary = await _compress_to_summary(to_compress, llm)
    return [HumanMessage(content=f"[압축된 이전 맥락]\n{summary}")] + to_keep


async def _compress_to_summary(messages: list[Any], llm: Any) -> str:
    formatted = _format_for_compact(messages)
    prompt = f"""Summarize the following R&D agent session into a structured handoff document.
The summary replaces the full conversation history — the agent must be able to continue work from it alone.
Write in Korean. Be specific: include exact IDs and names, not just counts.
Be concise — bullet lists only, no prose paragraphs.

Output the following sections (omit a section if not applicable):

## 원본 질문
<사용자가 요청한 내용 그대로>

## 수집된 데이터
<수집된 엔티티를 ID와 함께 나열>
- 연구자: 이름 (ID, 소속, 전문분야)
- 논문: 제목 (ID, 연도)
- 특허: 제목 (ID, 연도)
- 과제: 과제명 (ID, 기관)
- 기술: 기술명 (ID)

## 완료된 태스크
<어떤 검색·조회가 수행됐는지>

## 최종 답변 요약
<final_answer가 있었다면 핵심 내용>

## 미완료 사항
<아직 수집되지 않은 데이터나 남은 작업 — 해당 없으면 생략>

<conversation>
{formatted}
</conversation>"""
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return response.content


def _format_for_compact(messages: list[Any]) -> str:
    """압축 프롬프트용 메시지 포맷.

    ToolMessage/tool_calls 분기는 구버전 체크포인트(raw 도구 메시지 잔존) 호환용.
    """
    lines = []
    for m in messages:
        name = getattr(m, "name", None)

        if isinstance(m, ToolMessage):
            summary = _summarize_tool_content(str(m.content))
            lines.append(f"[도구 결과] {summary}")

        elif isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            calls = [
                f"{tc['name']}({_compact_args(tc.get('args', {}))})"
                for tc in m.tool_calls[:5]
            ]
            lines.append(f"[도구 호출] {', '.join(calls)}")

        elif name == "tool_results":
            # 내용이 이미 태스크별 요약 텍스트 — 압축 요약이 ID·엔티티명을 보존하도록 포함
            lines.append(str(m.content)[:800])

        elif name == "orchestrator":
            lines.append(f"[오케스트레이터 태스크]\n{str(m.content)[:300]}")

        elif name == "final_answer":
            lines.append(f"[최종 답변]\n{str(m.content)[:500]}")

        else:
            role = type(m).__name__.replace("Message", "")
            lines.append(f"[{role}]\n{str(m.content)[:400]}")

    return "\n\n".join(lines)


def _summarize_tool_content(content: str) -> str:
    """ToolMessage 내용을 한 줄로 요약."""
    if content.startswith("[ERROR]"):
        return f"오류: {content[7:60]}"
    if content.startswith("[SKIP]"):
        return "중복 호출 스킵"
    try:
        data = json.loads(content)
    except Exception:
        try:
            data = ast.literal_eval(content)
        except Exception:
            return content[:100]

    if isinstance(data, list):
        count = len(data)
        previews = []
        for d in data[:3]:
            if isinstance(d, dict):
                label = d.get("name") or d.get("title") or d.get("researcher_id") or ""
                if label:
                    previews.append(str(label)[:20])
        return f"{count}건" + (f": {', '.join(previews)}" if previews else "")

    if isinstance(data, dict):
        nodes = len(data.get("nodes", []))
        rels  = len(data.get("relationships", []))
        if nodes or rels:
            return f"nodes {nodes}개, rels {rels}개"

    return content[:100]


def _compact_args(args: dict) -> str:
    """도구 호출 args를 짧게 요약."""
    parts = []
    for k, v in list(args.items())[:3]:
        parts.append(f"{k}={str(v)[:30]}")
    return ", ".join(parts)
