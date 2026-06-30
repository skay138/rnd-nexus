import ast
import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

COMPACTION_THRESHOLD = 24_000


def should_compact(messages: list[Any], token_count: int) -> bool:
    return token_count > COMPACTION_THRESHOLD


async def compact_messages(messages: list[Any], llm: Any) -> list[Any]:
    """메시지 압축 전략:

    1. final_answer 경계가 있으면 → 완성된 이전 턴을 핵심 사실 요약으로 압축
    2. 단일 턴 진행 중 → 완료된 라운드(tool_results 마커 이전)를 압축, 현재 라운드 보존
    """
    final_answer_indices = [
        i for i, m in enumerate(messages)
        if getattr(m, "name", None) == "final_answer"
    ]

    if final_answer_indices:
        # 멀티턴: 마지막 final_answer까지를 압축, 이후(현재 턴)는 그대로 보존
        cut = final_answer_indices[-1] + 1
        to_compress = messages[:cut]
        to_keep     = messages[cut:]
    else:
        # 단일 턴 내 압축: 완료된 라운드(tool_results 마커 뒤까지)를 압축
        tool_result_markers = [
            i for i, m in enumerate(messages)
            if getattr(m, "name", None) == "tool_results"
        ]
        if len(tool_result_markers) > 1:
            # 마지막 라운드 직전까지 압축, 마지막 라운드 보존
            cut         = tool_result_markers[-1]
            to_compress = messages[:cut]
            to_keep     = messages[cut:]
        else:
            keep_count  = min(8, max(2, len(messages) - 2))
            to_compress = messages[:-keep_count]
            to_keep     = messages[-keep_count:]

    if not to_compress:
        return messages

    summary = await _compress_to_summary(to_compress, llm)
    return [HumanMessage(content=f"[압축된 이전 맥락]\n{summary}")] + to_keep


async def _compress_to_summary(messages: list[Any], llm: Any) -> str:
    formatted = _format_for_compact(messages)
    prompt = f"""<role>
R&D 에이전트 대화를 핵심 내용만 추출해 압축하세요.
</role>

<instructions>
반드시 보존:
- 사용자 원래 질문
- 수집된 엔티티: 연구자명·소속·논문/특허 제목·기술명·과제명 (ID 포함)
- 완료된 태스크 목록
- 도출된 최종 답변 (있을 경우)

제거:
- 중간 추론 과정
- 도구 호출 raw JSON (핵심 사실만 추출)
- 중복 데이터
</instructions>

<output_format>
평문 한국어로 작성하세요. 불릿 목록보다 서술 문장 형태를 선호합니다.
</output_format>

<conversation>
{formatted}
</conversation>
"""
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return response.content


def _format_for_compact(messages: list[Any]) -> str:
    """압축 프롬프트용 메시지 포맷 — ToolMessage는 요약본, AIMessage(tool_calls)는 호출 목록."""
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
            lines.append("[라운드 수집 완료]")

        elif name == "orchestrator":
            content = str(m.content)
            task_part = content.split("\n\n[계획한 태스크]", 1)
            tasks_text = task_part[1].strip()[:300] if len(task_part) > 1 else content[:300]
            lines.append(f"[오케스트레이터 태스크]\n{tasks_text}")

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
