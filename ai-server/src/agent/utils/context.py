from typing import Any

from langchain_core.messages import HumanMessage


def split_turns(messages: list[Any]) -> tuple[list[Any], list[Any]]:
    """마지막 final_answer를 경계로 (이전 턴들, 현재 턴)으로 분리한다."""
    last = -1
    for i, m in enumerate(messages):
        if getattr(m, "name", None) == "final_answer":
            last = i
    return messages[: last + 1], messages[last + 1:]


def previous_turn_context(messages: list[Any]) -> list[Any]:
    """이전 턴에서 대화 흐름 유지에 필요한 메시지만 남긴다.

    질문(HumanMessage — 압축 요약 포함)과 최종답변(final_answer)만 유지하고
    계획·수집 요약 등 중간 산출물은 제외한다.
    """
    return [
        m for m in messages
        if isinstance(m, HumanMessage) or getattr(m, "name", None) == "final_answer"
    ]
