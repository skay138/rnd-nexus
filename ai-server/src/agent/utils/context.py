from typing import Any

def get_turn_context(messages: list[Any]) -> tuple[list[Any], int, list[Any], list[Any]]:
    """
    전체 메시지에서 현재 턴의 시작 지점을 찾고, 이전 컨텍스트와 현재 턴 메시지들로 분리합니다.

    Returns:
        tuple: (prev_context, turn_start_index, current_msgs, prev_tool_results)
            prev_context      — 이전 턴 human 질문 + final_answer (대화 흐름)
            prev_tool_results — 이전 턴 tool_results AIMessage 목록 (원본 수집 데이터)
    """
    final_answer_indices = [
        i for i, m in enumerate(messages)
        if getattr(m, "name", None) == "final_answer"
    ]
    turn_start = (final_answer_indices[-1] + 1) if final_answer_indices else 0

    prev_messages = messages[:turn_start]

    prev_context = [
        m for m in prev_messages
        if (getattr(m, "type", None) == "human" and not getattr(m, "name", None))
        or getattr(m, "name", None) == "final_answer"
    ]

    prev_tool_results = [
        m for m in prev_messages
        if getattr(m, "name", None) == "tool_results"
    ]

    current_msgs = messages[turn_start:]

    return prev_context, turn_start, current_msgs, prev_tool_results
