import datetime
import re
from typing import Any

from langchain_core.messages import HumanMessage, AIMessage


def get_today_message() -> HumanMessage:
    """오늘 날짜를 담은 HumanMessage를 반환한다."""
    today = datetime.date.today().strftime("%Y년 %m월 %d일")
    return HumanMessage(content=f"[오늘 날짜: {today}]")


_CITE_RE = re.compile(r'\[#([A-Za-z0-9\-_.]+)\]')


def _strip_citations(text: str) -> str:
    """[#ID] 인용 마커에서 # 제거 → 오케스트레이터가 raw ID로 인식하도록.

    예) [#RS-2024-00123456] → (RS-2024-00123456)
    """
    return _CITE_RE.sub(lambda m: f"({m.group(1)})", text)


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

    final_answer의 [#ID] 인용 마커는 오케스트레이터가 '#' 포함 형태로 검색하지 않도록
    raw ID 형태 (ID) 로 치환한다.
    """
    result: list[HumanMessage | AIMessage] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            result.append(m)
        elif getattr(m, "name", None) == "final_answer":
            cleaned = _strip_citations(m.content if isinstance(m.content, str) else "")
            result.append(AIMessage(content=cleaned, name="final_answer"))
    return result
