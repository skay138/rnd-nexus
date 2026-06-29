import logging
import re
import time
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from common.llm import get_llm
from common.config.query_config import RequestConfig
from common.parsers import build_deduped_context
from agent.utils.context import get_turn_context
from agent.state import RDAgentState
from config import get_settings
from memory.compaction import should_compact, compact_messages

logger = logging.getLogger(__name__)


async def generate(state: RDAgentState, config: RunnableConfig) -> dict:
    # 오케스트레이터가 범위 외로 판단한 경우 LLM 호출 없이 즉시 반환
    if state.get("out_of_scope"):
        logger.debug("[generate] out_of_scope — 안내 반환")
        return {"messages": [AIMessage(
            content="죄송합니다. 해당 질문은 R&D 서비스의 지원 범위를 벗어납니다.\n논문·특허·연구자·기술·R&D 과제에 관한 질문을 입력해 주세요.",
            name="final_answer",
        )]}

    settings = get_settings()
    model = RequestConfig.current().generate_model or settings.rnd_model
    llm = get_llm(model=model, streaming=True)

    messages = list(state["messages"])
    approx_tokens = sum(len(str(m.content)) // 4 for m in messages)
    compaction_msgs: list = []
    if should_compact(messages, approx_tokens):
        llm_plain = get_llm(model=RequestConfig.current().compact_model or settings.rnd_model)
        compacted = await compact_messages(messages, llm_plain)
        kept_ids = {m.id for m in compacted if getattr(m, "id", None)}
        compaction_msgs = [RemoveMessage(id=m.id) for m in messages if getattr(m, "id", None) and m.id not in kept_ids]
        compaction_msgs.append(compacted[0])
        messages = compacted

    prev_context, turn_start, current_msgs, prev_tool_results = get_turn_context(messages)
    current_human = [
        m for m in current_msgs
        if getattr(m, "type", None) == "human" and not getattr(m, "name", None)
    ]
    current_tool_results = [
        m for m in current_msgs
        if getattr(m, "name", None) == "tool_results"
    ]

    if current_tool_results:
        # 이번 턴에 수집된 데이터가 있으면 그것만 사용
        merged = build_deduped_context(current_tool_results)
        data_msg = HumanMessage(content=f"[수집된 데이터]\n{merged}", name="tool_results")
        relevant = prev_context + current_human + [data_msg]
    elif prev_tool_results:
        # 오케스트레이터가 이전 턴 데이터로 충분하다 판단해 수집 스킵 → prev 데이터로 fallback
        prev_merged = build_deduped_context(prev_tool_results)
        prev_data_msg = HumanMessage(content=f"[수집된 데이터]\n{prev_merged}", name="tool_results")
        relevant = prev_context + current_human + [prev_data_msg]
    else:
        relevant = prev_context + current_human

    system_prompt = """<language>Korean</language>

<role>
당신은 R&D 전문 AI 어시스턴트입니다. 답변은 한국어로 작성하세요.
제공된 데이터를 바탕으로 사용자 질문에 직접 답하세요.
데이터에 없는 수치·사실·인물·기관은 절대 작성하지 마세요. 정보가 없거나 부족하면 "관련 정보를 찾을 수 없습니다"라고 답하세요.
출처 표기나 참고문헌 목록은 작성하지 마세요.
답변 끝에 "참고 사항", "추가 정보", "관련성 낮은 주제", "수집 범위 외", "주의", "제외 항목" 등의 제목으로 시작하는 섹션을 절대 추가하지 마세요. 특히 "~는 포함되지 않았습니다", "~는 범위를 벗어납니다" 같은 제외 이유 설명도 작성하지 마세요. 질문에 직접 해당하는 내용만 작성하세요.
</role>
"""

    logger.debug(
        "[generate] relevant_messages=%d\n%s",
        len(relevant),
        "\n".join(
            f"  [{getattr(m, 'name', None) or getattr(m, 'type', '?')}] "
            f"{str(m.content)[:300]}{'…' if len(str(m.content)) > 300 else ''}"
            for m in relevant
        ),
    )

    t0 = time.perf_counter()
    chunks: list[str] = []
    async for chunk in llm.astream([SystemMessage(content=system_prompt)] + relevant, config):
        chunks.append(chunk.content if isinstance(chunk.content, str) else "")
    full_content = re.sub(r"<think>.*?</think>", "", "".join(chunks), flags=re.DOTALL).strip()
    elapsed = time.perf_counter() - t0

    logger.debug("[generate] elapsed=%.2fs content_len=%d\n%s", elapsed, len(full_content), full_content[:500])
    return {"messages": compaction_msgs + [AIMessage(content=full_content, name="final_answer")]}
