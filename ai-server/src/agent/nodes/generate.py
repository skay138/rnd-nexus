import logging
import re
import time
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from common.llm import get_llm
from common.config.query_config import RequestConfig
from agent.utils.context import get_turn_context
from agent.state import RDAgentState
from config import get_settings
from memory.compaction import should_compact, compact_messages

logger = logging.getLogger(__name__)


async def generate(state: RDAgentState, config: RunnableConfig) -> dict:
    settings = get_settings()
    model = RequestConfig.current().generate_model or settings.rnd_model
    llm = get_llm(model=model, streaming=True)

    messages = list(state["messages"])
    approx_tokens = sum(len(str(m.content)) // 4 for m in messages)
    compaction_msgs: list = []
    if should_compact(messages, approx_tokens):
        llm_plain = get_llm(model=RequestConfig.current().compact_model or settings.rnd_model)
        compacted = await compact_messages(messages, llm_plain)
        # 새롭게 반환된 compacted에 포함되지 않은 과거 메시지의 ID만 추려내어 삭제
        kept_ids = {m.id for m in compacted if getattr(m, "id", None)}
        compaction_msgs = [RemoveMessage(id=m.id) for m in messages if getattr(m, "id", None) and m.id not in kept_ids]
        # 새롭게 생성된 요약 메시지(compacted[0])만 상태에 추가
        compaction_msgs.append(compacted[0])
        messages = compacted


    prev_context, turn_start, current_msgs = get_turn_context(messages)
    current_human = [
        m for m in current_msgs
        if getattr(m, "type", None) == "human" and not getattr(m, "name", None)
    ]
    current_tool_results = [
        m for m in current_msgs
        if getattr(m, "name", None) == "tool_results"
    ]

    # tool_results → 단일 HumanMessage (대화가 human turn으로 끝나도록)
    if current_tool_results:
        merged = "\n\n---\n\n".join(str(m.content) for m in current_tool_results)
        data_msg = HumanMessage(content=f"[수집된 데이터]\n{merged}", name="tool_results")
        relevant = prev_context + current_human + [data_msg]
    else:
        relevant = prev_context + current_human

    system_prompt = """<language>Korean</language>

<role>
당신은 R&D 전문 AI 어시스턴트입니다. 답변은 한국어로 작성하세요.
[tool_results] 메시지에 수집된 데이터가 있습니다. 이 데이터를 바탕으로 사용자 질문에 직접 답하세요.
수집된 데이터에 없는 수치·사실·인물·기관은 절대 작성하지 마세요. 데이터가 부족하면 "수집된 데이터 내에서는 확인되지 않습니다"로 명시하세요.
출처 표기나 참고문헌 목록은 작성하지 마세요.
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
