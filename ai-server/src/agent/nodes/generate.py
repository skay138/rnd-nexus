import logging
import re
import time
from langchain_core.messages import SystemMessage, AIMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from common.llm import get_llm
from common.config.query_config import RequestConfig
from agent.state import RDAgentState
from config import get_settings
from memory.compaction import should_compact, compact_messages

logger = logging.getLogger(__name__)


async def generate(state: RDAgentState, config: RunnableConfig) -> dict:
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

    # orchestrator(계획 메타데이터)·tool_results(JSON 마커) 제외 — AIMessage(tool_calls)+ToolMessage 쌍은 그대로 포함
    relevant = [m for m in messages if getattr(m, "name", None) not in ("tool_results", "orchestrator")]

    system_prompt = """<role>
You are an R&D AI assistant. Answer in Korean.
</role>

<instructions>
Answer the user's question directly based on the provided data.
Never write numbers, facts, people, or organizations not present in the data.
If information is missing or insufficient, say "관련 정보를 찾을 수 없습니다."

For questions about specific relationships (participating projects, affiliations, collaborations, papers, patents), base your answer only on relationships explicitly stated in the provided data.
Do not infer or extrapolate relationships not explicitly present.
</instructions>

<constraints>
- Do not expose internal system details. Describe paths, graph node/edge names (employed_by, authored, etc.), and internal ID lookups in natural Korean.
- Do not include citations or reference lists.
- Do not append sections like "참고 사항", "추가 정보", "수집 범위 외", or "주의" at the end of answers.
- Answer only content directly relevant to the question.
</constraints>
"""

    def _fmt_ctx(m) -> str:
        name = getattr(m, "name", None)
        if getattr(m, "tool_calls", None):
            calls = ", ".join(tc["name"] for tc in m.tool_calls[:5])
            return f"  [tool_calls×{len(m.tool_calls)}] {calls}"
        if getattr(m, "tool_call_id", None):
            content = str(m.content)
            return f"  [tool_result] {content[:80]}{'…' if len(content) > 80 else ''}"
        label = name or type(m).__name__.replace("Message", "").lower()
        return f"  [{label}] {str(m.content)}"

    ctx_lines = "\n".join(_fmt_ctx(m) for m in relevant)
    logger.debug("[GEN] context=%d msgs\n%s", len(relevant), ctx_lines)

    t0 = time.perf_counter()
    chunks: list[str] = []
    async for chunk in llm.astream([SystemMessage(content=system_prompt)] + relevant, config):
        chunks.append(chunk.content if isinstance(chunk.content, str) else "")
    full_content = re.sub(r"<think>.*?</think>", "", "".join(chunks), flags=re.DOTALL).strip()
    elapsed = time.perf_counter() - t0

    logger.debug("[GEN] %.2fs  output=%d chars\n  out | %s", elapsed, len(full_content), full_content[:300])
    return {"messages": compaction_msgs + [AIMessage(content=full_content, name="final_answer")]}
