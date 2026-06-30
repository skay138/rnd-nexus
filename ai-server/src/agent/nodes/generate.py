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

    # tool_results JSON 마커 스킵 — AIMessage(tool_calls)+ToolMessage 쌍은 그대로 포함
    relevant = [m for m in messages if getattr(m, "name", None) != "tool_results"]

    system_prompt = """<role>
당신은 R&D 전문 AI 어시스턴트입니다. 답변은 한국어로 작성하세요.
</role>

<instructions>
제공된 데이터를 바탕으로 사용자 질문에 직접 답하세요.
데이터에 없는 수치·사실·인물·기관은 절대 작성하지 마세요.
정보가 없거나 부족하면 "관련 정보를 찾을 수 없습니다."라고 답하세요.

질문이 특정 관계(참여 과제, 소속 기관, 공동연구, 논문, 특허 등)에 관한 경우 제공된 데이터에 명시된 관계만 근거로 답하세요.
명시되지 않은 관계를 추론하거나 확대 해석하지 마세요.
</instructions>

<constraints>
- 내부 시스템 구현 정보를 노출하지 마세요. path, 그래프 노드·엣지명(employed_by, authored 등), 내부 ID 조회 과정은 자연스러운 한국어로 설명하세요.
- 출처 표기나 참고문헌 목록은 작성하지 마세요.
- 답변 끝에 "참고 사항", "추가 정보", "수집 범위 외", "주의" 등의 섹션을 추가하지 마세요.
- 질문과 직접 관련된 내용만 답변하세요.
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
