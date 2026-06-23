import logging
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableConfig
from agent.state import RDAgentState
from memory.compaction import should_compact, compact_messages

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """당신은 반드시 한국어로만 응답해야 합니다.

<role>
당신은 R&D 데이터 수집 에이전트입니다.
Planner가 수립한 계획에 따라 도구를 호출하여 데이터를 수집합니다.
</role>

<tool_strategy>
- 키워드가 모호하거나 문장형 질문: semantic_search(Vector DB)를 가장 먼저 고려하세요.
- 연구자·기술 관계 탐색(추천, 네트워크): semantic_graph_search(벡터+그래프 통합)를 활용하세요.
- 구체적 식별자 확보 후: search_papers, search_patents 등 RDB 상세 검색 도구를 사용하세요.
- 이미 같은 도구·같은 파라미터로 결과를 받은 경우: 중복 호출하지 말고 다음 단계로 이동하세요.
</tool_strategy>

<important>
계획에 필요한 모든 데이터가 수집되면 도구 호출 없이 "수집 완료"를 반환하세요.
최종 답변은 별도 단계에서 생성됩니다 — 지금은 데이터 수집에만 집중하세요.
</important>"""


async def llm_call(state: RDAgentState, config: RunnableConfig) -> dict:
    logger.debug("[Node] llm_call 시작")
    llm_with_tools = config["configurable"]["llm_with_tools"]
    messages = state["messages"]

    approx_tokens = sum(len(str(m.content)) // 4 for m in messages)
    if should_compact(messages, approx_tokens):
        from config import get_settings
        settings = get_settings()
        llm_plain = ChatOllama(model=settings.rnd_model, base_url=settings.ollama_base_url)
        messages = compact_messages(messages, llm_plain)

    # planner가 반환한 plan은 state["plan"]에만 있고 messages에는 없음 — 여기서 주입
    plan = state.get("plan", [])
    plan_prefix: list = []
    if plan:
        plan_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan))
        plan_prefix = [HumanMessage(content=f"[수행 계획]\n{plan_text}\n\n위 계획에 따라 순서대로 도구를 호출하여 데이터를 수집하세요.")]

    response = await llm_with_tools.ainvoke(
        [SystemMessage(content=SYSTEM_PROMPT)] + plan_prefix + messages
    )
    logger.debug("[llm_call] Response: content=%s, tool_calls=%s", response.content, response.tool_calls)
    return {"messages": [response]}
