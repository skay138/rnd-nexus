import logging
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from agent.state import RDAgentState
from config import get_settings
from memory.compaction import should_compact, compact_messages

logger = logging.getLogger(__name__)


class ToolTask(BaseModel):
    tool: str  = Field(description="호출할 도구 이름")
    args: dict = Field(description="도구 인수 (JSON 객체)")
    label: str = Field(default="", description="이 태스크가 수집하는 것 (로깅용)")


class OrchestratorPlan(BaseModel):
    reasoning: str            = Field(description="수집 현황 평가 및 다음 전략 (한국어)")
    tasks: list[ToolTask]     = Field(description="병렬 실행할 태스크 목록. 데이터 수집이 완료됐으면 빈 리스트")


_STRATEGY = """
<strategy>
- 발견 먼저: semantic_search / semantic_graph_search로 관련 ID를 얻은 뒤, get_entities로 상세 조회 (ID 없이 get_entities 호출 금지)
- 입력이 이미 확보된 독립 태스크는 같은 라운드에 묶어 병렬 실행 (예: semantic_search 여러 쿼리, 또는 이전 라운드 ID/name을 활용하는 get_entities + get_researcher_network)
- 의존 태스크는 순서 보장 (semantic_search → get_entities 는 반드시 다른 라운드)
- 이미 충분한 데이터가 있으면 tasks=[]를 반환해 수집을 종료하세요
</strategy>

<follow_up_rule>
후속 질문인지 판단할 때 다음 기준을 따르세요:

tasks=[] (도구 호출 불필요):
  - 이전 답변에 이미 충분한 상세 정보가 있고, 단순 필터·정렬·요약 요청인 경우
  - 예: "그 중 KAIST만 보여줘", "위 목록 h-index 순으로 정렬해줘"

get_entities 호출 필요:
  - 이전 답변이 목록/요약 수준이었고, 특정 항목의 상세 정보를 요청하는 경우
  - 예: "한동현 연구자 자세히 알려줘", "첫 번째 특허 상세 내용 보여줘"
  - 이 경우 대화 히스토리에서 해당 항목의 ID를 찾아 get_entities로 조회하세요

새 RAG 검색 필요:
  - 이전 대화와 완전히 다른 주제이거나 새로운 발견이 필요한 경우
</follow_up_rule>
"""


def _build_tool_reference(tools_by_name: dict) -> str:
    lines = []
    for name, tool in tools_by_name.items():
        desc = (getattr(tool, "description", "") or "").strip()
        first_line = desc.splitlines()[0] if desc else ""
        lines.append(f"  {name}: {first_line}")
    tool_list = "\n".join(lines)
    return f"<available_tools>\n{tool_list}\n</available_tools>"


async def orchestrator(state: RDAgentState, config: RunnableConfig) -> dict:
    settings = get_settings()
    configurable = config.get("configurable", {})
    max_iterations: int = configurable.get("max_replan", 3)
    tools_by_name: dict = configurable.get("tools_by_name", {})
    iteration_count: int = state.get("iteration_count", 0)

    tool_results = state.get("tool_results", {})
    if tool_results:
        lines = [
            f"- {name}: {len([r for r in results if not str(r).startswith('[ERROR]')])}건 수집됨"
            for name, results in tool_results.items()
        ]
        collected = "\n".join(lines)
    else:
        collected = "없음"

    remaining = max_iterations - iteration_count

    tool_reference = _build_tool_reference(tools_by_name)

    system_prompt = f"""당신은 R&D 데이터 수집 오케스트레이터입니다. reasoning은 한국어로 작성하세요.
사용자 질문에 완전히 답하기 위해 필요한 데이터를 수집하고, 완료되면 tasks=[]를 반환하세요.
당신은 도구를 직접 실행하지 않습니다 — tasks 목록을 반환하면 워커가 병렬로 실행합니다.

{tool_reference}
{_STRATEGY}

<collection_status>
{collected}
</collection_status>

<history_guide>
대화 히스토리에서 이전 라운드의 [계획한 태스크] 항목을 확인하세요.
동일한 도구+인수 조합은 절대 반복하지 마세요.
이미 수집한 데이터로 답할 수 있거나, 새로운 검색 전략이 없다면 즉시 tasks=[]를 반환하세요.
</history_guide>

<constraints>
- 남은 수집 라운드: {remaining}회 (0이면 반드시 tasks=[] 반환)
- 한 라운드에 병렬 실행 가능한 독립 태스크를 최대한 묶어서 반환하세요
</constraints>"""

    messages = list(state["messages"])
    approx_tokens = sum(len(str(m.content)) // 4 for m in messages)
    if should_compact(messages, approx_tokens):
        llm_plain = ChatOllama(model=settings.rnd_model, base_url=settings.ollama_base_url)
        messages = compact_messages(messages, llm_plain)

    llm = ChatOllama(model=settings.rnd_model, base_url=settings.ollama_base_url)
    structured = llm.with_structured_output(OrchestratorPlan)

    messages_to_send = [SystemMessage(content=system_prompt)] + messages
    try:
        plan: OrchestratorPlan = await structured.ainvoke(messages_to_send)
        tasks = [t.model_dump() for t in plan.tasks]
        reasoning = plan.reasoning
    except Exception as e:
        logger.error("[orchestrator] structured output 실패: %s", e)
        tasks = []
        reasoning = "오류로 인해 수집 종료"

    if tasks:
        task_lines = "\n".join(f"  - {t['tool']}({t.get('args', {})})" for t in tasks)
        msg_content = f"{reasoning}\n\n[계획한 태스크]\n{task_lines}"
    else:
        msg_content = f"{reasoning}\n\n[수집 완료 — 생성 단계 진행]"

    logger.debug("[orchestrator] iter=%d/%d tasks=%d reasoning=%s",
                 iteration_count + 1, max_iterations, len(tasks), reasoning[:120])

    return {
        "messages":        [AIMessage(content=msg_content, name="orchestrator")],
        "pending_tasks":   tasks,
        "iteration_count": iteration_count + 1,
    }
