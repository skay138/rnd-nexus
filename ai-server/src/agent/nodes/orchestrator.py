import json
import logging
import time
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from agent.state import RDAgentState
from config import get_settings
from memory.compaction import should_compact, compact_messages

logger = logging.getLogger(__name__)


class TaskDescription(BaseModel):
    description: str = Field(description="워커에게 전달할 태스크 설명 — 무엇을 조사/수집할지 자연어로 기술")
    label: str       = Field(default="", description="태스크 레이블 (로깅·UI용)")


class OrchestratorPlan(BaseModel):
    reasoning: str              = Field(description="수집 현황 평가 및 다음 전략 (한국어)")
    tasks: list[TaskDescription] = Field(description="병렬 실행할 태스크 목록. 수집 완료 시 빈 리스트")


_STRATEGY = """
<strategy>
- 독립적인 조사 태스크는 같은 라운드에 묶어 병렬 실행 (예: "논문 조사"와 "특허 동향 분석")
- 워커가 내부에서 도구 순서를 스스로 결정하므로 의존관계 태스크도 하나의 태스크로 기술 가능
  (예: "관련 논문을 찾고 해당 저자의 연구자 네트워크를 파악해라" → 워커가 알아서 처리)
- 대화 히스토리의 [tool_results] 메시지를 보고 이미 충분한 데이터가 있으면 tasks=[]로 수집 종료
</strategy>

<follow_up_rule>
tasks=[] (재검색 불필요):
  - 이전 답변에 충분한 정보가 있고 단순 필터·정렬·요약 요청인 경우

새 조사 태스크:
  - 이전 대화와 다른 주제이거나 추가 데이터가 필요한 경우
  - 태스크 설명에 맥락을 충분히 포함해 워커가 독립적으로 실행 가능하게 작성
</follow_up_rule>
"""


def _build_capabilities(tools_by_name: dict) -> str:
    lines = []
    for name, tool in tools_by_name.items():
        desc = (getattr(tool, "description", "") or "").strip()
        first_line = desc.splitlines()[0] if desc else ""
        lines.append(f"  {name}: {first_line}")
    return "<available_capabilities>\n" + "\n".join(lines) + "\n</available_capabilities>"


async def orchestrator(state: RDAgentState, config: RunnableConfig) -> dict:
    settings = get_settings()
    configurable    = config.get("configurable", {})
    max_iterations  = configurable.get("max_replan", 3)
    tools_by_name   = configurable.get("tools_by_name", {})
    iteration_count = state.get("iteration_count", 0)

    system_prompt = f"""당신은 R&D 데이터 수집 오케스트레이터입니다. reasoning은 한국어로 작성하세요.
사용자 질문에 완전히 답하기 위해 필요한 데이터를 수집하고, 완료되면 tasks=[]를 반환하세요.
당신은 태스크를 기술하고 워커에게 위임합니다 — 도구를 직접 지정하지 마세요.
각 워커는 태스크 설명을 보고 스스로 적합한 도구를 선택해 실행합니다.

{_build_capabilities(tools_by_name)}
{_STRATEGY}

<constraints>
- 현재 라운드: {iteration_count + 1} / {max_iterations}
- 대화 히스토리에서 이미 다룬 주제의 태스크는 재지시하지 마세요
- 각 태스크 설명은 워커가 독립적으로 이해할 수 있을 만큼 구체적으로 작성하세요
</constraints>"""

    messages = list(state["messages"])
    approx_tokens = sum(len(str(m.content)) // 4 for m in messages)
    if should_compact(messages, approx_tokens):
        llm_plain = ChatOllama(model=settings.rnd_model, base_url=settings.ollama_base_url)
        messages = compact_messages(messages, llm_plain)

    llm = ChatOllama(model=settings.rnd_model, base_url=settings.ollama_base_url)
    structured = llm.with_structured_output(OrchestratorPlan)

    t0 = time.perf_counter()
    try:
        plan: OrchestratorPlan = await structured.ainvoke(
            [SystemMessage(content=system_prompt)] + messages
        )
        tasks = [t.model_dump() for t in plan.tasks]
        reasoning = plan.reasoning
    except Exception as e:
        logger.error("[orchestrator] structured output 실패: %s", e)
        tasks = []
        reasoning = f"계획 수립 실패 ({type(e).__name__}) — 현재까지 수집된 데이터로 답변합니다."
    elapsed = time.perf_counter() - t0

    if tasks:
        task_lines = "\n".join(
            f"  - [{t.get('label', '')}] {t['description']}" for t in tasks
        )
        msg_content = f"{reasoning}\n\n[계획한 태스크]\n{task_lines}"
    else:
        msg_content = f"{reasoning}\n\n[수집 완료 — 생성 단계 진행]"

    logger.debug(
        "[orchestrator] iter=%d/%d elapsed=%.2fs tasks=%d\nreasoning: %s\ntasks:\n%s",
        iteration_count + 1, max_iterations, elapsed, len(tasks),
        reasoning,
        json.dumps(tasks, ensure_ascii=False, indent=2) if tasks else "  (없음)",
    )

    return {
        "messages":        [AIMessage(content=msg_content, name="orchestrator")],
        "pending_tasks":   tasks,
        "iteration_count": iteration_count + 1,
    }
