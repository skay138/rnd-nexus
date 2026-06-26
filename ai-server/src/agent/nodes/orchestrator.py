import logging
import time
from pydantic import BaseModel, Field
from typing import Any
from langchain_core.messages import SystemMessage, AIMessage, RemoveMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from common.llm import get_llm, llm_ainvoke
from common.config.query_config import RequestConfig
from agent.utils.context import get_turn_context
from agent.state import RDAgentState
from config import get_settings
from memory.compaction import should_compact, compact_messages

logger = logging.getLogger(__name__)


class OrchestratorPlan(BaseModel):
    reasoning: str       = Field(description="수집 현황 평가 및 다음 전략 (한국어)")
    tasks: list[str]     = Field(description="병렬 실행할 '자연어' 태스크 지시문 목록. 수집 완료 시 빈 리스트")


_STRATEGY = """
<strategy>
- 독립적인 조사 태스크는 같은 라운드에 묶어 병렬 실행 (예: "논문 조사"와 "특허 동향 분석")
- 워커가 내부에서 도구 순서를 스스로 결정하므로 의존관계 태스크도 하나의 태스크로 기술 가능
  (예: "AI 반도체 분야 핵심 논문을 찾고 저자의 연구자 네트워크를 파악해라" → 워커가 알아서 처리)
- 대화 히스토리의 [tool_results] 메시지를 보고 이미 충분한 데이터가 있으면 tasks=[]로 수집 종료
- 워커는 서로 완전히 독립적으로 병렬 실행되므로, 다른 워커의 태스크를 참조할 수 없습니다.
</strategy>

<task_writing_guidelines>
- "관련된 주제", "해당 기술", "위에서 찾은"과 같은 지시 대명사나 문맥 의존적인 표현을 절대 사용하지 마세요.
- 각 태스크는 그 자체로 완전한 문맥(구체적인 키워드, 도메인, 목적 등)을 포함해야 합니다.
</task_writing_guidelines>

<topic_discipline>
- 사용자 원본 질문의 핵심 주제·키워드를 모든 라운드에서 유지하라
- 새로운 도메인·주제로 확장하지 마라 — 원본 질문 범위 내에서만 심화 조사
- 각 태스크에 핵심 키워드를 반드시 포함하라 (지시대명사·생략 금지)
</topic_discipline>

<follow_up_rule>
tasks=[] — 이미 수집된 데이터로 충분한 경우:

새 태스크 필요:
  - 원본 질문 범위 내에서 아직 수집되지 않은 측면
  - 추가 조사 명시 요청
  - 이전 결과의 특정 항목 상세 조회
</follow_up_rule>
"""


def _build_capabilities(tools_by_name: dict[str, Any]) -> str:
    lines = []
    for tool in tools_by_name.values():
        desc = (getattr(tool, "description", "") or "").strip()
        first_line = desc.splitlines()[0] if desc else ""
        if first_line:
            lines.append(f"  - {first_line}")
    return "<available_capabilities>\n" + "\n".join(lines) + "\n</available_capabilities>"


async def orchestrator(state: RDAgentState, config: RunnableConfig) -> dict:
    settings = get_settings()
    configurable: dict[str, Any] = config.get("configurable", {})
    max_iterations: int = configurable.get("max_iterations", 3)
    tools_by_name: dict[str, Any] = configurable.get("tools_by_name", {})
    iteration_count: int = state.get("iteration_count", 0)

    system_prompt = f"""당신은 R&D 데이터 수집 오케스트레이터입니다. 답변은 한국어로 작성하세요.
사용자 질문에 완전히 답하기 위해 필요한 데이터를 수집하고, 완료되면 tasks=[]를 반환하세요.
당신은 태스크를 기술하고 워커에게 위임합니다 — 도구를 직접 지정하지 마세요.
각 워커는 태스크 설명을 보고 스스로 적합한 도구를 선택해 실행합니다.

{_build_capabilities(tools_by_name)}
{_STRATEGY}

<constraints>
- 현재 라운드: {iteration_count + 1} / {max_iterations}
- 추가 데이터가 필요하면 다른 관점·키워드·범위로 접근하는 새로운 태스크를 계획하세요
- 각 태스크 설명은 워커가 독립적으로 이해할 수 있을 만큼 구체적으로 작성하세요.
</constraints>

<output_format>
반드시 아래 JSON 형식으로만 답변하세요. 다른 텍스트는 절대 포함하지 마세요.
{{"reasoning": "수집 현황 평가 및 전략 (한국어)", "tasks": ["태스크1", "태스크2"]}}
수집 완료 시: {{"reasoning": "완료 이유", "tasks": []}}
</output_format>"""

    messages = list(state["messages"])
    approx_tokens = sum(len(str(m.content)) // 4 for m in messages)
    compaction_msgs: list = []
    if should_compact(messages, approx_tokens):
        llm_plain = get_llm(model=RequestConfig.current().compact_model or settings.rnd_model, enable_thinking=False)
        compacted = await compact_messages(messages, llm_plain)
        # 새롭게 반환된 compacted에 포함되지 않은 과거 메시지의 ID만 추려내어 삭제
        kept_ids = {m.id for m in compacted if getattr(m, "id", None)}
        compaction_msgs = [RemoveMessage(id=m.id) for m in messages if getattr(m, "id", None) and m.id not in kept_ids]
        # 새롭게 생성된 요약 메시지(compacted[0])만 상태에 추가
        compaction_msgs.append(compacted[0])
        messages = compacted

    prev_context, turn_start, current_msgs = get_turn_context(messages)
    formatted_current = []
    for m in current_msgs:
        if getattr(m, "name", None) == "tool_results":
            formatted_current.append(
                HumanMessage(content=f"[수집된 데이터]\n{m.content}", name="tool_results")
            )
        else:
            formatted_current.append(m)

    relevant_messages = prev_context + formatted_current

    llm = get_llm(model=RequestConfig.current().orchestrator_model or settings.rnd_model)

    t0 = time.perf_counter()
    try:
        raw = await llm_ainvoke(llm, [SystemMessage(content=system_prompt)] + relevant_messages)
        plan = OrchestratorPlan.model_validate_json(raw)
        tasks = plan.tasks
        reasoning = plan.reasoning
    except Exception as e:
        logger.error("[orchestrator] structured output 실패: %s", e)
        tasks = []
        reasoning = f"계획 수립 실패 ({type(e).__name__}) — 현재까지 수집된 데이터로 답변합니다."
    elapsed = time.perf_counter() - t0

    if tasks:
        task_lines = "\n".join(f"  - {t}" for t in tasks)
        msg_content = f"{reasoning}\n\n[계획한 태스크]\n{task_lines}"
    else:
        msg_content = f"{reasoning}\n\n[수집 완료 — 생성 단계 진행]"

    logger.debug(
        "[orchestrator] iter=%d/%d elapsed=%.2fs tasks=%d\nreasoning: %s\ntasks:\n%s",
        iteration_count + 1, max_iterations, elapsed, len(tasks),
        reasoning,
        "\n".join(f"  - {t}" for t in tasks) if tasks else "  (없음)",
    )

    return {
        "messages":        compaction_msgs + [AIMessage(content=msg_content, name="orchestrator")],
        "pending_tasks":   tasks,
        "iteration_count": iteration_count + 1,
    }
