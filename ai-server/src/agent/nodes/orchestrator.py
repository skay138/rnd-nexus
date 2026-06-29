import datetime
import logging
import time
from pydantic import BaseModel, Field
from typing import Any
from langchain_core.messages import SystemMessage, AIMessage, RemoveMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from common.llm import get_llm, llm_ainvoke
from common.config.query_config import RequestConfig
from common.parsers import build_deduped_context
from agent.utils.context import get_turn_context
from agent.state import RDAgentState
from config import get_settings
from memory.compaction import should_compact, compact_messages

logger = logging.getLogger(__name__)


class OrchestratorPlan(BaseModel):
    reasoning: str       = Field(description="수집 현황 평가 및 다음 전략 (한국어)")
    tasks: list[str]     = Field(description="병렬 실행할 '자연어' 태스크 지시문 목록. 수집 완료 시 빈 리스트")
    out_of_scope: bool   = Field(default=False, description="R&D와 전혀 무관한 일상 대화, 날씨, 단순 번역, 코딩 질문이면 true. 논문·특허·연구자·기술·과제나 R&D 용어 및 개념 질문은 false")


_STRATEGY = """
<strategy>
- 서로 독립적인 태스크는 같은 라운드에 묶어 병렬 실행한다.
- 어떤 태스크의 실행 결과(검색으로 얻은 ID, 식별자, URL 등)가 다음 태스크의 입력으로 필요하면 절대 분리하지 말고 하나의 태스크로 합쳐라. 워커는 해당 태스크 내부에서 필요한 순차 작업을 수행한다.
- 동일한 사용자 입력(이름, 키워드 등)만 사용하는 태스크는 서로 독립이므로 병렬 실행한다.
- [tool_results]를 보고 충분한 데이터가 수집됐으면 tasks=[]로 종료하고, 부족하면 다른 관점이나 키워드로 추가 태스크를 계획한다.
</strategy>

<task_writing_rules>
- 각 태스크는 다른 태스크의 결과를 기대하지 않는 독립 실행 단위여야 한다.
- 각 태스크에는 사용자 질문의 핵심 키워드(이름, ID 등)를 반드시 포함한다.
- 이미 수집된 데이터에 ID 등 식별자가 있으면 태스크 설명에 직접 명시한다.
  (예: "R005 홍길동 연구자의 공동연구 네트워크를 조회하라")
- 아직 식별자를 모른다면 검색→조회 과정을 하나의 태스크에 포함한다.
  (예: "홍길동 연구자의 상세 프로필을 조사하라. 필요하면 연구자를 검색하여 ID를 찾은 뒤 상세 정보를 조회한다.")
- 원본 질문 범위를 벗어난 새로운 도메인이나 주제로 확장하지 않는다.

금지 예시 (앞 태스크 결과에 의존 → 독립 실행 불가):
✗ [
  "홍길동 연구자를 검색하라",
  "검색된 연구자 ID로 상세 정보를 조회하라"
]

올바른 예시 1 (검색→조회를 하나의 태스크로 결합):
✓ [
  "홍길동 연구자의 상세 프로필을 조사하라. 필요하면 검색하여 ID를 찾고 상세 정보를 조회한다."
]

올바른 예시 2 (동일 입력만 사용하므로 병렬 가능):
✓ [
  "홍길동 연구자의 논문을 조사하라",
  "홍길동 연구자의 특허를 조사하라",
  "홍길동 연구자의 연구과제를 조사하라"
]
</task_writing_rules>
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
- reasoning은 [수집된 데이터]에 명시된 사실만 근거로 작성하라. 데이터에 없는 기관·인물·관계·수치를 유추·창작하지 마라. 데이터가 없으면 수집 전략만 기술하라.
</constraints>

<output_format>
반드시 아래 JSON 형식으로만 답변하세요. 다른 텍스트는 절대 포함하지 마세요.
{{"reasoning": "...", "tasks": ["태스크1", "태스크2"], "out_of_scope": false}}
수집 완료 시: {{"reasoning": "완료 이유", "tasks": [], "out_of_scope": false}}
범위 외 질문(레시피·날씨·일반상식 등): {{"reasoning": "범위 외 이유", "tasks": [], "out_of_scope": true}}
</output_format>"""

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

    prev_context, turn_start, current_msgs, prev_tool_results = get_turn_context(messages)
    formatted_current = []
    for m in current_msgs:
        if getattr(m, "name", None) == "tool_results":
            formatted_current.append(
                HumanMessage(content=f"[수집된 데이터]\n{m.content}", name="tool_results")
            )
        elif getattr(m, "name", None) == "orchestrator":
            # reasoning 제거 — 틀린 추론이 다음 라운드로 재투입되어 누적되는 것 방지
            # 태스크 목록만 유지하여 "무엇을 계획했는가"만 전달
            content = str(m.content)
            if "\n\n[계획한 태스크]" in content:
                tasks_part = content.split("\n\n[계획한 태스크]", 1)[1].strip()
                formatted_current.append(
                    AIMessage(content=f"[계획한 태스크]\n{tasks_part}", name="orchestrator")
                )
            # 수집 완료/범위 외 메시지는 재투입 불필요 — skip
        else:
            formatted_current.append(m)

    # 날짜를 HumanMessage로 주입 — 오케스트레이터는 JSON 구조화 출력이므로 persona break 위험 낮음
    # system_prompt는 정적 유지 → KV prefix cache 최대 활용
    today = datetime.date.today().strftime("%Y년 %m월 %d일")
    date_msg = HumanMessage(content=f"[오늘 날짜: {today}]")

    # 이전 턴에서 수집된 raw 데이터를 compact summary로 주입
    # — 오케스트레이터가 이미 수집된 엔티티를 파악해 중복 태스크 방지
    if prev_tool_results:
        prev_data = build_deduped_context(prev_tool_results)
        prev_data_msg = HumanMessage(content=f"[이전 대화 수집 데이터]\n{prev_data}", name="tool_results")
        relevant_messages = [date_msg, prev_data_msg] + prev_context + formatted_current
    else:
        relevant_messages = [date_msg] + prev_context + formatted_current

    # json_mode=True: Ollama → format="json", OpenAI 호환 → response_format=json_object
    # <think> 블록 제거는 llm_ainvoke가 담당 (with_structured_output은 <think> 처리 불가)
    llm = get_llm(model=RequestConfig.current().orchestrator_model or settings.rnd_model, json_mode=True)

    _MAX_RETRIES = 2
    t0 = time.perf_counter()
    tasks = []
    reasoning = ""
    out_of_scope = False
    invoke_msgs = [SystemMessage(content=system_prompt)] + relevant_messages
    for attempt in range(_MAX_RETRIES + 1):
        try:
            raw = await llm_ainvoke(llm, invoke_msgs)
            plan = OrchestratorPlan.model_validate_json(raw)
            tasks = plan.tasks
            reasoning = plan.reasoning
            out_of_scope = plan.out_of_scope
            break
        except Exception as e:
            if attempt < _MAX_RETRIES:
                logger.warning("[orchestrator] JSON 파싱 실패, 재시도 (%d/%d): %s", attempt + 1, _MAX_RETRIES, e)
            else:
                logger.error("[orchestrator] structured output 최종 실패: %s", e)
                reasoning = f"계획 수립 실패 ({type(e).__name__}) — 현재까지 수집된 데이터로 답변합니다."
    elapsed = time.perf_counter() - t0

    if tasks:
        task_lines = "\n".join(f"  - {t}" for t in tasks)
        msg_content = f"{reasoning}\n\n[계획한 태스크]\n{task_lines}"
    elif out_of_scope:
        msg_content = f"{reasoning}\n\n[범위 외 질문]"
    else:
        msg_content = f"{reasoning}\n\n[수집 완료 — 생성 단계 진행]"

    logger.debug(
        "[orchestrator] iter=%d/%d elapsed=%.2fs tasks=%d out_of_scope=%s\nreasoning: %s\ntasks:\n%s",
        iteration_count + 1, max_iterations, elapsed, len(tasks), out_of_scope,
        reasoning,
        "\n".join(f"  - {t}" for t in tasks) if tasks else "  (없음)",
    )

    return {
        "messages":        compaction_msgs + [AIMessage(content=msg_content, name="orchestrator")],
        "pending_tasks":   tasks,
        "iteration_count": iteration_count + 1,
        "out_of_scope":    out_of_scope,
    }
