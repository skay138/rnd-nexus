import datetime
import hashlib
import logging
import time
from pydantic import BaseModel, Field
from typing import Any
from langchain_core.messages import SystemMessage, AIMessage, RemoveMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from common.llm import get_llm, llm_ainvoke
from common.config.query_config import RequestConfig
from agent.state import RDAgentState, TaskSpec
from config import get_settings
from memory.compaction import should_compact, compact_messages

logger = logging.getLogger(__name__)


class OrchestratorPlan(BaseModel):
    reasoning: str       = Field(description="수집 현황 평가 및 다음 전략 (한국어)")
    tasks: list[str]     = Field(description="병렬 실행할 '자연어' 태스크 지시문 목록. 수집 완료 시 빈 리스트")
    out_of_scope: bool   = Field(default=False, description="R&D와 전혀 무관한 일상 대화, 날씨, 단순 번역, 코딩 질문이면 true. 논문·특허·연구자·기술·과제나 R&D 용어 및 개념 질문은 false")


_DEPENDENCY_SIGNALS = (
    "검색된", "찾은", "조회된", "수집된", "발견된",
    "위의", "앞서", "이전 결과", "이전 단계", "위에서",
)

_STRATEGY = """
<instructions>
- 서로 독립적인 태스크는 같은 라운드에 묶어 병렬 실행하세요.
- 어떤 태스크의 결과(ID, 식별자 등)가 다음 태스크의 입력으로 필요하면 절대 분리하지 말고 하나의 태스크로 합치세요. 워커는 태스크 내부에서 순차 작업을 자율 처리합니다.
- 태스크 설명에 "검색된", "찾은", "조회된", "수집된", "위의", "앞서", "이전 결과" 같은 표현이 들어 있으면 이전 태스크 결과에 의존한다는 신호입니다 — 반드시 앞 태스크와 하나로 합쳐야 합니다.
- 동일한 사용자 입력(이름, 키워드)만 사용하는 태스크는 독립이므로 병렬 실행하세요.
- 이전 도구 호출 결과를 보고 충분한 데이터가 수집됐으면 tasks=[]로 종료하고, 부족하면 다른 관점·키워드로 추가 태스크를 계획하세요.
- 각 태스크에는 사용자 질문의 핵심 키워드(이름, ID 등)를 반드시 포함하세요.
- 이미 수집된 ID·식별자가 있으면 태스크 설명에 직접 명시하세요.
- 원본 질문 범위를 벗어난 새로운 도메인이나 주제로 확장하지 마세요.
</instructions>

<examples>
✗ 잘못된 분리 ("검색된"이 의존성 신호 — 앞 태스크 결과 없이 실행 불가):
["PIM 기술로 과제를 검색하세요", "검색된 PIM 과제와 연결된 주요 기관을 조회하세요"]
["홍길동 연구자를 검색하세요", "검색된 ID로 상세 정보를 조회하세요"]

✓ 올바른 예시 1 — 의존 순서가 있으면 하나의 태스크로:
["PIM 기술로 과제를 검색하고, 검색된 과제와 연결된 주요 기관을 조회하여 기관별 현황을 비교하세요."]
["홍길동 연구자의 상세 프로필을 조사하세요. 필요하면 검색하여 ID를 찾고 상세 정보를 조회하세요."]

✓ 올바른 예시 2 — 동일 입력이므로 병렬 실행:
["홍길동 연구자의 논문을 조사하세요", "홍길동 연구자의 특허를 조사하세요", "홍길동 연구자의 연구과제를 조사하세요"]
</examples>
"""


def _merge_dependent_tasks(tasks: list[str]) -> list[str]:
    """앞 태스크 결과에 의존하는 태스크를 이전 태스크와 자동 병합."""
    if len(tasks) <= 1:
        return tasks
    merged = [tasks[0]]
    for task in tasks[1:]:
        if any(sig in task for sig in _DEPENDENCY_SIGNALS):
            logger.warning("[orchestrator] 의존성 태스크 감지 → 이전 태스크와 병합: %.50s…", task)
            merged[-1] = f"{merged[-1]} 그리고 {task}"
        else:
            merged.append(task)
    return merged


def _make_task_id(description: str) -> str:
    """태스크 설명의 SHA1 앞 8자 — 중복 차단 및 결과 추적용."""
    return hashlib.sha1(description.strip().lower().encode()).hexdigest()[:8]


def _build_capabilities(tools_by_name: dict[str, Any]) -> str:
    lines = []
    for tool in tools_by_name.values():
        desc = (getattr(tool, "description", "") or "").strip()
        first_line = desc.splitlines()[0] if desc else ""
        if first_line:
            lines.append(f"  - {first_line}")
    return "<tools>\n" + "\n".join(lines) + "\n</tools>"


async def orchestrator(state: RDAgentState, config: RunnableConfig) -> dict:
    settings = get_settings()
    configurable: dict[str, Any] = config.get("configurable", {})
    max_iterations: int = configurable.get("max_iterations", 3)
    tools_by_name: dict[str, Any] = configurable.get("tools_by_name", {})
    iteration_count: int = state.get("iteration_count", 0)

    system_prompt = f"""<role>
당신은 R&D 데이터 수집 오케스트레이터입니다. 답변은 한국어로 작성하세요.
사용자 질문에 완전히 답하기 위해 필요한 데이터를 수집하고, 완료되면 tasks=[]를 반환하세요.
태스크를 기술하고 워커에게 위임합니다 — 도구를 직접 지정하지 마세요.
각 워커는 태스크 설명을 보고 스스로 적합한 도구를 선택해 실행합니다.
</role>

{_build_capabilities(tools_by_name)}

{_STRATEGY}

<constraints>
- 현재 라운드: {iteration_count + 1} / {max_iterations}
{"- 마지막 수집 라운드입니다. 이번 라운드 후 바로 답변 생성 단계로 전환됩니다." if iteration_count + 1 >= max_iterations else "- 추가 데이터가 필요하면 다른 관점·키워드·범위로 접근하는 새로운 태스크를 계획하세요."}
- 각 태스크 설명은 워커가 독립적으로 이해할 수 있을 만큼 구체적으로 작성하세요.
- reasoning은 이전 도구 호출 결과에 명시된 사실만 근거로 작성하세요. 데이터에 없는 기관·인물·관계·수치를 유추·창작하지 마세요. 데이터가 없으면 수집 전략만 기술하세요.
</constraints>

<output_format>
반드시 아래 JSON 형식으로만 답변하세요. 다른 텍스트는 절대 포함하지 마세요.
{{"reasoning": "...", "tasks": ["태스크1", "태스크2"], "out_of_scope": false}}
수집 완료 시: {{"reasoning": "완료 이유", "tasks": [], "out_of_scope": false}}
범위 외 질문(레시피·날씨·단순번역·코딩 등): {{"reasoning": "범위 외 이유", "tasks": [], "out_of_scope": true}}
R&D 용어·개념 질문: {{"reasoning": "용어 설명 요청 — 데이터 수집 불필요", "tasks": [], "out_of_scope": false}}
</output_format>"""

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

    # tool_results JSON 마커 스킵, orchestrator reasoning 제거 — 나머지(tool_calls/ToolMessage)는 그대로
    relevant_msgs: list = []
    for m in messages:
        if getattr(m, "name", None) == "tool_results":
            continue
        elif getattr(m, "name", None) == "orchestrator":
            content = str(m.content)
            if "\n\n[계획한 태스크]" in content:
                tasks_part = content.split("\n\n[계획한 태스크]", 1)[1].strip()
                relevant_msgs.append(AIMessage(content=f"[계획한 태스크]\n{tasks_part}", name="orchestrator"))
        else:
            relevant_msgs.append(m)

    today = datetime.date.today().strftime("%Y년 %m월 %d일")
    date_msg = HumanMessage(content=f"[오늘 날짜: {today}]")
    invoke_msgs = [SystemMessage(content=system_prompt), date_msg] + relevant_msgs
    # 컨텍스트가 길어질 경우 JSON 출력 지시를 잊지 않도록 마지막에 리마인더 추가
    invoke_msgs.append(HumanMessage(content="반드시 시스템 프롬프트에 지정된 JSON 형식으로만 답변하세요. 마크다운(```json)이나 다른 설명 텍스트 없이 순수 JSON만 출력하세요."))

    llm = get_llm(model=RequestConfig.current().orchestrator_model or settings.rnd_model, json_mode=True)

    _MAX_RETRIES = 2
    t0 = time.perf_counter()
    tasks: list[str] = []
    reasoning = ""
    out_of_scope = False
    for attempt in range(_MAX_RETRIES + 1):
        raw = ""
        try:
            raw = await llm_ainvoke(llm, invoke_msgs)
            plan = OrchestratorPlan.model_validate_json(raw)
            tasks = _merge_dependent_tasks(plan.tasks)
            reasoning = plan.reasoning
            out_of_scope = plan.out_of_scope
            break
        except Exception as e:
            if attempt < _MAX_RETRIES:
                logger.warning("[orchestrator] JSON 파싱 실패, 재시도 (%d/%d): %s\n[Raw Output]: %s", attempt + 1, _MAX_RETRIES, e, raw)
            else:
                logger.error("[orchestrator] structured output 최종 실패: %s\n[Raw Output]: %s", e, raw)
                reasoning = f"계획 수립 실패 ({type(e).__name__}) — 현재까지 수집된 데이터로 답변합니다."
    elapsed = time.perf_counter() - t0

    if tasks:
        task_lines = "\n".join(f"  - {t}" for t in tasks)
        msg_content = f"{reasoning}\n\n[계획한 태스크]\n{task_lines}"
    elif out_of_scope:
        msg_content = f"{reasoning}\n\n[범위 외 질문]"
    else:
        msg_content = f"{reasoning}\n\n[수집 완료 — 생성 단계 진행]"

    status = "→ generate" if not tasks else ("→ out_of_scope" if out_of_scope else "")
    task_lines = "\n".join(f"    {i+1}. {t}" for i, t in enumerate(tasks)) if tasks else "    (없음)"
    logger.debug(
        "[ORCH] round=%d/%d  %.2fs  tasks=%d  %s\n  reasoning | %s\n  tasks:\n%s",
        iteration_count + 1, max_iterations, elapsed, len(tasks), status,
        reasoning, task_lines,
    )

    pending: list[TaskSpec] = [
        {"id": _make_task_id(t), "description": t} for t in tasks
    ]

    return {
        "messages":        compaction_msgs + [AIMessage(content=msg_content, name="orchestrator")],
        "pending_tasks":   pending,
        "iteration_count": iteration_count + 1,
        "out_of_scope":    out_of_scope,
    }
