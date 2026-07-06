import datetime
import hashlib
import logging
import time
from pydantic import BaseModel, Field
from typing import Any
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from common.llm import get_llm, llm_ainvoke
from common.parsers import strip_code_fence
from common.config.query_config import RequestConfig
from agent.state import RDAgentState, TaskSpec
from agent.utils.context import split_turns, previous_turn_context, get_today_message
from config import get_settings
from memory.compaction import apply_compaction

logger = logging.getLogger(__name__)


class OrchestratorPlan(BaseModel):
    reasoning: str = Field(default="", description="수집 현황 평가와 다음 계획 전략 (한국어 1~2문장)")
    tasks: list[str]  = Field(description="병렬 실행할 '자연어' 태스크 지시문 목록. 수집 완료 시 빈 리스트")
    out_of_scope: bool = Field(default=False, description="날씨·요리·스포츠·코딩 등 R&D와 전혀 무관한 질문이면 true. R&D 용어·개념 질문, 논문·특허·연구자·기술·과제 관련 질문은 false")


_DEPENDENCY_SIGNALS = (
    "검색된", "찾은", "조회된", "수집된", "발견된",
    "위의", "앞서", "이전 결과", "이전 단계", "위에서",
)

_STRATEGY = """
<instructions>
- Bundle independent tasks in the same round for parallel execution.
- If one task's output (IDs, identifiers) is required as input for the next, never split them — merge into a single task; workers handle sequential steps autonomously. Signal words like "검색된", "찾은", "조회된", "수집된", "위의", "앞서", "이전 결과" in a task description indicate such a dependency.
- Tasks that use only the same user-supplied input (name, keyword) are independent and can run in parallel.
- Judge data sufficiency from the [수집 결과] messages collected in this turn. If they cover the user's question, return tasks=[] to finish; otherwise plan additional tasks from a different angle, keyword, or scope.
- If [수집 결과] shows empty results (빈 결과) or a failure report for a search, never plan the same search again — change keywords or scope, or finish with tasks=[].
- Each task must include the core keywords from the user's question (names, IDs, etc.).
- If IDs or identifiers have already been collected in previous rounds or from the user, include them directly in the task description.
- When a task carries collected IDs, ALWAYS keep the original selection criterion in the description so the worker can verify relevance and drop unrelated entities. Write "최유리(R004)가 저술한 논문 P004, P006의 상세를 조회하세요" — never a bare ID list like "논문 P004, P006의 상세를 조회하세요".
- If a task involves a specific entity, explicitly specify its entity type (e.g., 논문, 과제, 연구자) in the task description based on the conversation context. Example: Write "논문 12345의..." instead of "12345의...".
- Do not expand into new domains or topics beyond the scope of the original question.
</instructions>

<examples>
✗ Wrong — split with dependency signal ("검색된" cannot run without prior result):
["PIM 기술로 과제를 검색하세요", "검색된 PIM 과제와 연결된 주요 기관을 조회하세요"]
["홍길동 연구자를 검색하세요", "검색된 ID로 상세 정보를 조회하세요"]

✓ Correct 1 — sequential dependency → single task:
["PIM 기술로 과제를 검색하고, 검색된 과제와 연결된 주요 기관을 조회하여 기관별 현황을 비교하세요."]
["홍길동 연구자의 상세 프로필을 조사하세요. 필요하면 검색하여 ID를 찾고 상세 정보를 조회하세요."]

✓ Correct 2 — same input → parallel:
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
    """도구 docstring에서 첫 번째 실제 설명 문장을 추출.

    docstring이 <role> 등 태그로 시작하므로 태그·빈 줄을 건너뛴다.
    """
    lines = []
    for tool in tools_by_name.values():
        desc = (getattr(tool, "description", "") or "").strip()
        first_line = next(
            (ln.strip() for ln in desc.splitlines()
             if ln.strip() and not ln.strip().startswith("<")),
            "",
        )
        if first_line:
            lines.append(f"  - {first_line}")
    return "<tools>\n" + "\n".join(lines) + "\n</tools>"


async def orchestrator_node(state: RDAgentState, config: RunnableConfig) -> dict:
    settings = get_settings()
    configurable: dict[str, Any] = config.get("configurable", {})
    max_iterations: int = configurable.get("max_iterations", 3)
    tools_by_name: dict[str, Any] = configurable.get("tools_by_name", {})
    iteration_count: int = state.get("iteration_count", 0)

    final_round_note = (
        "\n- This is the FINAL collection round. Answer generation begins immediately after — plan only what is essential."
        if iteration_count + 1 >= max_iterations
        else ""
    )
    system_prompt = f"""<role>
You are an R&D data collection orchestrator. Write task descriptions in Korean.
Collect the data needed to fully answer the user's question, then return tasks=[] when done.
Describe tasks and delegate to workers — do not specify tool names or parameters.
Each worker reads the task description and autonomously selects the appropriate tools.
</role>

{_build_capabilities(tools_by_name)}

{_STRATEGY}

<constraints>
- Current round: {iteration_count + 1} / {max_iterations}{final_round_note}
- Each task description must be specific enough for a worker to execute independently.
</constraints>

<output_format>
Output ONLY a JSON object with EXACTLY these three keys, in this order — no other keys allowed:
  "reasoning": string (1~2 Korean sentences — evaluate collected data and state your strategy BEFORE deciding tasks)
  "tasks": array of strings (Korean task descriptions, or [] when done)
  "out_of_scope": boolean

Examples:
  Need data:    {{"reasoning": "아직 수집된 데이터가 없어 논문과 특허를 병렬로 조사한다.", "tasks": ["태스크1", "태스크2"], "out_of_scope": false}}
  Done:         {{"reasoning": "질문에 답하기에 충분한 데이터가 수집되었다.", "tasks": [], "out_of_scope": false}}
  Out of scope: {{"reasoning": "R&D와 무관한 질문이다.", "tasks": [], "out_of_scope": true}}
  Term/concept: {{"reasoning": "용어 설명 질문으로 데이터 수집이 불필요하다.", "tasks": [], "out_of_scope": false}}
</output_format>"""

    messages, compaction_msgs = await apply_compaction(
        list(state["messages"]),
        get_llm(model=RequestConfig.current().compact_model or settings.rnd_model),
    )

    # 턴 경계 분리: 이전 턴은 질문·최종답변만, 현재 턴은 질문·계획·수집 요약만
    prev_turns, current_turn = split_turns(messages)
    relevant_msgs = previous_turn_context(prev_turns) + [
        m for m in current_turn
        if isinstance(m, HumanMessage)
        or getattr(m, "name", None) in ("orchestrator", "tool_results")
    ]

    date_msg = get_today_message()
    invoke_msgs = [SystemMessage(content=system_prompt), date_msg] + relevant_msgs
    # 컨텍스트가 길어질 경우 JSON 출력 지시를 잊지 않도록 마지막에 리마인더 추가
    invoke_msgs.append(HumanMessage(content="Output ONLY valid JSON as specified. No markdown (```json), no explanation — pure JSON only."))

    llm = get_llm(model=RequestConfig.current().orchestrator_model or settings.rnd_model, json_mode=True)

    _MAX_RETRIES = 2
    t0 = time.perf_counter()
    tasks: list[str] = []
    out_of_scope = False
    plan_reasoning = ""
    retry_msgs = list(invoke_msgs)
    for attempt in range(_MAX_RETRIES + 1):
        raw = ""
        try:
            raw = await llm_ainvoke(llm, retry_msgs)
            plan = OrchestratorPlan.model_validate_json(strip_code_fence(raw))
            tasks = _merge_dependent_tasks(plan.tasks)
            out_of_scope = plan.out_of_scope
            plan_reasoning = plan.reasoning.strip()
            break
        except Exception as e:
            if attempt < _MAX_RETRIES:
                logger.warning("[orchestrator] JSON 파싱 실패, 재시도 (%d/%d): %s\n[Raw Output]: %s", attempt + 1, _MAX_RETRIES, e, raw)
                retry_msgs = list(invoke_msgs) + [
                    AIMessage(content=raw),
                    HumanMessage(content='Invalid output. Required: {"reasoning": "...", "tasks": [...], "out_of_scope": bool} — exactly these three keys only. Retry.'),
                ]
            else:
                logger.error("[orchestrator] structured output 최종 실패: %s\n[Raw Output]: %s", e, raw)
    elapsed = time.perf_counter() - t0

    if tasks:
        task_lines = "\n".join(f"  - {t}" for t in tasks)
        msg_content = f"{plan_reasoning}\n\n[계획한 태스크]\n{task_lines}".strip()
    elif out_of_scope:
        msg_content = f"{plan_reasoning}\n\n[범위 외 질문]".strip()
    else:
        msg_content = f"{plan_reasoning}\n\n[수집 완료 — 생성 단계 진행]".strip()

    status = "→ generate" if not tasks else "→ executor"
    task_lines = "\n".join(f"    {i+1}. {t}" for i, t in enumerate(tasks)) if tasks else "    (없음)"
    logger.debug(
        "[ORCH] round=%d/%d  %.2fs  tasks=%d  %s\n  reasoning: %s\n  tasks:\n%s",
        iteration_count + 1, max_iterations, elapsed, len(tasks), status,
        plan_reasoning or "(없음)",
        task_lines,
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
