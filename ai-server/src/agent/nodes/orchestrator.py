import datetime
import hashlib
import logging
import re
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
    tasks: list[str]  = Field(description="병렬 실행할 '자연어' 태스크 지시문 목록. 수집 완료 시 빈 리스트")
    out_of_scope: bool = Field(default=False, description="날씨·요리·스포츠·코딩 등 R&D와 전혀 무관한 질문이면 true. R&D 용어·개념 질문, 논문·특허·연구자·기술·과제 관련 질문은 false")


_DEPENDENCY_SIGNALS = (
    "검색된", "찾은", "조회된", "수집된", "발견된",
    "위의", "앞서", "이전 결과", "이전 단계", "위에서",
)

_STRATEGY = """
<instructions>
- Bundle independent tasks in the same round for parallel execution.
- If one task's output (IDs, identifiers) is required as input for the next, never split them — merge into a single task. Workers handle sequential steps autonomously within a task.
- If a task description contains "검색된", "찾은", "조회된", "수집된", "위의", "앞서", "이전 결과" (dependency signals in Korean), it depends on a prior task's results — merge it with the preceding task.
- Tasks that use only the same user-supplied input (name, keyword) are independent and can run in parallel.
- If previous tool results show sufficient data, return tasks=[] to finish; otherwise plan additional tasks from a different angle, keyword, or scope.
- Each task must include the core keywords from the user's question (names, IDs, etc.).
- If IDs or identifiers have already been collected, include them directly in the task description.
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


def _strip_code_fence(s: str) -> str:
    s = s.strip()
    m = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", s, re.DOTALL)
    return m.group(1).strip() if m else s


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
You are an R&D data collection orchestrator. Write task descriptions in Korean.
Collect the data needed to fully answer the user's question, then return tasks=[] when done.
Describe tasks and delegate to workers — do not specify tool names or parameters.
Each worker reads the task description and autonomously selects the appropriate tools.
</role>

{_build_capabilities(tools_by_name)}

{_STRATEGY}

<constraints>
- Current round: {iteration_count + 1} / {max_iterations}
{"- This is the final collection round. Answer generation begins immediately after." if iteration_count + 1 >= max_iterations else "- If more data is needed, plan new tasks from a different angle, keyword, or scope."}
- Each task description must be specific enough for a worker to execute independently.
</constraints>

<output_format>
Output ONLY a JSON object with EXACTLY these two keys — no other keys allowed:
  "tasks": array of strings (Korean task descriptions, or [] when done)
  "out_of_scope": boolean

Examples:
  Need data:    {{"tasks": ["태스크1", "태스크2"], "out_of_scope": false}}
  Done:         {{"tasks": [], "out_of_scope": false}}
  Out of scope: {{"tasks": [], "out_of_scope": true}}
  Term/concept: {{"tasks": [], "out_of_scope": false}}
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

    # tool_results JSON 마커 스킵 — orchestrator 메시지는 그대로 포함
    relevant_msgs = [m for m in messages if getattr(m, "name", None) != "tool_results"]

    today = datetime.date.today().strftime("%Y년 %m월 %d일")
    date_msg = HumanMessage(content=f"[오늘 날짜: {today}]")
    invoke_msgs = [SystemMessage(content=system_prompt), date_msg] + relevant_msgs
    # 컨텍스트가 길어질 경우 JSON 출력 지시를 잊지 않도록 마지막에 리마인더 추가
    invoke_msgs.append(HumanMessage(content="Output ONLY valid JSON as specified. No markdown (```json), no explanation — pure JSON only."))

    llm = get_llm(model=RequestConfig.current().orchestrator_model or settings.rnd_model, json_mode=True)

    _MAX_RETRIES = 2
    t0 = time.perf_counter()
    tasks: list[str] = []
    out_of_scope = False
    retry_msgs = list(invoke_msgs)
    for attempt in range(_MAX_RETRIES + 1):
        raw = ""
        try:
            raw = await llm_ainvoke(llm, retry_msgs)
            plan = OrchestratorPlan.model_validate_json(_strip_code_fence(raw))
            tasks = _merge_dependent_tasks(plan.tasks)
            out_of_scope = plan.out_of_scope
            break
        except Exception as e:
            if attempt < _MAX_RETRIES:
                logger.warning("[orchestrator] JSON 파싱 실패, 재시도 (%d/%d): %s\n[Raw Output]: %s", attempt + 1, _MAX_RETRIES, e, raw)
                retry_msgs = list(invoke_msgs) + [
                    AIMessage(content=raw),
                    HumanMessage(content=f'Invalid output. Required: {{"tasks": [...], "out_of_scope": bool}} — exactly these two keys only. Retry.'),
                ]
            else:
                logger.error("[orchestrator] structured output 최종 실패: %s\n[Raw Output]: %s", e, raw)
    elapsed = time.perf_counter() - t0

    if tasks:
        task_lines = "\n".join(f"  - {t}" for t in tasks)
        msg_content = f"[계획한 태스크]\n{task_lines}"
    elif out_of_scope:
        msg_content = "[범위 외 질문]"
    else:
        msg_content = "[수집 완료 — 생성 단계 진행]"

    status = "→ generate" if not tasks else "→ executor"
    task_lines = "\n".join(f"    {i+1}. {t}" for i, t in enumerate(tasks)) if tasks else "    (없음)"
    logger.debug(
        "[ORCH] round=%d/%d  %.2fs  tasks=%d  %s\n  tasks:\n%s",
        iteration_count + 1, max_iterations, elapsed, len(tasks), status,
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
