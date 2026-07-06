"""Worker Agent — 태스크 하나를 자율 수행하는 mini ReAct 루프.

도구 호출 트래픽(AIMessage(tool_calls)+ToolMessage)은 워커 내부에 격리되며
state.messages에는 올라가지 않는다. 전체 result_text는 TaskExecutionResult로만 전달.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from agent.utils.context import get_today_message
from agent.worker.enrichment import SEARCH_TOOLS, auto_join_details
from agent.state import TaskExecutionResult, TaskSpec, ToolCallRecord
from common.config.query_config import RequestConfig
from common.llm import get_llm
from common.parsers import (
    clean_tool_result,
    extract_tool_error,
    strip_code_fence,
    strip_think,
    summarize_tool_result,
    try_parse,
)

logger = logging.getLogger(__name__)

WORKER_MAX_STEPS = 10

WORKER_SYSTEM_PROMPT = """<role>
You are an autonomous R&D data collection worker. Your objective is to collect requested data by calling tools. Do NOT write analysis or conversational text.
</role>

<instructions>
# 1. Task Prioritization
- **Focus on [태스크]**: This is your primary objective. Use **[원본 질문]** only as supplementary context to resolve missing keywords or pronouns.
- **Do not repeat work**: If **[Completed tasks from previous rounds]** is provided, avoid repeating those exact searches.
- **Stop condition**: Stop calling tools when sufficient relevant data is collected or all reasonable search paths are exhausted.

# 2. Search Strategy
- **Explicit IDs First**: If **[태스크]** mentions specific entity IDs (e.g., "R004", "P002"), call `get_entities` on those IDs FIRST.
- **Anchor -> Target Traversal**: For relational searches (e.g., "A연구자의 논문", "B기술이 적용된 과제"):
  1. Find the **anchor entity** (A or B) first.
  2. Once the anchor's ID is found, use `run_graph_query` to traverse directly to the target entities.
  3. **DO NOT** run semantic search on the target entities unless the anchor entity cannot be found or graph traversal returns no results.
- **Entity Type Mapping**: Always use the correct `entity_type` (논문=Paper, 연구자=Researcher, 과제=Project, 기관=Organization, 기술=Technology, 특허=Patent).

# 3. Data Processing & Filtering
- **Use Joined Details**: Semantic search results automatically include detailed fields. Use them immediately for filtering.
- **Hydrate Bare IDs**: If graph queries or network tools return bare IDs without details, you **MUST** call `get_entities` on those IDs before finishing.
- **Strict Connection Filtering**: If the task requires a connection to a specific entity (e.g., "P002를 인용한 논문"), you MUST verify the direct connection via detailed fields (authors, citations, etc.). **Topical similarity is NOT relevance.**
- **Loose General Filtering**: For broad topical searches, exclude only CLEARLY unrelated entities. If details are insufficient to judge, **INCLUDE** the ID. Avoid losing potentially relevant data.

# 4. Final Output Format
When finished, reply with ONLY the following JSON object. Do not include any other text.
{"summary": "한 줄 보고 — 무엇을 수집했는지 또는 왜 찾지 못했는지 (Korean)", "relevant_ids": ["ID1", "ID2"]}

- `relevant_ids`: Contains ONLY the **relevant** IDs found in your tool results. Exclude irrelevant IDs. **NEVER** hallucinate IDs. Order by most relevant first. If none are relevant, use `[]`.
</instructions>"""


def parse_worker_final(text: str) -> tuple[str, list[str], bool]:
    """워커 최종 응답({"summary", "relevant_ids"} JSON) 파싱.

    반환: (보고 한 줄, 선별 ID 목록, 선별 유효 여부).
    유효 여부 True + 빈 목록 = 워커가 '관련 엔티티 없음'을 명시한 것 —
    파싱 실패(평문 응답)의 '선별 정보 없음'(False)과 구분된다.
    """
    data = try_parse(strip_code_fence(text))
    if isinstance(data, dict) and "relevant_ids" in data:
        summary = str(data.get("summary", "")).strip()
        ids = [str(i) for i in data.get("relevant_ids") or [] if i]
        return (summary or text), ids, True
    if isinstance(data, dict):
        summary = str(data.get("summary", "")).strip()
        return (summary or text), [], False
    return text, [], False


async def run_worker(
    task: TaskSpec,
    tools_by_name: dict[str, Any],
    settings: Any,
    current_round: int = 0,
    original_query: str = "",
    history_summary: str = "",
    sse_queue: asyncio.Queue | None = None,
) -> TaskExecutionResult:
    """TaskSpec을 받아 도구를 선택·실행하고 TaskExecutionResult를 반환한다."""
    task_id = task["id"]
    task_description = task["description"]
    tool_calls: list[ToolCallRecord] = []
    worker_note = ""                  # 워커 최종 보고 한 줄 — orchestrator 수집 완료 판단용
    selected_ids: list[str] = []      # 워커가 선별한 태스크 관련 엔티티 ID
    selection_valid = False           # relevant_ids가 유효 JSON으로 반환됐는지

    llm = get_llm(model=RequestConfig.current().worker_model or settings.rnd_model)
    llm_with_tools = llm.bind_tools(list(tools_by_name.values()))

    task_content = (
        f"[원본 질문]\n{original_query}\n\n[태스크]\n{task_description}"
        if original_query
        else task_description
    )
    if history_summary:
        task_content += f"\n\n{history_summary}"

    messages: list = [
        SystemMessage(content=WORKER_SYSTEM_PROMPT),
        get_today_message(),
        HumanMessage(content=task_content),
    ]
    seen_calls: set[str] = set()

    try:
        for step in range(WORKER_MAX_STEPS):
            response = await llm_with_tools.ainvoke(messages)
            messages.append(response)

            if not getattr(response, "tool_calls", None):
                worker_note, selected_ids, selection_valid = parse_worker_final(
                    strip_think(str(response.content))
                )
                logger.debug(
                    "[WORKER] %-38s  ✓ done  %d steps  %d tools  selected=%d  note=%s",
                    f'"{task_description[:35]}"', step + 1, len(tool_calls),
                    len(selected_ids), worker_note[:60],
                )
                break

            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc.get("args", {})
                call_key = f"{tool_name}:{json.dumps(tool_args, sort_keys=True, ensure_ascii=False)}"

                if call_key in seen_calls:
                    logger.debug("[worker:%s] 중복 호출 건너뜀: %s", task_description[:40], call_key[:80])
                    messages.append(ToolMessage(
                        content="[SKIP] 이미 동일한 호출 결과가 있습니다.",
                        tool_call_id=tc["id"],
                    ))
                    continue
                seen_calls.add(call_key)

                t0 = time.perf_counter()
                try:
                    result = await tools_by_name[tool_name].ainvoke(tool_args)
                    result_str = str(result)
                except KeyError:
                    result_str = f"[ERROR] 알 수 없는 도구: {tool_name}"
                except Exception as e:
                    result_str = f"[ERROR] {tool_name} 실패: {type(e).__name__}: {e}"
                elapsed = time.perf_counter() - t0

                # MCP 도구의 [{"error": ...}] 행을 [ERROR] 문자열로 정규화 — is_error 판정 일원화
                tool_err = extract_tool_error(result_str)
                if tool_err:
                    result_str = f"[ERROR] {tool_name}: {tool_err}"

                result_text = clean_tool_result(result_str)
                is_error = result_str.startswith("[ERROR]")
                if not is_error and tool_name in SEARCH_TOOLS:
                    # 검색 행에 상세 필드 자동 조인 — 워커가 루프 안에서 전체 필드를 보고 판단
                    result_text = await auto_join_details(result_text, tools_by_name)
                summary = summarize_tool_result(result_str)

                record: ToolCallRecord = {
                    "tool_name":   tool_name,
                    "args":        tool_args,
                    "result_text": result_text,
                    "summary":     summary,
                    "is_error":    is_error,
                }
                tool_calls.append(record)

                messages.append(ToolMessage(content=result_text, tool_call_id=tc["id"]))

                logger.debug(
                    "[WORKER] %-38s  step=%d  %s  %.2fs\n  in  | %s\n  out | %s",
                    f'"{task_description[:35]}"', step + 1, tool_name, elapsed,
                    json.dumps(tool_args, ensure_ascii=False),
                    summary,
                )

                if sse_queue is not None:
                    await sse_queue.put(("worker_result", {
                        "type":    "task_result",
                        "task_id": task_id,
                        "round":   current_round,
                        "task":    task_description,
                        "tools":   [{"name": tool_name, "summary": summary}],
                    }))

        has_data  = any(not r["is_error"] for r in tool_calls)
        has_error = any(r["is_error"] for r in tool_calls)
        status = "error" if (has_error and not has_data) else ("empty" if not tool_calls else "completed")

        result: TaskExecutionResult = {
            "task_id":          task_id,
            "task_description": task_description,
            "round":            current_round,
            "status":           status,
            "tool_calls":       tool_calls,
            "worker_note":      worker_note,
            "selected_ids":     selected_ids,
            "selection_valid":  selection_valid,
        }
        return result

    except Exception as e:
        logger.error("[worker:%s] 에러: %s", task_description[:40], e)
        err_result: TaskExecutionResult = {
            "task_id":          task_id,
            "task_description": task_description,
            "round":            current_round,
            "status":           "error",
            "tool_calls":       tool_calls + [{
                "tool_name":   "worker_error",
                "args":        {},
                "result_text": f"[ERROR] 워커 실행 중 에러 ({type(e).__name__}): {e}",
                "summary":     "워커 오류",
                "is_error":    True,
            }],
            "worker_note":      "",
            "selected_ids":     [],
            "selection_valid":  False,
        }
        return err_result
