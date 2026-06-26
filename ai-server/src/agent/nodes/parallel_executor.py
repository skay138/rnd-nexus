import ast
import asyncio
import json
import logging
import time
from typing import Any
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from common.llm import get_llm
from common.config.query_config import RequestConfig
from common.parsers import summarize_tool_result, iter_entities, item_to_ref
from collections import defaultdict

from agent.state import RDAgentState
from config import get_settings

logger = logging.getLogger(__name__)

_WORKER_MAX_STEPS = 5
_SEP = "─" * 50


def _task_key(task: str) -> str:
    return task.strip().lower()


def _clean_result(result_str: str) -> str:
    """MCP 결과에서 entity JSON을 추출해 LLM 친화적 텍스트로 변환."""
    if result_str.startswith("[ERROR]"):
        return result_str
    try:
        items = json.loads(result_str)
    except Exception:
        try:
            items = ast.literal_eval(result_str)
        except Exception:
            return result_str
    if not isinstance(items, list):
        return result_str
    entities = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            try:
                d = json.loads(item["text"])
                if isinstance(d, list):
                    entities.extend(d)
                else:
                    entities.append(d)
            except Exception:
                pass
        else:
            entities.append(item)
    return json.dumps(entities, ensure_ascii=False, indent=2) if entities else result_str


def _build_history_summary(state_tool_results: dict[str, list[str]]) -> str:
    """이전 라운드들의 도구 실행 결과에서 엔티티 요약(ID 및 이름) 추출"""
    entities_by_type = defaultdict(list)
    seen_ids = set()
    
    for tool_name, results in state_tool_results.items():
        for res in results:
            for d in iter_entities(res):
                ref = item_to_ref(d)
                if not ref:
                    continue
                eid = ref["id"]
                if not eid or eid in seen_ids:
                    continue
                seen_ids.add(eid)
                entities_by_type[ref["type"]].append(f"{ref['title']} ({eid})")
    
    if not entities_by_type:
        return ""
        
    lines = ["[이전 라운드 수집 데이터 요약]"]
    for ntype, items in entities_by_type.items():
        display_items = items[:15]
        suffix = f" 외 {len(items)-15}건" if len(items) > 15 else ""
        lines.append(f"- {ntype}: {', '.join(display_items)}{suffix}")
    return "\n".join(lines)


async def _run_worker(
    task: str,
    tools_by_name: dict[str, Any],
    settings,
    original_query: str = "",
    history_summary: str = "",
) -> list[tuple[str, str]]:
    """
    Mini ReAct agent — 태스크 설명을 받아 필요한 도구를 스스로 선택·실행하고
    [(tool_name, result_str), ...] 목록을 반환합니다.
    """
    llm = get_llm(model=RequestConfig.current().worker_model or settings.rnd_model)
    llm_with_tools = llm.bind_tools(list(tools_by_name.values()))

    system = SystemMessage(content="""당신은 R&D 데이터 수집 워커입니다. 도구 호출만 수행하세요 — 분석·요약·설명은 하지 않습니다.
[태스크]가 최우선입니다. [원본 질문]은 태스크에 키워드가 생략되거나 지시대명사가 있을 때만 보충 참고하세요.
충분한 데이터를 수집했거나 더 이상 조회할 내용이 없으면 종료하세요.""")
    if original_query:
        task_content = f"[원본 질문]\n{original_query}\n\n[태스크]\n{task}"
    else:
        task_content = task
        
    if history_summary:
        task_content += f"\n\n{history_summary}"
        
    messages: list = [system, HumanMessage(content=task_content)]
    collected: list[tuple[str, str]] = []
    seen_calls: set[str] = set()

    try:
        for step in range(_WORKER_MAX_STEPS):
            response = await llm_with_tools.ainvoke(messages)
            messages.append(response)

            if not getattr(response, "tool_calls", None):
                logger.debug("[worker:%s] ✓ 완료 (step=%d, 수집=%d건)", task[:40], step + 1, len(collected))
                break

            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc.get("args", {})
                call_key = f"{tool_name}:{json.dumps(tool_args, sort_keys=True, ensure_ascii=False)}"
                if call_key in seen_calls:
                    logger.debug("[worker:%s] 중복 호출 건너뜀: %s", task[:40], call_key[:80])
                    messages.append(ToolMessage(content="[SKIP] 이미 동일한 호출 결과가 있습니다.", tool_call_id=tc["id"]))
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
                logger.debug(
                    "[worker:%s] step=%d  %s  (%.2fs)\n  args: %s\n  → %s",
                    task[:40], step + 1, tool_name, elapsed,
                    json.dumps(tool_args, ensure_ascii=False, separators=(",", ":")),
                    summarize_tool_result(result_str),
                )
                collected.append((tool_name, result_str))
                messages.append(ToolMessage(content=result_str, tool_call_id=tc["id"]))

        return collected
    except Exception as e:
        logger.error("[worker:%s] 에러 발생: %s", task[:40], e)
        # 에러 발생 시 진행 중이던 collected 내용과 함께 에러 메시지를 반환하여 격리
        collected.append(("worker_error", f"[ERROR] 워커 실행 중 에러 발생 ({type(e).__name__}): {e}"))
        return collected


async def parallel_executor(state: RDAgentState, config: RunnableConfig) -> dict:
    settings = get_settings()
    tools_by_name: dict[str, Any] = config["configurable"]["tools_by_name"]
    pending_tasks: list[str] = state.get("pending_tasks", [])
    executed: list[str] = list(state.get("executed_tasks", []))

    if not pending_tasks:
        return {"pending_tasks": [], "executed_tasks": executed}

    already_run = {_task_key(t) for t in executed}
    fresh_tasks = [t for t in pending_tasks if _task_key(t) not in already_run]

    skipped = len(pending_tasks) - len(fresh_tasks)
    if skipped:
        logger.debug("[parallel_executor] %d개 태스크 중복으로 건너뜀", skipped)

    if not fresh_tasks:
        logger.debug("[parallel_executor] 실행할 새 태스크 없음 — generate로 단락")
        return {"pending_tasks": [], "executed_tasks": executed}

    logger.debug("[parallel_executor] %s\n  워커 %d개 시작: %s",
                 _SEP, len(fresh_tasks), " | ".join(t[:30] for t in fresh_tasks))

    t0_all = time.perf_counter()
    original_query = RequestConfig.current().original_query
    history_summary = _build_history_summary(state.get("tool_results", {}))
    
    worker_results = await asyncio.gather(*[
        _run_worker(t, tools_by_name, settings, original_query, history_summary)
        for t in fresh_tasks
    ])
    total_elapsed = time.perf_counter() - t0_all

    summary = "  " + "\n  ".join(
        f"{t[:25]}: {sum(1 for _ in pairs)}개 툴호출"
        for t, pairs in zip(fresh_tasks, worker_results)
    )
    logger.debug("[parallel_executor] 완료 %.2fs\n%s\n%s", total_elapsed, summary, _SEP)

    current_round = state.get("iteration_count", 0)
    updated: dict[str, list[str]] = dict(state.get("tool_results", {}))
    new_task_results: list[dict] = []
    msg_lines: list[str] = []

    for task, tool_pairs in zip(fresh_tasks, worker_results):
        tool_lines = [f"[{tool_name}]\n{_clean_result(result_str)}" for tool_name, result_str in tool_pairs]
        msg_lines.append(f"# {task[:40]}\n" + "\n\n".join(tool_lines))
        for tool_name, result_str in tool_pairs:
            updated[tool_name] = updated.get(tool_name, []) + [result_str]
        new_task_results.append({
            "round": current_round,
            "task":  task,
            "tools": [{"name": tn, "summary": summarize_tool_result(rs)} for tn, rs in tool_pairs],
        })

    executed.extend(fresh_tasks)

    result_message = AIMessage(
        content="\n\n---\n\n".join(msg_lines),
        name="tool_results",
    )

    return {
        "messages":       [result_message],
        "tool_results":   updated,
        "task_results":   list(state.get("task_results", [])) + new_task_results,
        "pending_tasks":  [],
        "executed_tasks": executed,
    }
