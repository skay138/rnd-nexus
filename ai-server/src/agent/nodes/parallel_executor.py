import ast
import asyncio
import json
import logging
import time
from typing import Any
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from agent.state import RDAgentState
from config import get_settings

logger = logging.getLogger(__name__)

_WORKER_MAX_STEPS = 5
_SEP = "─" * 50


def _task_key(task: str) -> str:
    return task.strip().lower()


def _fmt_result(result_str: str) -> str:
    """결과 문자열을 한 줄 요약으로."""
    if result_str.startswith("[ERROR]"):
        return result_str[:120]
    try:
        items = json.loads(result_str)
    except Exception:
        try:
            items = ast.literal_eval(result_str)
        except Exception:
            return f"{len(result_str)}c"
    if not isinstance(items, list):
        return f"{len(result_str)}c"
    texts = [i.get("text", "") for i in items if isinstance(i, dict) and i.get("type") == "text"]
    if not texts:
        return f"{len(items)}건"
    previews = []
    for text in texts[:3]:
        try:
            d = json.loads(text)
            entries = d if isinstance(d, list) else [d]
            for e in entries[:1]:
                name = str(e.get("name") or e.get("title") or e.get("id") or "")[:30]
                if name:
                    previews.append(name)
        except Exception:
            pass
    return f"{len(texts)}건" + (f": {', '.join(previews)}" if previews else "")


async def _run_worker(
    task: str,
    tools_by_name: dict[str, Any],
    settings,
) -> list[tuple[str, str]]:
    """
    Mini ReAct agent — 태스크 설명을 받아 필요한 도구를 스스로 선택·실행하고
    [(tool_name, result_str), ...] 목록을 반환합니다.
    """
    llm = ChatOllama(model=settings.rnd_model, base_url=settings.ollama_base_url)
    llm_with_tools = llm.bind_tools(list(tools_by_name.values()))

    system = SystemMessage(content="""<language>한국어</language>

당신은 R&D 데이터 수집 워커입니다. 주어진 태스크를 완료하기 위해 도구를 사용하세요.
도구 결과를 분석하고 추가 정보가 필요하면 계속 호출하세요. 충분한 데이터를 수집했으면 멈추세요.
답변은 반드시 도구 결과에 근거해서 작성하세요.""")
    messages: list = [system, HumanMessage(content=task)]
    collected: list[tuple[str, str]] = []

    for step in range(_WORKER_MAX_STEPS):
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)

        if not getattr(response, "tool_calls", None):
            logger.debug("[worker:%s] ✓ 완료 (step=%d, 수집=%d건)", task[:40], step + 1, len(collected))
            break

        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc.get("args", {})
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
                _fmt_result(result_str),
            )
            collected.append((tool_name, result_str))
            messages.append(ToolMessage(content=result_str, tool_call_id=tc["id"]))

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
        return {"pending_tasks": [], "executed_tasks": executed, "no_new_data": True}

    logger.debug("[parallel_executor] %s\n  워커 %d개 시작: %s",
                 _SEP, len(fresh_tasks), " | ".join(t[:30] for t in fresh_tasks))

    t0_all = time.perf_counter()
    worker_results = await asyncio.gather(*[
        _run_worker(t, tools_by_name, settings)
        for t in fresh_tasks
    ])
    total_elapsed = time.perf_counter() - t0_all

    summary = "  " + "\n  ".join(
        f"{t[:25]}: {sum(1 for _ in pairs)}개 툴호출"
        for t, pairs in zip(fresh_tasks, worker_results)
    )
    logger.debug("[parallel_executor] 완료 %.2fs\n%s\n%s", total_elapsed, summary, _SEP)

    updated: dict[str, list[str]] = dict(state.get("tool_results", {}))
    msg_lines: list[str] = []

    for task, tool_pairs in zip(fresh_tasks, worker_results):
        tool_lines = [f"[{tool_name}]\n{result_str}" for tool_name, result_str in tool_pairs]
        msg_lines.append(f"# {task[:40]}\n" + "\n\n".join(tool_lines))
        for tool_name, result_str in tool_pairs:
            updated[tool_name] = updated.get(tool_name, []) + [result_str]

    executed.extend(fresh_tasks)

    result_message = AIMessage(
        content="\n\n---\n\n".join(msg_lines),
        name="tool_results",
    )

    return {
        "messages":       [result_message],
        "tool_results":   updated,
        "pending_tasks":  [],
        "executed_tasks": executed,
        "no_new_data":    False,
    }
