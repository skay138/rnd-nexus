"""하네스 레벨 데이터 완전성 보강.

워커 LLM의 판단 편차(상세 조회 생략)를 코드가 결정적으로 보정하는 두 장치:
- auto_join_details:        검색 결과 행에 상세 필드를 즉시 조인 (워커 루프 안, 기본 경로)
- enrich_selected_entities: 선별됐지만 상세가 없는 ID를 사후 일괄 조회 (라운드 끝, 안전망)

둘 다 LLM 호출 없이 get_entities 도구만 사용한다 (DB ms급).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from agent.state import TaskExecutionResult, ToolCallRecord
from common.parsers import (
    clean_tool_result,
    entity_ids,
    extract_tool_error,
    iter_entities,
    summarize_tool_result,
)

logger = logging.getLogger(__name__)

ENRICH_MAX = 20   # 사후 보강 상한 (초과 시 score 상위만)
JOIN_MAX   = 10   # 검색 결과 상위 N건만 상세 자동 조인

DETAIL_TOOLS = ("get_entities", "filter_entities")
SEARCH_TOOLS = ("semantic_search", "semantic_graph_search")

# 검색 행이 원래 갖는 키 — 이 외의 키가 있으면 상세가 조인된 행으로 간주
SEARCH_ROW_KEYS = {"id", "entity_id", "node_type", "name", "score", "path", "text"}

ID_KEY_TO_TYPE = {
    "paper_id":      "Paper",
    "patent_id":     "Patent",
    "researcher_id": "Researcher",
    "tech_id":       "Technology",
    "technology_id": "Technology",
    "project_id":    "Project",
    "org_id":        "Organization",
}


async def auto_join_details(result_text: str, tools_by_name: dict[str, Any]) -> str:
    """검색 결과의 얕은 행(id·name·score)에 상세 필드를 즉시 조인한다.

    워커가 get_entities 호출을 생략해도 ReAct 루프 안에서 전체 필드를 보고
    선별·추가 탐색을 판단할 수 있게 한다.
    """
    tool = tools_by_name.get("get_entities")
    if tool is None:
        return result_text
    rows = list(iter_entities(result_text))
    if not rows:
        return result_text

    by_type: dict[str, list[str]] = {}
    for r in rows[:JOIN_MAX]:
        nt  = r.get("node_type")
        rid = str(r.get("id") or r.get("entity_id") or "")
        if nt and rid:
            by_type.setdefault(str(nt), []).append(rid)
    if not by_type:
        return result_text

    details: dict[str, dict] = {}
    for etype, ids in by_type.items():
        t0 = time.perf_counter()
        try:
            joined_str = clean_tool_result(
                str(await tool.ainvoke({"entity_type": etype, "ids": ids}))
            )
        except Exception as e:
            logger.debug("[EXEC] 자동 조인 실패 (%s): %s", etype, e)
            continue
        if extract_tool_error(joined_str):
            continue
        
        elapsed = time.perf_counter() - t0
        logger.debug("[EXEC] 자동 조인  %s %d건  %.2fs", etype, len(ids), elapsed)

        for d in iter_entities(joined_str):
            for i in entity_ids(d):
                details[i] = d

    if not details:
        return result_text

    for r in rows:
        d = details.get(str(r.get("id") or r.get("entity_id") or ""))
        if d:
            for k, v in d.items():
                if v and not r.get(k):
                    r[k] = v
    return json.dumps(rows, ensure_ascii=False)


def detail_fetched_ids(results: list[dict]) -> set[str]:
    """이번 턴에서 전체 필드가 이미 수집된 ID 집합.

    상세 조회 도구 결과 + 검색 결과 중 자동 조인으로 상세 필드를 보유한 행 포함.
    """
    fetched: set[str] = set()
    for r in results:
        for tc in r.get("tool_calls", []):
            if tc.get("is_error"):
                continue
            tname = tc.get("tool_name")
            if tname in DETAIL_TOOLS:
                for e in iter_entities(tc.get("result_text", "")):
                    fetched.update(entity_ids(e))
            elif tname in SEARCH_TOOLS:
                for e in iter_entities(tc.get("result_text", "")):
                    if set(e) - SEARCH_ROW_KEYS:   # 상세가 조인된 행
                        fetched.update(entity_ids(e))
    return fetched


def entity_score_map(results: list[dict]) -> dict[str, float]:
    """수집된 엔티티의 ID → 최고 검색 score 매핑 (score 필드가 있는 결과만)."""
    scores: dict[str, float] = {}
    for r in results:
        for tc in r.get("tool_calls", []):
            if tc.get("is_error"):
                continue
            for e in iter_entities(tc.get("result_text", "")):
                s = e.get("score")
                if isinstance(s, (int, float)):
                    for i in entity_ids(e):
                        if i not in scores or s > scores[i]:
                            scores[i] = float(s)
    return scores


def entity_type_map(results: list[dict]) -> dict[str, str]:
    """수집된 엔티티로부터 ID → entity_type 매핑 구성 (node_type 필드 + 도메인 ID 키)."""
    mapping: dict[str, str] = {}
    for r in results:
        for tc in r.get("tool_calls", []):
            if tc.get("is_error"):
                continue
            for e in iter_entities(tc.get("result_text", "")):
                nt = e.get("node_type")
                if nt:
                    for i in entity_ids(e):
                        mapping.setdefault(i, str(nt))
                for key, etype in ID_KEY_TO_TYPE.items():
                    if e.get(key):
                        mapping.setdefault(str(e[key]), etype)
    return mapping


async def enrich_selected_entities(
    new_results: list[TaskExecutionResult],
    all_results: list[dict],
    tools_by_name: dict[str, Any],
    current_round: int,
    sse_queue: asyncio.Queue | None = None,
) -> TaskExecutionResult | None:
    """워커가 선별(relevant_ids)했지만 상세 조회를 생략한 ID를 사후 일괄 보강.

    자동 조인이 못 덮는 구멍(그래프 유래 ID, 조인 상한 밖 행, 조인 실패)의 안전망.
    보강할 것이 없으면 DB 호출 없이 None을 반환한다.
    """
    tool = tools_by_name.get("get_entities")
    if tool is None:
        return None

    selected: list[str] = []
    for r in new_results:
        for i in r.get("selected_ids", []):
            if i not in selected:
                selected.append(i)
    if not selected:
        return None

    fetched  = detail_fetched_ids(all_results)
    type_map = entity_type_map(all_results)
    missing  = [i for i in selected if i not in fetched and i in type_map]
    if not missing:
        return None
    if len(missing) > ENRICH_MAX:
        # score 높은 순으로 상위만 보강 — 점수 미상 ID는 워커 선별 순서를 유지한 채 뒤로 (stable sort)
        score_map = entity_score_map(all_results)
        missing.sort(key=lambda i: score_map.get(i, -1.0), reverse=True)
        logger.warning("[EXEC] 상세 보강 대상 %d건 → score 상위 %d건만 보강", len(missing), ENRICH_MAX)
        missing = missing[:ENRICH_MAX]

    by_type: dict[str, list[str]] = {}
    for i in missing:
        by_type.setdefault(type_map[i], []).append(i)

    tool_calls: list[ToolCallRecord] = []
    for etype, ids in by_type.items():
        args = {"entity_type": etype, "ids": ids}
        t0 = time.perf_counter()
        try:
            result_str = str(await tool.ainvoke(args))
        except Exception as e:
            result_str = f"[ERROR] get_entities 실패: {type(e).__name__}: {e}"
        elapsed = time.perf_counter() - t0
        tool_err = extract_tool_error(result_str)
        if tool_err:
            result_str = f"[ERROR] get_entities: {tool_err}"
        summary = summarize_tool_result(result_str)
        tool_calls.append({
            "tool_name":   "get_entities",
            "args":        args,
            "result_text": clean_tool_result(result_str),
            "summary":     summary,
            "is_error":    result_str.startswith("[ERROR]"),
        })
        logger.debug("[EXEC] 상세 보강  %s %d건  %.2fs  → %s", etype, len(ids), elapsed, summary)

    result: TaskExecutionResult = {
        "task_id":          f"enrich-r{current_round}",
        "task_description": "선별 엔티티 상세 정보 보강",
        "round":            current_round,
        "status":           "completed",
        "tool_calls":       tool_calls,
        "worker_note":      "",
        "selected_ids":     missing,
        "selection_valid":  True,
    }
    if sse_queue is not None:
        await sse_queue.put(("worker_result", {
            "type":    "task_result",
            "task_id": result["task_id"],
            "round":   current_round,
            "task":    result["task_description"],
            "tools":   [{"name": tc["tool_name"], "summary": tc["summary"]} for tc in tool_calls],
        }))
    return result
