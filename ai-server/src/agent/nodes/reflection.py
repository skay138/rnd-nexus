import ast as _ast
import json as _json
import logging
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from agent.state import RDAgentState
from config import get_settings

logger = logging.getLogger(__name__)


def _get_entity_id(entry: dict) -> str:
    return (
        entry.get("paper_id") or entry.get("patent_id") or
        entry.get("researcher_id") or entry.get("technology_id") or
        entry.get("project_id") or
        str(entry.get("id") or entry.get("entity_id") or "")
    )


_CACHE_MARKER = "[데이터가 길어 캐시되었습니다]"
_REDIS_KEY_PREFIX = "* Redis Key: "


def _extract_redis_key(s: str) -> str:
    for line in s.splitlines():
        if line.startswith(_REDIS_KEY_PREFIX):
            return line[len(_REDIS_KEY_PREFIX):].strip()
    return ""


def _filter_items(s: str, discard_ids: set) -> str:
    """MCP 응답 문자열에서 discard_ids에 해당하는 엔티티를 제거 후 재직렬화."""
    try:
        items = _ast.literal_eval(s)
        if not isinstance(items, list):
            return s
        filtered: list = []
        for item in items:
            if not isinstance(item, dict) or item.get("type") != "text":
                filtered.append(item)
                continue
            try:
                data = _json.loads(item.get("text", ""))
                entries = data if isinstance(data, list) else [data]
                kept = [e for e in entries if isinstance(e, dict) and _get_entity_id(e) not in discard_ids]
                if kept:
                    new_item = dict(item)
                    new_item["text"] = _json.dumps(kept if isinstance(data, list) else kept[0], ensure_ascii=False)
                    filtered.append(new_item)
            except Exception:
                filtered.append(item)
        return str(filtered) if filtered else ""
    except Exception:
        return s


async def _prune_by_ids(tool_results: dict, discard_ids: set) -> dict:
    if not discard_ids:
        return dict(tool_results)

    from memory.tool_cache import get_tool_result, delete_tool_result, save_tool_result

    pruned: dict = {}
    for tool_name, results in tool_results.items():
        new_results: list = []
        for result_str in results:
            s = str(result_str)
            if s.startswith("[ERROR]"):
                new_results.append(result_str)
                continue

            if _CACHE_MARKER in s:
                redis_key = _extract_redis_key(s)
                if not redis_key:
                    new_results.append(result_str)
                    
                    continue
                full = await get_tool_result(redis_key)
                filtered = _filter_items(full, discard_ids)
                await delete_tool_result(redis_key)
                if filtered and filtered != "[]":
                    new_key = await save_tool_result(filtered)
                    if new_key:
                        preview = filtered[:2500]
                        new_results.append(
                            f"{_CACHE_MARKER}\n{_REDIS_KEY_PREFIX}{new_key}\n* 미리보기:\n{preview}..."
                        )
            else:
                filtered = _filter_items(s, discard_ids)
                if filtered and filtered != "[]":
                    new_results.append(filtered)

        if new_results:
            pruned[tool_name] = new_results
    return pruned


class ReflectionOutput(BaseModel):
    result: Literal["sufficient", "insufficient"]
    feedback: str = Field(default="", description="insufficient인 경우 부족한 데이터 설명")
    discard_ids: list[str] = Field(default_factory=list, description="사용자 질문과 무관한 엔티티 ID 목록")


async def reflection(state: RDAgentState) -> dict:
    settings = get_settings()
    max_replan = settings.rnd_max_replan

    if state.get("replan_count", 0) >= max_replan:
        return {
            "reflection_result": "sufficient",
            "reflection_feedback": f"최대 Replan 횟수({max_replan}) 도달. 현재 데이터로 진행합니다.",
        }

    tool_results = state.get("tool_results", {})
    plan = state.get("plan", [])

    data_sections = []
    for name, results in tool_results.items():
        if not results:
            continue
        parts = [str(r)[:600] for r in results]
        data_sections.append(f"[{name}]\n" + "\n---\n".join(parts))
    data_block = "\n\n".join(data_sections) if data_sections else "(수집된 데이터 없음)"

    plan_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan))

    llm = ChatOllama(model=settings.rnd_model, base_url=settings.ollama_base_url)
    structured = llm.with_structured_output(ReflectionOutput)

    from memory.compaction import should_compact, compact_messages

    messages = state["messages"]
    approx_tokens = sum(len(str(m.content)) // 4 for m in messages)
    if should_compact(messages, approx_tokens):
        messages = compact_messages(messages, llm)

    system_prompt = f"""<role>
당신은 R&D Nexus의 품질 검증 AI(리뷰어)입니다.
대화 맥락과 수집된 데이터를 바탕으로, 최종 답변을 작성하기에 데이터의 '질(Quality)'이 충분한지 평가하세요.
</role>

<executed_plan>
{plan_text}
</executed_plan>

<collected_data>
{data_block}
</collected_data>

<evaluation_criteria>
1. 계획된 도구들이 실행되었는가?
2. 단순히 건수만 많은 것이 아니라, 내용 자체가 사용자의 대화 맥락과 질문 의도에 부합하는 유효한 결과인가?
3. 최종 답변을 작성하기 위해 더 찾거나 교차 검증해야 할 정보(다른 도메인 등)는 없는가?
4. 수집된 결과 중 질문과 명백히 무관한 엔티티가 있으면 discard_ids에 해당 ID를 포함하세요.
</evaluation_criteria>"""

    evaluation_request = """<task>
위 계획과 수집된 데이터를 바탕으로 최종 답변 작성에 충분한지 평가하세요.
충분하면 result를 "sufficient"로, 부족하면 "insufficient"로 설정하고 구체적인 피드백을 작성하세요.
</task>

<examples>
<example>
<context>질문: AI 반도체 연구자 추천 / 수집 데이터: 연구자 5명(소속·전문분야 포함), 기술 3개, 논문 10편</context>
<output>{"result": "sufficient", "feedback": ""}</output>
</example>
<example>
<context>질문: 뉴로모픽 특허 동향 / 수집 데이터: semantic_search 결과만 있고 search_patents 미실행</context>
<output>{"result": "insufficient", "feedback": "search_patents 도구가 실행되지 않았습니다. 연도별 특허 출원 현황 데이터가 필요합니다."}</output>
</example>
<example>
<context>질문: PIM 기술 국가과제 현황 / 수집 데이터: 기술 2건, 과제 검색 결과 0건(오류)</context>
<output>{"result": "insufficient", "feedback": "과제 검색이 실패했습니다. search_projects로 'Processing-In-Memory'로 키워드를 확장하여 재시도하세요."}</output>
</example>
</examples>"""

    try:
        output: ReflectionOutput = await structured.ainvoke(
            [SystemMessage(content=system_prompt)] + messages + [HumanMessage(content=evaluation_request)]
        )
        result = output.result
        feedback = output.feedback
        discard_ids = set(output.discard_ids)
    except Exception:
        result = "sufficient"
        feedback = "구조화 출력 실패. 현재 데이터로 진행합니다."
        discard_ids = set()

    logger.debug("[reflection] Evaluation result: %s, feedback: %s, discard_ids: %s", result, feedback, discard_ids)

    updated_tool_results = await _prune_by_ids(state.get("tool_results", {}), discard_ids)

    return {
        "reflection_result": result,
        "reflection_feedback": feedback,
        "replan_count": state.get("replan_count", 0) + (1 if result == "insufficient" else 0),
        "tool_results": updated_tool_results,
    }
