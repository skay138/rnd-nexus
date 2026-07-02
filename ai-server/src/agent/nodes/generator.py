import json
import logging
import time
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from common.llm import get_llm
from common.parsers import entity_ids, iter_entities, strip_think
from common.config.query_config import RequestConfig
from agent.state import RDAgentState
from agent.utils.context import split_turns, previous_turn_context
from config import get_settings
from memory.compaction import apply_compaction

logger = logging.getLogger(__name__)


def _format_collected_data(task_execution_results: list) -> str:
    """task_execution_results → generate 컨텍스트용 데이터 블록.

    - 엔티티 리스트 결과: 전역 entity-ID 단위 dedup (워커 간 중복 수집 제거).
      워커가 선별한 selected_ids가 있으면 해당 엔티티만 포함하되,
      선별 ID가 실제 수집 ID와 전혀 매칭되지 않으면(환각 ID) 선별을 무시하고 전문 사용.
    - 비엔티티 결과(그래프 네트워크 dict 등): 문자열 동일성 dedup으로 그대로 포함.
    """
    blocks: list[str] = []
    seen_ids: set[str] = set()
    seen_text: set[str] = set()

    for r in task_execution_results:
        calls = [
            (tc, list(iter_entities(tc.get("result_text", ""))))
            for tc in r.get("tool_calls", [])
            if not tc.get("is_error") and tc.get("result_text")
        ]

        sel = {str(i) for i in r.get("selected_ids", []) if i}
        if sel:
            present = {i for _, ents in calls for e in ents for i in entity_ids(e)}
            if not (sel & present):
                sel = set()

        parts: list[str] = []
        for tc, entities in calls:
            if entities:
                kept: list[dict] = []
                for e in entities:
                    ids = entity_ids(e)
                    if sel and ids and not (set(ids) & sel):
                        continue
                    keys = ids or [json.dumps(e, sort_keys=True, ensure_ascii=False)]
                    if any(k in seen_ids for k in keys):
                        continue
                    seen_ids.update(keys)
                    kept.append(e)
                if kept:
                    parts.append(json.dumps(kept, ensure_ascii=False))
            else:
                text = tc["result_text"]
                if text in seen_text:
                    continue
                seen_text.add(text)
                parts.append(text)

        if parts:
            blocks.append(f"### {r.get('task_description', '')}\n" + "\n".join(parts))
    return "\n\n".join(blocks)


async def generate(state: RDAgentState, config: RunnableConfig) -> dict:
    if state.get("out_of_scope"):
        logger.debug("[generate] out_of_scope — 안내 반환")
        return {"messages": [AIMessage(
            content="죄송합니다. 해당 질문은 R&D 서비스의 지원 범위를 벗어납니다.\n논문·특허·연구자·기술·R&D 과제에 관한 질문을 입력해 주세요.",
            name="final_answer",
        )]}

    settings = get_settings()
    model = RequestConfig.current().generate_model or settings.rnd_model
    llm = get_llm(model=model, streaming=True)

    messages, compaction_msgs = await apply_compaction(
        list(state["messages"]),
        get_llm(model=RequestConfig.current().compact_model or settings.rnd_model),
    )

    # 턴 경계 분리: 이전 턴은 질문·최종답변만 유지, 현재 턴 데이터는
    # task_execution_results에서 HumanMessage로 구성 (Human→AI 교차 구조 보장)
    prev_turns, current_turn = split_turns(messages)
    history = previous_turn_context(prev_turns)

    current_humans = [m for m in current_turn if isinstance(m, HumanMessage)]
    if current_humans:
        *lead, last_human = current_humans
    else:
        lead, last_human = [], HumanMessage(content=RequestConfig.current().original_query)

    data_block = _format_collected_data(state.get("task_execution_results", []))
    if data_block:
        last_human = HumanMessage(
            content=f"<수집된 데이터>\n{data_block}\n</수집된 데이터>\n\n[질문]\n{last_human.content}"
        )
    relevant = history + lead + [last_human]

    system_prompt = """<role>
You are an R&D AI assistant. Answer in Korean.
</role>

<instructions>
Answer the user's question using only the data inside <수집된 데이터> and, for follow-up questions, facts stated in your own previous answers in this conversation.

Never introduce any people, organizations, projects, papers, numbers, or facts that are not present in the provided data.

If there is no <수집된 데이터> block and the question asks for a definition or explanation of a general R&D term or concept, give a brief factual explanation from general knowledge.
If the provided data does not contain information that answers the user's question, respond with "관련 정보를 찾을 수 없습니다."
Otherwise, answer using only the available information without filling in missing parts.

For questions about relationships (participating projects, affiliations, collaborations, papers, patents, etc.), use only relationships explicitly supported by the provided data. Do not infer new relationships.

Relevance:
- Include only entities that directly answer the user's question based on the provided data.
- Exclude entities that are only tangentially or broadly related.

When combining information from multiple tool results:
- Combine facts only when the resulting statement is fully supported by the provided data.
- Do not introduce new conclusions, relationships, or assumptions.

Answer only what the user asked.
Do not add background information, related topics, or additional entities.
</instructions>

<constraints>
- Do not expose internal implementation details such as graph nodes, edge names, retrieval steps, tool calls, or internal IDs. Describe internal concepts naturally in Korean when necessary.
- Do not include citations, references, or source lists.
- Do not append generic closing sections such as "참고 사항", "추가 정보", "주의", or "수집 범위 외".
- Do not add generic concluding sentences that summarize the field or introduce additional entities (e.g. "이 외에도 X, Y, Z가 관련 분야에 참여하고 있습니다" is forbidden).
- Do not explain what was excluded or why. Simply omit irrelevant results without comment. Sentences like "기타 기관(A, B, C 등)은 관련성이 없습니다" or "A, B, C는 다른 분야에 집중하고 있습니다" are forbidden.
- Do not mention data limitations, missing fields, or what the data does not contain. If information is unavailable, omit it silently — do not write "제공된 데이터에 따르면 ~가 명시되어 있지 않습니다" or similar.
</constraints>
"""

    logger.debug(
        "[GEN] context=%d msgs (history=%d)  data=%d chars",
        len(relevant), len(history), len(data_block),
    )

    t0 = time.perf_counter()
    chunks: list[str] = []
    async for chunk in llm.astream([SystemMessage(content=system_prompt)] + relevant, config):
        chunks.append(chunk.content if isinstance(chunk.content, str) else "")
    full_content = strip_think("".join(chunks))
    if not full_content:
        full_content = "관련 정보를 찾을 수 없습니다."
    elapsed = time.perf_counter() - t0

    logger.debug("[GEN] %.2fs  output=%d chars\n  out | %s", elapsed, len(full_content), full_content[:300])
    return {"messages": compaction_msgs + [AIMessage(content=full_content, name="final_answer")]}
