import json
import logging
import time
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from common.llm import get_llm
from common.parsers import collect_relevant_data, strip_think
from common.config.query_config import RequestConfig
from agent.state import RDAgentState
from agent.utils.context import split_turns, previous_turn_context
from config import get_settings
from memory.compaction import apply_compaction

logger = logging.getLogger(__name__)


def _format_collected_data(task_execution_results: list) -> str:
    """task_execution_results → generate 컨텍스트용 데이터 블록.

    선별·dedup 규칙은 collect_relevant_data(출처 생성과 공유)에 위임한다.
    """
    blocks: list[str] = []
    for b in collect_relevant_data(task_execution_results):
        parts = [
            json.dumps(item, ensure_ascii=False) if isinstance(item, list) else item
            for item in b["items"]
        ]
        blocks.append(f"### {b['task_description']}\n" + "\n".join(parts))
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

Answer only what the user asked. If an entity or fact does not directly answer the question, omit it — do not mention it, qualify it, or explain why it was excluded.
</instructions>

<constraints>
- Citation: when a statement is based on an entity from <수집된 데이터>, append that entity's ID marker [#ID] immediately after the statement (e.g. "…를 개발했습니다 [#P002].", multiple: [#P002][#R001]). Use ONLY IDs that appear in <수집된 데이터> — never invent an ID.
- NEVER write raw IDs in prose (e.g. "ID는 RS-2024-...입니다" or "ID: P001"). If you need to identify an entity, use its natural name/title in the text and append the [#ID] citation marker.
- Do not expose internal implementation details such as graph nodes, edge names, retrieval steps, or tool calls in prose. Describe internal concepts naturally in Korean when necessary.
- Do not write any other citation format or a source/reference list section — [#ID] markers are the only citation.
- Do not append generic closing sections such as "참고 사항", "추가 정보", "주의", or "수집 범위 외".
- Do not add or reference anything not directly answering the question — no extra entities, no exclusion explanations, no data-limitation comments. This applies even when [#ID] citations are available.
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
    elapsed = time.perf_counter() - t0

    logger.debug("[GEN] %.2fs  output=%d chars\n  out | %s", elapsed, len(full_content), full_content[:300])
    return {"messages": compaction_msgs + [AIMessage(content=full_content, name="final_answer")]}
