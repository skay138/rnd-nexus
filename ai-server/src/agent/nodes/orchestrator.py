import logging
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from agent.state import RDAgentState
from config import get_settings
from memory.compaction import should_compact, compact_messages

logger = logging.getLogger(__name__)


class ToolTask(BaseModel):
    tool: str  = Field(description="호출할 도구 이름")
    args: dict = Field(description="도구 인수 (JSON 객체)")
    label: str = Field(default="", description="이 태스크가 수집하는 것 (로깅용)")


class OrchestratorPlan(BaseModel):
    reasoning: str            = Field(description="수집 현황 평가 및 다음 전략 (한국어)")
    tasks: list[ToolTask]     = Field(description="병렬 실행할 태스크 목록. 데이터 수집이 완료됐으면 빈 리스트")


_TOOL_REFERENCE = """
<available_tools>
발견 (RAG — 항상 먼저 사용):
  semantic_search(query, node_type?, top_k?)
    - 자연어로 관련 엔티티 탐색. node_type: Researcher|Paper|Patent|Technology|Project (빈값=전체)
    - 반환: [{id, node_type, name, score}, ...]  ← id를 get_entities에 전달

  semantic_graph_search(concept, entry_type?, hops?, top_k?)
    - 벡터 탐색 후 그래프 관계(협업·인용·사용)로 확장
    - 반환: [{id, node_type, name, score}, ...]

상세 조회 (DB — 발견 후 필요 시):
  get_entities(entity_type, ids)
    - semantic_search/graph로 얻은 ID로 DB 상세 정보 조회
    - entity_type: Researcher|Paper|Patent|Technology|Project

관계 탐색 (Neo4j):
  get_researcher_network(researcher_name)  - 연구자 협업·논문·특허 네트워크
  get_citation_graph(paper_title, depth?)  - 논문 인용 그래프
  run_graph_query(cypher)                  - READ 전용 Cypher 직접 실행
</available_tools>

<strategy>
1단계 발견: semantic_search 또는 semantic_graph_search로 관련 ID 획득
2단계 상세: 필요한 엔티티만 get_entities로 상세 조회 (전체 조회 불필요)
3단계 관계: 연구자·논문 관계가 필요할 때 graph 도구 사용
→ 이미 충분한 데이터가 있으면 tasks를 비워 수집 완료를 신호하세요
</strategy>

<follow_up_rule>
후속 질문인지 판단할 때 다음 기준을 따르세요:

tasks=[] (도구 호출 불필요):
  - 이전 답변에 이미 충분한 상세 정보가 있고, 단순 필터·정렬·요약 요청인 경우
  - 예: "그 중 KAIST만 보여줘", "위 목록 h-index 순으로 정렬해줘"

get_entities 호출 필요:
  - 이전 답변이 목록/요약 수준이었고, 특정 항목의 상세 정보를 요청하는 경우
  - 예: "한동현 연구자 자세히 알려줘", "첫 번째 특허 상세 내용 보여줘"
  - 이 경우 대화 히스토리에서 해당 항목의 ID를 찾아 get_entities로 조회하세요

새 RAG 검색 필요:
  - 이전 대화와 완전히 다른 주제이거나 새로운 발견이 필요한 경우
</follow_up_rule>
"""


async def orchestrator(state: RDAgentState, config: RunnableConfig) -> dict:
    settings = get_settings()
    max_iterations: int = config.get("configurable", {}).get("max_replan", 3)
    iteration_count: int = state.get("iteration_count", 0)

    tool_results = state.get("tool_results", {})
    if tool_results:
        lines = [
            f"- {name}: {len([r for r in results if not str(r).startswith('[ERROR]')])}건 수집됨"
            for name, results in tool_results.items()
        ]
        collected = "\n".join(lines)
    else:
        collected = "없음"

    remaining = max_iterations - iteration_count

    system_prompt = f"""당신은 R&D 데이터 수집 오케스트레이터입니다.
사용자 질문에 완전히 답하기 위해 필요한 데이터를 수집 계획을 세우세요.
당신은 도구를 직접 실행하지 않습니다 — tasks 목록을 반환하면 워커가 병렬로 실행합니다.

{_TOOL_REFERENCE}

<수집 현황>
{collected}
</수집 현황>

<제약>
- 남은 수집 라운드: {remaining}회 (0이면 반드시 tasks=[] 반환)
- 이미 수집된 도구와 동일한 쿼리로 중복 실행 금지
- 한 라운드에 병렬 실행 가능한 독립 태스크를 최대한 묶어서 반환하세요
</제약>"""

    messages = list(state["messages"])
    approx_tokens = sum(len(str(m.content)) // 4 for m in messages)
    if should_compact(messages, approx_tokens):
        llm_plain = ChatOllama(model=settings.rnd_model, base_url=settings.ollama_base_url)
        messages = compact_messages(messages, llm_plain)

    llm = ChatOllama(model=settings.rnd_model, base_url=settings.ollama_base_url)
    structured = llm.with_structured_output(OrchestratorPlan)

    messages_to_send = [SystemMessage(content=system_prompt)] + messages
    try:
        plan: OrchestratorPlan = await structured.ainvoke(messages_to_send)
        tasks = [t.model_dump() for t in plan.tasks]
        reasoning = plan.reasoning
    except Exception as e:
        logger.error("[orchestrator] structured output 실패: %s", e)
        tasks = []
        reasoning = "오류로 인해 수집 종료"

    logger.debug("[orchestrator] iter=%d/%d tasks=%d reasoning=%s",
                 iteration_count + 1, max_iterations, len(tasks), reasoning[:120])

    return {
        "messages":       [AIMessage(content=reasoning, name="orchestrator")],
        "pending_tasks":  tasks,
        "iteration_count": iteration_count + 1,
    }
