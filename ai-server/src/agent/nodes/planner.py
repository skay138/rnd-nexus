import logging
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from agent.state import RDAgentState
from config import get_settings
from memory.compaction import should_compact, compact_messages

logger = logging.getLogger(__name__)


class PlanOutput(BaseModel):
    reasoning: str = Field(description="사용자 질문의 핵심 의도 분석 및 필요한 데이터 도메인(논문, 특허 등) 추론 과정")
    semantic_entry: Optional[str] = Field(
        default=None,
        description=(
            "시맨틱 벡터 검색 쿼리. "
            "개념 탐색·추천·동향 파악처럼 폭넓은 발견이 필요한 경우 작성. "
            "이전 대화에서 이미 시맨틱 검색을 완료했거나, "
            "정확한 ID/번호 조회·이전 결과의 단순 필터/정렬인 경우 null로 설정."
        ),
    )
    steps: list[str] = Field(description="수행할 단계 목록 (1-4개). semantic_entry 지정 시 그 이후 단계를 작성.")


async def planner(state: RDAgentState) -> dict:
    settings = get_settings()
    llm = ChatOllama(model=settings.rnd_model, base_url=settings.ollama_base_url)
    structured = llm.with_structured_output(PlanOutput)

    feedback = state.get("reflection_feedback", "")
    replan_count = state.get("replan_count", 0)

    system_prompt = """<role>
당신은 국가 R&D 데이터를 다루는 AI 데이터 분석가입니다.
사용자의 질문과 대화 맥락을 바탕으로 어떤 데이터를 우선적으로 검색하고 분석할지 계획을 세우세요.
</role>

<available_domains>
논문, 특허, 국가 R&D 과제, 연구자, 기술
</available_domains>

<search_strategy>
1. 발견(Semantic): 개념 탐색·추천·동향 파악 질문이면 semantic_entry를 작성하세요.
   단, 이전 대화에서 이미 시맨틱 검색을 수행했거나 이전 결과의 단순 필터·정렬이면 semantic_entry는 null로 설정하세요.
2. 확장(Relational): 발견된 핵심 노드를 바탕으로 그래프 네트워크(연구자 협업, 논문 인용)를 탐색합니다.
3. 검증(Exact): 특정 대상의 상세 정보가 필요할 때만 RDB 도구(search_papers, search_patents 등)를 사용합니다.
</search_strategy>"""

    if replan_count > 0 and feedback:
        instruction = f"""<feedback>
{feedback}
</feedback>

<task>
이전 계획으로는 충분한 데이터를 얻지 못했습니다.
'reasoning'에 왜 이전 접근이 실패했는지 분석하고, 대화 맥락을 참고하여 검색 키워드를 확장하거나 다른 도메인을 탐색하는 새로운 전략을 세우세요.
그에 따른 새로운 실행 단계를 'steps' 배열로 반환하세요.
각 단계는 MECE 원칙을 준수하여 중복 없이 다각도로 데이터를 수집하도록 구성하세요.
모든 분석(reasoning)과 실행 단계(steps)는 반드시 한국어로 작성하세요.
</task>

<replan_examples>
<example>
<feedback_given>search_patents가 결과를 반환했으나 발명인 연구자 정보가 없습니다.</feedback_given>
<output>{{"reasoning": "특허 데이터는 확보됐으나 발명인-연구자 연결이 누락됐습니다. semantic_graph_search로 특허→연구자 관계를 추가 탐색합니다.", "semantic_entry": null, "steps": ["semantic_graph_search(entry_type=Patent, hops=[invented_by→Researcher])로 발명인 탐색", "search_researchers로 발명인 상세 프로필 조회"]}}</output>
</example>
<example>
<feedback_given>search_technologies 결과가 0건입니다. 키워드를 확장해야 합니다.</feedback_given>
<output>{{"reasoning": "기술 검색 키워드가 너무 좁았습니다. 상위 개념어로 재시도하고 semantic_search로 보완합니다.", "semantic_entry": "프로세서인메모리 아키텍처 연산", "steps": ["search_technologies(query='PIM OR Processing-In-Memory')로 키워드 확장", "search_papers로 관련 논문 크로스체크"]}}</output>
</example>
</replan_examples>"""
    else:
        instruction = """<task>
먼저 질문의 핵심 의도와 현재까지의 대화 맥락을 파악하고 어떤 도메인의 데이터가 교차 검증되어야 하는지 'reasoning'에 작성하세요.
이후 논리적 흐름에 따라 구체적인 실행 단계를 'steps' 배열로 반환하세요.
각 단계는 MECE 원칙을 준수하여 중복 없이 다각도로 데이터를 수집하도록 구성하세요.
모든 분석(reasoning)과 실행 단계(steps)는 반드시 한국어로 작성하세요.
</task>

<examples>
<example>
<query>AI 반도체 분야 핵심 연구자를 추천해줘</query>
<output>{"reasoning": "연구자 추천 질문입니다. 기술 도메인 시맨틱 탐색 후 연구자 네트워크로 확장하는 전략이 적합합니다.", "semantic_entry": "AI 반도체 저전력 설계 연구", "steps": ["semantic_graph_search로 기술→연구자 관계 탐색", "search_researchers로 전문분야·소속 상세 조회"]}</output>
</example>
<example>
<query>뉴로모픽 컴퓨팅 특허 동향을 알려줘</query>
<output>{"reasoning": "특허 동향 분석 요청입니다. 벡터 검색으로 관련 특허 발굴 후 출원인·연도별 분포를 파악합니다.", "semantic_entry": "뉴로모픽 컴퓨팅 회로 설계", "steps": ["search_patents로 연도별 특허 현황 조회", "search_technologies로 관련 기술 성숙도 확인"]}</output>
</example>
<example>
<query>그 중에서 KAIST 출원 특허만 보여줘</query>
<output>{"reasoning": "이전 결과에 대한 기관 필터링 요청입니다. 이미 시맨틱 검색이 완료되었으므로 RDB 필터 조회만 수행합니다.", "semantic_entry": null, "steps": ["search_patents(assignee='KAIST')로 출원인 필터링"]}</output>
</example>
</examples>"""

    messages = state["messages"]
    approx_tokens = sum(len(str(m.content)) // 4 for m in messages)
    if should_compact(messages, approx_tokens):
        messages = compact_messages(messages, llm)

    messages_to_send = [SystemMessage(content=system_prompt)] + messages + [HumanMessage(content=instruction)]

    try:
        output: PlanOutput = await structured.ainvoke(messages_to_send)
        semantic_entry = (output.semantic_entry or "").strip()
        follow_steps = [s for s in output.steps if s.strip()]
        plan = ([f"[시맨틱 진입] semantic_search('{semantic_entry}')"] if semantic_entry else []) + follow_steps
        reasoning = getattr(output, 'reasoning', 'No reasoning provided')
    except Exception as e:
        logger.error("[planner] Failed to generate plan: %s", e)
        plan = []
        reasoning = "Error occurred"

    if not plan:
        plan = ["관련 논문 검색", "관련 기술 추천"]

    logger.debug("[planner] Reasoning: %s", reasoning)
    logger.debug("[planner] Established plan: %s", plan)
    return {"plan": plan}
