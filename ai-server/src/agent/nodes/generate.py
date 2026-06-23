import logging
import time
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from agent.state import RDAgentState
from config import get_settings
from memory.compaction import should_compact, compact_messages

logger = logging.getLogger(__name__)


async def generate(state: RDAgentState, config: RunnableConfig) -> dict:
    settings = get_settings()
    model = config.get("configurable", {}).get("generate_model", settings.rnd_model_generate)
    llm = ChatOllama(model=model, base_url=settings.ollama_base_url, streaming=True)

    messages = list(state["messages"])
    approx_tokens = sum(len(str(m.content)) // 4 for m in messages)
    if should_compact(messages, approx_tokens):
        messages = compact_messages(messages, llm)


    # 사용자 질문 + 도구 결과 + 이전 최종 답변만 전달 (오케스트레이터 계획 메시지 제외)
    relevant = [
        m for m in messages
        if getattr(m, "type", None) == "human"
        or getattr(m, "name", None) in ("tool_results", "final_answer")
    ]

    system_prompt = """<language>Korean</language>

<role>
당신은 R&D 전문 AI 어시스턴트입니다. 답변은 한국어로 작성하세요.
[tool_results] 메시지에 수집된 데이터가 있습니다. 이 데이터를 바탕으로 사용자 질문에 직접 답하세요.
데이터가 충분하지 않더라도 수집된 내용 내에서 최선의 답변을 작성하세요.
</role>

<format_guide>
질문 유형별 답변 형식:
- 연구자·기술 추천: 이름/TRL → 소속/분야 → 추천 근거 순으로 번호 목록 작성
- 목록 조회: 번호 매긴 목록, 각 항목에 핵심 속성(기관, 연도, 분야 등) 포함
- 동향·분석: 핵심 인사이트 먼저, 뒷받침 데이터 후술
- 상세 조회: 항목별 구조화 출력 (이름, 소속, 전문분야, 관련 논문/특허 등)
데이터가 없는 항목은 "정보 없음"으로 명시하세요.
출처 표기나 참고문헌 목록은 작성하지 마세요 — 시스템이 자동으로 추가합니다.
</format_guide>

<examples>
<example type="researcher_recommendation">
<query>AI 반도체 분야 핵심 연구자를 추천해줘</query>
<answer>
AI 반도체 분야 핵심 연구자 추천 결과입니다.

1. **홍길동** (KAIST 전기전자공학과)
   - 전문분야: 저전력 뉴로모픽 칩 설계, PIM 아키텍처
   - 추천 근거: 관련 논문 15편, 특허 7건 보유. 국가과제 주관 2건 수행 중.

2. **김영희** (서울대 컴퓨터공학과)
   - 전문분야: AI 가속기 컴파일러, 메모리 계층 최적화
   - 추천 근거: 인용 지수 상위 10%, 주요 기업과 공동연구 다수.
</answer>
</example>
<example type="trend_analysis">
<query>뉴로모픽 컴퓨팅 특허 동향을 알려줘</query>
<answer>
뉴로모픽 컴퓨팅 특허 동향 분석 결과입니다.

**핵심 인사이트**: 2021년 이후 국내 출원이 연평균 23% 증가하며 급성장 중.

- 주요 출원인: 삼성전자(34건), ETRI(21건), KAIST(15건)
- 기술 분류: 스파이킹 신경망 회로(42%), 메모리 소자(31%), 학습 알고리즘(27%)
- 주목 동향: 엣지 디바이스용 저전력 설계 특허가 2023년부터 급증.
</answer>
</example>
</examples>"""

    logger.debug(
        "[generate] relevant_messages=%d\n%s",
        len(relevant),
        "\n".join(
            f"  [{getattr(m, 'name', None) or getattr(m, 'type', '?')}] "
            f"{str(m.content)[:300]}{'…' if len(str(m.content)) > 300 else ''}"
            for m in relevant
        ),
    )

    t0 = time.perf_counter()
    parts: list[str] = []
    async for chunk in llm.astream([SystemMessage(content=system_prompt)] + relevant, config):
        if chunk.content:
            parts.append(chunk.content)
    elapsed = time.perf_counter() - t0
    full_content = "".join(parts)

    logger.debug("[generate] elapsed=%.2fs content_len=%d\n%s", elapsed, len(full_content), full_content[:500])
    return {"messages": [AIMessage(content=full_content, name="final_answer")]}
