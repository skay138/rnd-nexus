import ast as _ast
import json as _json
import logging
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from agent.state import RDAgentState
from config import get_settings
from memory.compaction import should_compact, compact_messages

logger = logging.getLogger(__name__)


def _dedup_tool_results(tool_results: dict) -> dict:
    """같은 도구의 여러 호출 결과를 엔티티 ID 기준으로 중복 제거."""
    deduped: dict = {}
    for name, results in tool_results.items():
        seen_ids: set = set()
        unique_texts: list = []
        for result_str in results:
            s = str(result_str)
            if s.startswith("[ERROR]"):
                continue
            try:
                raw = _ast.literal_eval(s)
                if not isinstance(raw, list):
                    unique_texts.append(s[:800])
                    continue
                new_items = []
                for item in raw:
                    if not isinstance(item, dict) or item.get("type") != "text":
                        continue
                    text = item.get("text", "")
                    if not text:
                        continue
                    try:
                        entries = _json.loads(text)
                        if not isinstance(entries, list):
                            entries = [entries]
                        for entry in entries:
                            if not isinstance(entry, dict):
                                continue
                            eid = (
                                entry.get("paper_id") or entry.get("patent_id") or
                                entry.get("researcher_id") or entry.get("technology_id") or
                                entry.get("project_id") or
                                str(entry.get("id") or entry.get("entity_id") or "")
                            )
                            if eid and eid not in seen_ids:
                                seen_ids.add(eid)
                                new_items.append(entry)
                    except Exception:
                        continue
                if new_items:
                    unique_texts.append(_json.dumps(new_items, ensure_ascii=False))
                elif not new_items:
                    unique_texts.append(s[:800])
            except Exception:
                unique_texts.append(s[:800])
        if unique_texts:
            deduped[name] = unique_texts
    return deduped


async def generate(state: RDAgentState, config: RunnableConfig) -> dict:
    settings = get_settings()
    model = config.get("configurable", {}).get("generate_model", settings.rnd_model_generate)
    llm = ChatOllama(model=model, base_url=settings.ollama_base_url, streaming=True)

    # 도구 결과 취합 — ERROR 제외, 엔티티 ID 기준 중복 제거
    raw_results = state.get("tool_results", {})
    deduped = _dedup_tool_results(raw_results)
    data_sections = []
    for name, texts in deduped.items():
        data_sections.append(f"[{name}]\n" + "\n---\n".join(texts))
    data_block = "\n\n".join(data_sections) if data_sections else "(수집된 데이터 없음)"

    all_messages = state["messages"]
    approx_tokens = sum(len(str(m.content)) // 4 for m in all_messages)
    if should_compact(all_messages, approx_tokens):
        all_messages = compact_messages(all_messages, llm)

    # ToolMessage·중간 AIMessage 제외: 사용자 질문과 이전 최종 답변만 유지
    clean_messages = [
        m for m in all_messages
        if getattr(m, "type", None) == "human"
        or (getattr(m, "type", None) == "ai" and getattr(m, "name", None) == "final_answer")
    ]

    system_prompt = f"""당신은 반드시 한국어로만 답변해야 합니다.

<role>
당신은 R&D 전문 AI 어시스턴트입니다.
수집된 데이터를 바탕으로 사용자 질문에 직접 답하세요.
</role>

<collected_data>
{data_block}
</collected_data>

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

    messages_to_send = [SystemMessage(content=system_prompt)] + clean_messages
    response = await llm.ainvoke(messages_to_send)

    logger.debug("[generate] Final Response:\n%s", response.content)
    return {
        "messages": [AIMessage(content=response.content, name="final_answer")],
    }
