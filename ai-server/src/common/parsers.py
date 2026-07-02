import ast
import json
import re
from typing import Any

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """추론 모델의 <think> 블록 제거.

    닫히지 않은 <think>(토큰 한도로 잘린 응답)는 태그 이후 전체를 제거한다.
    """
    text = _THINK_RE.sub("", text)
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text.strip()


_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def strip_code_fence(s: str) -> str:
    """```json ... ``` 래핑 제거."""
    s = s.strip()
    m = _CODE_FENCE_RE.match(s)
    return m.group(1).strip() if m else s


_ENTITY_ID_KEYS = (
    "id", "entity_id", "paper_id", "patent_id", "researcher_id",
    "tech_id", "technology_id", "project_id", "org_id",
)


def entity_ids(d: dict) -> list[str]:
    """entity dict에서 식별 가능한 모든 ID 값을 추출."""
    return [str(d[k]) for k in _ENTITY_ID_KEYS if d.get(k)]


def extract_tool_error(result_str: str) -> str | None:
    """MCP 도구가 [{"error": ...}] 행 형태로 반환한 오류 메시지를 추출 — 정상 결과면 None.

    MCP 도구들은 예외 대신 error 행을 반환하므로, [ERROR] 접두어 기반 판정만으로는
    이런 오류가 정상 엔티티로 오인되어 generate 컨텍스트까지 유입된다.
    """
    entities = list(iter_entities(result_str))
    if entities and all(isinstance(e, dict) and "error" in e for e in entities):
        return "; ".join(str(e["error"]) for e in entities[:3])
    return None


def collect_relevant_data(task_execution_results: list) -> list[dict]:
    """generate 컨텍스트에 포함할 데이터를 태스크별로 선별한다.

    generator(<수집된 데이터> 블록)와 references(출처)가 공유하는 단일 규칙:
    - 엔티티 리스트 결과: 워커 selected_ids로 필터링하되, 선별 ID가 실제 수집 ID와
      전혀 매칭되지 않으면(환각 ID) 선별을 무시하고 전문 사용. 전역 entity-ID dedup.
    - 비엔티티 결과(그래프 네트워크 dict 등): 문자열 동일성 dedup으로 그대로 포함.

    반환: [{"task_description": str, "items": [...]}]
      items 원소가 list면 엔티티 그룹(list[dict]), str이면 비엔티티 result_text.
    """
    blocks: list[dict] = []
    seen_entities: dict[str, dict] = {}
    seen_text: set[str] = set()

    for r in task_execution_results:
        calls = [
            (tc, list(iter_entities(tc.get("result_text", ""))))
            for tc in r.get("tool_calls", [])
            if not tc.get("is_error") and tc.get("result_text")
        ]

        sel = {str(i) for i in r.get("selected_ids", []) if i}
        # 워커가 유효한 JSON으로 '관련 엔티티 없음'을 명시한 경우 — 선별 없음 fallback과 구분
        deliberate_empty = (not sel) and bool(r.get("selection_valid"))
        if sel:
            present = {i for _, ents in calls for e in ents for i in entity_ids(e)}
            if not (sel & present):
                sel = set()

        items: list = []
        for tc, entities in calls:
            if entities:
                if deliberate_empty:
                    continue   # 워커가 전부 무관하다고 판단한 태스크의 엔티티는 제외
                kept: list[dict] = []
                for e in entities:
                    ids = entity_ids(e)
                    if sel and ids and not (set(ids) & sel):
                        continue
                    keys = ids or [json.dumps(e, sort_keys=True, ensure_ascii=False)]
                    
                    existing_entity = next((seen_entities[k] for k in keys if k in seen_entities), None)
                    if existing_entity is not None:
                        for k, v in e.items():
                            if not v:
                                continue
                            ev = existing_entity.get(k)
                            if not ev:
                                # 값 중복 가드는 ID 계열 키에만 적용 — name/title 같은 정규 키는
                                # 같은 값이 별칭 키(researcher 등)로 있어도 항상 병합한다
                                if k in _ENTITY_ID_KEYS and v in existing_entity.values():
                                    continue
                                existing_entity[k] = v
                            elif isinstance(ev, list) and isinstance(v, list):
                                for item in v:
                                    if item not in ev:
                                        ev.append(item)
                        # 이 레코드로 새로 알게 된 ID도 같은 엔티티로 등록 (multi-ID 행 대응)
                        for k in keys:
                            seen_entities.setdefault(k, existing_entity)
                        continue

                    for k in keys:
                        seen_entities[k] = e
                    kept.append(e)
                if kept:
                    items.append(kept)
            else:
                text = tc["result_text"]
                if text in seen_text:
                    continue
                parsed = try_parse(text)
                if parsed is not None and not parsed:
                    continue   # "[]", "{}" 등 빈 구조는 컨텍스트에 넣지 않음
                seen_text.add(text)
                items.append(text)

        if items:
            blocks.append({"task_description": r.get("task_description", ""), "items": items})
    return blocks


def try_parse(s: str) -> Any:
    """JSON 우선, 실패 시 ast.literal_eval로 파싱."""
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def iter_entities(result_str: str):
    """tool result 문자열에서 entity dict를 순서대로 yield."""
    if str(result_str).startswith("[ERROR]"):
        return
    items = try_parse(str(result_str))
    if not isinstance(items, list):
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            data = try_parse(item.get("text", ""))
            if data is None:
                continue
            for d in (data if isinstance(data, list) else [data])[:5]:
                if isinstance(d, dict):
                    yield d
        else:
            yield item



def clean_tool_result(result_str: str) -> str:
    """MCP text-block 래퍼를 제거하고 순수 entity JSON 문자열로 변환.
    iter_entities가 파싱 가능한 형태로 정규화한다."""
    if str(result_str).startswith("[ERROR]"):
        return result_str
    items = try_parse(str(result_str))
    if not isinstance(items, list):
        return result_str
    entities = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            data = try_parse(item.get("text", ""))
            if data is None:
                continue
            if isinstance(data, list):
                entities.extend(data)
            else:
                entities.append(data)
        else:
            entities.append(item)
    # compact JSON — generate 컨텍스트 토큰 절약 (indent 불필요)
    return json.dumps(entities, ensure_ascii=False) if entities else result_str


def summarize_tool_result(result_str: str) -> str:
    if str(result_str).startswith("[ERROR]"):
        return "오류"
    entities = list(iter_entities(result_str))
    if entities:
        previews = []
        for d in entities[:3]:
            label = str(d.get("name") or d.get("title") or d.get("id") or "")[:25]
            score = d.get("score")
            if score is not None:
                label += f"({score:.2f})"
            if label:
                previews.append(label)
        return f"{len(entities)}건" + (f": {', '.join(previews)}" if previews else "")

    parsed = try_parse(str(result_str))
    if parsed is None:
        return "결과 있음"
    if isinstance(parsed, list):
        return f"{len(parsed)}건" if parsed else "빈 결과"
    if isinstance(parsed, dict):
        nodes = parsed.get("nodes", [])
        rels = parsed.get("relationships", [])
        if nodes or rels:
            return f"nodes {len(nodes)}개, rels {len(rels)}개"
        return "결과 있음" if parsed else "빈 결과"
    return "빈 결과"
