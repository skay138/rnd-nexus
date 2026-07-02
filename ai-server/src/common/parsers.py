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
