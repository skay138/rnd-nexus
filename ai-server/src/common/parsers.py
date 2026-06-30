import ast
import json
from typing import Any


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
    return json.dumps(entities, ensure_ascii=False, indent=2) if entities else result_str


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
