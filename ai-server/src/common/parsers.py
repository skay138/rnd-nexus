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

def item_to_ref(d: dict) -> dict | None:
    """엔티티 딕셔너리를 프론트엔드 레퍼런스 포맷으로 변환."""
    if "paper_id" in d:
        return {"type": "논문", "id": d["paper_id"], "title": d.get("title", "")}
    if "patent_id" in d:
        return {"type": "특허", "id": d["patent_id"], "title": d.get("title", "")}
    if "researcher_id" in d:
        return {"type": "연구자", "id": d["researcher_id"], "title": d.get("name", "")}
    if "technology_id" in d:
        return {"type": "기술", "id": d["technology_id"], "title": d.get("name", "")}
    if "project_id" in d:
        return {"type": "과제", "id": d["project_id"], "title": d.get("title", d.get("name", ""))}
    if "node_type" in d:
        return {
            "type": d["node_type"],
            "id": str(d.get("id", "") or d.get("entity_id", "")),
            "title": d.get("name", d.get("title", "")),
        }
    return None

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

    # entity 포맷이 아닌 경우 (그래프 네트워크, Cypher 결과 등) — 비어있는지 확인
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
        # 다른 dict 구조
        return "결과 있음" if parsed else "빈 결과"
    return "빈 결과"
