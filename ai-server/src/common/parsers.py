import ast
import json
import re
from typing import Any

# tool_results 메시지 내 툴명 줄 패턴: [semantic_search], [get_entities] 등
_TOOL_LINE_RE = re.compile(r'^\[[a-z_]+\]$')

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
    if "tech_id" in d:
        return {"type": "기술", "id": d["tech_id"], "title": d.get("name", "")}
    if "technology_id" in d:
        return {"type": "기술", "id": d["technology_id"], "title": d.get("name", "")}
    if "project_id" in d:
        return {"type": "과제", "id": d["project_id"], "title": d.get("title", d.get("name", ""))}
    if "org_id" in d:
        return {"type": "기관", "id": d["org_id"], "title": d.get("name", "")}
    if "node_type" in d:
        return {
            "type": d["node_type"],
            "id": str(d.get("id", "") or d.get("entity_id", "")),
            "title": d.get("name", d.get("title", "")),
        }
    return None

def build_deduped_context(tool_result_messages: list) -> str:
    """
    tool_results AIMessage 목록에서 엔티티를 ID 기준 dedup하여 단일 컨텍스트 반환.
    엔티티가 없는 결과(그래프 쿼리 등)는 문자열 중복 제거 후 그대로 보존.
    """
    seen_ids: set[str] = set()
    unique_entities: list[dict] = []
    seen_other: set[str] = set()
    other_parts: list[str] = []

    for msg in tool_result_messages:
        content = str(msg.content)
        for section in content.split("\n\n---\n\n"):
            for block in section.split("\n\n"):
                lines = block.strip().splitlines()
                tool_idx = next(
                    (i for i, ln in enumerate(lines) if _TOOL_LINE_RE.match(ln.strip())),
                    None,
                )
                if tool_idx is None:
                    continue
                result_str = "\n".join(lines[tool_idx + 1:]).strip()
                if not result_str or result_str.startswith("[ERROR]"):
                    continue
                entities = list(iter_entities(result_str))
                if entities:
                    for entity in entities:
                        ref = item_to_ref(entity)
                        eid = (ref["id"] if ref else "") or str(entity.get("id") or entity.get("entity_id", ""))
                        if not eid or eid in seen_ids:
                            continue
                        seen_ids.add(eid)
                        unique_entities.append(entity)
                elif result_str not in seen_other:
                    seen_other.add(result_str)
                    other_parts.append(result_str)

    parts: list[str] = []
    if unique_entities:
        parts.append(json.dumps(unique_entities, ensure_ascii=False, indent=2))
    parts.extend(other_parts)
    return "\n\n---\n\n".join(parts) if parts else "\n\n---\n\n".join(str(m.content) for m in tool_result_messages)


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
