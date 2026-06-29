import ast
import json
import re
from typing import Any

# tool_results 메시지 내 툴명 줄 패턴: [semantic_search], [get_entities] 등
_TOOL_LINE_RE = re.compile(r'^\[[a-z_]+\]$')

# 벡터 검색 내부 필드 — LLM 컨텍스트에서 제외 (도메인 메트릭으로 오해 방지)
_SEARCH_INTERNAL_FIELDS = {"score", "distance", "dense_score", "sparse_score"}

def _drop_internal_fields(entity: dict) -> dict:
    return {k: v for k, v in entity.items() if k not in _SEARCH_INTERNAL_FIELDS}

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
        return {"type": "Paper", "id": d["paper_id"], "title": d.get("title", "")}
    if "patent_id" in d:
        return {"type": "Patent", "id": d["patent_id"], "title": d.get("title", "")}
    if "researcher_id" in d:
        return {"type": "Researcher", "id": d["researcher_id"], "title": d.get("name", d.get("researcher", ""))}
    if "tech_id" in d:
        return {"type": "Technology", "id": d["tech_id"], "title": d.get("name", "")}
    if "technology_id" in d:
        return {"type": "Technology", "id": d["technology_id"], "title": d.get("name", "")}
    if "project_id" in d:
        return {"type": "Project", "id": d["project_id"], "title": d.get("title", d.get("name", ""))}
    if "org_id" in d:
        return {"type": "Organization", "id": d["org_id"], "title": d.get("name", "")}
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
    seen_ids: dict[str, int] = {}  # id → unique_entities 인덱스
    unique_entities: list[dict] = []
    seen_other: set[str] = set()
    other_parts: list[str] = []

    def process_block(lines: list[str]):
        if not lines: return
        result_str = "\n".join(lines).strip()
        if not result_str or result_str.startswith("[ERROR]"):
            return
            
        entities = list(iter_entities(result_str))
        if entities:
            added_any = False
            for entity in entities:
                ref = item_to_ref(entity)
                eid = (ref["id"] if ref else "") or str(entity.get("id") or entity.get("entity_id", ""))
                if not eid:
                    continue
                added_any = True
                clean = _drop_internal_fields(entity)
                
                # 불필요한 스키마/중복 키 제거 (토큰 낭비 방지)
                if ref:
                    clean["id"] = ref["id"]
                    clean["type"] = ref["type"]
                    
                    # 중복 ID 키 제거
                    for k in ["researcher_id", "tech_id", "technology_id", "project_id", "patent_id", "org_id", "paper_id", "entity_id"]:
                        if k in clean and clean[k] == clean["id"]:
                            del clean[k]
                            
                    # node_type은 type으로 대체되었으므로 제거
                    if "node_type" in clean:
                        del clean["node_type"]
                        
                    # 중복 이름 키 제거
                    primary_name = clean.get("title") or clean.get("name") or ref.get("title")
                    if primary_name:
                        for k in ["researcher", "technology", "project", "organization", "patent"]:
                            if k in clean and clean[k] == primary_name:
                                del clean[k]
                                
                    # name과 title이 동일하면 하나만 유지
                    if "name" in clean and "title" in clean and clean["name"] == clean["title"]:
                        del clean["name"]
                        
                # 서로 다른 타입(예: 논문 P001과 과제 P001)의 ID 충돌을 막기 위해 복합 키 사용
                entity_type = clean.get("type", "Unknown")
                unique_key = f"{entity_type}::{eid}"
                
                if unique_key in seen_ids:
                    # 병합 로직 고도화: 단순 키 존재 여부가 아니라, 더 '풍부한' 데이터로 스마트 업데이트
                    idx = seen_ids[unique_key]
                    for k, v in clean.items():
                        old_v = unique_entities[idx].get(k)
                        # 1. 기존 값이 없거나 비어있으면 무조건 채움
                        if not old_v:
                            unique_entities[idx][k] = v
                        # 2. 타입이 다를 경우 구조화된 데이터(list, dict)를 더 우선시함
                        elif type(v) != type(old_v):
                            if isinstance(v, (list, dict)):
                                unique_entities[idx][k] = v
                        # 3. 리스트인 경우 덮어쓰지 않고 중복 없이 병합(Union)하여 네트워크 관계 유실 방지
                        elif isinstance(v, list) and isinstance(old_v, list):
                            for item in v:
                                if item not in old_v:
                                    old_v.append(item)
                        # 4. 딕셔너리인 경우 키 기준으로 병합
                        elif isinstance(v, dict) and isinstance(old_v, dict):
                            for sub_k, sub_v in v.items():
                                if sub_k not in old_v:
                                    old_v[sub_k] = sub_v
                        # 5. 문자열일 경우 내용이 더 긴 상세 데이터로 덮어쓰기
                        elif isinstance(v, str) and isinstance(old_v, str):
                            if len(v) > len(old_v):
                                unique_entities[idx][k] = v
                else:
                    seen_ids[unique_key] = len(unique_entities)
                    unique_entities.append(clean)
            # eid를 추출할 수 없는 엔티티만 있는 경우(run_graph_query Cypher 결과 등) → other_parts로 보존
            if not added_any and result_str not in seen_other:
                seen_other.add(result_str)
                other_parts.append(result_str)
        elif result_str not in seen_other:
            seen_other.add(result_str)
            other_parts.append(result_str)

    for msg in tool_result_messages:
        content = str(msg.content)
        current_tool_output = []
        in_tool_block = False
        
        for line in content.splitlines():
            line_stripped = line.strip()
            # 태스크 구분선이나 제목은 무시
            if line_stripped.startswith("---") or line_stripped.startswith("# "):
                if current_tool_output:
                    process_block(current_tool_output)
                    current_tool_output = []
                in_tool_block = False
                continue
                
            # [tool_name] 마커 발견
            if _TOOL_LINE_RE.match(line_stripped):
                if current_tool_output:
                    process_block(current_tool_output)
                    current_tool_output = []
                in_tool_block = True
                continue
                
            if in_tool_block:
                current_tool_output.append(line)
                
        # 마지막 블록 처리
        if current_tool_output:
            process_block(current_tool_output)

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
