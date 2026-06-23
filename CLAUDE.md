# CLAUDE.md — R&D Nexus

> Claude Code가 이 프로젝트를 읽을 때 따라야 할 규칙, 아키텍처 맥락, 코드 작성 기준을 정의합니다.

---

## 1. 프로젝트 개요

**목적:** R&D 전반(연구자 추천, 기술 추천, 과제 기획, 논문·특허 동향 분석)을 지원하는 멀티에이전트 AI 서비스

**핵심 질문 예시:**
```
AI 반도체 분야에서 향후 3년 투자 가치가 높은 연구 주제를 추천해줘
AI 반도체 핵심 연구자를 추천해줘
뉴로모픽 컴퓨팅 특허 동향을 알려줘
```

**기술 스택:**
- Runtime: Python 3.11+, 패키지 관리: uv
- Agent 프레임워크: LangGraph (`StateGraph`, `AsyncRedisSaver`)
- LLM: Ollama (`qwen2.5:7b`) via `langchain-ollama`
- MCP: `mcp>=1.28.0`, `langchain-mcp-adapters>=0.3.0` (SSE 프로토콜)
- Memory: `AsyncRedisSaver` (`langgraph-checkpoint-redis`)
- DB: MariaDB (선택), 미설정 시 인메모리 mock fallback
- Vector DB: Milvus (선택) — Dense(KR-SBERT 768-dim) + BM25 하이브리드 검색
- Graph DB: Neo4j (선택) — 연구자·논문·특허·기술·과제 관계 그래프
- API: FastAPI + `sse-starlette` (SSE 스트리밍), port 8080
- UI: `ai-server/src/static/index.html` (다크 테마, 퍼플 계열)

---

## 2. 아키텍처 원칙

### 결정론적 인프라가 LLM을 감싼다

- **LLM이 담당:** planner(계획 수립), llm_call(tool 선택), reflection(충분성 판단), generate(최종 답변)
- **코드가 담당:** tool_node MCP 라우팅, 루프 제어·종료, 에러 복구, Memory 관리

`rnd_max_replan` 초과 시 LLM 판단 없이 코드가 `sufficient`로 강제 처리합니다.

### 도구는 RunnableConfig로 주입

`llm_with_tools`와 `tools_by_name`은 FastAPI lifespan에서 MCP 세션으로 동적 획득한 뒤 `config["configurable"]`로 주입됩니다. 노드 내에서 직접 import하지 않습니다.

```python
# api/app.py lifespan 패턴
async with mcp_server_session() as session:
    llm_with_tools, tools_by_name = await get_llm_and_tools(session)
    app.state.graph = build_graph(memory)
    app.state.llm_with_tools = llm_with_tools
    app.state.tools_by_name = tools_by_name
```

### Config 우선순위

**API 파라미터 > MariaDB system_config > CONFIG_DEFAULTS**

`RequestConfig.set_current()` 가 요청 시작 시 ContextVar에 설정합니다. 노드는 `RequestConfig.current()`로 조회합니다.

---

## 3. 전체 흐름

```
HTTP POST /agent/query  (SSE 스트리밍)
    ↓
RequestConfig.set_current()       ← API params > MariaDB > defaults
    ↓
planner                           ← PlanOutput (with_structured_output, JSON 강제)
    ↓
llm_call ──┐                      ← llm_with_tools via config["configurable"]
           ↓
     should_continue               ← 결정론적: tool_calls 유무만 확인
      ├─ tool_calls 있음 → tool_node → [MCP 서버 SSE] ──┐
      └─ 없음 → reflection                              │
                   ↓        ←────────────────────────────┘
            ReflectionOutput (with_structured_output, JSON 강제)
            result == "sufficient"? ──No──→ planner (replan_count += 1)
                   ↓ Yes
              generate                ← rnd_model_generate (config override 가능)
                   ↓
            SSE done 이벤트 (eval_score 포함)
```

**SSE 이벤트 타입:** `plan` | `tool_call` | `token` | `done` | `error`

---

## 4. 디렉터리 구조 (모노레포)

```
rnd-nexus/
├── CLAUDE.md
├── pyproject.toml             ← uv workspace root (members: ai-server, mcp-server)
├── .env / .env.example
├── ai-server/                 ← LangGraph 에이전트 + FastAPI 서비스
│   ├── pyproject.toml
│   ├── Dockerfile
│   └── src/
│       ├── config.py          ← Settings: ollama, redis, mcp_server_url, mariadb_url, api_host/port
│       ├── main.py            ← uvicorn 실행만 (python -m main)
│       ├── api/
│       │   ├── app.py         ← FastAPI factory, lifespan (MCP session + memory 초기화)
│       │   ├── schemas.py     ← QueryRequest, ConfigOverride, HealthResponse
│       │   └── routes/
│       │       ├── health.py  ← GET /health
│       │       └── query.py   ← POST /agent/query (SSE 스트리밍)
│       ├── common/
│       │   └── config/
│       │       └── query_config.py  ← CONFIG_DEFAULTS, QueryConfig, RequestConfig (ContextVar)
│       ├── infrastructure/
│       │   ├── config_repository.py  ← MemoryConfigRepository, MariaDBConfigRepository
│       │   └── mariadb.py            ← parse_mariadb_url()
│       ├── agent/
│       │   ├── graph.py       ← build_graph(memory), should_continue, should_replan
│       │   ├── mcp_client.py  ← mcp_server_session(), get_llm_and_tools()
│       │   ├── state.py       ← RDAgentState (typing_extensions.TypedDict)
│       │   ├── nodes/
│       │   │   ├── planner.py
│       │   │   ├── llm_call.py
│       │   │   ├── tool_node.py
│       │   │   ├── reflection.py
│       │   │   └── generate.py
│       │   └── edges/
│       │       ├── should_continue.py
│       │       └── should_replan.py
│       ├── memory/
│       │   ├── session.py     ← AsyncRedisSaver, asynccontextmanager
│       │   └── compaction.py  ← should_compact(), compact_messages()
│       └── static/
│           └── index.html     ← 다크 테마 UI (퍼플 #7c3aed, SSE 수신)
├── mcp-server/                ← FastMCP 데이터 서버 (SSE, port 8000)
│   ├── pyproject.toml
│   ├── Dockerfile
│   └── src/
│       ├── config.py          ← MCPSettings: mariadb_url, milvus_*, neo4j_*, sentence_transformer_model
│       ├── mcp_server/
│       │   ├── server.py      ← FastMCP 조립, register_*_tools() (budget 제외)
│       │   └── tools/
│       │       ├── paper.py, patent.py, project.py
│       │       ├── researcher.py, technology.py
│       │       ├── vector.py  ← semantic_search() — MILVUS_HOST 미설정 시 None 반환
│       │       └── graph.py   ← get_researcher_network(), get_citation_graph(), run_graph_query()
│       └── infrastructure/
│           ├── component_factory.py  ← RepositoryFactory 싱글톤 + Milvus/Neo4j lazy-init
│           ├── database.py           ← ensure_schema, seed_from_fixtures
│           ├── milvus.py             ← make_vector_search_fn(), ensure_collection()
│           ├── neo4j.py              ← make_graph_query_fn(), make_fetch_*_fn()
│           └── repositories/         ← MariaDB/InMemory 구현체 (각 도메인)
├── domain/                    ← 공유 도메인 모델
│   ├── entities/              ← Paper, Patent, Project, Researcher, Technology
│   └── repositories/          ← Protocol 인터페이스
├── data/fixtures/             ← 인메모리 fallback JSON + Milvus/Neo4j 시딩 소스
│   ├── papers.json            ← id, text, cites 필드 포함
│   ├── patents.json           ← id, text 필드 포함
│   ├── researchers.json       ← id, text, authored_papers, invented_patents, researches_technologies
│   ├── technologies.json      ← id, text 필드 포함
│   └── projects.json          ← id, text, employs_researchers, uses_technologies
├── scripts/
│   └── seed_data.py           ← MariaDB/Milvus/Neo4j 통합 시딩 (--target, --clear 옵션)
└── docker-compose.yml         ← redis, mariadb, ollama, mcp-server, ai-server, neo4j, etcd, minio, milvus
```

---

## 5. State 설계

```python
# src/agent/state.py
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
import operator

class RDAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    plan: list[str]
    completed_steps: Annotated[list[str], operator.add]
    reflection_result: str          # "sufficient" | "insufficient"
    reflection_feedback: str
    replan_count: int               # 결정론적 루프 종료 카운터
    tool_results: dict[str, list[str]]  # 도구명 → 결과 문자열 목록
    eval_score: float               # 0.0 ~ 1.0 (generate가 기록)
    eval_feedback: str
```

- `typing_extensions.TypedDict` 직접 사용 (pyrefly 호환, `MessagesState` 상속 금지)
- `StateGraph(RDAgentState)` 라인에 `# pyrefly: ignore[bad-specialization]` 추가

---

## 6. Config 시스템

```python
# src/common/config/query_config.py
CONFIG_DEFAULTS = {
    "generate_model": "qwen2.5:7b",
    "max_replan": 3,
    "temperature": 0.0,
    "semantic_top_k": 20,
    "dense_weight": 0.3,
    "sparse_weight": 0.7,
}

# 우선순위: API 파라미터 > MariaDB system_config > CONFIG_DEFAULTS
class RequestConfig:
    @classmethod
    def set_current(cls, repo, override: QueryConfig) -> None: ...
    @classmethod
    def current(cls) -> dict: ...  # ContextVar 기반 per-request 격리
```

MariaDB `system_config` 테이블 (key VARCHAR, value JSON):
- `MARIADB_URL` 설정 시 자동 생성·사용
- 미설정 시 `MemoryConfigRepository` (프로세스 메모리) fallback

---

## 7. 노드 설계

### planner

```python
class PlanOutput(BaseModel):
    steps: list[str] = Field(description="수행할 단계 목록 (2-5개)")

async def planner(state: RDAgentState) -> dict:
    structured = llm.with_structured_output(PlanOutput)
    # Replan 시 reflection_feedback 프롬프트에 포함
    # 실패 시 fallback: ["관련 논문 검색", "관련 기술 추천"]
    return {"plan": plan, "completed_steps": []}
```

- 프롬프트에 도구 목록 없음 — 조사 방향(의도)만 기술

### llm_call

```python
async def llm_call(state: RDAgentState, config: RunnableConfig) -> dict:
    llm_with_tools = config["configurable"]["llm_with_tools"]
    # Context Compaction: 80,000 토큰 초과 시 compact_messages() 자동 실행
```

### tool_node

```python
async def tool_node(state: RDAgentState, config: RunnableConfig) -> dict:
    tools_by_name = config["configurable"]["tools_by_name"]
    # 예외 → ToolMessage("[ERROR] ...") 반환, raise 금지
    # tool_results 캐시 업데이트
```

### reflection

```python
class ReflectionOutput(BaseModel):
    result: Literal["sufficient", "insufficient"]
    feedback: str = Field(default="")

async def reflection(state: RDAgentState) -> dict:
    # replan_count >= rnd_max_replan → sufficient 강제 (무한루프 방지)
    # tool_results.items() 동적 순회 — 도구명 하드코딩 없음
    # 파싱 실패 시 sufficient 폴백
```

### generate

```python
async def generate(state: RDAgentState, config: RunnableConfig) -> dict:
    model = config.get("configurable", {}).get("generate_model", settings.rnd_model_generate)
    # tool_results.items() 동적 순회
    # "SCORE: X.X" 파싱하여 eval_score 기록
```

---

## 8. MCP 서버 도구 목록

| 도구 | 파일 | 주요 파라미터 |
|------|------|------|
| semantic_search | tools/vector.py | query, node_type, top_k — 단일 도메인 벡터 검색 (Milvus 필요) |
| semantic_graph_search | tools/graph_search.py | concept, entry_type, hops, top_k — **벡터→그래프 멀티홉** (Milvus+Neo4j 필요) |
| search_papers | tools/paper.py | query, year_from, year_to, author, limit |
| search_patents | tools/patent.py | query, country, year_from, assignee, limit |
| search_projects | tools/project.py | keyword, institution, status, year_from, limit |
| search_researchers | tools/researcher.py | query, specialty, affiliation, top_k |
| search_technologies | tools/technology.py | query, trl_min, top_k |
| get_researcher_network | tools/graph.py | researcher_name — Neo4j 연구자 협업·논문·특허 네트워크 (NEO4J_URI 필요) |
| get_citation_graph | tools/graph.py | paper_title, depth — Neo4j 논문 인용 그래프 |
| run_graph_query | tools/graph.py | cypher — Neo4j READ 전용 Cypher (WRITE 차단) |

새 도구는 `mcp-server/src/mcp_server/server.py`에 `register_*_tools(mcp)` 추가 → 클라이언트 자동 반영.

---

## 9. Milvus 벡터 검색

```python
# mcp-server/src/infrastructure/milvus.py
# 컬렉션: rnd_nodes
# 필드: id(int), entity_id(str), node_type(str), text(str)
#       dense_vector(FLOAT_VECTOR 768), sparse_vector(SPARSE_FLOAT_VECTOR BM25)
# 인덱스: dense → HNSW COSINE, sparse → BM25
```

- `MILVUS_HOST` 미설정 시 `semantic_search` 도구는 None 반환 (graceful degradation)
- 임베딩 모델: `snunlp/KR-SBERT-V40K-klueNLI-augSTS` (768-dim)
- 기본 가중치: dense 0.3 + sparse 0.7 (QueryConfig으로 per-request 오버라이드 가능)

---

## 10. Neo4j 그래프

**노드 레이블:** `Paper`, `Patent`, `Researcher`, `Technology`, `Project`, `Organization`

**관계:**
| 관계 | 방향 |
|------|------|
| AUTHORED | Researcher → Paper |
| INVENTED | Researcher → Patent |
| RESEARCHES | Researcher → Technology |
| WORKS_AT | Researcher → Organization |
| CITES | Paper → Paper |
| EMPLOYS | Project → Researcher |
| USES | Project → Technology |

- `NEO4J_URI` 미설정 시 graph 도구들은 None 반환 (graceful degradation)

---

## 11. Fixtures 구조

모든 fixture 파일은 `id` (= 도메인 ID), `text` (Milvus BM25용 결합 텍스트) 필드를 포함합니다.

```json
// researchers.json 추가 필드 (Neo4j 관계 시딩용)
{
  "authored_papers": ["P001"],
  "invented_patents": ["KR10-2024-0012345"],
  "researches_technologies": ["T002", "T007"]
}

// projects.json 추가 필드
{
  "employs_researchers": ["R001", "R007"],
  "uses_technologies": ["T002", "T007"]
}

// papers.json 추가 필드
{
  "cites": ["P001", "P003"]
}
```

---

## 12. 모델 설정

```python
# src/config.py
rnd_model: str = "qwen2.5:7b"          # planner / llm_call / reflection
rnd_model_generate: str = "qwen2.5:7b" # generate
ollama_base_url: str = "http://localhost:11430"  # Docker 내부: 11434
api_host: str = "0.0.0.0"
api_port: int = 8080
```

| 노드 | 오버라이드 방법 |
|------|----------------|
| planner / llm_call / reflection | `.env` RND_MODEL |
| generate | `.env` RND_MODEL_GENERATE 또는 QueryRequest.config.generate_model |

---

## 13. Memory

```python
# src/memory/session.py
@asynccontextmanager
async def create_memory():
    async with AsyncRedisSaver.from_conn_string(settings.redis_url) as memory:
        yield memory
```

- `thread_id` 단위 상태 격리, time-travel 지원
- `build_graph(memory)` — memory는 외부에서 주입

### Context Compaction

```python
COMPACTION_THRESHOLD = 80_000  # 토큰 초과 시 압축 트리거
```

---

## 14. 실행 방법

```bash
# 인프라 전체 실행 (repo root에서)
docker compose up -d

# 개별 서비스
docker compose up -d redis mariadb ollama mcp_server

# AI 서버 (web API, port 8080)
cd ai-server
python -m src.main

# 시드 데이터 투입
python scripts/seed_data.py --target mariadb --mariadb-url "mysql+pymysql://..."
python scripts/seed_data.py --target milvus --milvus-host localhost
python scripts/seed_data.py --target neo4j --neo4j-uri bolt://localhost:7687
python scripts/seed_data.py --clear   # 전체 초기화 후 재시딩

# 테스트
pytest tests/

# 그래프 시각화
python -c "
import asyncio
from src.memory.session import create_memory
from src.agent.graph import build_graph

async def show():
    async with create_memory() as m:
        g = build_graph(m)
        print(g.get_graph().draw_mermaid())

asyncio.run(show())
"
```

---

## 15. 환경 변수

```bash
# AI Server
OLLAMA_BASE_URL=http://localhost:11430  # Docker 내부: http://ollama:11434
RND_MODEL=qwen2.5:7b
RND_MODEL_GENERATE=qwen2.5:7b
RND_MAX_REPLAN=3
RND_LOG_LEVEL=INFO
REDIS_URL=redis://localhost:6379
MCP_SERVER_URL=http://localhost:8000/sse
API_HOST=0.0.0.0
API_PORT=8080
# MARIADB_URL=mysql+pymysql://rnd:rnd_password@localhost:3306/rnd_nexus

# MCP Server
# MARIADB_URL=mysql+pymysql://rnd:rnd_password@mariadb:3306/rnd_nexus
# MILVUS_HOST=milvus-standalone
# MILVUS_PORT=19530
# MILVUS_COLLECTION=rnd_nodes
# SENTENCE_TRANSFORMER_MODEL=snunlp/KR-SBERT-V40K-klueNLI-augSTS
# NEO4J_URI=bolt://neo4j:7687
# NEO4J_USERNAME=neo4j
# NEO4J_PASSWORD=password
```

---

## 16. 코드 작성 규칙

### 금지 사항
- `time.sleep()`으로 루프를 대기시키지 않습니다
- `rnd_max_replan` 없이 루프를 열어두지 않습니다
- State 필드를 노드 내부에서 직접 뮤테이션하지 않습니다 (항상 `return dict`)
- `tool_node`에서 예외를 `raise`하지 않습니다 (ToolMessage로 반환)
- SYSTEM_PROMPT에 도구 목록을 하드코딩하지 않습니다 (`bind_tools()` 스키마가 담당)
- `generate`, `reflection`, `tool_node`에서 도구명을 하드코딩하지 않습니다 (`tool_results.items()` 사용)
- 노드 함수를 동기(`def`)로 작성하지 않습니다 (항상 `async def`)
- `MessagesState`를 상속하지 않습니다 (`typing_extensions.TypedDict` 직접 사용)
- budget 관련 도구·모델·fixture를 추가하지 않습니다 (제거된 도메인)

### 필수 사항
- 새 MCP 도구는 `mcp-server/src/mcp_server/server.py`에 추가합니다 (클라이언트 자동 반영)
- `planner`, `reflection`의 LLM 호출은 `with_structured_output(PydanticModel)` 사용합니다
- `build_graph(memory)`는 항상 외부에서 memory를 주입받습니다
- `llm_with_tools`, `tools_by_name`은 `config["configurable"]`로 주입합니다
- Milvus/Neo4j 기능은 해당 환경변수 미설정 시 graceful degradation (None 반환, 예외 없음)
- Config 변경은 QueryRequest의 `config` 필드로 전달하고, `RequestConfig.set_current()`로 등록합니다
