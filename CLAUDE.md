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
- LLM: Ollama (`qwen2.5:32b` 이상 권장) via `langchain-ollama`
- MCP: `mcp>=1.28.0`, `langchain-mcp-adapters>=0.3.0` (SSE 프로토콜)
- Memory: `AsyncRedisSaver` (`langgraph-checkpoint-redis`)
- DB: MariaDB (선택), 미설정 시 인메모리 mock fallback
- Vector DB: Milvus (선택) — Dense(KR-SBERT 768-dim) + BM25 하이브리드 검색
- Graph DB: Neo4j (선택) — 연구자·논문·특허·기술·과제 관계 그래프
- API: FastAPI + `sse-starlette` (SSE 스트리밍), port 8080
- UI: `ai-server/src/static/index.html` (다크 테마, 퍼플 계열)

---

## 2. 아키텍처 원칙

### 멀티에이전트: Orchestrator → Worker Agents

- **Orchestrator:** 고수준 태스크 계획만 수립. 도구 이름·파라미터를 직접 지정하지 않음
- **Worker Agent:** 각 태스크를 받아 어떤 도구를 어떻게 쓸지 스스로 판단 (mini ReAct loop)
- **코드가 담당:** 루프 제어·종료, 중복 태스크 차단, 에러 격리, Memory 관리

`iteration_count >= max_iterations` 도달 시 LLM 판단 없이 코드가 `generate`로 강제 라우팅합니다.

### 도구는 RunnableConfig로 주입

`tools_by_name`은 FastAPI lifespan에서 MCP 세션으로 동적 획득한 뒤 `config["configurable"]`로 주입됩니다.

```python
# api/app.py lifespan 패턴
async with mcp_server_session() as session:
    app.state.tools_by_name = await get_llm_and_tools(session)
    app.state.graph = build_graph(memory)
```

### Messages가 Source of Truth

모든 노드 간 컨텍스트는 `state["messages"]`를 통해 전달됩니다. `generate`는 `orchestrator`, `tool_results`, `final_answer` 이름의 메시지를 필터링해 사용합니다.

### Config 우선순위

**API 파라미터 > MariaDB system_config > CONFIG_DEFAULTS**

---

## 3. 전체 흐름

```
HTTP POST /agent/query  (SSE 스트리밍)
    ↓
RequestConfig.set_current()       ← API params > MariaDB > defaults
    ↓
orchestrator                      ← OrchestratorPlan (with_structured_output)
                                    고수준 태스크 목록 반환 list[str]
    ↓
should_continue                   ← pending_tasks 유무만 확인
    ├─ tasks 있음 → parallel_executor
    └─ 없음 → generate
         ↓
parallel_executor                 ← 태스크별 Worker Agent 병렬 실행
    각 Worker: get_llm().bind_tools() + ReAct loop (최대 5스텝)
    의존관계 도구 호출도 워커 내부에서 자율 처리
         ↓
_after_executor                   ← iteration_count >= max_iterations
    ├─ 조건 충족 → generate
    └─ 아니면 → orchestrator
         ↓
generate                          ← messages 히스토리 기반 최종 답변
    relevant: human | orchestrator | tool_results | final_answer
         ↓
SSE done 이벤트 (references 포함)
```

**SSE 이벤트 타입:** `orchestrator` | `tool_result` | `token` | `done` | `error`

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
│       │   ├── config/
│       │   │   └── query_config.py  ← CONFIG_DEFAULTS, QueryConfig, RequestConfig (ContextVar)
│       │   ├── llm.py               ← get_llm() 팩토리 (ollama / openai 호환 서버)
│       │   └── parsers.py           ← iter_entities(), item_to_ref(), summarize_tool_result()
│       ├── infrastructure/
│       │   ├── config_repository.py  ← MemoryConfigRepository, MariaDBConfigRepository
│       │   └── mariadb.py            ← parse_mariadb_url()
│       ├── agent/
│       │   ├── graph.py       ← build_graph(memory), should_continue, _after_executor
│       │   ├── mcp_client.py  ← mcp_server_session(), get_llm_and_tools()
│       │   ├── state.py       ← RDAgentState (typing_extensions.TypedDict)
│       │   ├── nodes/
│       │   │   ├── orchestrator.py   ← 고수준 태스크 계획
│       │   │   ├── parallel_executor.py  ← Worker Agent 병렬 실행
│       │   │   └── generate.py       ← 최종 답변 생성
│       │   ├── edges/
│       │   │   └── should_continue.py
│       │   └── utils/
│       │       └── context.py        ← get_turn_context() 멀티턴 경계 분리
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
│       │   ├── server.py      ← FastMCP 조립, register_*_tools()
│       │   └── tools/
│       │       ├── entities.py    ← get_entities() — ID 기반 상세 조회 (도메인 공통)
│       │       ├── vector.py      ← semantic_search() — MILVUS_HOST 미설정 시 빈 목록 반환
│       │       ├── vector_graph.py ← semantic_graph_search() — 벡터→그래프 멀티홉
│       │       ├── graph.py       ← get_researcher_network(), get_citation_graph(), run_graph_query()
│       │       └── filter.py      ← filter_entities() — 연도·기관·상태 필터 (repository 경유)
│       └── infrastructure/
│           ├── component_factory.py  ← RepositoryFactory 싱글톤 + Milvus/Neo4j lazy-init (False sentinel)
│           ├── graph_compiler.py     ← _NEO4J_RELATIONS, SCHEMA_HINT, compile_hop()
│           ├── database.py           ← ensure_schema, seed_from_fixtures
│           ├── milvus.py             ← make_vector_search_fn(), ensure_collection()
│           ├── neo4j.py              ← make_graph_query_fn(), make_fetch_*_fn()
│           └── repositories/         ← MariaDB/InMemory 구현체 (각 도메인)
├── domain/                    ← 공유 도메인 모델
│   ├── entities/              ← Paper, Patent, Project, Researcher, Technology (budget 제외)
│   └── repositories/          ← Protocol 인터페이스 (budget 제외)
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

class RDAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    tool_results: dict[str, list[str]]  # {tool_name: [result_str, ...]} — _build_references용
    iteration_count: int                # 오케스트레이터 호출 횟수
    pending_tasks: list[str]            # 이번 라운드 실행 태스크 설명 목록
    executed_tasks: list[str]           # 코드 레벨 dedup용 (실행된 태스크 설명)
    task_results: list[dict]            # [{round, task, tools:[{name,summary}]}] — UI per-task 표시용
```

- `typing_extensions.TypedDict` 직접 사용 (pyrefly 호환, `MessagesState` 상속 금지)
- `StateGraph(RDAgentState)` 라인에 `# pyrefly: ignore[bad-specialization]` 추가

---

## 6. Config 시스템

```python
# src/common/config/query_config.py
CONFIG_DEFAULTS = {
    "temperature":    0.0,
    "semantic_top_k": 20,
    "keyword_weight": 0.5,   # BM25(sparse) 가중치. dense = 1.0 - keyword_weight
}
# 모델명·max_iterations는 DB → config.py(env) 순으로 fallback

# 우선순위: API 파라미터 > MariaDB system_config > config.py(env) > CONFIG_DEFAULTS
class RequestConfig:
    @classmethod
    def set_current(cls, resolved, original_query) -> None: ...
    @classmethod
    def current(cls) -> dict: ...  # ContextVar 기반 per-request 격리
```

MariaDB `system_config` 테이블 (key VARCHAR, value JSON):
- `MARIADB_URL` 설정 시 자동 생성·사용
- 미설정 시 `MemoryConfigRepository` (프로세스 메모리) fallback
- 저장 키: `orchestrator_model`, `worker_model`, `generate_model`, `compact_model`, `max_iterations`, `temperature`, `semantic_top_k`, `keyword_weight`
- `/settings` 페이지 또는 `PATCH /api/v1/admin/config`로 재기동 없이 변경 가능

---

## 7. 노드 설계

### orchestrator

```python
class OrchestratorPlan(BaseModel):
    reasoning: str   # 수집 현황 평가 및 전략 (한국어)
    tasks: list[str] # 병렬 실행 태스크 설명 목록. 수집 완료 시 []

async def orchestrator(state, config) -> dict:
    # _build_capabilities(tools_by_name) — 동적 capabilities 주입
    # with_structured_output(OrchestratorPlan)
    # _merge_dependent_tasks(tasks) — 의존 키워드("검색된", "위의" 등) 감지 시 단일 태스크로 병합
    # 실패 시: tasks=[], meaningful error reasoning 반환
    return {"messages": [AIMessage(name="orchestrator")], "pending_tasks": tasks, "iteration_count": n}
```

- 도구 이름·파라미터를 직접 지정하지 않음 — 워커가 자율 결정
- 대화 히스토리의 `[tool_results]` 메시지를 보고 수집 완료 여부 판단
- `_merge_dependent_tasks`: 태스크 설명에 이전 결과 참조 키워드가 있으면 병렬 실행 불가 → 앞 태스크에 병합

### parallel_executor (Worker Agent)

```python
_WORKER_MAX_STEPS = 5

async def _run_worker(task: str, tools_by_name, settings) -> list[tuple[str, str]]:
    # get_llm().bind_tools(all_tools) + ReAct loop
    # seen_calls 기반 동일 (tool, args) 중복 호출 차단
    # 도구 호출 → 결과 분석 → 추가 호출 여부 자율 판단
    return [(tool_name, result_str), ...]

async def parallel_executor(state, config) -> dict:
    # _task_key(task) 기준 중복 태스크 차단 (executed_tasks: list[str])
    # asyncio.gather(*[_run_worker(t, ...) for t in fresh_tasks])
    # tool_results[tool_name] += [result_str]
    # AIMessage(name="tool_results") + task_results 생성
```

### generate

```python
async def generate(state, config) -> dict:
    # get_turn_context() 로 멀티턴 경계 분리 (마지막 final_answer 이후 = 현재 턴)
    # tool_results → HumanMessage 변환 (로컬 LLM Human→AI 교차 구조 보장)
    # get_llm(streaming=True) — SSE 토큰 스트리밍
    return {"messages": [AIMessage(name="final_answer")]}
```

---

## 8. MCP 클라이언트

```python
# agent/mcp_client.py
async def get_llm_and_tools(session: ClientSession) -> dict:
    mcp_tools = await load_mcp_tools(session)
    tools_by_name = {t.name: t for t in mcp_tools}
    return tools_by_name  # llm_with_tools 없음 — 워커가 자체 bind_tools
```

새 도구는 `mcp-server/src/mcp_server/server.py`에 `register_*_tools(mcp)` 추가 → 클라이언트 자동 반영.

---

## 9. MCP 서버 도구 목록

| 도구 | 파일 | 주요 파라미터 |
|------|------|------|
| semantic_search | tools/vector.py | query, node_type, top_k, keyword_weight, year_from, year_to — 하이브리드 벡터 검색 (Milvus 필요) |
| semantic_graph_search | tools/vector_graph.py | query, entry_type, hops, top_k — **벡터→그래프 멀티홉** (Milvus+Neo4j 필요) |
| get_entities | tools/entities.py | entity_type, ids — ID 기반 엔티티 상세 조회 → `list[dict]` 반환 |
| filter_entities | tools/filter.py | entity_type, year_from, year_to, affiliation 등 — 연도·기관·상태 필터 (repository 경유) |
| get_researcher_network | tools/graph.py | researcher_name, researcher_id — Neo4j 연구자 논문·특허·기술·기관 네트워크 |
| get_citation_graph | tools/graph.py | paper_title, paper_id, depth — Neo4j 논문 인용 그래프 |
| run_graph_query | tools/graph.py | cypher — Neo4j READ 전용 Cypher (WRITE 차단, 알 수 없는 관계 타입 차단) |

> 도메인별 키워드 검색(search_papers 등)은 제거됨. 워커는 `semantic_search`로 ID를 수집하고 `get_entities`로 상세 조회하는 패턴을 사용합니다.

---

## 10. Milvus 벡터 검색

```python
# mcp-server/src/infrastructure/milvus.py
# 컬렉션: rnd_nodes
# 필드: id(int), entity_id(str), node_type(str), text(str)
#       dense_vector(FLOAT_VECTOR 768), sparse_vector(SPARSE_FLOAT_VECTOR BM25)
# 인덱스: dense → HNSW COSINE, sparse → BM25
```

- `MILVUS_HOST` 미설정 시 `semantic_search` 도구는 빈 목록 반환 (graceful degradation)
- 임베딩 모델: `snunlp/KR-SBERT-V40K-klueNLI-augSTS` (768-dim)
- `keyword_weight`: BM25(sparse) 가중치 (기본 0.5), dense = `1.0 - keyword_weight`
- `year_from` / `year_to`: Milvus 필드 레벨 필터 (Milvus 컬렉션에 `year` 필드 필요)

---

## 11. Neo4j 그래프

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

## 12. Fixtures 구조

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

## 13. 모델 설정

```python
# src/config.py
llm_provider: str = "ollama"       # "ollama" 또는 "openai" (vLLM/Triton 호환)
rnd_model:    str = "qwen2.5:7b"   # 최초 기동 시 모든 역할의 시드값
llm_base_url: str = "http://localhost:11434"  # Ollama: http://localhost:11434 / Triton·vLLM: http://server:8000/v1
api_host: str = "0.0.0.0"
api_port: int = 8080
```

역할별 모델은 기동 시 `rnd_model` 값으로 `system_config`에 `INSERT IGNORE` 시드되며, 이후 `/settings` 페이지 (`PATCH /api/v1/admin/config`)에서 재기동 없이 독립 변경 가능.

| 노드 | system_config 키 | 최소 | 권장 |
|------|-----------------|------|------|
| orchestrator | `orchestrator_model` | 14b | 32b |
| worker | `worker_model` | 14b | 32b |
| generate | `generate_model` | 7b | 32b+ |
| compaction | `compact_model` | 3b | 7b |

---

## 14. Memory

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
COMPACTION_THRESHOLD = 80_000  # 토큰 초과 시 압축 트리거 (orchestrator, generate)
```

---

## 15. 로깅 및 디버깅

`RND_LOG_LEVEL=DEBUG` 설정 시 각 노드에서 상세 로그 출력:

```
[orchestrator] iter=1/3 elapsed=1.2s tasks=2
  reasoning: ...
  tasks: ["논문 조사 ...", "특허 동향 분석 ..."]

[worker:태스크설명앞40자] semantic_search elapsed=0.4s
  args: {"query": "..."}
  result: [전체 결과]

[parallel_executor] 전체 2개 완료 elapsed=0.8s

[generate] state.messages 전체 (7개):
  [0] name=human ...
  [1] name=orchestrator ...
  [2] name=tool_results ...
[generate] elapsed=3.1s
```

**노드별 로그 색상:**
| 노드 | 색상 |
|------|------|
| orchestrator | magenta |
| parallel_executor / worker | yellow |
| generate | green |
| api | bright white |
| infrastructure | bright cyan |

---

## 16. 실행 방법

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

## 17. 환경 변수

```bash
# AI Server
LLM_BASE_URL=http://localhost:11434   # Ollama: http://localhost:11434 / Triton·vLLM: http://server:8000/v1
RND_MODEL=qwen2.5:7b   # 최초 기동 시 시드값 — 이후엔 /settings 에서 역할별 관리
RND_MAX_ITERATIONS=3
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

## 18. 코드 작성 규칙

### 금지 사항
- `time.sleep()`으로 루프를 대기시키지 않습니다
- `max_iterations` 없이 루프를 열어두지 않습니다
- State 필드를 노드 내부에서 직접 뮤테이션하지 않습니다 (항상 `return dict`)
- worker에서 예외를 `raise`하지 않습니다 (`[ERROR] ...` 문자열로 반환)
- orchestrator 프롬프트에 도구 이름·파라미터를 하드코딩하지 않습니다 (`_build_capabilities()` 사용)
- 노드 함수를 동기(`def`)로 작성하지 않습니다 (항상 `async def`)
- `MessagesState`를 상속하지 않습니다 (`typing_extensions.TypedDict` 직접 사용)
- budget 관련 도구·모델·fixture를 추가하지 않습니다 (제거된 도메인)

### 필수 사항
- 새 MCP 도구는 `mcp-server/src/mcp_server/server.py`에 `register_*_tools(mcp)` 추가합니다 (클라이언트 자동 반영)
- `orchestrator`의 LLM 호출은 `with_structured_output(OrchestratorPlan)` 사용합니다
- `build_graph(memory)`는 항상 외부에서 memory를 주입받습니다
- `tools_by_name`은 `config["configurable"]`로 주입합니다 (`llm_with_tools` 없음)
- Milvus/Neo4j 기능은 해당 환경변수 미설정 시 graceful degradation (None 반환, 예외 없음)
- Config 변경은 QueryRequest의 `config` 필드로 전달하고, `RequestConfig.set_current()`로 등록합니다
