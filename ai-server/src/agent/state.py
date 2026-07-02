from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class TaskSpec(TypedDict):
    id: str           # description SHA1 앞 8자
    description: str


class ToolCallRecord(TypedDict):
    tool_name: str
    args: dict
    result_text: str   # MCP 래퍼 제거 후 순수 entity JSON — iter_entities 파싱 가능
    summary: str       # 예: "3건: 김민준, 이서연"
    is_error: bool


class TaskExecutionResult(TypedDict):
    task_id: str
    task_description: str
    round: int         # 실행 시점의 iteration_count
    status: str        # "completed" | "empty" | "error"
    tool_calls: list[ToolCallRecord]
    worker_note: str          # 워커 최종 보고 한 줄 — orchestrator 수집 완료 판단용
    selected_ids: list[str]   # 워커가 태스크와 직접 관련하다고 선별한 엔티티 ID (빈 리스트 = 선별 없음 → 전문 사용)


class RDAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    pending_tasks: list[TaskSpec]               # orchestrator → parallel_executor
    task_execution_results: list[TaskExecutionResult]  # 단일 소스 (현재 턴)
    iteration_count: int
    out_of_scope: bool
