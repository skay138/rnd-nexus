from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class RDAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    tool_results: dict[str, list[str]]
    iteration_count: int
    pending_tasks: list[str]    # orchestrator → parallel_executor (태스크 설명 문자열)
    executed_tasks: list[str]   # 중복 차단용 (실행된 태스크 설명)
    task_results: list[dict]    # [{round, task, tools:[{name,summary}]}] — UI per-task 표시용
    no_new_data: bool           # parallel_executor에서 새 태스크 없을 때 → generate로 단락
