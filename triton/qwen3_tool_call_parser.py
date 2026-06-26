import json
import re
import uuid
from typing import Union

from engine.utils.tokenizer import AnyTokenizer
from engine.utils.tool_call_parsers.tool_call_parser import (
    ToolCallParser,
    ToolParserManager,
)
from schemas.openai import (
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallChunk,
    ChatCompletionMessageToolCalls,
    ChatCompletionResponseMessage,
    ChatCompletionStreamResponseDelta,
    Function1,
    Function2,
)

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@ToolParserManager.register_module("qwen3")
class QwenToolCallParser(ToolCallParser):
    """Tool call parser for Qwen3 models using Hermes <tool_call> format."""

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)
        self._emitted_calls: int = 0

    def parse_tool_calls(
        self, full_text: str, role: str, backend: str
    ) -> ChatCompletionResponseMessage:
        clean = _THINK_RE.sub("", full_text).strip()
        matches = _TOOL_CALL_RE.findall(clean)
        if not matches:
            return ChatCompletionResponseMessage(
                tool_calls=None, content=full_text, role=role
            )
        calls = []
        for raw in matches:
            try:
                obj = json.loads(raw.strip())
                calls.append(
                    ChatCompletionMessageToolCall(
                        id=f"call-{uuid.uuid4().hex[:8]}",
                        type="function",
                        function=Function1(
                            name=obj["name"],
                            arguments=json.dumps(
                                obj.get("arguments", obj.get("parameters", {}))
                            ),
                        ),
                    )
                )
            except Exception:
                continue
        if not calls:
            return ChatCompletionResponseMessage(
                tool_calls=None, content=full_text, role=role
            )
        content_before = _TOOL_CALL_RE.split(clean)[0].strip()
        return ChatCompletionResponseMessage(
            tool_calls=ChatCompletionMessageToolCalls(root=calls),
            content=content_before or "",
            role=role,
        )

    def parse_tool_calls_streaming(
        self, current_text: str, delta_text: str, backend: str
    ) -> Union[ChatCompletionStreamResponseDelta, None]:
        clean = _THINK_RE.sub("", current_text).strip()

        # No tool call started yet
        if "<tool_call>" not in clean:
            if "<think>" not in current_text:
                return ChatCompletionStreamResponseDelta(content=delta_text)
            if "</think>" in current_text:
                after = current_text.split("</think>", 1)[1]
                if delta_text in after and "<tool_call>" not in after:
                    return ChatCompletionStreamResponseDelta(content=delta_text)
            return None

        # Count complete tool calls
        complete = _TOOL_CALL_RE.findall(clean)
        if len(complete) <= self._emitted_calls:
            return None

        # Emit next complete tool call
        idx = self._emitted_calls
        self._emitted_calls += 1
        try:
            obj = json.loads(complete[idx].strip())
            return ChatCompletionStreamResponseDelta(
                tool_calls=[
                    ChatCompletionMessageToolCallChunk(
                        index=idx,
                        type="function",
                        id=f"call-{uuid.uuid4().hex[:8]}",
                        function=Function2(
                            name=obj["name"],
                            arguments=json.dumps(
                                obj.get("arguments", obj.get("parameters", {}))
                            ),
                        ).model_dump(exclude_none=True),
                    )
                ]
            )
        except Exception:
            return None
