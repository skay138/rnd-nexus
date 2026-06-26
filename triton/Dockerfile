FROM nvcr.io/nvidia/tritonserver:26.05-vllm-python-py3

# Patch Qwen2Tokenizer: all_special_tokens_extended attribute missing in slow tokenizer
RUN sed -i \
    's/tokenizer\.all_special_tokens_extended/getattr(tokenizer, "all_special_tokens_extended", tokenizer.all_special_tokens)/g' \
    /opt/tritonserver/python/openai/openai_frontend/engine/utils/tokenizer.py

# Add Qwen3 tool call parser (Hermes <tool_call> format)
COPY triton/qwen_tool_call_parser.py \
    /opt/tritonserver/python/openai/openai_frontend/engine/utils/tool_call_parsers/qwen_tool_call_parser.py

# Register qwen parser in __init__.py
RUN sed -i \
    '/^from .tool_call_parser import/a from .qwen_tool_call_parser import QwenToolCallParser' \
    /opt/tritonserver/python/openai/openai_frontend/engine/utils/tool_call_parsers/__init__.py && \
    sed -i \
    's/"MistralToolParser",/"MistralToolParser",\n    "QwenToolCallParser",/' \
    /opt/tritonserver/python/openai/openai_frontend/engine/utils/tool_call_parsers/__init__.py
