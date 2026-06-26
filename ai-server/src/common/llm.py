import logging
import re
from typing import Any
from langchain_core.language_models import BaseChatModel
from config import get_settings

logger = logging.getLogger(__name__)

def get_llm(model: str, streaming: bool = False, enable_thinking: bool = True, **kwargs) -> BaseChatModel:
    """
    공통 LLM 팩토리 함수.
    추후 Triton, vLLM, OpenAI 등 다양한 모델 제공자를 지원하기 위해 확장 가능하도록 설계되었습니다.
    
    Args:
        model (str): 사용할 LLM 모델 이름
        streaming (bool): 스트리밍 모드 활성화 여부
        **kwargs: 추가 LLM 설정 (temperature 등)
        
    Returns:
        BaseChatModel: 설정된 LangChain Chat 모델 인스턴스
    """
    settings = get_settings()
    provider = settings.llm_provider.lower()
    base_url = settings.llm_base_url
    api_key = settings.llm_api_key
    
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model, 
            base_url=base_url, 
            streaming=streaming, 
            **kwargs
        )
        
    elif provider == "openai":
        # Triton, vLLM 등 OpenAI API 스펙을 호환하는 서버에서 사용
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model,
                base_url=base_url,
                streaming=streaming,
                api_key=api_key or "EMPTY",
                timeout=None,
                max_tokens=kwargs.pop("max_tokens", 8192),
                **kwargs
            )
        except ImportError:
            logger.error("langchain-openai 패키지가 설치되어 있지 않습니다. 'pip install langchain-openai'를 실행하세요.")
            raise
            
    else:
        raise ValueError(f"지원하지 않는 LLM 제공자입니다: {provider}")


async def llm_ainvoke(llm: BaseChatModel, messages: list, config: Any = None) -> str:
    """astream으로 청크를 직접 누적하고 <think> 블록을 제거하여 반환.

    config는 LangGraph 상태 전달용으로만 시그니처에 유지하며, astream에는 넘기지 않는다.
    넘기면 LangGraph가 스트림을 인터셉트해 Qwen3 thinking 응답이 중간에 잘린다.
    """
    chunks: list[str] = []
    async for chunk in llm.astream(messages):
        chunks.append(chunk.content if isinstance(chunk.content, str) else "")
    full = "".join(chunks)
    return re.sub(r"<think>.*?</think>", "", full, flags=re.DOTALL).strip()
