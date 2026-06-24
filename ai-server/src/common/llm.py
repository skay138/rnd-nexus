import logging
from langchain_core.language_models import BaseChatModel
from config import get_settings

logger = logging.getLogger(__name__)

def get_llm(model: str, streaming: bool = False, **kwargs) -> BaseChatModel:
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
    base_url = settings.ollama_base_url
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
                **kwargs
            )
        except ImportError:
            logger.error("langchain-openai 패키지가 설치되어 있지 않습니다. 'pip install langchain-openai'를 실행하세요.")
            raise
            
    else:
        raise ValueError(f"지원하지 않는 LLM 제공자입니다: {provider}")
