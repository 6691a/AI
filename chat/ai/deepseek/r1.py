from enum import Enum

from langchain_ollama import ChatOllama


class DeepSeekR1Params(str, Enum):
    TINY = "deepseek-r1:1.5b"  # 1.5B 파라미터
    SMALL = "deepseek-r1:7b"  # 7B 파라미터
    MEDIUM = "deepseek-r1:8b"  # 8B 파라미터
    LARGE = "deepseek-r1:14b"  # 14B 파라미터
    XL = "deepseek-r1:32b"  # 32B 파라미터
    XXL = "deepseek-r1:70b"  # 70B 파라미터


class DeepSeekR1:
    def __init__(
            self,
            parameter: DeepSeekR1Params,
            temperature: float = None,
            stop: list[str] = None
    ):
        self.llm = ChatOllama(model=parameter.value, temperature=temperature, stop=stop)

    def __call__(self) -> ChatOllama:
        return self.llm