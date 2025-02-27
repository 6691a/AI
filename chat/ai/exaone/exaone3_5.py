from enum import Enum

from langchain_ollama import ChatOllama


class Exaone3_5Params(str, Enum):
    # 2.4b 파라미터
    TINY = "exaone3.5:2.4b"
    # 7.8b 파라미터
    SMALL = "exaone3.5"
    # 32b 파라미터
    MEDIUM = "exaone3.5:32b"


class Exaone3_5:
    def __init__(self,
         parameter: Exaone3_5Params,
        temperature: float = None,
        stop: list[str] = None
    ):
        self.llm = ChatOllama(model=parameter.value, temperature=temperature, stop=stop)

    def __call__(self) -> ChatOllama:
        return self.llm