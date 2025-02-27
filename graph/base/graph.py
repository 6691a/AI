from abc import ABC, abstractmethod

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from graph.base.state import Language, BaseState


class BaseGraph(ABC):
    def __init__(self, models: dict[str, BaseChatModel], state: BaseState, default_model_key: str = "default"):
        if not models:
            raise ValueError("최소한 하나의 LLM 모델은 제공해야 합니다.")

        self.models = models
        self.default_model = self.models[default_model_key]
        self.graph_builder = StateGraph(state)
        self.compiled_graph = None

    def __call__(self, *args, **kwargs) -> CompiledStateGraph:
        self.graph_node_setup()
        self.graph_edge_setup()
        self.compiled_graph = self.graph_complete(*args, **kwargs)
        return self.compiled_graph

    @abstractmethod
    def graph_node_setup(self):
        """
        그래프 노드를 설정합니다.
        """

    @abstractmethod
    def graph_edge_setup(self):
        """
        그래프 엣지를 설정합니다.
        """

    @abstractmethod
    def get_language_model(self, language: Language, **kwargs) -> BaseChatModel:
        """
        BaseState["language"]의 설정을 통해 언어 번역 시 사용될 모델 반환

        Example:
        def get_language_model(self, state: BaseState, **kwargs):
            if language == Language.KOR:
                return ChatOllama(model="llama3.2")
        """

    def graph_complete(self, *args, **kwargs) -> CompiledStateGraph:
        return self.graph_builder.compile()


    def get_think_model(self) -> BaseChatModel:
        return self.models.get("think", self.default_model)
