from typing import Annotated

from pydantic import BaseModel

from langchain_ollama import ChatOllama
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

"""
LangGraph 맛보기
"""
class AgentState(BaseModel):
    messages: list[Annotated[AnyMessage, add_messages]]


graph_builder = StateGraph(AgentState)

llm = ChatOllama(model="deepseek-r1:8b")

def generate(state: AgentState) -> AgentState:
    """
    `generate` 노드는 사용자의 질문을 받아서 응답을 생성하는 노드입니다.
    """
    ai_message: BaseMessage = llm.invoke(state.messages)
    return AgentState(messages=[ai_message])


graph_builder.add_node("generate", generate)

graph_builder.add_edge(START, "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()


message = {
    "messages": [
        SystemMessage("질문이 한국어로 주어질 경우, 한국어로 답변을 생성합니다."),
        HumanMessage("대한민국의 수도는 어디인가요?")
    ]
}

ai_message = graph.invoke(message)

"""
SystemMessage를 통해 한국어 답변 생성을 유도함 -> 항상 한국어로 대답하지 않음.
답변 예시)
<think>
답변 생성을 위한 AI의 생각...
</think>
The capital of South Korea is **Seoul**. <- 최종 답변
"""
print(ai_message["messages"][0].content)
