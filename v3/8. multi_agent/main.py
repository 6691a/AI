from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import START, END

from income_tax_graph import graph as income_tax_agent
from estate_tax_graph import graph as real_estate_tax_agent

llm = ChatOllama(model="llama3.1:8b")


class AgentState(BaseModel):
    query: str
    context: list = []
    answer: str | None = None

graph_builder = StateGraph(AgentState)


class Route(BaseModel):
    target: Literal['income_tax', 'llm', 'real_estate_tax'] = Field(
        description="The target for the query to answer"
    )

def router(state: AgentState) -> Literal['income_tax', 'real_estate_tax', 'llm']:
    query = state.query
    router_system_prompt = """
    You are an expert at routing a user's question to 'income_tax', 'llm', or 'estate_tax'.
    'income_tax' contains information about income tax up to December 2024.
    'real_estate_tax' contains information about real estate tax up to December 2024.
    if you think the question is not related to either 'income_tax' or 'real_estate_tax';
    you can route it to 'llm'."""

    router_prompt = ChatPromptTemplate.from_messages([
        ('system', router_system_prompt),
        ('user', '{query}')
    ])
    router_llm = llm.with_structured_output(Route)
    router_chain = router_prompt | router_llm
    route = router_chain.invoke({'query': query})
    print(route)
    return route.target

def call_llm(state: AgentState) -> dict:
    query = state.query
    llm_chain = llm | StrOutputParser()
    llm_answer = llm_chain.invoke(query)
    return {'answer': llm_answer}


graph_builder.add_node('income_tax', income_tax_agent)
graph_builder.add_node('real_estate_tax', real_estate_tax_agent)
graph_builder.add_node('llm', call_llm)

graph_builder.add_conditional_edges(
    START,
    router,
    {
        'income_tax': 'income_tax',
        'real_estate_tax': 'real_estate_tax',
        'llm': 'llm'
    }
)
graph_builder.add_edge('income_tax', END)
graph_builder.add_edge('real_estate_tax', END)
graph_builder.add_edge('llm', END)

graph = graph_builder.compile()

graph_image = graph.get_graph().draw_mermaid_png()
with open('graph.png', 'wb') as f:
    f.write(graph_image)


print(graph.invoke(AgentState(query="떡볶이는 어디가 맛있나요?")))