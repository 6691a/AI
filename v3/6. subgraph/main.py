from pprint import pprint
from typing import Literal

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langchain import hub
from langchain_community.tools import DuckDuckGoSearchRun

from income_tax_graph import graph as income_tax_graph

def create_income_tax_vector_store() -> Chroma:
    file_path = "./income_tax.txt"
    text_splitter = text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap = 100,
        separators=['\n\n', '\n']
    )
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter)

    embeddings = OllamaEmbeddings(
        model="snowflake-arctic-embed2:latest",
    )

    store = Chroma(
        embedding_function=embeddings,
        collection_name='income_tax',
        persist_directory='./income_tax'
    )
    # 생성
    # store = Chroma.from_documents(
    #     documents=docs,
    #     embedding=embeddings,
    #     collection_name='income_tax',
    #     persist_directory='./income_tax',
    # )
    return store


vector_store = create_income_tax_vector_store()

retriever = vector_store.as_retriever(search_kwargs={'k': 3})
# print(retriever.invoke("연봉 5천만원 거주자의 종합소득는?"))

llm = ChatOllama(model="llama3.1:8b")

class AgentState(BaseModel):
    query: str
    context: list = []
    answer: str | None = None


graph_builder = StateGraph(AgentState)

def web_search(state: AgentState) -> AgentState:
    """
    사용자의 질문을 웹 검색을 통해 답변을 찾습니다.
    """
    pprint("running web search")
    search = DuckDuckGoSearchRun()
    query = state.query
    pprint(f"query = {query}")
    response = search.invoke(query)
    pprint(f"response = {response}")
    state.context = [response]
    return state


def web_generate(state: AgentState) -> AgentState:
    """
    웹 검색을 통해 찾은 답변을 기반으로 응답을 생성합니다.
    """
    pprint("running web generate")
    prompt = hub.pull("rlm/rag-prompt")
    context = state.context
    query = state.query

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({'question': query, 'context': context})
    pprint(f"response => {response}")
    state.answer = response
    return state

def basic_generate(state: AgentState) -> AgentState:
    query = state.query

    chain = llm | StrOutputParser()

    response = chain.invoke(query)

    state.answer = response

    return state

class Route(BaseModel):
    target: Literal['vector_store', 'llm', 'web_search'] = Field(
        description="The target for the query to answer"
    )

def router(state: AgentState) -> Literal['vector_store', 'llm', 'web_search']:
    system_prompt = """
    You are an expert at routing a user's question to 'vector_store', 'llm', or 'web_search'.
    'vector_store' contains information about income tax up to December 2024.
    if you think the question is simple enough use 'llm'
    if you think you need to search the web to answer the question use 'web_search'
    """
    pprint("running route")
    router_prompt = ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        ('user', '{query}')
    ])

    router_llm = ChatOllama(model="llama3.2")
    structured_router_llm = router_llm.with_structured_output(Route)
    query = state.query
    chain = router_prompt | structured_router_llm

    route = chain.invoke({'query': query})
    print(route)
    return route.target


graph_builder.add_node("web_search", web_search)
graph_builder.add_node("web_generate", web_generate)
graph_builder.add_node("basic_generate", basic_generate)
graph_builder.add_node("income_tax_agent", income_tax_graph)


graph_builder.add_conditional_edges(
    START,
    router,
    {
        'vector_store': 'income_tax_agent',
        'llm': 'basic_generate',
        'web_search': 'web_search'
    }
)
graph_builder.add_edge("web_search", "web_generate")
graph_builder.add_edge('web_generate', END)
graph_builder.add_edge('basic_generate', END)
graph_builder.add_edge('income_tax_agent', END)

graph = graph_builder.compile()

graph_image = graph.get_graph().draw_mermaid_png()
with open('graph.png', 'wb') as f:
    f.write(graph_image)

ai_message = graph.invoke(AgentState(query="Obama's first name?"))
pprint(ai_message)
