from typing import Literal

from pydantic import BaseModel
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langchain import hub


def create_income_tax_vector_store() -> Chroma:
    file_path = "./income_tax.txt"
    text_splitter = RecursiveCharacterTextSplitter( chunk_size = 1500, chunk_overlap = 100)
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter)

    embeddings = OllamaEmbeddings(
        model="snowflake-arctic-embed2:latest",
    )
    try:
        store = Chroma(
            embedding_function=embeddings,
            collection_name='income_tax',
            persist_directory='./income_tax'
        )
    except Exception as e:
        store = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name='income_tax',
            persist_directory='./income_tax',
        )
    return store


vector_store = create_income_tax_vector_store()

retriever = vector_store.as_retriever(search_kwargs={'k': 3})
# print(retriever.invoke("연봉 5천만원 직장인의 소득세는?"))

llm = ChatOllama(model="llama3.2")

class AgentState(BaseModel):
    query: str
    context: list[Document] = []
    answer: str | None = None


graph_builder = StateGraph(AgentState)

def retrieve(state: AgentState) -> AgentState:
    """
    사용자의 질문에 기반하여 벡터 스토어에서 관련 문서를 검색합니다.
    """
    query = state.query
    docs = retriever.invoke(query)
    state.context = docs
    print(f"retrieve => {state}")
    return state


def generate(state: AgentState) -> AgentState:
    """
    주어진 state를 기반으로 RAG 체인을 사용하여 응답을 생성합니다.

    Args:
        state (AgentState): 사용자의 질문과 문맥을 포함한 에이전트의 현재 state.

    Returns:
        AgentState: 생성된 응답을 포함하는 state를 반환합니다.
    """
    context = state.context
    query = state.query
    prompt = hub.pull("rlm/rag-prompt")

    chain = prompt | llm

    response = chain.invoke({'question': query, 'context': context})
    state.answer = response.content
    print(f"generate => {state}")
    return state

def check_doc_relevance(state: AgentState) -> Literal['generate', 'rewrite']:
    """
    주어진 문서가 질문에 관련이 있는지 확인합니다.

    score가 1이면 generate로, 그렇지 않으면 rewrite로 분기합니다.

    Args:
        state (AgentState): 사용자의 질문과 문맥을 포함한 에이전트의 현재 state.

    Returns:
        Literal['generate', 'rewrite']: 문서의 관련성에 따라 generate 또는 rewrite로 분기합니다.
    """
    prompt = hub.pull("langchain-ai/rag-document-relevance")
    context = state.context
    query = state.query
    print(f"check_doc_relevance context", context)
    print("query", query)
    chain = prompt | llm

    response = chain.invoke({'question': query, 'documents': context})
    print("check_doc_relevance response", response)


    # ollama기준 response가 None일 경우가 있음 -> 이 경우에는 rewrite로 분기
    if not response:
        return 'rewrite'

    score = response['Score']
    if score == 1:
        return 'generate'
    return 'rewrite'

def rewrite(state: AgentState) -> AgentState:
    """
    사용자의 질문을 사전을 참고하여 변경합니다.

    Args:
        state (AgentState): 사용자의 질문을 포함한 에이전트의 현재 state.

    Returns:
        AgentState: 변경된 질문을 포함하는 state를 반환합니다.
    """
    print(f"rewrite => {state}")
    dictionary = ['사람과 관련된 표현 -> 거주자']

    prompt = PromptTemplate.from_template(f"""
    사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요
    사전: {dictionary}
    질문: {{query}}
    """)
    query = state.query

    chain = prompt | llm | StrOutputParser()

    response: str = chain.invoke({'query': query})
    state.query = response
    return state


graph_builder.add_node('retrieve', retrieve)
graph_builder.add_node('generate', generate)
graph_builder.add_node('rewrite', rewrite)


graph_builder.add_edge(START, 'retrieve')
graph_builder.add_conditional_edges('retrieve', check_doc_relevance)
graph_builder.add_edge('rewrite', 'retrieve')
graph_builder.add_edge('generate', END)

graph = graph_builder.compile()


# print(graph.get_graph().draw_mermaid())

ai_message = graph.invoke(AgentState(query="연봉 5천만원 직장인의 소득세는?"))

print(ai_message["answer"])