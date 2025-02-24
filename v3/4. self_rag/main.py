from pprint import pprint
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
    return state


def generate(state: AgentState) -> AgentState:
    """
    주어진 state를 기반으로 RAG 체인을 사용하여 응답을 생성합니다.

    Args:
        state (AgentState): 사용자의 질문과 문맥을 포함한 에이전트의 현재 state.

    Returns:
        AgentState: 생성된 응답을 포함하는 state를 반환합니다.
    """
    generate_llm = ChatOllama(model="llama3.1:8b", num_predict=100)

    context = state.context
    query = state.query

    prompt = hub.pull("rlm/rag-prompt")

    chain = prompt | generate_llm

    response = chain.invoke({'question': query, 'context': context})
    state.answer = response.content
    return state

def check_doc_relevance(state: AgentState) -> Literal['relevant', 'irrelevant']:
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
    chain = prompt | llm

    response = chain.invoke({'question': query, 'documents': context})

    # ollama기준 response가 None일 경우가 있음 -> 이 경우에는 rewrite로 분기
    if not response or response["Score"] == 0:
        print("END!!")
        return 'irrelevant'
    return 'relevant'

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




def check_hallucination(state: AgentState) -> Literal['hallucinated', 'not hallucinated']:
    hallucination_prompt = PromptTemplate.from_template("""
    You are a teacher tasked with evaluating whether a student's answer is based on documents or not,
    Given documents, which are excerpts from income tax law, and a student's answer;
    If the student's answer is based on documents, respond with "not hallucinated",
    If the student's answer is not based on documents, respond with "hallucinated".

    documents: {documents}
    student_answer: {student_answer}
    """)

    hallucination_llm = ChatOllama(model="llama3.1:8b", temperature=0)
    answer = state.answer
    context = state.context
    context = [doc.page_content for doc in context]
    hallucination_chain = hallucination_prompt | hallucination_llm | StrOutputParser()
    response = hallucination_chain.invoke({'student_answer': answer, 'documents': context})
    return response

# def check_hallucination(state: AgentState) -> Literal['hallucinated', 'not hallucinated']:
#     """
#     주어진 응답이 환각인지 확인합니다.
#
#     Args:
#         state (AgentState): 사용자의 질문과 문맥을 포함한 에이전트의 현재 state.
#
#     Returns:
#         Literal['hallucination', 'not_hallucination']: 응답이 환각인지 여부를 반환합니다.
#     """
#     hallucination_llm = ChatOllama(model="llama3.1:8b", temperature=0)
#     prompt = hub.pull("langchain-ai/rag-answer-hallucination")
#     context = [doc.page_content for doc in state.context]
#     answer = state.answer
#     chain = prompt | hallucination_llm
#
#     response = chain.invoke({'student_answer': answer, 'documents': context})
#
#     if not response or response['Score'] == 0:
#         return 'hallucinated'
#     return 'not hallucinated'

def check_helpfulness_grader(state: AgentState) -> Literal['helpful', 'unhelpful']:
    """
    주어진 응답이 유용한지 확인합니다.

    Args:
        state (AgentState): 사용자의 질문과 문맥을 포함한 에이전트의 현재 state.

    Returns:
        str: 응답의 유용성을 반환합니다.
    """
    prompt = hub.pull("langchain-ai/rag-answer-helpfulness")
    query = state.query
    answer = state.answer
    chain = prompt | llm

    response = chain.invoke({'question': query, 'student_answer': answer})

    if not response or response['Score'] == 0:
        return 'helpful'

    return 'unhelpful'

def check_helpfulness(state: AgentState) -> AgentState:
    """
    유용성을 확인하는 자리 표시자 함수입니다.
    graph에서 conditional_edge를 연속으로 사용하지 않고 node를 추가해
    가독성을 높이기 위해 사용합니다

    Args:
        state (AgentState): 에이전트의 현재 state.

    Returns:
        AgentState: 변경되지 않은 state를 반환합니다.
    """
    # 이 함수는 현재 아무 작업도 수행하지 않으며 state를 그대로 반환합니다
    return state

graph_builder.add_node('retrieve', retrieve)
graph_builder.add_node('generate', generate)
graph_builder.add_node('rewrite', rewrite)
graph_builder.add_node('check_helpfulness', check_helpfulness)


graph_builder.add_edge(START, 'retrieve')
graph_builder.add_conditional_edges(
    'retrieve',
    check_doc_relevance,
    {
        'relevant': 'generate',
        'irrelevant': END
    }
)
graph_builder.add_conditional_edges(
    'generate',
    check_hallucination,
    {
        'not hallucinated': 'check_helpfulness',
        'hallucinated': 'generate'
    }
)

graph_builder.add_conditional_edges(
    'check_helpfulness',
    check_helpfulness_grader,
    {
        'helpful': END,
        'unhelpful': 'rewrite'
    }
)
graph_builder.add_edge('rewrite', 'retrieve')
graph = graph_builder.compile()

graph_image = graph.get_graph().draw_mermaid_png()
with open('graph.png', 'wb') as f:
    f.write(graph_image)

ai_message = graph.invoke(AgentState(query="연봉 5천만원인 거주자가 납부해야 하는 소득세는 얼마인가요?"))
pprint(ai_message)

