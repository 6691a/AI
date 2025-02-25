from datetime import date
from pprint import pprint

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langgraph.graph import StateGraph, START, END
from langchain import hub
from langchain_community.tools import DuckDuckGoSearchRun

from pydantic import BaseModel

llm = ChatOllama(model="llama3.1:8b")

class AgentState(BaseModel):
    query: str  # 사용자 질문
    answer: str | None = None  # 세율
    tax_base_equation: str | None = None # 과세표준 계산 수식
    tax_deduction: str | None = None # 공제액
    market_ratio: str | None = None # 공정시장가액비율
    tax_base: str | None = None # 과세표준 계산

graph_builder = StateGraph(AgentState)

def create_income_tax_vector_store() -> Chroma:
    file_path = "./estate_tax.txt"
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
        collection_name='estate_tax.txt',
        persist_directory='./estate_tax'
    )
    # 생성
    # store = Chroma.from_documents(
    #     documents=docs,
    #     embedding=embeddings,
    #     collection_name='estate_tax',
    #     persist_directory='./estate_tax',
    # )
    return store


vector_store = create_income_tax_vector_store()
retriever = vector_store.as_retriever(search_kwargs={'k': 3})

def get_tax_base_equation(state: AgentState) -> AgentState:
    prompt = hub.pull('rlm/rag-prompt')
    tax_base_retrieval_chain = (
        {
            "context": retriever,
            'question': RunnablePassthrough(),
        } |
        prompt |
        llm |
        StrOutputParser()
    )
    tax_base_equation_prompt = ChatPromptTemplate.from_messages([
        ('system', '사용자의 질문에서 과세표준을 계산하는 방법을 수식으로 나타내주세요 부연설명 없이 수식만 리턴해주세요'),
        ('human', '{tax_base_equation_information}')
    ])
    tax_base_equation_chain = (
        {
            "tax_base_equation_information": RunnablePassthrough(),
        } |
        tax_base_equation_prompt |
        llm |
        StrOutputParser()
    )
    chain = {"tax_base_equation_information": tax_base_retrieval_chain} | tax_base_equation_chain
    response = chain.invoke("주택에 대한 종합부동산세 계산시 과세표준을 계산하는 방법을 알려주세요")

    return {"tax_base_equation": response}

def get_tax_deduction(state: AgentState) -> AgentState:
    prompt = hub.pull('rlm/rag-prompt')
    tax_deduction_chain = (
        {'context': retriever, 'question': RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    tax_deduction_question = '주택에 대한 종합부동산세 계산시 공제금액을 알려주세요'

    tax_deduction = tax_deduction_chain.invoke(tax_deduction_question)
    return state.model_copy(update={"tax_deduction": tax_deduction})


def get_market_ratio(state: AgentState) -> AgentState:
    search = DuckDuckGoSearchRun()
    q = f"{date.today()}에 해당하는 주택 공시가격 공정시장가액 비울은 몇%인가요?"
    prompt = ChatPromptTemplate.from_messages([
        ("system", f'아래 정보를 기반으로 공정시장 가액비율을 계산해주세요\n\nContext:\n{{context}}'),
        ("human", "{query}",)
    ])
    search_response = search.invoke(q)
    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({"context": search_response, "query": q})

    pprint(f"market_ratio => {response}")
    return {"market_ratio": response}

def calculate_tax_base(state: AgentState) -> AgentState:
    tax_base_equation = state.tax_base_equation
    tax_deduction = state.tax_deduction
    market_ratio = state.market_ratio
    query = state.query

    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', """
            주어진 내용을 기반으로 과세표준을 계산해주세요
        
            과세표준 계산 공식: {tax_base_equation}
            공제금액: {tax_deduction}
            공정시장가액비율: {market_ratio}"""),
            ('human', '사용자 주택 공시가격 정보: {query}')
        ]
    )
    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        'tax_base_equation': tax_base_equation,
        'tax_deduction': tax_deduction,
        'market_ratio': market_ratio,
        'query': query
    })
    return {"tax_base": response}


def calculate_tax_rate(state: AgentState):
    query = state.query
    tax_base = state.tax_base
    tax_rate_calculation_prompt = ChatPromptTemplate.from_messages([
        ('system', '''당신은 종합부동산세 계산 전문가입니다. 아래 문서를 참고해서 사용자의 질문에 대한 종합부동산세를 계산해주세요

    종합부동산세 세율:{context}'''),
        ('human', '''과세표준과 사용자가 소지한 주택의 수가 아래와 같을 때 종합부동산세를 계산해주세요

    과세표준: {tax_base}
    주택 수:{query}''')
    ])

    context = retriever.invoke(query)

    tax_rate_chain = (
        tax_rate_calculation_prompt
        | llm
        | StrOutputParser()
    )

    tax_rate = tax_rate_chain.invoke({
        'context': context,
        'tax_base': tax_base,
        'query': query
    })

    print(tax_rate)
    return state.model_copy(update={"answer": tax_rate})

graph_builder.add_node('get_tax_base_equation', get_tax_base_equation)
graph_builder.add_node('get_tax_deduction', get_tax_deduction)
graph_builder.add_node('get_market_ratio', get_market_ratio)
graph_builder.add_node('calculate_tax_base', calculate_tax_base)
graph_builder.add_node('calculate_tax_rate', calculate_tax_rate)

graph_builder.add_edge(START, 'get_tax_base_equation')
graph_builder.add_edge(START, 'get_tax_deduction')
graph_builder.add_edge(START, 'get_market_ratio')
graph_builder.add_edge('get_tax_base_equation', 'calculate_tax_base')
graph_builder.add_edge('get_tax_deduction', 'calculate_tax_base')
graph_builder.add_edge('get_market_ratio', 'calculate_tax_base')
graph_builder.add_edge('calculate_tax_base', 'calculate_tax_rate')
graph_builder.add_edge('calculate_tax_rate', END)

graph = graph_builder.compile()

graph_image = graph.get_graph().draw_mermaid_png()
with open('graph.png', 'wb') as f:
    f.write(graph_image)
query = '5억짜리 집 1채, 10억짜리 집 1채, 20억짜리 집 1채를 가지고 있을 때 세금을 얼마나 내나요?'
print(graph.invoke(AgentState(query=query)))