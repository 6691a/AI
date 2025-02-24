
from pydantic import BaseModel
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain import hub
from langgraph.graph import StateGraph, START, END
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

"""
LangGraph 맛보기
"""

def create_paik_dabang_vector_store():
    file_path = "./빽다방.txt"

    loader = TextLoader(file_path).load()
    spliter = RecursiveCharacterTextSplitter(chunk_size=140, chunk_overlap=0,  separators=["\n\n", "\n"])

    docs = spliter.split_documents(loader)

    embeddings = OllamaEmbeddings(
        model="snowflake-arctic-embed2:latest",
    )

    try:
        vector_store = Chroma(
            embedding_function=embeddings,
            collection_name='paik_dabang',
            persist_directory='./paik_dabang'
        )
    except Exception as e:
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name = 'paik_dabang',
            persist_directory = './paik_dabang',
        )
    return vector_store

def create_income_tax_vector_store(vector_store: Chroma):
    file_path = "./income_tax.txt"
    text_splitter = RecursiveCharacterTextSplitter( chunk_size = 1500, chunk_overlap = 100)
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter)

    embeddings = OllamaEmbeddings(
        model="snowflake-arctic-embed2:latest",
    )
    vector_store = Chroma(
        embedding_function=embeddings,
        collection_name='income_tax',
        persist_directory='./income_tax',
    )
    vector_store.add_documents(docs)

vector_store = create_paik_dabang_vector_store()
create_income_tax_vector_store(vector_store)


retriever = vector_store.as_retriever(search_kwargs={'k': 3})
"""
page_content='5. 빽사이즈 원조커피(ICED)
  - 빽다방의 BEST메뉴를 더 크게 즐겨보세요 :) [주의. 564mg 고카페인으로 카페인에 민감한 어린이, 임산부는 섭취에 주의바랍니다]
  - 가격: 4000원' metadata={'source': './빽다방.txt'}
page_content='6. 빽사이즈 원조커피 제로슈거(ICED)
  - 빽다방의 BEST메뉴를 더 크게, 제로슈거로 즐겨보세요 :) [주의. 686mg 고카페인으로 카페인에 민감한 어린이, 임산부는 섭취에 주의바랍니다]
  - 가격: 4000원' metadata={'source': './빽다방.txt'}
page_content='10. 빽사이즈 초코라떼(ICED)
  - 빽다방의 BEST메뉴를 더 크게 즐겨보세요 :) 진짜~완~전 진한 초코라떼
  - 가격 : 5500원' metadata={'source': './빽다방.txt'}
"""
# for i in retriever.invoke("빽다방에서 높은 카페인 순서"):
#     print(i)


# for i in retriever.invoke("연봉 5천만원 직장인의 소득세는?"):
#     print(i)

class AgentState(BaseModel):
    query: str
    context: list[Document] = []
    answer: str | None = None


graph_builder = StateGraph(AgentState)

llm = ChatOllama(model="deepseek-r1:14b")

def retrieve(state: AgentState) -> AgentState:
    """
    사용자의 질문에 기반하여 벡터 스토어에서 관련 문서를 검색합니다.
    """
    query = state.query
    docs = retriever.invoke(query)
    return AgentState(query=query, context=docs, answer=None)



def generate(state: AgentState) -> AgentState:
    """
    `generate` 노드는 사용자의 질문을 받아서 응답을 생성하는 노드입니다.
    """
    context = state.context
    query = state.query
    prompt = hub.pull("rlm/rag-prompt")
    """ 이런 형식의 템플릿
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Question: {question}
    Context: {context}
    Answer:
    """

    rag_chain = prompt | llm
    response = rag_chain.invoke({
        "context": context,
        "question": query
    })
    state.answer = response.content
    return state



graph_builder.add_node("generate", generate)
graph_builder.add_node("retrieve", retrieve)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)
graph = graph_builder.compile()

# 다른 방법
# sequence_graph_builder = StateGraph(AgentState).add_sequence([retrieve, generate])
# sequence_graph_builder.add_edge(START, 'retrieve')
# sequence_graph_builder.add_edge('generate', END)
# sequence_graph = sequence_graph_builder.compile()

ai_message = graph.invoke(AgentState(query="빽다방에서 높은 카페인 순서는?"))

print(ai_message["answer"])


ai_message = graph.invoke(AgentState(query="연봉 5천만원 직장인의 소득세는?"))

print(ai_message["answer"])