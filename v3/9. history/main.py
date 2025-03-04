from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import RemoveMessage, SystemMessage, HumanMessage
from langgraph.graph import MessagesState
llm = ChatOllama(model="llama3.1:8b")


class AgentState(MessagesState):
    summary: str

graph_builder = StateGraph(AgentState)


def agent(state: AgentState) -> AgentState:
    messages = state['messages']
    summary = state['summary']

    if summary != '':
        messages = [SystemMessage(content=f'Here is the summary of the earlier conversation: {summary}')] + messages

    response = llm.invoke(messages)

    return {'messages': [response]}


def summarize_messages(state: AgentState) -> AgentState:
    messages = state['messages']
    summary = state['summary']

    summary_prompt = f'summarize this chat history below: \n\nchat_history:{messages}'

    if summary != '':
        summary_prompt = f'''summarize this chat history below while looking at the summary of earlier conversations
chat_history:{messages}
summary:{summary}'''

    summary = llm.invoke(summary_prompt)

    response = llm.invoke(messages)

    # 요약된 메시지를 반환합니다.
    return {'summary': response.content}


def delete_messages(state: AgentState) -> AgentState:
    messages = state['messages']
    delete_messages = [RemoveMessage(id=message.id) for message in messages[:-3]]
    return {'messages': delete_messages}


graph_builder.add_node('agent', agent)
graph_builder.add_node('delete_messages', delete_messages)
graph_builder.add_node('summarize_messages', summarize_messages)

graph_builder.add_edge(START, "agent")
graph_builder.add_edge('agent', "summarize_messages")
graph_builder.add_edge('summarize_messages', 'delete_messages')
graph_builder.add_edge('delete_messages', END)

checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

# graph_image = graph.get_graph().draw_mermaid_png()
# with open('graph.png', 'wb') as f:
#     f.write(graph_image)

config = {
    "configurable": {
        "thread_id": "user_0"
    }
}

query = "점심 매뉴 추천해주세요"
for chunk in graph.stream({'messages': [HumanMessage(query)], 'summary': ''}, config=config, stream_mode='values'):
    if chunk["messages"]:
        print(chunk['messages'][-1].pretty_print())

update_query = "답변 중 가장 마음에 드는 것을 골라주세요"
for chunk in graph.stream({'messages': [HumanMessage(update_query)]}, config=config, stream_mode='values'):
    if chunk["messages"]:
        print(chunk['messages'][-1].pretty_print())


