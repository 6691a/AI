from enum import Enum
from typing import AsyncIterator
from uuid import uuid4

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, BaseMessageChunk
from pydantic import BaseModel

from vector import VectorList

class MessageType(str, Enum):
    AI = "ai"
    SYSTEM = "system"


class StreamResponse(BaseModel):
    message_id: str
    message: str
    message_type: MessageType


class PaikdabangAI:
    def __init__(self):
        try:
            self.vectors = VectorList().load("pickle")
        except FileNotFoundError:
            self.vectors = VectorList().make("빽다방.txt")
            self.vectors.save("pickle")

    async def aget_response(self, query: str, stream: bool = False) -> AsyncIterator[BaseMessageChunk] | BaseMessage:
        search_docs = self.vectors.search(query)

        search_docs_str = [doc.page_content for doc in search_docs]
        llm = ChatOllama(model='exaone3.5:7.8b')

        if stream:
            return llm.astream(f"""
            사전지식: {search_docs_str}
            질문: {query}
            """)

        return await llm.ainvoke(f"""
        사전지식: {search_docs_str}
        질문: {query}
        """)

    async def ainvoke(self, query: str) -> str:
        response: BaseMessage = await self.aget_response(query, stream=False)
        return response.content

    async def astream(self, query: str) -> AsyncIterator[StreamResponse]:
        response: AsyncIterator[BaseMessageChunk] = await self.aget_response(query, stream=True)

        message_id = uuid4().hex
        async for chunk in response:
            chunk: BaseMessageChunk
            yield StreamResponse(
                message_id=message_id,
                message=chunk.content,
                message_type=MessageType.AI,
            )

