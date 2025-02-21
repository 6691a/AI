from enum import Enum
from typing import AsyncIterator
from uuid import uuid4

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, BaseMessageChunk
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from model import PaikdabangMenuDocument
from config import settings


class MessageType(str, Enum):
    AI = "ai"
    SYSTEM = "system"


class StreamResponse(BaseModel):
    message_id: str
    message: str
    message_type: MessageType


class PaikdabangAI:
    async def aget_response(self, query: str, session: AsyncSession, stream: bool = False) -> AsyncIterator[BaseMessageChunk] | BaseMessage:
        embed_query = PaikdabangMenuDocument.embeddings(query)
        queryset = (
            select(PaikdabangMenuDocument)
            .order_by(PaikdabangMenuDocument.embedding.cosine_distance(embed_query))
            .limit(4)
        )
        items = await session.execute(queryset)

        search_docs_str = "\n\n".join(item.page_content for item in items.scalars())
        llm = ChatOllama(model=settings.LLM_MODEL)

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

    async def astream(self, query: str, session: AsyncSession) -> AsyncIterator[StreamResponse]:
        response: AsyncIterator[BaseMessageChunk] = await self.aget_response(query, session, stream=True)

        message_id = uuid4().hex
        async for chunk in response:
            chunk: BaseMessageChunk
            yield StreamResponse(
                message_id=message_id,
                message=chunk.content,
                message_type=MessageType.AI,
            )

