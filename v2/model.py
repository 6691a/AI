import tiktoken
import ollama

from ollama._types import EmbeddingsResponse, EmbedResponse
from langchain_core.documents import Document
from sqlalchemy import event
from sqlmodel import Field, SQLModel, Index
from pgvector.sqlalchemy import Vector, HALFVEC
from typing import Any, Sequence
from sqlmodel import JSON
from pydantic import field_validator
from config import settings

class Item(SQLModel, table=True):
    id: int = Field(primary_key=True)
    embedding: Any = Field(sa_type=Vector(3))

    __table_args__ = (
        Index(
            'item_embedding_idx',
            'embedding',
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={'embedding': 'vector_cosine_ops'}
        ),
    )

class Document(SQLModel):
    id: int | None = Field(primary_key=True, default=None)
    embedding: Any | None = Field(sa_type=Vector(settings.EMBEDDING_DIMENSIONS), nullable=True, default=None)
    page_content: str
    meta_data: dict | None = Field(sa_type=JSON, nullable=True, default=None)

    @classmethod
    def embeddings(cls, input: str) -> list[float]:
        response: EmbeddingsResponse = ollama.embeddings(
            model=settings.EMBEDDING_MODEL,
            prompt=input,
        )
        return response["embedding"]

    @classmethod
    def embed(cls, input: Sequence[str] | str) -> list[float]:
        response: EmbedResponse = ollama.embed(
            model=settings.EMBEDDING_MODEL,
            input=input,
        )
        return response["embeddings"]

    @field_validator('page_content', mode='before')
    @classmethod
    def validate_page_content(cls, value: str) -> str:
        encoding: tiktoken.Encoding = tiktoken.encoding_for_model(settings.TOKEN_EMBEDDING_MODEL)
        num_tokens = len(encoding.encode(value or ""))
        if num_tokens >= settings.MAX_TOKENS:
            raise ValueError(f"Number of tokens {num_tokens} exceeds the maximum of {settings.MAX_TOKENS}")
        return value

    @classmethod
    def load(cls, file_path: str) -> list[Document]:
        with open(file_path, "r") as file:
            return [
                Document(
                    page_content=file.read(),
                    metadata={"source": file_path}
                )
            ]

    @classmethod
    def split(cls, docs: list[Document]) -> list[Document]:
        new_doc_list = []
        for doc in docs:
            for new_page_content in doc.page_content.split("\n\n"):
                new_doc_list.append(
                    Document(
                        metadata=doc.metadata.copy(),
                        page_content=new_page_content,
                    )
                )
        return new_doc_list

class PaikdabangMenuDocument(Document, table=True):
    __table_args__ = (
        Index(
            'paikdabang_menu_document_embedding_idx',
            'embedding',
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={'embedding': 'vector_cosine_ops'}
        ),
    )

@event.listens_for(PaikdabangMenuDocument, 'before_insert')
def before_insert(mapper, connection, target):
    if target.embedding is None:
        target.embedding = PaikdabangMenuDocument.embeddings(target.page_content)

@event.listens_for(PaikdabangMenuDocument, 'before_update')
def before_update(mapper, connection, target):
    # page_content가 변경되었을 때만 임베딩을 다시 생성
    if target.page_content:
        target.embedding = PaikdabangMenuDocument.embeddings(target.page_content)



class TaxLawDocument(Document, table=True):
    embedding: Any | None = Field(sa_type=HALFVEC(settings.EMBEDDING_DIMENSIONS), nullable=True, default=None)

    __table_args__ = (
        Index(
            'tax_law_document_embedding_idx',
            'embedding',
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={'embedding': 'halfvec_cosine_ops'}
        ),
    )