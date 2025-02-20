import pickle
import numpy as np

import ollama
from typing import Self

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from ollama._types import EmbeddingsResponse
from sklearn.metrics.pairwise import cosine_similarity

class VectorList(list):
    def __init__(self, model: str = "snowflake-arctic-embed2:latest"):
        self.mode = model

    def _split(self, doc: list[Document]) -> list:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=140,  # 문서를 나눌 최소 글자 수 (디폴트: 4000)
            chunk_overlap=0,  # 문서를 나눌 때 겹치는 글자 수 (디폴트: 200)
        )
        new_text_list = text_splitter.split_documents(doc)
        return new_text_list

    def _read_doc(self, file_path: str) -> list[Document]:
        with open(file_path, "r") as file:
            return [Document(page_content=file.read())]

    def _embedding(self, doc: Document) -> EmbeddingsResponse:
        response: EmbeddingsResponse = ollama.embeddings(
            model="snowflake-arctic-embed2:latest",
            prompt=doc.page_content,
        )
        return response

    def make(self, file_path: str) -> Self:
        docs = self._read_doc(file_path)
        split_docs = self._split(docs)

        for doc in split_docs:
            response: EmbeddingsResponse = self._embedding(doc)

            self.append(
                {
                    "text": doc.model_copy(),
                    "vector": response["embedding"],
                }
            )

        return self

    def save(self, file_path: str):
        with open(file_path, "wb") as file:
            pickle.dump(self, file)

    def load(self, file_path: str) -> Self:
        with open(file_path, "rb") as file:
            return pickle.load(file)

    def search(self, query: str, k: int = 4) -> list[Document]:
        query_vector = self._embedding(Document(page_content=query))["embedding"]
        embedding_list = [vector["vector"] for vector in self]

        similarities = cosine_similarity([query_vector], embedding_list)[0]

        top_k = np.argsort(similarities)[::-1][:k]

        return [
            self[i]["text"].model_copy()
            for i in top_k
        ]

