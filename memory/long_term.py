# -*- coding: utf-8 -*-
"""长期记忆 — ChromaDB 向量检索 + Ollama bge-m3 Embedding"""
import hashlib
import time

import chromadb
from chromadb.config import Settings

from llm.ollama import OllamaClient


class LongTermMemory:
    def __init__(self, llm: OllamaClient, collection_name: str = "agent_memory", persist_dir: str = "./memory_db"):
        self.llm = llm
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def store(self, text: str, metadata: dict | None = None):
        doc_id = hashlib.md5(f"{text}{time.time()}".encode()).hexdigest()[:16]
        embedding = self.llm.embed(text)
        self._collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata or {}],
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        embedding = self.llm.embed(query)
        results = self._collection.query(query_embeddings=[embedding], n_results=top_k)
        return results.get("documents", [[]])[0] or []

    def clear(self):
        try:
            self._client.delete_collection(self._collection.name)
            self._collection = self._client.get_or_create_collection(name=self._collection.name)
        except Exception:
            pass
