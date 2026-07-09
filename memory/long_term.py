# -*- coding: utf-8 -*-
"""长期记忆 — ChromaDB 向量检索 + BM25 关键词回退"""
import hashlib
import time
import math
from collections import Counter

import chromadb
from chromadb.config import Settings


class BM25Scorer:
    """简易 BM25 评分器 — 当 embedding 不可用时做关键词检索"""

    def __init__(self):
        self._docs: list[list[str]] = []
        self._avgdl: float = 0
        self._df: Counter = Counter()
        self._N: int = 0

    def fit(self, documents: list[str], tokenizer):
        self._N = len(documents)
        if self._N == 0:
            return
        self._docs = []
        df_raw: dict[str, set] = {}
        total_len = 0
        for doc in documents:
            tokens = list(tokenizer(doc))
            self._docs.append(tokens)
            total_len += len(tokens)
            seen = set()
            for t in tokens:
                if t not in seen:
                    df_raw.setdefault(t, set()).add(len(self._docs) - 1)
                    seen.add(t)
        self._avgdl = total_len / self._N if self._N > 0 else 1
        self._df = Counter({t: len(idxs) for t, idxs in df_raw.items()})

    def search(self, query: str, tokenizer, top_k: int = 5) -> list[tuple[int, float]]:
        if self._N == 0:
            return []
        query_tokens = list(tokenizer(query))
        scores: list[tuple[int, float]] = []
        k1, b = 1.2, 0.75
        for i, doc_tokens in enumerate(self._docs):
            dl = len(doc_tokens) if doc_tokens else 1
            score = 0.0
            tf = Counter(doc_tokens)
            for qt in query_tokens:
                if qt not in tf:
                    continue
                f = tf[qt]
                df = self._df.get(qt, 0)
                idf = math.log((self._N - df + 0.5) / (df + 0.5) + 1.0)
                numerator = f * (k1 + 1)
                denominator = f + k1 * (1 - b + b * dl / self._avgdl)
                score += idf * numerator / denominator
            if score > 0:
                scores.append((i, score))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


class LongTermMemory:
    def __init__(self, llm, collection_name: str = "agent_memory", persist_dir: str = "./memory_db"):
        self.llm = llm
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._get_or_create()
        self._bm25 = BM25Scorer()
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = self._make_tokenizer()
        return self._tokenizer

    @staticmethod
    def _make_tokenizer():
        try:
            import jieba
            return lambda text: jieba.cut_for_search(text)
        except ImportError:
            return lambda text: text.lower().split()

    def _get_or_create(self):
        try:
            return self._client.get_collection(name=self.collection_name)
        except Exception:
            return self._client.create_collection(name=self.collection_name)

    def store(self, text: str, metadata: dict | None = None):
        if self.llm is None:
            return
        doc_id = hashlib.md5(f"{text}{time.time()}".encode()).hexdigest()[:16]
        embedding = self.llm.embed(text)
        try:
            self._collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata or {}],
            )
        except Exception:
            self._collection = self._get_or_create()
            self._collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata or {}],
            )

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        if self.llm is None:
            return []
        embedding = self.llm.embed(query)

        if all(v == 0.0 for v in embedding):
            return self._keyword_retrieve(query, top_k)

        try:
            results = self._collection.query(query_embeddings=[embedding], n_results=top_k)
            docs = results.get("documents", [[]])[0] or []
            if not docs:
                return self._keyword_retrieve(query, top_k)
            return docs
        except Exception:
            self._collection = self._get_or_create()
            return self._keyword_retrieve(query, top_k)

    def _keyword_retrieve(self, query: str, top_k: int = 5) -> list[str]:
        try:
            stored = self._collection.get()
            docs = stored.get("documents", []) or []
            if not docs:
                return []
            self._bm25.fit(docs, self.tokenizer)
            scored = self._bm25.search(query, self.tokenizer, top_k)
            return [docs[i] for i, _ in scored]
        except Exception:
            return []

    def clear(self):
        try:
            self._client.delete_collection(self._collection.name)
            self._collection = self._client.get_or_create_collection(name=self._collection.name)
        except Exception:
            pass
