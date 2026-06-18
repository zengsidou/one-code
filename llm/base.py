# -*- coding: utf-8 -*-
"""LLM 抽象基类"""
from abc import ABC, abstractmethod
from agent.models import Message


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, messages: list[Message], tools: list[dict] | None = None) -> Message:
        """生成回复，支持工具调用"""
        ...

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """生成文本 embedding"""
        ...
