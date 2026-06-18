# -*- coding: utf-8 -*-
"""记忆管理"""
from .short_term import ShortTermMemory
from .long_term import LongTermMemory
from .token_counter import TokenCounter
from agent.models import Message


class MemoryManager:
    def __init__(self, short: ShortTermMemory, long: LongTermMemory):
        self.short_term = short
        self.long_term = long

    def add_message(self, msg):
        self.short_term.add(msg)
        if msg.content and msg.role in ("user", "assistant"):
            self.long_term.store(msg.content, {"role": msg.role})

    def get_context(self, query: str = "", recall_n: int = 3):
        messages = self.short_term.get_messages()
        if query and recall_n > 0:
            recalled = self.long_term.retrieve(query, top_k=recall_n)
            context = "相关历史记忆:\n" + "\n---\n".join(recalled)
            return [Message(role="system", content=context)] + messages
        return messages

    def clear(self):
        self.short_term.clear()
        self.long_term.clear()
