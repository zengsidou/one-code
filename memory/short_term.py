# -*- coding: utf-8 -*-
"""短期记忆 — Token 感知滑动窗口"""
from collections import deque
from agent.models import Message

_ZH_RANGE = ("\u4e00", "\u9fff")


class ShortTermMemory:
    def __init__(self, max_tokens: int = 4096):
        self.max_tokens = max_tokens
        self._messages: deque[Message] = deque()

    def add(self, message: Message):
        self._messages.append(message)
        self._trim()

    def get_messages(self) -> list[Message]:
        return list(self._messages)

    def _token_count(self, text: str) -> int:
        total = 0
        for ch in text:
            if "\u4e00" <= ch <= "\u9fff":
                total += 2
            elif ch.isascii() and ch.isalpha():
                total += 1
            else:
                total += 1
        return total

    def _trim(self):
        total = sum(self._token_count(str(m.content or "")) for m in self._messages)
        while total > self.max_tokens and len(self._messages) > 2:
            old = self._messages.popleft()
            total -= self._token_count(str(old.content or ""))
            # 确保不留下孤立的 tool 消息（前一个带 tool_calls 的 assistant 已被裁剪）
            while self._messages and self._messages[0].role == "tool":
                orphan = self._messages.popleft()
                total -= self._token_count(str(orphan.content or ""))

    def clear(self):
        self._messages.clear()

    def __len__(self):
        return len(self._messages)
