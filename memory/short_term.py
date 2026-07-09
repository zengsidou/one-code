# -*- coding: utf-8 -*-
"""短期记忆 — Token 计数 + 2层压缩：LLM摘要 + 硬截断"""
from collections import deque
from agent.models import Message
from .token_counter import TokenCounter


class ShortTermMemory:
    def __init__(self, max_tokens: int = 65536, max_messages: int = 200, long_term_store=None):
        self.max_tokens = max_tokens
        self.max_messages = max_messages
        self._messages: deque[Message] = deque()
        self._counter = TokenCounter()
        self._llm = None
        self._long_term_store = long_term_store

    def add(self, message: Message):
        self._messages.append(message)
        self._manage()

    def get_messages(self) -> list[Message]:
        return list(self._messages)

    def __len__(self):
        return len(self._messages)

    def get_token_count(self) -> int:
        return self._counter.count_messages(list(self._messages))

    def get_context(self, limit: int = 30) -> list[Message]:
        return list(self._messages)[-limit:]

    def set_llm(self, llm):
        self._llm = llm

    def clear(self):
        self._messages.clear()

    def _manage(self):
        """2层压缩策略：
        1. 超过 max_tokens 的 70% → LLM 摘要压缩最旧消息
        2. 超过 max_tokens → 硬截断（FIFO pop，保留至少 2 条）
        """
        total = self.get_token_count()
        soft_limit = int(self.max_tokens * 0.7)

        # 层1: LLM 摘要（在 70% 阈值提前触发）
        if total > soft_limit and self._llm and len(self._messages) > 8:
            self._summarize_oldest()

        # 层2: 硬截断
        total = self.get_token_count()
        while total > self.max_tokens and len(self._messages) > 2:
            old = self._messages.popleft()
            total = self.get_token_count()
            while self._messages and self._messages[0].role == "tool":
                self._messages.popleft()
                total = self.get_token_count()

        while len(self._messages) > self.max_messages:
            self._messages.popleft()
            while self._messages and self._messages[0].role == "tool":
                self._messages.popleft()

    def _summarize_oldest(self):
        """LLM 摘要最旧的消息块，释放上下文窗口"""
        msgs = list(self._messages)
        old = msgs[:10]
        if not old:
            return
        summary = self._llm_summarize(old)
        # 替换最旧的 10 条消息为摘要
        for _ in range(len(old)):
            if self._messages:
                self._messages.popleft()
        self._messages.appendleft(summary)

    def _llm_summarize(self, messages: list[Message]) -> Message:
        if not self._llm or len(messages) < 4:
            text = "\n".join(f"{m.role}: {str(m.content or '')[:100]}" for m in messages)
            return Message(role="system", content=f"[摘要] {text[:400]}")

        conversation = []
        for m in messages:
            c = str(m.content or "")[:300]
            if getattr(m, "tool_calls", None):
                c += f" [工具: {', '.join(tc.name for tc in m.tool_calls)}]"
            conversation.append(f"[{m.role}] {c}")

        prompt = (
            "将以下对话历史压缩为结构化摘要:\n\n"
            + "\n".join(conversation[-30:])
            + "\n\n格式: 目标: <用户想要什么> | 已完成: <做了什么> | 发现: <关键发现> | 文件: <涉及的文件>"
        )
        try:
            resp = self._llm.generate(
                [Message(role="user", content=prompt)], tools=None)
            return Message(role="system", content=f"[Compressed] {(resp.content or '')[:500]}")
        except Exception:
            text = "\n".join(f"{m.role}: {str(m.content or '')[:100]}" for m in messages)
            return Message(role="system", content=f"[摘要] {text[:400]}")
