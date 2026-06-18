# -*- coding: utf-8 -*-
"""短期记忆 — 准确 Token 计数 + 智能压缩旧消息"""
from collections import deque
from agent.models import Message
from .token_counter import TokenCounter


class ShortTermMemory:
    def __init__(self, max_tokens: int = 65536, max_messages: int = 200):
        self.max_tokens = max_tokens
        self.max_messages = max_messages
        self._messages: deque[Message] = deque()
        self._counter = TokenCounter()

    def add(self, message: Message):
        self._messages.append(message)
        self._manage()

    def get_messages(self) -> list[Message]:
        return list(self._messages)

    def get_token_count(self) -> int:
        return self._counter.count_messages(list(self._messages))

    def _manage(self):
        """管理窗口大小：超过 token 上限先摘要，再裁剪"""
        total = self.get_token_count()

        if total > self.max_tokens:
            self._summarize_old_pairs()

        # 如果摘要后仍然超限，裁剪最旧消息
        total = self.get_token_count()
        while total > self.max_tokens and len(self._messages) > 2:
            old = self._messages.popleft()
            total = self.get_token_count()
            # 级联删除孤立的 tool 消息
            while self._messages and self._messages[0].role == "tool":
                self._messages.popleft()
                total = self.get_token_count()

        # 消息数量上限保护
        while len(self._messages) > self.max_messages:
            self._messages.popleft()
            while self._messages and self._messages[0].role == "tool":
                self._messages.popleft()

    def _summarize_old_pairs(self):
        """智能压缩：把最旧的 user+assistant 对话对压缩为一条摘要消息

        只压缩普通对话对（跳过 tool 链），保留最近的上下文不压缩。
        只在确实能减少 token 时才执行压缩。
        """
        msgs = list(self._messages)
        if len(msgs) < 6:
            return  # 太少，不值得压缩

        # 只在非 tool 链区域找 old_pair
        compression_count = 0
        max_compressions = 5  # 每次最多压缩5对，防止过度

        # 从旧到新扫描
        i = 1  # 跳过第一条（通常是 system）
        while i < len(msgs) - 4 and compression_count < max_compressions:
            a = msgs[i]
            b = msgs[i + 1] if i + 1 < len(msgs) else None

            # 找 user → assistant 对话对（无 tool_calls）
            if (a.role == "user" and b and b.role == "assistant"
                    and not getattr(b, "tool_calls", None)):
                a_text = str(a.content or "")[:150]
                b_text = str(b.content or "")[:200]
                summary_text = f"[对话摘要] 用户: {a_text} | 助手: {b_text}"
                summary = Message(role="system", content=summary_text)

                # 替换：移除旧对，在当前位置插入摘要
                new_msgs = msgs[:i] + [summary] + msgs[i + 2:]
                msgs = new_msgs
                compression_count += 1
                i += 1  # 跳过刚插入的摘要
            else:
                i += 1

        if compression_count > 0:
            self._messages = deque(msgs)

    def clear(self):
        self._messages.clear()

    def __len__(self):
        return len(self._messages)
