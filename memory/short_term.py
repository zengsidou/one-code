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
        """管理窗口大小：基于 token 计数分层压缩，优先压缩 tool 链，再压缩旧对话对"""
        total = self.get_token_count()

        # 第一层：如果超限，先压缩 tool 链（通常 token 多且价值低）
        if total > self.max_tokens:
            self._compress_tool_chains()
            total = self.get_token_count()

        # 第二层：如果仍然超限，压缩旧对话对
        if total > self.max_tokens:
            self._summarize_old_pairs()
            total = self.get_token_count()

        # 第三层：如果仍然超限，裁剪最旧消息（保留至少 2 条）
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

    def _compress_tool_chains(self):
        """压缩 tool 链：将 assistant[tool_calls] + 后续 tool 消息合并为摘要。
        
        保留 DeepSeek API 兼容性：tool 链必须完整移除（assistant+tool_calls 与其 tool 响应一起），
        避免留下孤立的 tool 消息。
        """
        msgs = list(self._messages)
        if len(msgs) < 4:
            return

        compressed = 0
        i = 2  # 跳过 system + 第一条 user
        while i < len(msgs) - 1 and compressed < 3:
            msg = msgs[i]
            if msg.role == "assistant" and getattr(msg, "tool_calls", None):
                j = i + 1
                tool_summaries = []
                while j < len(msgs) and msgs[j].role == "tool":
                    tc = msgs[j].content or ""
                    tool_summaries.append(tc[:150])
                    j += 1
                if tool_summaries:
                    combined = " | ".join(tool_summaries)
                    summary = Message(
                        role="system",
                        content=f"[工具结果摘要] {combined[:400]}"
                    )
                    msgs = msgs[:i] + [summary] + msgs[j:]
                    compressed += 1
                i += 1
            else:
                i += 1

        if compressed > 0:
            self._messages = deque(msgs)

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
