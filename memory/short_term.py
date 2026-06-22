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
        self._llm = None  # 注入 LLM 用于智能压缩

    def add(self, message: Message):
        self._messages.append(message)
        self._manage()

    def get_messages(self) -> list[Message]:
        return list(self._messages)

    def get_token_count(self) -> int:
        return self._counter.count_messages(list(self._messages))

    def _manage(self):
        """管理窗口大小：基于 token 计数分层压缩，优先压缩 tool 链，再压缩旧对话对，最后转移至长期记忆。
        同时动态调整 max_tokens 以缓解上下文过小瓶颈。
        """
        self._adjust_max_tokens()
        total = self.get_token_count()

        # 第一层：如果超限，先压缩 tool 链（通常 token 多且价值低）
        if total > self.max_tokens:
            self._compress_tool_chains()
            total = self.get_token_count()

        # 第二层：如果仍然超限，压缩旧对话对
        if total > self.max_tokens:
            self._summarize_old_pairs()
            total = self.get_token_count()

        # 第三层：如果仍然超限，转移至长期记忆
        if total > self.max_tokens:
            self._transfer_to_long_term()
            total = self.get_token_count()

        # 第四层：如果仍然超限，使用LLM摘要压缩最旧消息块
        if total > self.max_tokens:
            self._aggressive_compress()
            total = self.get_token_count()

        # 第五层：如果仍然超限，裁剪最旧消息（保留至少 2 条）
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

    def _transfer_to_long_term(self):
        """将最旧的消息转移到长期记忆存储，以释放短期记忆空间。
        转移策略：保留最近的 N 条消息（例如 50 条），将更旧的消息打包为摘要并存储到长期记忆。
        """
        if len(self._messages) < 50:
            return
        # 保留最近 50 条消息
        keep_count = 50
        old_messages = list(self._messages)[:-keep_count]
        self._messages = deque(list(self._messages)[-keep_count:])
        # 将旧消息打包为摘要（简单拼接，实际可调用 LLM 生成摘要）
        summary_content = "\n".join(
            f"{msg.role}: {str(msg.content or '')[:200]}" for msg in old_messages
        )
        # 假设存在长期记忆存储模块（需导入或注入）
        # 这里使用一个简单的内存字典模拟，实际应集成到持久化存储
        if not hasattr(self, '_long_term_store'):
            self._long_term_store = []
        self._long_term_store.append({
            'summary': summary_content,
            'token_count': self._counter.count_messages(old_messages)
        })

    def retrieve_from_long_term(self, query: str) -> list[Message]:
        """根据查询从长期记忆中检索相关消息。
        简单实现：返回所有长期记忆摘要作为 system 消息。
        """
        if not hasattr(self, '_long_term_store') or not self._long_term_store:
            return []
        # 将所有摘要合并为一条 system 消息
        combined = "\n---\n".join(item['summary'] for item in self._long_term_store)
        return [Message(role="system", content=f"[长期记忆摘要]\n{combined}")]

    def _adjust_max_tokens(self):
        """根据历史使用模式动态调整最大token数，以缓解上下文过小瓶颈。"""
        if not hasattr(self, '_usage_history'):
            self._usage_history = []
        current_count = self.get_token_count()
        self._usage_history.append(current_count)
        if len(self._usage_history) > 100:
            self._usage_history.pop(0)
        if len(self._usage_history) >= 10:
            avg_usage = sum(self._usage_history[-10:]) / 10
            if avg_usage > self.max_tokens * 0.9 and self.max_tokens < 131072:
                self.max_tokens = min(self.max_tokens * 2, 131072)
            elif avg_usage < self.max_tokens * 0.3 and self.max_tokens > 16384:
                self.max_tokens = max(self.max_tokens // 2, 16384)

    def set_llm(self, llm):
        """注入 LLM 适配器用于智能压缩"""
        self._llm = llm

    def _llm_summarize(self, messages: list[Message]) -> Message:
        """LLM 驱动对话压缩 — 对标 MiMoCode compaction

        将一段对话历史压缩为结构化摘要：
        - 目标(Goal): 用户想要什么
        - 已完成(Accomplished): 做了哪些操作
        - 发现(Findings): 关键发现
        - 相关文件(Files): 涉及的文件
        """
        if not self._llm or len(messages) < 4:
            text = "\n".join(f"{m.role}: {str(m.content or '')[:100]}" for m in messages)
            return Message(role="system", content=f"[对话摘要] {text[:400]}")

        conversation = []
        for m in messages:
            role = m.role
            c = str(m.content or "")[:300]
            if getattr(m, "tool_calls", None):
                c += f" [调用了: {', '.join(tc.name for tc in m.tool_calls)}]"
            conversation.append(f"[{role}] {c}")

        prompt = (
            "将以下对话历史压缩为结构化摘要。只输出摘要，不要解释。\n\n"
            + "\n".join(conversation[-30:])
            + "\n\n压缩格式:\n"
            "目标: <用户想达成什么>\n"
            "已完成: <已经做了哪些操作>\n"
            "发现: <关键发现或错误>\n"
            "涉及文件: <文件列表>"
        )
        try:
            resp = self._llm.generate(
                [Message(role="user", content=prompt)],
                tools=None,
            )
            summary = (resp.content or "")[:600]
            return Message(role="system", content=f"[LLM压缩] {summary}")
        except Exception:
            text = "\n".join(f"{m.role}: {str(m.content or '')[:100]}" for m in messages)
            return Message(role="system", content=f"[对话摘要] {text[:400]}")

    def _aggressive_compress(self):
        """激进压缩：当常规压缩后仍超限时，使用LLM摘要压缩最旧的消息块。"""
        if len(self._messages) < 10:
            return
        # 取最旧的10条消息（跳过system）
        msgs = list(self._messages)
        # 找到第一条非system消息
        start_idx = 0
        for idx, msg in enumerate(msgs):
            if msg.role != "system":
                start_idx = idx
                break
        if start_idx >= len(msgs) - 2:
            return
        # 压缩从start_idx开始的10条消息（或剩余消息的一半，取较小值）
        compress_count = min(10, (len(msgs) - start_idx) // 2)
        if compress_count < 2:
            return
        old_chunk = msgs[start_idx:start_idx + compress_count]
        summary = self._llm_summarize(old_chunk)
        # 替换为摘要
        new_msgs = msgs[:start_idx] + [summary] + msgs[start_idx + compress_count:]
        self._messages = deque(new_msgs)

