# -*- coding: utf-8 -*-
"""准确 Token 计数器 — tiktoken 精确计数 + 词数估算 fallback"""
import re


class TokenCounter:
    """Token 计数工具

    - 优先使用 tiktoken（OpenAI 标准库，128K+ 模型准确）
    - tiktoken 不可用时使用词数估算 fallback
    - 支持中文/英文混合文本
    """

    def __init__(self, model: str = "gpt-4"):
        self._encoder = None
        self._model = model
        try:
            import tiktoken
            self._encoder = tiktoken.encoding_for_model(model)
        except Exception:
            self._encoder = None

    def count(self, text: str) -> int:
        """计算文本的 token 数"""
        if not text:
            return 0
        if self._encoder:
            try:
                return len(self._encoder.encode(text))
            except Exception:
                return self._estimate(text)
        return self._estimate(text)

    @staticmethod
    def _estimate(text: str) -> int:
        """词数估算 fallback

        英文：按空格分词（1 word ≈ 1.3 tokens）
        中文：每个字符 ≈ 1.5-2 tokens
        """
        total = 0
        buf = ""
        for ch in text:
            if "\u4e00" <= ch <= "\u9fff" or "\u3000" <= ch <= "\u303f":
                # CJK character: flush buffer, count char
                if buf:
                    total += max(1, int(len(buf.split()) * 1.3))
                    buf = ""
                total += 1
            elif ch.isspace():
                buf += " "
            else:
                buf += ch
        if buf:
            total += max(1, int(len(buf.split()) * 1.3))
        return total

    def count_messages(self, messages: list) -> int:
        """计算消息列表的总 token 数，额外计入 role/format 开销"""
        total = 0
        for m in messages:
            # role + 格式开销 ~4 tokens
            total += 4
            content = getattr(m, "content", "") or ""
            total += self.count(str(content))
            if getattr(m, "tool_calls", None):
                import json
                for tc in m.tool_calls:
                    total += self.count(tc.name)
                    total += self.count(json.dumps(getattr(tc, "arguments", {}), ensure_ascii=False))
        return total
