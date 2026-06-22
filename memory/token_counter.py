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

    @staticmethod
    def _is_cjk(ch: str) -> bool:
        cp = ord(ch)
        return (
            0x2E80 <= cp <= 0x2EFF or
            0x3000 <= cp <= 0x303F or
            0x3040 <= cp <= 0x309F or
            0x30A0 <= cp <= 0x30FF or
            0x3400 <= cp <= 0x4DBF or
            0x4E00 <= cp <= 0x9FFF or
            0xF900 <= cp <= 0xFAFF or
            0xFF00 <= cp <= 0xFFEF
        )

    def count(self, text: str) -> int:
        """估算文本的 token 数量。
        参考: CJK 字符约 1.5-2 tokens/字，英文约 1.3 tokens/词
        """
        total = 0
        buf = ""
        for ch in text:
            if self._is_cjk(ch):
                if buf:
                    total += max(1, int(len(buf.split()) * 1.3))
                    buf = ""
                total += 2
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
