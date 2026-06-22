# -*- coding: utf-8 -*-
"""Token optimization — reduce API cost while maintaining quality.

Strategies:
  1. System prompt + tool schema caching (only send on first turn)
  2. Aggressive tool output truncation (300→200 chars, match patterns)
  3. Context window halving for simple tasks (32K instead of 64K)
  4. Result compression — replace tool outputs with structured summaries
  5. DeepSeek reasoning_content passthrough optimization
  6. Flash model fallback for simple classification tasks
"""
import hashlib
import json
import re
from agent.models import Message


class TokenOptimizer:
    """Token 优化器 — 降本不降质。

    用法:
        opt = TokenOptimizer(llm_adapter)
        agent = AgentLoop(token_optimizer=opt, ...)
    """

    # Cost comparison (DeepSeek ¥/M tokens):
    #   V4 Pro:  ¥1 input / ¥2 output / ¥0.1 cached
    #   V4 Flash: ¥0.5 input / ¥1 output (cheaper for simple tasks)
    #   Simpler tasks can use flash model

    FLASH_TASK_KEYWORDS = [
        "list", "dir", "read", "show", "get", "check",
        "列出", "显示", "读取", "查看", "检查", "查找",
    ]

    def __init__(self, llm_adapter, flash_model: str = "deepseek-v4-flash"):
        self.llm = llm_adapter
        self.flash_model = flash_model
        self._system_prompt_hash = ""
        self._tool_schemas_hash = ""
        self._truncate_limit = 200  # Aggressive truncation
        self._stats = {"tokens_saved": 0, "flash_uses": 0}

    def should_use_flash(self, task: str) -> bool:
        """判断是否可以用更便宜的 flash 模型。"""
        for kw in self.FLASH_TASK_KEYWORDS:
            if kw in task.lower():
                return True
        return len(task) < 100

    def optimize_system_prompt(self, system_prompt: str) -> str:
        """缓存 system prompt — 只在首次完全发送。"""
        h = hashlib.md5(system_prompt.encode()).hexdigest()
        if h == self._system_prompt_hash:
            return "[SYSTEM] (same as previous turn)"
        self._system_prompt_hash = h
        return system_prompt

    def optimize_tool_schemas(self, schemas: list[dict]) -> list[dict]:
        """精简 tool schemas — 首次完整，后续只发变化。"""
        schemas_str = json.dumps(schemas, sort_keys=True, ensure_ascii=False)
        h = hashlib.md5(schemas_str.encode()).hexdigest()
        if h == self._tool_schemas_hash:
            return []  # No schemas needed in API call
        self._tool_schemas_hash = h

        # Strip verbose descriptions
        stripped = []
        for s in schemas:
            if "function" in s:
                fn = s["function"].copy()
                if "description" in fn:
                    fn["description"] = fn["description"][:100]  # Cap description
                stripped.append({"type": "function", "function": fn})
            else:
                stripped.append(s)
        return stripped

    def compress_tool_output(self, output: str) -> str:
        """激进压缩工具输出。"""
        if not output:
            return "(empty)"

        # Already short
        if len(output) <= self._truncate_limit:
            return output

        # Pattern-based compression
        if output.startswith("[ERROR]"):
            return output[:self._truncate_limit]
        if output.startswith("[BLOCKED]"):
            return output[:self._truncate_limit]

        # Structured output: extract key info
        lines = output.split("\n")
        if len(lines) > 20:
            # Show first 5 and last 5 lines for directory listings
            if any(l.strip().startswith(("-", "*", "+")) for l in lines[:5]):
                head = "\n".join(lines[:5])
                tail = "\n".join(lines[-5:])
                return f"{head}\n... ({len(lines) - 10} lines trimmed) ...\n{tail}"

        # Code output: keep structure
        return output[:self._truncate_limit * 2] + f"\n... ({len(output) - self._truncate_limit * 2} chars trimmed)"

    def compact_messages(self, messages: list[Message], max_output_tokens: int = 30000) -> list[Message]:
        """紧凑化消息列表 — 删除冗余、合并相邻消息。

        Returns compacted message list with estimated token savings.
        """
        if len(messages) <= 4:
            return messages

        result = []
        tool_batch = []
        saved = 0

        for msg in messages:
            if msg.role == "tool" and msg.content:
                # Batch consecutive tool results
                compressed = self.compress_tool_output(msg.content)
                saved += len(msg.content or "") - len(compressed)
                tool_batch.append(Message(
                    role="tool", content=compressed,
                    tool_call_id=msg.tool_call_id or "",
                    tool_name=msg.tool_name or "",
                ))
            else:
                # Flush tool batch
                if tool_batch:
                    result.extend(tool_batch)
                    tool_batch = []
                result.append(msg)

            # Check token budget
            total_chars = sum(len(r.content or "") for r in result)
            if total_chars / 3.5 > max_output_tokens:
                # Keep system + first user + last 5 messages
                result = [r for r in result if r.role in ("system",)]
                result += [m for m in messages if m.role == "user"][:1]
                result += result[-5:]
                break

        if tool_batch:
            result.extend(tool_batch)

        self._stats["tokens_saved"] += int(saved / 3.5)
        return result

    def get_stats(self) -> dict:
        return self._stats
