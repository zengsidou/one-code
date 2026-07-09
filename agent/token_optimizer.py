# -*- coding: utf-8 -*-
"""Token optimizer — smart tool output compression and work summarization"""
import hashlib
import re
from agent.models import Message


class TokenOptimizer:
    """Token 优化器"""

    KEEP_PATTERNS = [
        (r"\[ERROR\].*", 300),
        (r"\[BLOCKED\].*", 200),
        (r"\[STOPPED\].*", 200),
        (r"(File .*written|已(替换|删除|重命名)).*", 100),
        (r"OK:.*", 100),
        (r"^\s*\d+.*passed.*", 200),
        (r"^\s*\d+.*failed.*", -1),
    ]

    def __init__(self, llm_adapter, flash_model: str = "deepseek-v4-flash"):
        self.llm = llm_adapter
        self.flash_model = flash_model
        self._truncate_limit = 800
        self._stats = {"tokens_saved": 0, "summaries": 0}

    def compress_tool_output(self, output: str) -> str:
        if not output or len(output) <= 400:
            return output

        for pattern, limit in self.KEEP_PATTERNS:
            if re.search(pattern, output, re.IGNORECASE):
                return output if limit == -1 else output[:limit]

        lines = output.split("\n")

        if len(lines) > 30:
            list_markers = sum(1 for l in lines[:20] if re.match(
                r"^\s*[-*+│└├╔═║\[]|^[a-zA-Z]:[\\/]|^\s+\d+:", l.strip()))
            if list_markers >= 5:
                total = len(lines)
                head = lines[:6]
                tail = lines[-6:]
                return "\n".join(head) + f"\n... [{total - 12} lines] ...\n" + "\n".join(tail)

        if len(output) > 2000 and any(kw in output[:200].lower() for kw in ["diff", "import ", "class ", "def "]):
            if len(lines) > 30:
                head = "\n".join(lines[:12])
                tail = "\n".join(lines[-8:])
                return f"{head}\n... [{len(lines) - 20} lines] ...\n{tail}"

        if len(output) > self._truncate_limit:
            return output[:self._truncate_limit // 2] + f"\n... [截断 {len(output) - self._truncate_limit} 字符] ...\n" + output[-200:]

        return output

    def summarize_work(self, messages: list[Message], task: str) -> str:
        files_touched = []
        errors = []
        last_result = ""
        step_count = 0

        for msg in messages:
            if msg.role == "user":
                step_count += 1
            elif msg.role == "tool" and msg.content:
                ct = msg.content
                for m in re.finditer(r"(?:File written:|已替换|修复|已删除|OK:)\s*(\S+)", ct):
                    files_touched.append(m.group(1))
                if "[ERROR]" in ct or "[STOPPED]" in ct:
                    errors.append(ct[:120])
                last_result = ct

        files_uniq = list(dict.fromkeys(files_touched))[-10:]
        summary = f"[进度] {task[:120]}\n  步骤: {step_count}\n"
        if files_uniq:
            summary += f"  文件: {', '.join(files_uniq)}\n"
        if errors:
            summary += f"  错误: {errors[-1][:100]}\n"
        if last_result and not errors:
            summary += f"  最新: {last_result[:200]}\n"

        self._stats["summaries"] += 1
        self._stats["tokens_saved"] += int(len("\n".join(m.content or "" for m in messages[-20:])) / 3.5 - len(summary) / 3.5)
        return summary

    def get_stats(self) -> dict:
        return self._stats
