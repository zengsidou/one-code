# -*- coding: utf-8 -*-
"""Token optimization — context compression and smart truncation"""
import hashlib
import json
import re
from agent.models import Message


class TokenOptimizer:
    """Token 优化器 — 降本不降质"""

    FLASH_TASK_KEYWORDS = [
        "list", "dir", "read", "show", "get", "check",
        "列出", "显示", "读取", "查看", "检查", "查找",
    ]

    # 关键信息保留模式 (content, max_chars)
    KEEP_PATTERNS = [
        (r"\[ERROR\].*", 300),      # 错误信息全保留
        (r"\[BLOCKED\].*", 200),    # 拦截信息
        (r"\[STOPPED\].*", 200),    # 停止信号
        (r"File .*written.*", 100), # 文件写入确认
        (r"已(替换|删除|重命名|创建).*", 100),  # 操作确认
        (r"OK:.*", 100),            # 操作成功
        (r"^\s*\d+.*passed.*", 200), # 测试结果
        (r"^\s*\d+.*failed.*", -1),  # 测试失败全保留 (max_chars=-1)
    ]

    def __init__(self, llm_adapter, flash_model: str = "deepseek-v4-flash"):
        self.llm = llm_adapter
        self.flash_model = flash_model
        self._system_prompt_hash = ""
        self._tool_schemas_hash = ""
        self._truncate_limit = 800  # default per-tool cap
        self._stats = {"tokens_saved": 0, "flash_uses": 0, "summaries": 0}

    def should_use_flash(self, task: str) -> bool:
        for kw in self.FLASH_TASK_KEYWORDS:
            if kw in task.lower():
                return True
        return len(task) < 100

    def optimize_system_prompt(self, system_prompt: str) -> str:
        h = hashlib.md5(system_prompt.encode()).hexdigest()
        if h == self._system_prompt_hash:
            return "[SYSTEM] (same as previous turn)"
        self._system_prompt_hash = h
        return system_prompt

    def optimize_tool_schemas(self, schemas: list[dict]) -> list[dict]:
        schemas_str = json.dumps(schemas, sort_keys=True, ensure_ascii=False)
        h = hashlib.md5(schemas_str.encode()).hexdigest()
        if h == self._tool_schemas_hash:
            return []
        self._tool_schemas_hash = h
        stripped = []
        for s in schemas:
            if "function" in s:
                fn = s["function"].copy()
                if "description" in fn:
                    fn["description"] = fn["description"][:100]
                stripped.append({"type": "function", "function": fn})
            else:
                stripped.append(s)
        return stripped

    def compress_tool_output(self, output: str) -> str:
        """智能压缩工具输出 — 保留关键信息，丢弃冗余

        优先级:
          1. 错误信息和关键信号 → 完整保留
          2. 结构化输出(文件列表/diff) → 去重 + 取头尾
          3. 冗长文本 → 按模式截断
        """
        if not output or len(output) <= 400:
            return output

        # 1. 检查是否匹配关键保留模式
        for pattern, limit in self.KEEP_PATTERNS:
            if re.search(pattern, output, re.IGNORECASE):
                if limit == -1:  # unlimited
                    return output
                return output[:limit]

        lines = output.split("\n")

        # 2. 目录/文件列表类 — 保头尾去中间
        if len(lines) > 30:
            list_patterns = sum(
                1 for l in lines[:20]
                if re.match(r"^\s*[-*+│├└┌┐╔═╚║\[]|^[a-zA-Z]:[\\/]|^\s+\d+:", l.strip())
            )
            if list_patterns >= 5:
                return self._trim_list(output, lines)

        # 3. 代码/Diff 类 — 保留首尾关键部分
        if len(output) > 2000:
            if any(kw in output[:200].lower() for kw in ["diff", "patch", "import ", "class ", "def ", "from "]):
                return self._trim_code(output)

        # 4. 通用截断
        if len(output) > self._truncate_limit:
            head = output[:self._truncate_limit // 2]
            tail = self._extract_tail_key_lines(output, 8)
            return f"{head}\n... [截断 {len(output) - len(head) - len(tail)} 字符] ...\n{tail}"

        return output

    @staticmethod
    def _trim_list(output: str, lines: list[str]) -> str:
        total = len(lines)
        if total <= 12:
            return output
        head = lines[:6]
        tail = lines[-6:]
        # 去重 — 过滤完全重复行
        seen = set()
        mid_unique = 0
        for l in lines[6:-6]:
            if l.strip() and l.strip() not in seen:
                seen.add(l.strip())
                mid_unique += 1
        return "\n".join(head) + f"\n... [{total - 12} lines, {mid_unique} unique items] ...\n" + "\n".join(tail)

    @staticmethod
    def _trim_code(output: str) -> str:
        lines = output.split("\n")
        if len(lines) <= 30:
            return output
        head = "\n".join(lines[:12])
        tail_lines = _extract_error_lines(output[-60:])
        if not tail_lines:
            tail_lines = lines[-8:]
        tail = "\n".join(tail_lines)
        return f"{head}\n... [{len(lines) - 12 - len(tail_lines)} lines] ...\n{tail}"

    @staticmethod
    def _extract_tail_key_lines(output: str, n: int) -> str:
        """从末尾提取关键行（错误、警告、结论）"""
        lines = output.split("\n")
        key_patterns = [
            r"\[?(ERROR|WARN|FAIL|PASS|STOPPED|BLOCKED)\]?",
            r"^\d+.*(passed|failed|error)",
            r"^(OK|FAILED|SUCCESS|DONE|完成|结果|总结)",
        ]
        result = []
        for line in reversed(lines):
            if any(re.search(p, line) for p in key_patterns):
                result.append(line)
                if len(result) >= n:
                    break
        if len(result) < n:
            result = lines[-n:] + result
        return "\n".join(reversed(result[-n:]))

    def compact_messages(self, messages: list[Message], max_output_tokens: int = 30000) -> list[Message]:
        """紧凑化消息列表"""
        if len(messages) <= 4:
            return messages

        result = []
        tool_batch = []
        saved = 0

        for msg in messages:
            if msg.role == "tool" and msg.content:
                compressed = self.compress_tool_output(msg.content)
                saved += len(msg.content or "") - len(compressed)
                tool_batch.append(Message(
                    role="tool", content=compressed,
                    tool_call_id=msg.tool_call_id or "",
                    tool_name=msg.tool_name or "",
                ))
            else:
                if tool_batch:
                    # 合并相邻工具结果为一条摘要
                    if len(tool_batch) >= 3:
                        summary = self._batch_tool_summary(tool_batch)
                        result.append(Message(role="tool", content=summary, tool_name="batch"))
                    else:
                        result.extend(tool_batch)
                    tool_batch = []
                result.append(msg)

            total_chars = sum(len(r.content or "") for r in result)
            if total_chars / 3.5 > max_output_tokens:
                result = [r for r in result if r.role in ("system",)]
                result += [m for m in messages if m.role == "user"][:1]
                result += [m for m in messages if m.role != "system"][-5:]
                break

        if tool_batch:
            result.extend(tool_batch)

        self._stats["tokens_saved"] += int(saved / 3.5)
        return result

    def summarize_work(self, messages: list[Message], task: str) -> str:
        """将已完成工作压缩为结构化摘要

        格式: [TASK] → [FILES] → [STATUS] → [NEXT]
        """
        files_touched = []
        errors = []
        last_result = ""
        step_count = 0

        for msg in messages:
            if msg.role == "user":
                step_count += 1
            elif msg.role == "tool" and msg.content:
                ct = msg.content
                # 提取文件名
                for m in re.finditer(r"(?:File written:|已替换|修复|已删除|OK:)\s*(\S+)", ct):
                    files_touched.append(m.group(1))
                # 提取错误
                if "[ERROR]" in ct or "[STOPPED]" in ct:
                    errors.append(ct[:120])
                last_result = ct

        files_uniq = list(dict.fromkeys(files_touched))[-10:]
        summary = f"[进度] 任务: {task[:120]}\n"
        summary += f"  步骤: {step_count}\n"
        if files_uniq:
            summary += f"  文件: {', '.join(files_uniq)}\n"
        if errors:
            summary += f"  错误: {errors[-1][:100]}\n"
        if last_result and not errors:
            summary += f"  最新: {last_result[:200]}\n"

        self._stats["summaries"] += 1
        self._stats["tokens_saved"] += int(len("\n".join(m.content or "" for m in messages[-20:])) / 3.5 - len(summary) / 3.5)
        return summary

    @staticmethod
    def _batch_tool_summary(batch: list[Message]) -> str:
        tools = list(set(m.tool_name for m in batch if m.tool_name))
        outputs = [m.content for m in batch if m.content]
        errors = sum(1 for o in outputs if o and ("[ERROR]" in o or "failed" in o.lower()))
        return f"[批量] {len(batch)} 次工具调用 → {', '.join(tools)} | {errors} 失败 | {len(outputs) - errors} 成功"

    def get_stats(self) -> dict:
        return self._stats


def _extract_error_lines(text: str) -> list[str]:
    """从文本末尾提取错误相关行"""
    lines = text.split("\n")
    result = []
    for line in reversed(lines):
        lower = line.lower()
        if any(kw in lower for kw in ["error", "fail", "traceback", "exception", "assert", "warning"]):
            result.insert(0, line)
            if len(result) >= 6:
                break
    return result
