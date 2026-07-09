# -*- coding: utf-8 -*-
"""系统约束硬化 — 将 system prompt 中的 17 条硬约束转化为运行时检查

不再依赖 prompt 建议，而是在代码层面围堵常见 Agent 错误模式。
"""
import re
from agent.models import Message


class ConstraintEnforcer:
    """运行时约束执行器

    在 Agent 执行过程中插入检查点，拦截违反约束的行为。
    """

    def __init__(self):
        self._edit_count: dict[str, int] = {}
        self._tool_call_history: list[dict] = []

    def before_tool_call(self, tool_name: str, arguments: dict, messages: list[Message]) -> str | None:
        """工具调用前置检查 — 返回错误消息则拦截，返回 None 则放行"""
        check = getattr(self, f"_check_{tool_name}", None)
        if check:
            return check(arguments, messages)
        return None

    def after_tool_call(self, tool_name: str, arguments: dict, result: str, messages: list[Message]) -> str | None:
        """工具调用后置建议 — 返回提示注入到下一轮上下文"""
        self._tool_call_history.append({"tool": tool_name, "args": arguments, "result": result[:200]})
        if len(self._tool_call_history) > 20:
            self._tool_call_history.pop(0)

        check = getattr(self, f"_after_{tool_name}", None)
        if check:
            return check(arguments, result, messages)
        return None

    # ━━ 前置检查 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _check_edit_file(self, args: dict, _) -> str | None:
        """约束: 优先用 edit_file，不要用 write_file 重写整个文件"""
        path = args.get("path", "")
        old = args.get("old_string", "")
        # 检测是否在用 edit_file 做整文件替换（old_string 是文件开头）
        if old and len(old) > 500:
            return "[约束提醒] 建议用更精确的 old_string 做精准编辑，而非大段替换。请缩小匹配范围。"
        return None

    def _check_write_file(self, args: dict, messages: list[Message]) -> str | None:
        """约束: write_file 仅在创建新文件时使用"""
        import os
        path = args.get("path", "")
        if os.path.exists(path):
            return f"[约束违规] 文件 {path} 已存在。请使用 edit_file 做精准修改，不要用 write_file 覆盖已有文件。"
        return None

    # ━━ 后置建议 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _after_edit_file(self, args: dict, result: str, _) -> str | None:
        if "OK" in result:
            return None
        if "未找到" in result or "ERROR" in result:
            return "[建议] 编辑匹配失败。请用 read_file 确认当前文件内容后再重试。不要凭记忆猜测代码内容。"
        return None

    def _after_run_shell(self, args: dict, result: str, _) -> str | None:
        if "[ERROR]" in result:
            return "[约束] 命令执行失败。请仔细阅读错误输出，分析根因，换个方法重试。不要重复相同操作。"
        return None

    def _after_read_file(self, _args, result, __) -> str | None:
        if "[ERROR]" in result and "not found" in result.lower():
            return "[建议] 文件不存在。请用 grep 或 glob 搜索正确的文件路径后再读取。"
        return None

    # ━━ 循环内检查 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def check_repeated_same_tool(self, tool_name: str, args: dict, history_limit: int = 3) -> bool:
        """检测是否连续 3 次用相同的工具和参数 — 回路检测增强"""
        sig = f"{tool_name}:{_safe_args(args)}"
        recent = [f"{h['tool']}:{_safe_args(h['args'])}" for h in self._tool_call_history[-history_limit:]]
        return recent.count(sig) >= history_limit

    def get_constraint_hint(self) -> str:
        """生成当前上下文的约束提示，注入到 system prompt 末尾"""
        hints = []
        edit_count = sum(1 for h in self._tool_call_history if h["tool"] == "edit_file")
        if edit_count > 5:
            hints.append("已进行多次编辑，请在完成所有修改后运行验证。")
        shell_errors = sum(1 for h in self._tool_call_history
                          if h["tool"] == "run_shell" and "[ERROR]" in str(h.get("result", "")))
        if shell_errors >= 2:
            hints.append("多次命令失败，请检查环境配置后再继续。不要盲目重试。")
        return "\n".join(hints) if hints else ""


def _safe_args(args: dict) -> str:
    return str({k: args[k][:40] for k in sorted(args.keys()) if k not in ("content", "old_string", "new_string")})
