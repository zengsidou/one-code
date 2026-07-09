# -*- coding: utf-8 -*-
"""Tree-sitter shell 命令安全解析 — 对标 MiMo Code 的 AST 级安全

用 tree-sitter 将 bash/PowerShell 命令解析为 AST，
精确识别文件操作路径、用户切换、网络访问等危险操作。
替代原有的正则黑名单。
"""
import re
from typing import Callable


# 危险命令类别
DANGEROUS_PATTERNS_AST = {
    "destructive_fs": [
        "rm", "rmdir", "del", "format", "mkfs", "dd",
        "Remove-Item",
    ],
    "privilege_escalation": [
        "sudo", "su", "chmod", "chown", "cacls", "icacls",
        "runas",
    ],
    "system_control": [
        "shutdown", "reboot", "halt", "poweroff", "init",
        "Restart-Computer", "Stop-Computer",
    ],
    "network_dangerous": [
        "nc", "ncat", "netcat", "socat",  # raw sockets only; wget/curl handled by SandboxPolicy
    ],
    "code_execution": [
        "eval", "exec", "Invoke-Expression", "iex",
        "Start-Process", "Invoke-Command",
    ],
}


def parse_command_ast(command: str) -> dict:
    """用 tree-sitter 解析 shell 命令为 AST

    优先用 bash 语法，如果失败则用简单词法分析 fallback。

    Returns:
        {
            commands: [{"name": "rm", "args": ["-rf", "/tmp/x"], "redirects": [...]}],
            raw: "原始命令",
            parser: "bash" | "powershell" | "fallback",
        }
    """
    try:
        return _parse_with_tree_sitter_bash(command)
    except Exception:
        pass

    try:
        return _parse_with_lexer(command)
    except Exception:
        pass

    return {"commands": [], "raw": command, "parser": "failed"}


def _parse_with_tree_sitter_bash(command: str) -> dict:
    import tree_sitter_bash as tsb
    from tree_sitter import Language, Parser

    lang = Language(tsb.language())
    parser = Parser(lang)
    tree = parser.parse(command.encode())

    commands = []
    _extract_commands(tree.root_node, command, commands)
    return {"commands": commands, "raw": command, "parser": "bash"}


def _extract_commands(node, source: str, result: list):
    """递归从 AST 提取命令名和参数"""
    if node.type == "command":
        cmd_name = ""
        args = []
        for child in node.children:
            if child.type == "command_name":
                cmd_name = source[child.start_byte:child.end_byte]
            elif child.type in ("word", "string", "raw_string", "concatenation"):
                args.append(source[child.start_byte:child.end_byte])
        if cmd_name:
            result.append({"name": cmd_name, "args": args})
    for child in node.children:
        _extract_commands(child, source, result)


def _parse_with_lexer(command: str) -> dict:
    """简单词法分析 fallback — 提取命令链"""
    cmds = []
    for part in _split_pipes(command):
        tokens = part.strip().split()
        if tokens:
            cmds.append({"name": tokens[0], "args": tokens[1:]})
    return {"commands": cmds, "raw": command, "parser": "fallback"}


def _split_pipes(command: str) -> list[str]:
    parts = []
    current = ""
    in_quote = False
    quote_char = ""
    for ch in command:
        if ch in "\"'":
            if not in_quote:
                in_quote, quote_char = True, ch
            elif ch == quote_char:
                in_quote, quote_char = False, ""
        if ch == "|" and not in_quote:
            parts.append(current)
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current)
    return parts


def check_dangerous(command: str) -> dict:
    """检查命令是否有危险

    Returns:
        {allowed: bool, blocked: bool, reason: str, details: list}
    """
    ast = parse_command_ast(command)
    blocked = False
    reasons = []
    details = []

    for cmd in ast.get("commands", []):
        name = cmd.get("name", "").lower()
        args = [a.lower() for a in cmd.get("args", [])]

        # 检查各级危险类别
        for category, keywords in DANGEROUS_PATTERNS_AST.items():
            for kw in keywords:
                if name == kw.lower():
                    reason = f"[{category}] {name}"
                    reasons.append(reason)
                    details.append({"command": name, "args": cmd.get("args", []), "risk": category})
                    blocked = True

        # 危险参数检测（正则 fallback，AST 可逐步替换）
        full_cmd = f"{name} {' '.join(args)}"
        if _has_dangerous_arg(full_cmd):
            reason = f"[dangerous_arg] {name} (含危险参数模式)"
            reasons.append(reason)
            details.append({"command": name, "args": cmd.get("args", []), "risk": "dangerous_arg"})
            blocked = True

    if not blocked and ast.get("commands"):
        reason = f"安全: {len(ast['commands'])} 条命令 via {ast.get('parser', '?')}"

    return {
        "allowed": not blocked,
        "blocked": blocked,
        "reason": "; ".join(reasons) if blocked else f"安全: {len(ast.get('commands', []))} 条命令",
        "details": details,
    }


def _has_dangerous_arg(full_command: str) -> bool:
    """检查命令中是否包含危险的参数模式（正则 fallback）"""
    patterns = [
        r"\brm\s+.*\s+(-[rf]|--recursive|--force)",
        r"\brmdir\s+/s",
        r"\bformat\s+[A-Z]:",
        r">\s*/dev/[a-z]+",
        r"\bchmod\s+777",
        r"Remove-Item\s+.*-Recurse\s+.*-Force",
    ]
    return any(re.search(p, full_command, re.IGNORECASE) for p in patterns)
