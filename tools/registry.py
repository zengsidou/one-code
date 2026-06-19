# -*- coding: utf-8 -*-
"""工具注册引擎 — 装饰器注册 + 自动 Schema 生成 + 危险命令拦截"""
import inspect
import re
import subprocess
import platform
from typing import Any, Callable

from .schema import generate_tool_schema, python_type_to_json_type

DANGEROUS_PATTERNS = [
    r"\brm\s+(-[rf]|--recursive|--force)",
    r"\brmdir\s+/s",
    r"\bdel\s+/[FSQ]",
    r"\bformat\s+[A-Z]:",
    r"\bmkfs\.",
    r"\bdd\s+if=",
    r">\s*/dev/",
    r"\b(sudo|su)\s",
    r"\bchmod\s+777",
    r"\b(shutdown|reboot|halt|poweroff)\b",
    r":\(\)\s*\{",
    r"Remove-Item\s+.*-Recurse\s+.*-Force",
]


class ToolRegistry:
    def __init__(self, safe_mode: bool = True):
        self._tools: dict[str, Callable] = {}
        self._tool_metadata: dict[str, dict] = {}
        self.safe_mode = safe_mode

    def register(self, name: str, description: str):
        def decorator(func: Callable):
            self._tools[name] = func
            hints = getattr(func, "__annotations__", {})
            sig = inspect.signature(func)
            params = {}
            for pname, param in sig.parameters.items():
                if pname in ("self", "cls"):
                    continue
                ptype = hints.get(pname, str)
                pdesc = f"Parameter: {pname}"
                params[pname] = (ptype, pdesc)
            schema = generate_tool_schema(name, description, params)
            self._tool_metadata[name] = {"schema": schema, "description": description}
            return func
        return decorator

    def get_schemas(self) -> list[dict]:
        return [m["schema"] for m in self._tool_metadata.values()]

    def get_tools_description(self) -> str:
        lines = []
        for name, meta in self._tool_metadata.items():
            lines.append(f"- **{name}**: {meta['description']}")
        return "\n".join(lines)

    def execute(self, name: str, arguments: dict) -> str:
        func = self._tools.get(name)
        if func is None:
            return f"[ERROR] Unknown tool: {name}"
        try:
            result = func(**arguments)
            output = str(result) if result is not None else "Tool executed successfully."
            if self.safe_mode:
                danger = self._intercept_dangerous(output)
                if danger:
                    return danger
            return output
        except Exception as e:
            return f"[ERROR] Tool '{name}' failed: {e}"

    def _intercept_dangerous(self, command: str) -> str | None:
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return f"[BLOCKED] Dangerous pattern detected: {pattern}"
        return None

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())


def run_shell(command: str, timeout: int = 30) -> str:
    shell_cmd = ["powershell", "-Command", command] if platform.system() == "Windows" else ["bash", "-c", command]
    try:
        result = subprocess.run(shell_cmd, capture_output=True, timeout=timeout)
        raw = result.stdout or result.stderr or b"(no output)"
        output = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
        return output[:4000]
    except subprocess.TimeoutExpired:
        return f"[ERROR] Command timed out after {timeout}s"
    except Exception as e:
        return f"[ERROR] {e}"
