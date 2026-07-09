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
    def __init__(self, safe_mode: bool = True, permissions=None, audit=None):
        self._tools: dict[str, Callable] = {}
        self._tool_metadata: dict[str, dict] = {}
        self._tool_aliases: dict[str, str] = {}
        self.safe_mode = safe_mode
        self.permissions = permissions
        self.audit = audit

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

    def add_alias(self, alias: str, target: str):
        self._tool_aliases[alias] = target

    def get_tools_description(self) -> str:
        lines = []
        for name, meta in self._tool_metadata.items():
            lines.append(f"- **{name}**: {meta['description']}")
        return "\n".join(lines)

    def execute(self, name: str, arguments: dict) -> str:
        import inspect

        name = self._tool_aliases.get(name, name)
        func = self._tools.get(name)
        if func is None:
            return f"[ERROR] Unknown tool: {name}"

        # ━━━ 参数类型校验与强制转换 ━━━
        try:
            sig = inspect.signature(func)
            cleaned = {}
            for pname, param in sig.parameters.items():
                val = arguments.get(pname, param.default if param.default is not inspect.Parameter.empty else None)
                if val is inspect.Parameter.empty or val is None:
                    cleaned[pname] = val
                    continue
                anno = param.annotation
                if anno is not inspect.Parameter.empty:
                    val = self._coerce_arg(val, anno, pname)
                cleaned[pname] = val
            arguments = cleaned
        except Exception:
            pass  # 签名解析失败，用原始参数

        # ━━━ 权限检查 ━━━
        if self.permissions:
            allowed, reason = self.permissions.check(name, arguments)
            if self.audit:
                risk = self.permissions.get_risk(name).value
                self.audit.record(
                    agent_id="default", action="tool_call",
                    tool=name, details=str(arguments)[:200],
                    risk_level=risk, allowed=allowed,
                )
            if not allowed:
                return f"[BLOCKED] {reason}"

        try:
            if self.safe_mode:
                danger = self._intercept_dangerous(arguments)
                if danger:
                    return danger
            result = func(**arguments)
            output = str(result) if result is not None else "Tool executed successfully."
            if len(output) > 4000:
                import tempfile, os
                f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
                f.write(output)
                path = f.name
                f.close()
                head = output[:2000]
                tail = output[-1000:]
                output = f"{head}\n... [{len(output)-3000} chars truncated, full output at {path}]\n{tail}"
                output += f"\n[Use read_file offset to read the full file: {path}]"
            return output
        except Exception as e:
            return f"[ERROR] Tool '{name}' failed: {e}"

    @staticmethod
    def _coerce_arg(val, expected_type, param_name: str):
        origin = getattr(expected_type, "__origin__", None)
        if origin is not None:
            args = getattr(expected_type, "__args__", ())
            if origin in (list, tuple):
                if isinstance(val, str):
                    return [val]
                return val
            return val
        if expected_type in (str, int, float, bool):
            if isinstance(val, expected_type):
                return val
            if expected_type is str and isinstance(val, dict):
                import json
                try:
                    return json.dumps(val, ensure_ascii=False)
                except Exception:
                    return str(val)
            if expected_type is str and not isinstance(val, str):
                return str(val)
            if expected_type is int and isinstance(val, str) and val.isdigit():
                return int(val)
            if expected_type is float and isinstance(val, (int, str)):
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
        return val

    def _intercept_dangerous(self, arguments: dict) -> str | None:
        for val in arguments.values():
            if isinstance(val, str):
                for pattern in DANGEROUS_PATTERNS:
                    if re.search(pattern, val, re.IGNORECASE):
                        return f"[BLOCKED] Dangerous pattern detected: {pattern}"
        return None

    def subset(self, names: list[str]) -> "ToolRegistry":
        """创建仅包含指定工具的子注册表"""
        sub = ToolRegistry(safe_mode=False)
        for name in names:
            if name in self._tools:
                sub._tools[name] = self._tools[name]
                if name in self._tool_metadata:
                    sub._tool_metadata[name] = self._tool_metadata[name]
        return sub

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
