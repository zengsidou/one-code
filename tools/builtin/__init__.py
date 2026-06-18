# -*- coding: utf-8 -*-
"""内置工具集"""
import os
from tools.registry import run_shell


def register_builtin_tools(registry, sandbox=None) -> None:
    @registry.register("read_file", "读取指定路径的文件内容")
    def read_file(path: str) -> str:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                content = f.read()
            return content[:5000] if len(content) > 5000 else content
        except FileNotFoundError:
            return f"[ERROR] File not found: {path}"
        except PermissionError:
            return f"[ERROR] Permission denied: {path}"
        except Exception as e:
            return f"[ERROR] Read file failed: {e}"

    @registry.register("write_file", "写入内容到指定路径的文件")
    def write_file(path: str, content: str) -> str:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"File written: {path} ({len(content)} chars)"
        except PermissionError:
            return f"[ERROR] Permission denied: {path}"
        except Exception as e:
            return f"[ERROR] Write file failed: {e}"

    @registry.register("list_dir", "列出目录内容，默认为当前目录")
    def list_dir(path: str = ".") -> str:
        try:
            entries = os.listdir(path)
            if not entries:
                return f"Directory '{path}' is empty."
            lines = [f"  {'D' if os.path.isdir(os.path.join(path, e)) else 'F'}  {e}" for e in sorted(entries)]
            return f"Contents of '{path}':\n" + "\n".join(lines[:100])
        except FileNotFoundError:
            return f"[ERROR] Directory not found: {path}"
        except PermissionError:
            return f"[ERROR] Permission denied: {path}"
        except Exception as e:
            return f"[ERROR] List dir failed: {e}"

    @registry.register("run_shell", "执行 Shell 命令并返回输出，超时默认 30 秒")
    def _run_shell(command: str, timeout: int = 30) -> str:
        if sandbox:
            result = sandbox.execute(command)
            if result["ok"]:
                out = result["output"]
                if result["error"]:
                    out += "\n[stderr] " + result["error"]
                return f"{out}\n[done in {result['duration_ms']}ms]"
            else:
                return f"[{result['blocked_by'] or 'ERROR'}] {result['error']}"
        return run_shell(command, timeout)

    @registry.register("search_web", "网络搜索占位 — 实际可接入 Tavily/SerpAPI")
    def search_web(query: str) -> str:
        return f"[PLACEHOLDER] Web search for '{query}': not implemented. Connect Tavily or SerpAPI."

    @registry.register("calculate", "执行数学计算，支持加减乘除、幂运算、三角函数等")
    def calculate(expression: str) -> str:
        import math
        allowed = {"__builtins__": {}, **{k: getattr(math, k) for k in dir(math) if not k.startswith("_")}}
        try:
            result = eval(expression, allowed)
            return f"计算结果: {expression} = {result}"
        except Exception as e:
            return f"[ERROR] 计算失败: {e}"
