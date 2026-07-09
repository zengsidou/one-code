# -*- coding: utf-8 -*-
"""内置工具集"""
import os
from tools.registry import run_shell


def register_builtin_tools(registry, sandbox=None, llm=None) -> None:
    import os
    import sys
    import re
    import shutil
    import subprocess
    from datetime import datetime

    @registry.register("read_file", "读取文件内容，支持 offset(起始行，1开始)和 limit(行数上限)")
    def read_file(path: str, offset: int = 0, limit: int = 2000) -> str:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            total = len(lines)
            if offset > 0:
                lines = lines[offset - 1:]
            if limit > 0:
                lines = lines[:limit]
            content = "".join(lines)
            if total > offset + len(lines):
                content += f"\n... (共 {total} 行，已显示第 {max(offset,1)}-{min(offset+len(lines)-1,total)} 行)"
            return content[:8000] if len(content) > 8000 else content
        except FileNotFoundError:
            return f"[ERROR] File not found: {path}"
        except PermissionError:
            return f"[ERROR] Permission denied: {path}"
        except Exception as e:
            return f"[ERROR] Read file failed: {e}"

    @registry.register("grep", "在目录中搜索匹配正则表达式的内容，返回文件路径和行号。支持 include 参数过滤文件名(如 '*.py')")
    def grep(pattern: str, path: str = ".", include: str = "*") -> str:
        import re
        import fnmatch
        from pathlib import Path

        try:
            compiled = re.compile(pattern)
        except re.error as e:
            return f"[ERROR] 正则表达式错误: {e}"

        results = []
        try:
            base = Path(path).resolve()
            for f in base.rglob("*"):
                if not f.is_file():
                    continue
                if include != "*" and not fnmatch.fnmatch(f.name, include):
                    continue
                if f.suffix.lower() in (".exe", ".dll", ".pyd", ".pyc", ".so", ".o", ".obj", ".bin", ".png", ".jpg", ".zip", ".tar", ".gz"):
                    continue
                try:
                    if f.stat().st_size > 1024 * 1024:
                        continue
                    with open(f, encoding="utf-8", errors="replace") as fh:
                        for i, line in enumerate(fh, 1):
                            if compiled.search(line):
                                results.append(f"{f.relative_to(base)}:{i}: {line.rstrip()[:200]}")
                                if len(results) >= 100:
                                    break
                except Exception:
                    continue
                if len(results) >= 100:
                    break
        except Exception as e:
            return f"[ERROR] 搜索失败: {e}"

        if not results:
            return f"未找到匹配 '{pattern}' 的内容 (路径: {path}, 文件过滤: {include})"
        suffix = f"\n... (超过 100 条结果，已截断)" if len(results) >= 100 else ""
        return f"找到 {len(results)} 处匹配:\n" + "\n".join(results) + suffix

    @registry.register("glob", "按 glob 模式匹配文件，如 '**/*.py' 或 '*.json'")
    def glob_files(pattern: str, path: str = ".") -> str:
        from pathlib import Path
        if not isinstance(pattern, str):
            return f"[ERROR] glob 参数错误: pattern 应为字符串，实际为 {type(pattern).__name__}"
        if not isinstance(path, str):
            return f"[ERROR] glob 参数错误: path 应为字符串，实际为 {type(path).__name__}"
        try:
            matches = sorted(Path(path).rglob(pattern))
            if not matches:
                return f"未找到匹配 '{pattern}' 的文件 (路径: {path})"
            items = []
            for m in matches[:100]:
                items.append(str(m))
            suffix = f"\n... (共 {len(matches)} 个结果，已截断至前 100 个)" if len(matches) > 100 else ""
            return f"找到 {len(matches)} 个匹配:\n" + "\n".join(items) + suffix
        except Exception as e:
            return f"[ERROR] glob 失败: {e}"

    @registry.register("edit_file", "智能编辑文件。old_string 尽量精确匹配；找不到时会尝试缩进修正、空白标准化等多策略模糊匹配")
    def edit_file(path: str, old_string: str, new_string: str) -> str:
        from tools.edit_smart import edit_smart
        return edit_smart(path, old_string, new_string)

    @registry.register("delete_file", "删除文件（自动备份到 .trash/ 目录，可恢复）")
    def delete_file(path: str) -> str:
        if not os.path.exists(path):
            return f"[ERROR] 文件不存在: {path}"
        trash_dir = ".trash"
        os.makedirs(trash_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = os.path.basename(path)
        backup = os.path.join(trash_dir, f"{ts}_{name}")
        try:
            os.rename(path, backup)
            return f"已删除 {path} (备份到 {backup})"
        except OSError:
            shutil.copy2(path, backup)
            os.remove(path)
            return f"已删除 {path} (跨盘备份到 {backup})"

    @registry.register("rename_file", "重命名/移动文件或目录")
    def rename_file(old_path: str, new_path: str) -> str:
        if not os.path.exists(old_path):
            return f"[ERROR] 源文件不存在: {old_path}"
        os.makedirs(os.path.dirname(new_path) or ".", exist_ok=True)
        try:
            os.rename(old_path, new_path)
            return f"已重命名 {old_path} → {new_path}"
        except OSError as e:
            return f"[ERROR] 重命名失败: {e}"

    @registry.register("diff_file", "查看文件与 git HEAD 或暂存区的差异")
    def diff_file(path: str, staged: bool = False) -> str:
        cmd = ["git", "diff"]
        if staged:
            cmd.append("--staged")
        cmd.extend(["--", path])
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            output = (result.stdout or result.stderr or "(无差异)")[:3000]
            return output if output.strip() else "(无差异)"
        except Exception as e:
            return f"[ERROR] diff 失败: {e}"

    @registry.register("git", "执行 Git 操作: status/diff/log/add/commit。commit 需要 message 参数")
    def git(action: str, repo: str = ".", message: str = "") -> str:
        """安全 git 操作"""
        # Only allow safe operations
        allowed = ["status", "diff", "log", "add", "commit", "branch", "stash"]
        if action not in allowed:
            return f"[ERROR] 不安全的 git 操作: {action}。允许: {', '.join(allowed)}"
        cmd = ["git", "-C", repo, action]
        if action == "commit" and message:
            cmd.extend(["-m", message])
        elif action == "diff":
            cmd.append("--stat")
        elif action == "log":
            cmd.extend(["--oneline", "-10"])
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            raw = result.stdout or result.stderr
            return raw[:3000] if raw.strip() else "(无输出)"
        except Exception as e:
            return f"[ERROR] git {action} 失败: {e}"

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

    @registry.register("search_web", "Bing 网络搜索（免费，国内可用，超时 10 秒）")
    def search_web(query: str) -> str:
        import re, os, threading

        # DuckDuckGo 备用（优先避免被封 IP）
        try:
            from ddgs import DDGS
            results_holder = []
            def _ddg():
                try:
                    results_holder.extend(list(DDGS().text(query, max_results=5)))
                except Exception:
                    pass
            t = threading.Thread(target=_ddg, daemon=True)
            t.start()
            t.join(timeout=8)
            if results_holder:
                lines = [f"搜索 '{query}':"]
                for i, r in enumerate(results_holder[:5], 1):
                    lines.append(f"{i}. {r.get('title','')[:80]}\n   {r.get('href','')}\n   {r.get('body','')[:200]}")
                return "\n".join(lines)
        except ImportError:
            pass

        # Tavily 备用
        tavily_key = os.environ.get("TAVILY_API_KEY") or os.environ.get("tavily_api_key")
        if tavily_key:
            try:
                import httpx
                resp = httpx.post("https://api.tavily.com/search", json={"api_key": tavily_key, "query": query, "max_results": 5}, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get("results", [])
                    if results:
                        lines = [f"Tavily 搜索结果:"]
                        for i, r in enumerate(results[:5], 1):
                            lines.append(f"{i}. {r.get('title','')}\n   {r.get('url','')}\n   {r.get('content','')[:200]}")
                        return "\n".join(lines)
            except Exception:
                pass

        return f"搜索 '{query}' 无结果。DuckDuckGo/Bing/Tavily 均不可用。"

    @registry.register("fetch_url", "抓取指定 URL 的完整网页内容（Markdown 格式）。配合 search_web 使用：搜出结果→选 URL→fetch 全文。format 可选 markdown/text/html")
    def fetch_url(url: str, format: str = "markdown") -> str:
        import re, httpx

        try:
            r = httpx.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; OneCode/1.0)"},
                follow_redirects=True,
                timeout=15,
            )
            if r.status_code != 200:
                return f"[ERROR] HTTP {r.status_code}"

            content_type = r.headers.get("content-type", "")
            charset = "utf-8"
            m = re.search(r'charset=([^\s;]+)', content_type)
            if m:
                charset = m.group(1)
            html = r.content.decode(charset, errors="replace")

            if format == "html":
                return html[:8000]

            if format == "text":
                text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"\s+", " ", text).strip()
                return text[:6000]

            # Default: markdown
            try:
                import html2text
                h = html2text.HTML2Text()
                h.ignore_links = False
                h.ignore_images = True
                h.body_width = 0
                h.ignore_emphasis = False
                md = h.handle(html)
                return md[:6000]
            except ImportError:
                text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"\s+", " ", text).strip()
                return f"(html2text 未安装，返回纯文本)\n{text[:6000]}"

        except httpx.ConnectTimeout:
            return f"[ERROR] 连接 {url} 超时"
        except Exception as e:
            return f"[ERROR] {e}"

    @registry.register("calculate", "安全执行数学表达式。支持 +-*/**%// 和 math 模块函数")
    def calculate(expression: str) -> str:
        import ast
        import math
        try:
            node = ast.parse(expression.strip(), mode="eval")
            allowed = {"__builtins__": {}, **{k: getattr(math, k) for k in dir(math) if not k.startswith("_")}}
            code = compile(node, "<calc>", "eval")
            result = eval(code, allowed)
            return f"计算结果: {expression} = {result}"
        except Exception as e:
            return f"[ERROR] 计算失败: {e}"

    @registry.register("delegate_task", "委派子 Agent 执行独立子任务，可指定工具白名单。参数: task=任务描述, tools=允许的工具名列表(逗号分隔), 可选 max_steps=最大步骤数")
    def delegate_task(task: str, tools: str = "", max_steps: int = 5) -> str:
        """Spawn a SubAgent to handle a specific subtask"""
        from tools.registry import ToolRegistry
        from agent.subagent import SubAgent

        tool_names = [t.strip() for t in tools.split(",") if t.strip()] if tools else []
        sub_registry = ToolRegistry(safe_mode=False)

        if not tool_names:
            readable_tools = ["read_file", "list_dir", "calculate"]
            tool_names = readable_tools

        for name in tool_names:
            if name in registry._tools:
                # Copy existing tool to sub-registry
                func = registry._tools[name]
                meta = registry._tool_metadata.get(name, {})
                sub_registry._tools[name] = func
                sub_registry._tool_metadata[name] = meta

        if not sub_registry._tools:
            return "[ERROR] No valid tools available for sub-agent"

        sub = SubAgent(
            llm=llm,
            registry=sub_registry,
            prompt="你是子任务执行 Agent。用赋予的工具完成任务，然后给出简洁中文结果。",
            max_steps=max_steps,
        )
        try:
            result = sub.run(task)
            return f"[SubAgent完成] {result}"
        except Exception as e:
            return f"[ERROR] SubAgent 执行失败: {e}"

    @registry.register("lsp", "代码智能: action=def/jump/refs/references/hover/diag/diagnostics/impact/analyze")
    def lsp(file: str, action: str = "def", line: int = 1, col: int = 1) -> str:
        from tools.lsp_client import get_lsp
        am = {"def": "go_to_definition", "jump": "go_to_definition",
              "refs": "find_references", "references": "find_references",
              "hover": "hover", "diag": "diagnostics", "diagnostics": "diagnostics",
              "impact": "impact_analysis", "analyze": "impact_analysis"}
        method = am.get(action)
        if not method:
            return f"[ERROR] unknown action: {action}, use def/refs/hover/diag/impact"
        try:
            c = get_lsp(); fn = getattr(c, method)
            if method in ("hover",):
                return str(fn(file, line, col))[:1000]
            if method == "diagnostics":
                diags = fn(file)
                if not diags: return f"{file}: no issues"
                ls = []; sv = {1: "ERR", 2: "WARN", 3: "INFO", 4: "HINT"}
                for d in diags[:15]:
                    ls.append(f"  {file}:{d['line']}:{d['col']} [{sv.get(d['severity'],'?')}] {d['message']}")
                return "\n".join([f"{file}: {len(diags)} diagnostics"] + ls)
            if method == "impact_analysis":
                a = fn(file); syms = a.get("symbols", [])
                if not syms: return f"{file}: no cross-file refs"
                ls = [f"{file}: {len(syms)} symbols referenced:"]
                for s in syms[:8]:
                    rf = list(set(r["file"] for r in s["refs"]))
                    ls.append(f"  {s['name']}(:{s['line']}) -> {len(rf)} files")
                    for f_ in rf[:3]: ls.append(f"    - {f_}")
                return "\n".join(ls)
            data = fn(file, line, col)
            if not data: return f"no results for {file}:{line}:{col}"
            items = [f"{r['file']}:{r['line']}:{r['col']}" for r in data[:10]]
            return f"{len(data)} results:\n" + "\n".join(items)
        except Exception as e:
            return f"[ERROR] LSP: {e}"
    # ━━━ 工具别名（LLM 可能用不同名称调用）━━━
    registry.add_alias("search_content", "grep")
    registry.add_alias("search_file", "grep")
    registry.add_alias("find_files", "glob")
    registry.add_alias("read", "read_file")
    registry.add_alias("write", "write_file")
    registry.add_alias("edit", "edit_file")
    registry.add_alias("execute", "run_shell")
    registry.add_alias("bash", "run_shell")
    registry.add_alias("shell", "run_shell")
    registry.add_alias("delete", "delete_file")
    registry.add_alias("rm", "delete_file")
    registry.add_alias("rename", "rename_file")
    registry.add_alias("mv", "rename_file")
    registry.add_alias("lsp_def", "lsp")
    registry.add_alias("lsp_refs", "lsp")
    registry.add_alias("goto_def", "lsp")
    registry.add_alias("find_refs", "lsp")
